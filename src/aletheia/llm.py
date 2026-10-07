"""Claim normalizer.

Maps raw comments from a governance thread to canonical position
indices, without knowing how many positions exist a priori.

Three backends are provided behind a single interface:

- ``MockClaimNormalizer`` — deterministic keyword-based clustering,
  used in tests and CI. No network, no API keys.
- ``OpenAIClaimNormalizer`` — calls OpenAI with ``temperature=0``.
- ``GeminiClaimNormalizer`` — calls Google Gemini with ``temperature=0``.

The LLM backends cache every response on disk (keyed by a hash of the
prompt) so that re-running an evaluation with the same inputs costs
zero API credits after the first run.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from aletheia.config import get_default_normalizer, get_gemini_key, get_openai_key

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class ClaimNormalizer(Protocol):
    """Assigns each comment to a canonical position index.

    The indices are arbitrary labels local to the call: ``[0, 1, 0, 2]``
    means the first and third comments share a position, the second and
    fourth are distinct from each other and from the first.
    """

    def normalize(self, comments: list[str]) -> list[int]:
        """Return a list of cluster indices, one per comment."""
        ...


# ---------------------------------------------------------------------------
# Mock normalizer (deterministic, no network)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on",
        "at", "for", "with", "by", "from", "as", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "should", "could", "can", "may", "might", "must",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "its", "our", "their", "this",
        "that", "these", "those", "so", "not", "no", "yes", "very", "just",
        "about", "into", "than", "then", "there", "here", "when", "where",
        "why", "how", "what", "which", "who", "whom", "whose", "all", "any",
        "some", "many", "much", "more", "most", "other", "another", "such",
        "only", "also", "too", "now", "still", "already", "yet", "again",
    }
)


def _tokenize(text: str) -> set[str]:
    """Lowercase alphabetic tokens, minus stopwords and single letters."""
    tokens: set[str] = set()
    for word in text.lower().replace("'", " ").split():
        cleaned = "".join(c for c in word if c.isalpha())
        if len(cleaned) > 1 and cleaned not in _STOPWORDS:
            tokens.add(cleaned)
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def _cluster_by_similarity(
    token_sets: list[set[str]], threshold: float
) -> list[int]:
    """Union-find clustering by Jaccard similarity above threshold."""
    n = len(token_sets)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(token_sets[i], token_sets[j]) >= threshold:
                union(i, j)

    # Relabel into contiguous indices in order of first appearance.
    labels: list[int] = []
    seen: dict[int, int] = {}
    for i in range(n):
        root = find(i)
        if root not in seen:
            seen[root] = len(seen)
        labels.append(seen[root])
    return labels


class MockClaimNormalizer:
    """Deterministic keyword-based normalizer for tests and CI."""

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold

    def normalize(self, comments: list[str]) -> list[int]:
        if not comments:
            return []
        token_sets = [_tokenize(c) for c in comments]
        return _cluster_by_similarity(token_sets, self.threshold)


# ---------------------------------------------------------------------------
# LLM-backed normalizers
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are analyzing a governance thread. Below are {n} \
comments, each prefixed by its index in brackets.

Your task: group the comments into clusters of *conceptually equivalent*
positions. Comments that express the same underlying position, even with
different wording, structure, or framing, belong to the same cluster.
Comments that are off-topic or do not express a position on the main
topic should each go into their own singleton cluster.

Do not decide how many clusters there should be. Let the data decide.

Return your answer as a JSON object with a single key "clusters" whose
value is a list of lists of comment indices. Every comment index from 0
to {last} must appear in exactly one cluster.

Example output format:
{{"clusters": [[0, 5, 12], [1, 3], [2], [4, 7, 8, 9, 10, 11]]}}

Comments:
{comments}

Your JSON answer:"""


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from a possibly noisy LLM response."""
    # Strip markdown code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in LLM response: {text!r}")
        text = text[start : end + 1]
    return json.loads(text)


def _clusters_to_labels(clusters: list[list[int]], n: int) -> list[int]:
    """Convert a list of index clusters into a flat label list of length n.

    Indices outside ``[0, n)`` are ignored. Comments not mentioned in any
    cluster are assigned their own singleton cluster.
    """
    labels = [-1] * n
    next_label = 0
    for cluster in clusters:
        for idx in cluster:
            if 0 <= idx < n:
                labels[idx] = next_label
        next_label += 1
    # Any comment not mentioned gets its own singleton cluster.
    for i in range(n):
        if labels[i] == -1:
            labels[i] = next_label
            next_label += 1
    return labels


class _LLMClaimNormalizerBase(ABC):
    """Shared caching and prompt logic for LLM-backed normalizers."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or ".cache/llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def normalize(self, comments: list[str]) -> list[int]:
        if not comments:
            return []
        prompt = self._build_prompt(comments)
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"

        if cache_path.exists():
            raw = cache_path.read_text(encoding="utf-8")
        else:
            raw = self._call_api(prompt)
            cache_path.write_text(raw, encoding="utf-8")

        clusters = _extract_json(raw)["clusters"]
        return _clusters_to_labels(clusters, len(comments))

    def _build_prompt(self, comments: list[str]) -> str:
        numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(comments))
        return _PROMPT_TEMPLATE.format(
            n=len(comments),
            last=len(comments) - 1,
            comments=numbered,
        )

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Return the raw response text from the LLM."""


class OpenAIClaimNormalizer(_LLMClaimNormalizerBase):
    """Claim normalizer backed by OpenAI chat completions."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.api_key = api_key
        self.model = model

    def _call_api(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI returned an empty response")
        return content


class GeminiClaimNormalizer(_LLMClaimNormalizerBase):
    """Claim normalizer backed by Google Gemini."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.api_key = api_key
        self.model = model

    def _call_api(self, prompt: str) -> str:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": 0.0, "response_mime_type": "application/json"},
        )
        text = response.text
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        return text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_normalizer(name: str | None = None) -> ClaimNormalizer:
    """Return the normalizer selected by ``name`` or by config.

    Args:
        name: One of ``"mock"``, ``"openai"``, ``"gemini"``. If ``None``,
            falls back to the ``ALETHEIA_NORMALIZER`` environment variable
            (default: ``"mock"``).

    Raises:
        RuntimeError: If the selected backend requires an API key that is
            not configured.
        ValueError: If ``name`` is not a known backend.
    """
    name = name or get_default_normalizer()
    if name == "mock":
        return MockClaimNormalizer()
    if name == "openai":
        key = get_openai_key()
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return OpenAIClaimNormalizer(api_key=key)
    if name == "gemini":
        key = get_gemini_key()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return GeminiClaimNormalizer(api_key=key)
    raise ValueError(f"Unknown normalizer: {name!r}")