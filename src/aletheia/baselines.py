"""Naive baselines for plurality estimation.

Each baseline implements the same ``normalize`` interface as the claim
normalizers, so all of them can be evaluated through the same code path.

The four baselines are deliberately simple:

- ``UniqueAccountsBaseline`` — treats every comment as coming from a
  distinct account, hence every comment is its own position. This is
  what a reviewer who counts voices without reading them would do.
- ``UniqueCommentsBaseline`` — groups comments whose text is byte-for-byte
  identical. A reviewer's mental deduplication of obvious repeats.
- ``RawEmbeddingsBaseline`` — clusters raw comment embeddings, with no
  LLM normalization. Similarity search as a straw man for Aletheia.
- ``JaccardKeywordsBaseline`` — clusters comments by Jaccard similarity
  over keyword sets. Classical lexical baseline.

Aletheia must beat the *best* of these four on median relative error on
the inflation factor ``I``. See ``PROTOCOL.md``.
"""

from __future__ import annotations

from typing import Protocol

from aletheia.cluster import cluster_embeddings
from aletheia.embedders import Embedder, HashingEmbedder
from aletheia.llm import MockClaimNormalizer


class Baseline(Protocol):
    """Common interface: comments in, cluster labels out."""

    name: str

    def normalize(self, comments: list[str]) -> list[int]:
        """Return one cluster label per comment."""
        ...


# ---------------------------------------------------------------------------
# Baseline 1 — Unique accounts
# ---------------------------------------------------------------------------


class UniqueAccountsBaseline:
    """Every comment is treated as a distinct voice.

    This is the weakest possible baseline: it assumes no two voices ever
    express the same position. Its estimated inflation factor is always
    ``1.0``, regardless of the true value.
    """

    name = "unique_accounts"

    def normalize(self, comments: list[str]) -> list[int]:
        return list(range(len(comments)))


# ---------------------------------------------------------------------------
# Baseline 2 — Unique comments
# ---------------------------------------------------------------------------


class UniqueCommentsBaseline:
    """Groups only byte-for-byte identical comments.

    Captures exact duplicates but nothing else. Paraphrases of the same
    position end up in different clusters.
    """

    name = "unique_comments"

    def normalize(self, comments: list[str]) -> list[int]:
        seen: dict[str, int] = {}
        labels: list[int] = []
        for c in comments:
            if c not in seen:
                seen[c] = len(seen)
            labels.append(seen[c])
        return labels


# ---------------------------------------------------------------------------
# Baseline 3 — Raw embeddings
# ---------------------------------------------------------------------------


class RawEmbeddingsBaseline:
    """Clusters raw comment embeddings with no LLM normalization.

    Uses the same clustering routine as Aletheia, but skips the LLM step.
    This is the baseline that most directly tests whether the LLM
    normalization contributes anything (hypothesis H2 in ``PROTOCOL.md``).
    """

    name = "raw_embeddings"

    def __init__(
        self,
        embedder: Embedder | None = None,
        similarity_threshold: float = 0.7,
    ) -> None:
        self.embedder = embedder if embedder is not None else HashingEmbedder()
        self.similarity_threshold = similarity_threshold

    def normalize(self, comments: list[str]) -> list[int]:
        if not comments:
            return []
        embeddings = self.embedder.embed(comments)
        return cluster_embeddings(embeddings, self.similarity_threshold)


# ---------------------------------------------------------------------------
# Baseline 4 — Jaccard keywords
# ---------------------------------------------------------------------------


class JaccardKeywordsBaseline:
    """Clusters comments by Jaccard similarity over keyword sets.

    Reuses the keyword tokenizer and union-find clustering from the
    mock normalizer. The default threshold matches the mock's default.
    """

    name = "jaccard_keywords"

    def __init__(self, threshold: float = 0.3) -> None:
        self._mock = MockClaimNormalizer(threshold=threshold)

    def normalize(self, comments: list[str]) -> list[int]:
        return self._mock.normalize(comments)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def get_baselines(
    embedder: Embedder | None = None,
    raw_embeddings_threshold: float = 0.7,
    jaccard_threshold: float = 0.3,
) -> list[Baseline]:
    """Return the four baselines, ready to be evaluated.

    Args:
        embedder: Passed to ``RawEmbeddingsBaseline``. Defaults to
            ``HashingEmbedder``.
        raw_embeddings_threshold: Cosine similarity threshold for the
            raw-embeddings baseline.
        jaccard_threshold: Jaccard threshold for the keyword baseline.

    Returns:
        A list of four baseline instances.
    """
    return [
        UniqueAccountsBaseline(),
        UniqueCommentsBaseline(),
        RawEmbeddingsBaseline(
            embedder=embedder, similarity_threshold=raw_embeddings_threshold
        ),
        JaccardKeywordsBaseline(threshold=jaccard_threshold),
    ]