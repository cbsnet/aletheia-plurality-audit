"""Tests for the claim normalizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aletheia.llm import (
    GeminiClaimNormalizer,
    MockClaimNormalizer,
    OpenAIClaimNormalizer,
    _clusters_to_labels,
    _extract_json,
    get_normalizer,
)

# ---------------------------------------------------------------------------
# MockClaimNormalizer
# ---------------------------------------------------------------------------


def test_mock_normalizer_empty_input() -> None:
    """Empty input returns empty output."""
    assert MockClaimNormalizer().normalize([]) == []


def test_mock_normalizer_groups_identical_comments() -> None:
    """Identical comments are grouped into one cluster."""
    comments = ["I support the reform", "I support the reform", "I support the reform"]
    labels = MockClaimNormalizer().normalize(comments)
    assert labels == [0, 0, 0]


def test_mock_normalizer_separates_dissimilar_comments() -> None:
    """Completely different comments end up in different clusters."""
    comments = [
        "Adopting the reform is the right move",
        "I reject this proposal entirely",
        "Postpone the decision until next year",
    ]
    labels = MockClaimNormalizer().normalize(comments)
    assert len(set(labels)) == 3


def test_mock_normalizer_groups_shared_keywords() -> None:
    """Comments sharing content words are grouped together."""
    comments = [
        "The reform addresses our concerns",
        "Our concerns are addressed by the reform",
        "I reject the entire proposal",
    ]
    labels = MockClaimNormalizer().normalize(comments)
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_mock_normalizer_labels_are_contiguous_from_zero() -> None:
    """Labels are a contiguous range starting at 0."""
    comments = [
        "Alpha beta gamma",
        "Alpha beta gamma",
        "Delta epsilon zeta",
        "Eta theta iota",
    ]
    labels = MockClaimNormalizer().normalize(comments)
    assert min(labels) == 0
    assert max(labels) == len(set(labels)) - 1


def test_mock_normalizer_is_deterministic() -> None:
    """Same input -> same output, across calls."""
    comments = ["Alpha beta", "Gamma delta", "Alpha beta"]
    a = MockClaimNormalizer().normalize(comments)
    b = MockClaimNormalizer().normalize(comments)
    assert a == b


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_plain() -> None:
    """Plain JSON object is parsed correctly."""
    text = '{"clusters": [[0, 1], [2]]}'
    assert _extract_json(text) == {"clusters": [[0, 1], [2]]}


def test_extract_json_with_markdown_fence() -> None:
    """JSON wrapped in ```json ... ``` is unwrapped."""
    text = 'Here is the answer:\n```json\n{"clusters": [[0, 1]]}\n```\nDone.'
    assert _extract_json(text) == {"clusters": [[0, 1]]}


def test_extract_json_with_surrounding_text() -> None:
    """JSON embedded in prose is extracted."""
    text = 'Sure! Here you go: {"clusters": [[0], [1]]} — hope that helps.'
    assert _extract_json(text) == {"clusters": [[0], [1]]}


def test_extract_json_invalid_raises() -> None:
    """No JSON object -> ValueError."""
    with pytest.raises(ValueError, match="No complete JSON object found"):
        _extract_json("no json here at all")


# ---------------------------------------------------------------------------
# _clusters_to_labels
# ---------------------------------------------------------------------------


def test_clusters_to_labels_basic() -> None:
    """Standard case: every index appears exactly once."""
    labels = _clusters_to_labels([[0, 2], [1]], 3)
    assert labels == [0, 1, 0]


def test_clusters_to_labels_missing_indices_become_singletons() -> None:
    """Indices not mentioned get their own clusters."""
    labels = _clusters_to_labels([[0]], 3)
    assert labels[0] == 0
    assert labels[1] != labels[2]
    assert labels[1] != 0
    assert labels[2] != 0


def test_clusters_to_labels_empty() -> None:
    """Empty clusters list with n=0 returns empty list."""
    assert _clusters_to_labels([], 0) == []


def test_clusters_to_labels_out_of_range_ignored() -> None:
    """Indices outside [0, n) are ignored without crashing."""
    labels = _clusters_to_labels([[0, 99], [1]], 2)
    assert labels == [0, 1]


# ---------------------------------------------------------------------------
# get_normalizer factory
# ---------------------------------------------------------------------------


def test_get_normalizer_mock_explicit() -> None:
    """name='mock' returns a MockClaimNormalizer."""
    assert isinstance(get_normalizer("mock"), MockClaimNormalizer)


def test_get_normalizer_openai_without_key_raises(monkeypatch) -> None:
    """name='openai' without OPENAI_API_KEY raises RuntimeError."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        get_normalizer("openai")


def test_get_normalizer_gemini_without_key_raises(monkeypatch) -> None:
    """name='gemini' without GEMINI_API_KEY raises RuntimeError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        get_normalizer("gemini")


def test_get_normalizer_unknown_raises() -> None:
    """Unknown name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown normalizer"):
        get_normalizer("nonsense")


# ---------------------------------------------------------------------------
# Caching (using a subclass that never calls the network)
# ---------------------------------------------------------------------------


class _StubNormalizer(OpenAIClaimNormalizer):
    """OpenAI normalizer whose API call is replaced by a fixed response."""

    call_count: int = 0

    def _call_api(self, prompt: str) -> str:
        type(self).call_count += 1
        return json.dumps({"clusters": [[0, 1], [2]]})


def test_llm_normalizer_caches_responses(tmp_path: Path) -> None:
    """The second call with identical input hits the cache, not the API."""
    _StubNormalizer.call_count = 0
    normalizer = _StubNormalizer(api_key="test", cache_dir=tmp_path)
    comments = ["alpha", "beta", "gamma"]

    labels_first = normalizer.normalize(comments)
    labels_second = normalizer.normalize(comments)

    assert labels_first == labels_second == [0, 0, 1]
    assert _StubNormalizer.call_count == 1


def test_llm_normalizer_empty_input_short_circuits(tmp_path: Path) -> None:
    """Empty input returns empty output without any API call."""
    _StubNormalizer.call_count = 0
    normalizer = _StubNormalizer(api_key="test", cache_dir=tmp_path)
    assert normalizer.normalize([]) == []
    assert _StubNormalizer.call_count == 0


def test_gemini_normalizer_constructs(tmp_path: Path) -> None:
    """GeminiClaimNormalizer can be instantiated without network access."""
    normalizer = GeminiClaimNormalizer(api_key="fake", cache_dir=tmp_path)
    assert normalizer.api_key == "fake"