"""Tests for text embedders."""

from __future__ import annotations

import numpy as np
import pytest

from aletheia.embedders import (
    HashingEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)

# ---------------------------------------------------------------------------
# HashingEmbedder
# ---------------------------------------------------------------------------


def test_hashing_embedder_empty_input() -> None:
    """Empty input returns shape (0, n_features)."""
    emb = HashingEmbedder(n_features=64)
    result = emb.embed([])
    assert result.shape == (0, 64)


def test_hashing_embedder_output_shape() -> None:
    """Output shape is (n_texts, n_features)."""
    emb = HashingEmbedder(n_features=128)
    result = emb.embed(["alpha beta", "gamma delta", "epsilon"])
    assert result.shape == (3, 128)


def test_hashing_embedder_l2_normalized() -> None:
    """Every non-empty vector has unit L2 norm."""
    emb = HashingEmbedder(n_features=128)
    result = emb.embed(["alpha beta gamma", "delta epsilon"])
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-10)


def test_hashing_embedder_is_deterministic() -> None:
    """Same input -> identical output, across calls and instances."""
    emb_a = HashingEmbedder(n_features=64)
    emb_b = HashingEmbedder(n_features=64)
    texts = ["alpha beta", "gamma delta"]
    np.testing.assert_array_equal(emb_a.embed(texts), emb_b.embed(texts))


def test_hashing_embedder_similar_texts_are_closer() -> None:
    """Texts sharing words have higher cosine similarity than unrelated ones."""
    emb = HashingEmbedder(n_features=512)
    vectors = emb.embed(
        [
            "the reform addresses our concerns",
            "our concerns are addressed by the reform",
            "completely unrelated sentence about cats",
        ]
    )
    sim_related = float(np.dot(vectors[0], vectors[1]))
    sim_unrelated = float(np.dot(vectors[0], vectors[2]))
    assert sim_related > sim_unrelated


def test_hashing_embedder_invalid_n_features_raises() -> None:
    """n_features < 1 is rejected."""
    with pytest.raises(ValueError, match="n_features must be >= 1"):
        HashingEmbedder(n_features=0)


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder (no network in tests)
# ---------------------------------------------------------------------------


def test_sentence_transformer_constructs_without_loading() -> None:
    """Construction does not trigger a model download."""
    emb = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    assert emb.model_name == "all-MiniLM-L6-v2"
    assert emb._model is None  # not loaded yet


def test_sentence_transformer_empty_input_short_circuits() -> None:
    """Empty input returns shape (0, dim) without loading the model."""
    emb = SentenceTransformerEmbedder()
    result = emb.embed([])
    assert result.shape[0] == 0
    # Model must still not be loaded.
    assert emb._model is None


# ---------------------------------------------------------------------------
# OpenAIEmbedder (no network in tests)
# ---------------------------------------------------------------------------


def test_openai_embedder_constructs() -> None:
    """Construction stores the API key and model name."""
    emb = OpenAIEmbedder(api_key="fake-key", model="text-embedding-3-small")
    assert emb.api_key == "fake-key"
    assert emb.model == "text-embedding-3-small"


def test_openai_embedder_empty_input_short_circuits() -> None:
    """Empty input returns shape (0, dim) without any API call."""
    emb = OpenAIEmbedder(api_key="fake-key")
    result = emb.embed([])
    assert result.shape == (0, 1536)


# ---------------------------------------------------------------------------
# get_embedder factory
# ---------------------------------------------------------------------------


def test_get_embedder_hashing() -> None:
    """name='hashing' returns a HashingEmbedder."""
    assert isinstance(get_embedder("hashing"), HashingEmbedder)


def test_get_embedder_sentence_transformer() -> None:
    """name='sentence-transformer' returns a SentenceTransformerEmbedder."""
    assert isinstance(
        get_embedder("sentence-transformer"), SentenceTransformerEmbedder
    )


def test_get_embedder_openai() -> None:
    """name='openai' returns an OpenAIEmbedder."""
    assert isinstance(get_embedder("openai", api_key="x"), OpenAIEmbedder)


def test_get_embedder_unknown_raises() -> None:
    """Unknown name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown embedder"):
        get_embedder("nonsense")