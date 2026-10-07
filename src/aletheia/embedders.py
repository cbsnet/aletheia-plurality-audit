"""Text embedders.

Three backends behind a single interface:

- ``HashingEmbedder`` — deterministic, no downloads, based on sklearn's
  ``HashingVectorizer``. Used in tests, CI, and as the "raw embeddings"
  baseline. No network access.
- ``SentenceTransformerEmbedder`` — local sentence-transformers model.
  Higher quality, requires a one-time model download (~80 MB).
- ``OpenAIEmbedder`` — OpenAI embeddings API. Optional.

All embedders return L2-normalized vectors, so that cosine similarity
reduces to a dot product.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class Embedder(Protocol):
    """Maps a list of texts to a matrix of L2-normalized vectors."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape ``(len(texts), dim)``."""
        ...


# ---------------------------------------------------------------------------
# HashingEmbedder — deterministic, no network
# ---------------------------------------------------------------------------


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Uses sklearn's ``HashingVectorizer`` to map text to a fixed-size
    feature space. Words and bigrams are hashed into ``n_features``
    dimensions. Output vectors are L2-normalized.
    """

    def __init__(self, n_features: int = 512) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        self.n_features = n_features
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm=None,  # normalization done explicitly below
            lowercase=True,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.n_features), dtype=np.float64)
        matrix = self._vectorizer.transform(texts)
        dense = matrix.toarray().astype(np.float64)
        return normalize(dense, norm="l2", axis=1, copy=False)


# ---------------------------------------------------------------------------
# SentenceTransformerEmbedder — local model, high quality
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """Embedder backed by sentence-transformers.

    The default model (``all-MiniLM-L6-v2``) is small, fast, and
    produces 384-dimensional embeddings. The model is downloaded on
    first use (~80 MB) and cached by the library.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float64)
        self._lazy_load()
        assert self._model is not None
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return normalize(vectors.astype(np.float64), norm="l2", axis=1, copy=False)


# ---------------------------------------------------------------------------
# OpenAIEmbedder — API-backed, optional
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """Embedder backed by OpenAI's embeddings API."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        self.api_key = api_key
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1536), dtype=np.float64)
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.embeddings.create(model=self.model, input=texts)
        vectors = np.array([item.embedding for item in response.data], dtype=np.float64)
        return normalize(vectors, norm="l2", axis=1, copy=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedder(name: str = "hashing", **kwargs) -> Embedder:
    """Return an embedder by name.

    Args:
        name: One of ``"hashing"``, ``"sentence-transformer"``,
            ``"openai"``.
        **kwargs: Forwarded to the embedder constructor.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    if name == "hashing":
        return HashingEmbedder(**kwargs)
    if name == "sentence-transformer":
        return SentenceTransformerEmbedder(**kwargs)
    if name == "openai":
        return OpenAIEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder: {name!r}")