"""Semantic clustering of embeddings.

Groups a set of L2-normalized vectors into clusters based on cosine
similarity. Uses agglomerative clustering with average linkage and a
distance threshold, so the number of clusters is determined by the data
rather than fixed a priori.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def cluster_embeddings(
    embeddings: np.ndarray,
    similarity_threshold: float = 0.7,
) -> list[int]:
    """Cluster L2-normalized embeddings by cosine similarity.

    Two vectors are merged when their cosine similarity is at or above
    ``similarity_threshold``. Because embeddings are L2-normalized, the
    cosine distance is ``1 - dot(a, b)``, and sklearn's ``cosine`` metric
    handles this internally.

    Args:
        embeddings: Array of shape ``(n, dim)``. Rows should be
            L2-normalized (as produced by the embedders in this package),
            though the function does not enforce it.
        similarity_threshold: Similarity level at or above which two
            vectors are considered the same cluster. Must be in [-1, 1].
            Higher values mean more, smaller clusters.

    Returns:
        A list of integer cluster labels, one per row of ``embeddings``.
        Labels are contiguous starting from 0.

    Raises:
        ValueError: If ``similarity_threshold`` is outside [-1, 1].
        ValueError: If ``embeddings`` has fewer than 1 row when non-empty.
    """
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError(
            f"similarity_threshold must be in [-1, 1], got {similarity_threshold}"
        )

    if embeddings.size == 0:
        return []

    n = embeddings.shape[0]
    if n == 1:
        return [0]

    distance_threshold = 1.0 - similarity_threshold
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(embeddings)
    return labels.tolist()


def n_clusters(labels: list[int]) -> int:
    """Number of distinct clusters in a label list."""
    return len(set(labels))