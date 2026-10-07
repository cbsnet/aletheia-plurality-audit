"""Tests for semantic clustering."""

from __future__ import annotations

import numpy as np
import pytest

from aletheia.cluster import cluster_embeddings, n_clusters


def _unit(v: list[float]) -> np.ndarray:
    """Return an L2-normalized row vector."""
    arr = np.array(v, dtype=np.float64)
    return arr / np.linalg.norm(arr)


# ---------------------------------------------------------------------------
# cluster_embeddings — basic cases
# ---------------------------------------------------------------------------


def test_cluster_empty_input() -> None:
    """Empty matrix -> empty label list."""
    assert cluster_embeddings(np.zeros((0, 4))) == []


def test_cluster_single_vector() -> None:
    """One vector -> one cluster with label 0."""
    result = cluster_embeddings(_unit([1.0, 0.0, 0.0]).reshape(1, -1))
    assert result == [0]


def test_cluster_identical_vectors_merge() -> None:
    """Identical vectors are merged into one cluster."""
    v = _unit([1.0, 0.0, 0.0])
    matrix = np.vstack([v, v, v])
    labels = cluster_embeddings(matrix, similarity_threshold=0.9)
    assert n_clusters(labels) == 1
    assert labels == [0, 0, 0]


def test_cluster_orthogonal_vectors_separate() -> None:
    """Orthogonal vectors (similarity 0) end up in different clusters."""
    matrix = np.vstack(
        [
            _unit([1.0, 0.0, 0.0]),
            _unit([0.0, 1.0, 0.0]),
            _unit([0.0, 0.0, 1.0]),
        ]
    )
    labels = cluster_embeddings(matrix, similarity_threshold=0.7)
    assert n_clusters(labels) == 3


def test_cluster_two_groups_of_identical_vectors() -> None:
    """Two groups of identical vectors yield exactly two clusters."""
    a = _unit([1.0, 0.0, 0.0])
    b = _unit([0.0, 1.0, 0.0])
    matrix = np.vstack([a, a, b, b])
    labels = cluster_embeddings(matrix, similarity_threshold=0.9)
    assert n_clusters(labels) == 2
    # The first two share a label, the last two share a different one.
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cluster_labels_are_contiguous_from_zero() -> None:
    """Labels form a contiguous range starting at 0."""
    matrix = np.vstack(
        [
            _unit([1.0, 0.0, 0.0]),
            _unit([1.0, 0.0, 0.0]),
            _unit([0.0, 1.0, 0.0]),
            _unit([0.0, 0.0, 1.0]),
        ]
    )
    labels = cluster_embeddings(matrix, similarity_threshold=0.9)
    assert set(labels) == set(range(n_clusters(labels)))
    assert min(labels) == 0


def test_cluster_is_deterministic() -> None:
    """Same input -> same output, across calls."""
    matrix = np.vstack(
        [
            _unit([1.0, 0.0, 0.0]),
            _unit([0.9, 0.1, 0.0]),
            _unit([0.0, 1.0, 0.0]),
        ]
    )
    a = cluster_embeddings(matrix, similarity_threshold=0.8)
    b = cluster_embeddings(matrix, similarity_threshold=0.8)
    assert a == b


# ---------------------------------------------------------------------------
# cluster_embeddings — threshold behavior
# ---------------------------------------------------------------------------


def test_cluster_high_threshold_splits_more() -> None:
    """A higher threshold yields at least as many clusters as a lower one."""
    v1 = _unit([1.0, 0.0, 0.0])
    v2 = _unit([0.8, 0.6, 0.0])  # cosine ~ 0.8 with v1
    matrix = np.vstack([v1, v2])

    labels_low = cluster_embeddings(matrix, similarity_threshold=0.5)
    labels_high = cluster_embeddings(matrix, similarity_threshold=0.99)

    assert n_clusters(labels_low) <= n_clusters(labels_high)
    assert n_clusters(labels_high) == 2


def test_cluster_invalid_threshold_raises() -> None:
    """Threshold outside [-1, 1] is rejected."""
    matrix = _unit([1.0, 0.0, 0.0]).reshape(1, -1)
    with pytest.raises(ValueError, match="similarity_threshold must be in"):
        cluster_embeddings(matrix, similarity_threshold=1.5)
    with pytest.raises(ValueError, match="similarity_threshold must be in"):
        cluster_embeddings(matrix, similarity_threshold=-1.5)


# ---------------------------------------------------------------------------
# n_clusters
# ---------------------------------------------------------------------------


def test_n_clusters_empty() -> None:
    """Empty label list -> 0 clusters."""
    assert n_clusters([]) == 0


def test_n_clusters_singleton() -> None:
    """One label -> 1 cluster."""
    assert n_clusters([0]) == 1


def test_n_clusters_multiple() -> None:
    """Distinct labels counted correctly."""
    assert n_clusters([0, 0, 1, 2, 2, 2]) == 3