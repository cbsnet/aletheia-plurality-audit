"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from aletheia.metrics import (
    ClusteringMetrics,
    compute_metrics,
    median_relative_error,
    purity_score,
    summary_across_threads,
)

# ---------------------------------------------------------------------------
# purity_score
# ---------------------------------------------------------------------------


def test_purity_perfect_clustering() -> None:
    """Perfect clustering -> purity 1.0."""
    import numpy as np

    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    assert purity_score(true, pred) == 1.0


def test_purity_single_cluster() -> None:
    """All in one cluster -> purity equals majority class fraction."""
    import numpy as np

    true = np.array([0, 0, 0, 1])
    pred = np.array([0, 0, 0, 0])
    # Majority is class 0 with 3 of 4
    assert purity_score(true, pred) == pytest.approx(0.75)


def test_purity_empty() -> None:
    """Empty input returns 0.0."""
    import numpy as np

    assert purity_score(np.array([]), np.array([])) == 0.0


# ---------------------------------------------------------------------------
# compute_metrics — basic behavior
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_clustering() -> None:
    """Perfect recovery: ARI=1, NMI=1, purity=1, error=0."""
    true = [0, 0, 1, 1, 2, 2]
    pred = [0, 0, 1, 1, 2, 2]
    m = compute_metrics(true, pred)
    assert m.ari == pytest.approx(1.0)
    assert m.nmi == pytest.approx(1.0)
    assert m.purity == pytest.approx(1.0)
    assert m.relative_error_inflation == pytest.approx(0.0)
    assert m.n_predicted == 3
    assert m.n_true == 3


def test_compute_metrics_all_in_one_cluster() -> None:
    """Everything collapsed: ARI low, high inflation underestimation."""
    true = [0, 0, 1, 1, 2, 2]
    pred = [0, 0, 0, 0, 0, 0]
    m = compute_metrics(true, pred)
    assert m.n_predicted == 1
    assert m.n_true == 3
    assert m.inflation_true == pytest.approx(2.0)
    assert m.inflation_predicted == pytest.approx(6.0)
    assert m.relative_error_inflation == pytest.approx(2.0)


def test_compute_metrics_label_permutation_invariant() -> None:
    """Renaming clusters does not change ARI, NMI, or purity."""
    true = [0, 0, 1, 1, 2, 2]
    pred_a = [0, 0, 1, 1, 2, 2]
    pred_b = [2, 2, 0, 0, 1, 1]
    m_a = compute_metrics(true, pred_a)
    m_b = compute_metrics(true, pred_b)
    assert m_a.ari == pytest.approx(m_b.ari)
    assert m_a.nmi == pytest.approx(m_b.nmi)
    assert m_a.purity == pytest.approx(m_b.purity)


def test_compute_metrics_length_mismatch_raises() -> None:
    """Different lengths -> ValueError."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_metrics([0, 1], [0, 1, 2])


# ---------------------------------------------------------------------------
# compute_metrics — noise handling
# ---------------------------------------------------------------------------


def test_compute_metrics_ignores_noise_in_metrics() -> None:
    """Noise comments (-1) are excluded from ARI/NMI/purity."""
    true = [0, 0, 1, 1, -1, -1]
    pred = [0, 0, 1, 1, 5, 5]  # noise grouped into a spurious cluster
    m = compute_metrics(true, pred)
    # Only the four clean comments matter; their clustering is perfect.
    assert m.ari == pytest.approx(1.0)
    assert m.purity == pytest.approx(1.0)
    # n_predicted and n_true count only non-noise comments.
    assert m.n_predicted == 2
    assert m.n_true == 2


def test_compute_metrics_inflation_uses_only_non_noise() -> None:
    """Inflation is computed on non-noise voices only."""
    true = [0, 0, 1, 1, -1]
    pred = [0, 0, 1, 1, 2]
    m = compute_metrics(true, pred)
    # 4 non-noise voices, 2 true positions -> inflation_true = 2.0
    assert m.inflation_true == pytest.approx(2.0)
    assert m.inflation_predicted == pytest.approx(2.0)
    assert m.relative_error_inflation == pytest.approx(0.0)


def test_compute_metrics_all_noise() -> None:
    """All-noise thread returns zeros without crashing."""
    true = [-1, -1, -1]
    pred = [0, 1, 2]
    m = compute_metrics(true, pred)
    assert m.n_predicted == 0
    assert m.n_true == 0
    assert m.relative_error_inflation == 0.0


# ---------------------------------------------------------------------------
# compute_metrics — over-splitting (high n_predicted)
# ---------------------------------------------------------------------------


def test_compute_metrics_over_splitting() -> None:
    """Every comment in its own cluster -> inflation_predicted == n_voices."""
    true = [0, 0, 0, 0]
    pred = [0, 1, 2, 3]
    m = compute_metrics(true, pred)
    assert m.n_predicted == 4
    assert m.n_true == 1
    assert m.inflation_true == pytest.approx(4.0)
    assert m.inflation_predicted == pytest.approx(1.0)
    # Relative error |1 - 4| / 4 = 0.75
    assert m.relative_error_inflation == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# median_relative_error
# ---------------------------------------------------------------------------


def test_median_relative_error_empty() -> None:
    """Empty list -> 0.0."""
    assert median_relative_error([]) == 0.0


def test_median_relative_error_odd_length() -> None:
    """Median of three values."""
    assert median_relative_error([0.1, 0.2, 0.3]) == pytest.approx(0.2)


def test_median_relative_error_even_length() -> None:
    """Median of four values is the mean of the middle two."""
    assert median_relative_error([0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# summary_across_threads
# ---------------------------------------------------------------------------


def _make_metrics(ari: float, rel_err: float) -> ClusteringMetrics:
    return ClusteringMetrics(
        ari=ari,
        nmi=ari,
        purity=ari,
        n_predicted=3,
        n_true=3,
        inflation_predicted=10.0,
        inflation_true=10.0,
        relative_error_inflation=rel_err,
        paraphrase_f1=ari,
    )


def test_summary_empty_list() -> None:
    """Empty list -> dict of zeros."""
    s = summary_across_threads([])
    assert s["median_ari"] == 0.0
    assert s["median_relative_error"] == 0.0


def test_summary_medians() -> None:
    """Medians over a small list of metrics."""
    m1 = _make_metrics(0.9, 0.1)
    m2 = _make_metrics(0.8, 0.2)
    m3 = _make_metrics(0.7, 0.3)
    s = summary_across_threads([m1, m2, m3])
    assert s["median_ari"] == pytest.approx(0.8)
    assert s["median_relative_error"] == pytest.approx(0.2)
    assert s["median_nmi"] == pytest.approx(0.8)
    assert s["median_purity"] == pytest.approx(0.8)
    assert s["mean_inflation_predicted"] == pytest.approx(10.0)
    assert s["mean_inflation_true"] == pytest.approx(10.0)