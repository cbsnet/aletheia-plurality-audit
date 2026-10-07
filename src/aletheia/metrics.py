"""Evaluation metrics for plurality audit.

Given a ground-truth assignment of comments to positions, and a
predicted assignment produced by a normalizer, this module computes:

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Purity of predicted clusters with respect to true positions
- Relative error on the inflation factor ``I``
- F1 score for paraphrase grouping

Noise comments (ground truth ``-1``) are excluded from ARI, NMI,
purity, and F1, because the normalizer does not know which comments are
off-topic and should not be penalized for grouping them arbitrarily.
They are also excluded from the inflation factor: ``P_real`` counts only
true positions, not the noise bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)


@dataclass(frozen=True)
class ClusteringMetrics:
    """Container for all metrics computed on one thread.

    Attributes:
        ari: Adjusted Rand Index between predicted labels and ground truth.
        nmi: Normalized Mutual Information, same inputs.
        purity: Fraction of non-noise comments that fall in the majority
            true position of their predicted cluster.
        n_predicted: Number of predicted clusters (excluding noise bucket).
        n_true: Number of true positions (ground truth).
        inflation_predicted: ``n_voices_nonnoise / n_predicted``.
        inflation_true: ``n_voices_nonnoise / n_true``.
        relative_error_inflation: ``|pred - true| / true``.
        paraphrase_f1: Macro F1 over clusters, treating each cluster as
            a class. Only meaningful when true positions are known.
    """

    ari: float
    nmi: float
    purity: float
    n_predicted: int
    n_true: int
    inflation_predicted: float
    inflation_true: float
    relative_error_inflation: float
    paraphrase_f1: float


def _non_noise_mask(true_positions: list[int]) -> np.ndarray:
    return np.array([p != -1 for p in true_positions], dtype=bool)


def _n_clusters(labels: list[int] | np.ndarray) -> int:
    """Number of distinct labels, ignoring the sentinel ``-1``."""
    arr = np.asarray(labels)
    unique = set(arr.tolist())
    unique.discard(-1)
    return len(unique)


def purity_score(true: np.ndarray, predicted: np.ndarray) -> float:
    """Purity of predicted clusters with respect to true labels.

    For each predicted cluster, count the size of its majority true
    label. Purity is the sum of these counts divided by the total number
    of points.
    """
    if len(true) == 0:
        return 0.0
    total = 0
    for pred_label in set(predicted.tolist()):
        mask = predicted == pred_label
        true_in_cluster = true[mask]
        if len(true_in_cluster) == 0:
            continue
        counts = np.bincount(true_in_cluster)
        total += int(counts.max())
    return total / len(true)


def compute_metrics(
    true_positions: list[int],
    predicted_labels: list[int],
) -> ClusteringMetrics:
    """Compute all clustering metrics for one thread.

    Args:
        true_positions: Ground-truth position index per comment, with
            ``-1`` marking off-topic noise.
        predicted_labels: Predicted cluster label per comment. Must have
            the same length as ``true_positions``.

    Returns:
        A :class:`ClusteringMetrics` instance.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(true_positions) != len(predicted_labels):
        raise ValueError(
            f"Length mismatch: true_positions has {len(true_positions)} "
            f"elements, predicted_labels has {len(predicted_labels)}"
        )

    true = np.asarray(true_positions)
    pred = np.asarray(predicted_labels)
    mask = _non_noise_mask(true_positions)

    true_clean = true[mask]
    pred_clean = pred[mask]

    if len(true_clean) == 0:
        # Degenerate: no non-noise comments. All metrics default to 0.
        return ClusteringMetrics(
            ari=0.0,
            nmi=0.0,
            purity=0.0,
            n_predicted=0,
            n_true=_n_clusters(true),
            inflation_predicted=0.0,
            inflation_true=0.0,
            relative_error_inflation=0.0,
            paraphrase_f1=0.0,
        )

    ari = float(adjusted_rand_score(true_clean, pred_clean))
    nmi = float(normalized_mutual_info_score(true_clean, pred_clean))
    purity = float(purity_score(true_clean, pred_clean))

    n_predicted = _n_clusters(pred_clean.tolist())
    n_true = _n_clusters(true_clean.tolist())

    n_voices = len(true_clean)
    inflation_true = n_voices / n_true if n_true > 0 else 0.0
    inflation_predicted = n_voices / n_predicted if n_predicted > 0 else 0.0
    if inflation_true > 0:
        rel_err = abs(inflation_predicted - inflation_true) / inflation_true
    else:
        rel_err = 0.0

    # F1 for paraphrase grouping: each true position is treated as a
    # class, and we compute macro-F1 with the predicted clusters as
    # the candidate partition.
    paraphrase_f1 = float(
        f1_score(true_clean, pred_clean, average="macro", zero_division=0)
    )

    return ClusteringMetrics(
        ari=ari,
        nmi=nmi,
        purity=purity,
        n_predicted=n_predicted,
        n_true=n_true,
        inflation_predicted=inflation_predicted,
        inflation_true=inflation_true,
        relative_error_inflation=rel_err,
        paraphrase_f1=paraphrase_f1,
    )


def median_relative_error(errors: list[float]) -> float:
    """Median of a list of relative errors. Empty list returns 0.0."""
    if not errors:
        return 0.0
    return float(np.median(errors))


def summary_across_threads(
    metrics_list: list[ClusteringMetrics],
) -> dict[str, float]:
    """Aggregate a list of per-thread metrics into a summary dict.

    Returns median ARI, NMI, purity, and relative error, plus mean
    inflation estimates.
    """
    if not metrics_list:
        return {
            "median_ari": 0.0,
            "median_nmi": 0.0,
            "median_purity": 0.0,
            "median_relative_error": 0.0,
            "mean_inflation_predicted": 0.0,
            "mean_inflation_true": 0.0,
        }
    aris = [m.ari for m in metrics_list]
    nmis = [m.nmi for m in metrics_list]
    purities = [m.purity for m in metrics_list]
    errors = [m.relative_error_inflation for m in metrics_list]
    infl_pred = [m.inflation_predicted for m in metrics_list]
    infl_true = [m.inflation_true for m in metrics_list]
    return {
        "median_ari": float(np.median(aris)),
        "median_nmi": float(np.median(nmis)),
        "median_purity": float(np.median(purities)),
        "median_relative_error": float(np.median(errors)),
        "mean_inflation_predicted": float(np.mean(infl_pred)),
        "mean_inflation_true": float(np.mean(infl_true)),
    }