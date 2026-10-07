"""Evaluation orchestration.

Runs the Aletheia pipeline (claim normalization) and the four naive
baselines on a set of synthetic threads, computes clustering metrics
for each, and aggregates the results into a summary that supports the
hypotheses in ``PROTOCOL.md``.

This module does not itself decide whether Aletheia "wins". It produces
the numbers; the interpretation lives in the notebook and the protocol.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from aletheia.baselines import Baseline, get_baselines
from aletheia.llm import ClaimNormalizer, get_normalizer
from aletheia.metrics import ClusteringMetrics, compute_metrics
from aletheia.synthetic import SyntheticThread, generate_batch


@dataclass(frozen=True)
class ThreadResult:
    """Evaluation outcome for a single thread.

    Attributes:
        seed: The thread's random seed.
        n_voices: Number of comments in the thread.
        n_positions: Number of true positions.
        true_inflation: The ground-truth inflation factor ``I``.
        aletheia: Metrics produced by the Aletheia pipeline.
        baselines: Mapping from baseline name to its metrics.
    """

    seed: int
    n_voices: int
    n_positions: int
    true_inflation: float
    aletheia: ClusteringMetrics
    baselines: dict[str, ClusteringMetrics]


def evaluate_thread(
    thread: SyntheticThread,
    normalizer: ClaimNormalizer,
    baselines: Iterable[Baseline],
) -> ThreadResult:
    """Run Aletheia and all baselines on one thread.

    Args:
        thread: A synthetic thread with ground truth.
        normalizer: The Aletheia claim normalizer.
        baselines: The baselines to evaluate against.

    Returns:
        A :class:`ThreadResult` with metrics for Aletheia and each
        baseline.
    """
    true = thread.true_positions

    aletheia_labels = normalizer.normalize(thread.comments)
    aletheia_metrics = compute_metrics(true, aletheia_labels)

    baseline_metrics: dict[str, ClusteringMetrics] = {}
    for baseline in baselines:
        labels = baseline.normalize(thread.comments)
        baseline_metrics[baseline.name] = compute_metrics(true, labels)

    return ThreadResult(
        seed=thread.seed,
        n_voices=thread.n_voices,
        n_positions=thread.n_positions,
        true_inflation=thread.true_inflation,
        aletheia=aletheia_metrics,
        baselines=baseline_metrics,
    )


def evaluate_batch(
    threads: list[SyntheticThread],
    normalizer: ClaimNormalizer | None = None,
    baselines: list[Baseline] | None = None,
) -> list[ThreadResult]:
    """Evaluate a batch of threads.

    Args:
        threads: The threads to evaluate.
        normalizer: Aletheia normalizer. Defaults to the one selected by
            ``ALETHEIA_NORMALIZER`` (usually ``"mock"``).
        baselines: Baselines to run. Defaults to :func:`get_baselines`.

    Returns:
        A list of :class:`ThreadResult`, one per thread.
    """
    if normalizer is None:
        normalizer = get_normalizer()
    if baselines is None:
        baselines = get_baselines()
    return [evaluate_thread(t, normalizer, baselines) for t in threads]


def summarize_results(results: list[ThreadResult]) -> dict[str, object]:
    """Aggregate a batch of results into a comparative summary.

    The summary reports, for Aletheia and for each baseline:

    - median relative error on the inflation factor ``I``
    - median ARI
    - median NMI

    It also identifies the best baseline (lowest median relative error)
    and computes Aletheia's relative improvement over it. A negative
    improvement means Aletheia is *worse* than the best baseline.

    Args:
        results: List of per-thread results.

    Returns:
        A dict with keys ``"n_threads"``, ``"aletheia"``,
        ``"baselines"``, ``"best_baseline"``, and
        ``"improvement_over_best_baseline"``.
    """
    if not results:
        return {
            "n_threads": 0,
            "aletheia": {},
            "baselines": {},
            "best_baseline": None,
            "improvement_over_best_baseline": 0.0,
        }

    def _agg(metrics_list: list[ClusteringMetrics]) -> dict[str, float]:
        errors = [m.relative_error_inflation for m in metrics_list]
        aris = [m.ari for m in metrics_list]
        nmis = [m.nmi for m in metrics_list]
        return {
            "median_relative_error": float(np.median(errors)),
            "median_ari": float(np.median(aris)),
            "median_nmi": float(np.median(nmis)),
        }

    aletheia_metrics = [r.aletheia for r in results]
    aletheia_summary = _agg(aletheia_metrics)

    baseline_names: set[str] = set()
    for r in results:
        baseline_names.update(r.baselines.keys())

    baselines_summary: dict[str, dict[str, float]] = {}
    for name in sorted(baseline_names):
        metrics_list = [r.baselines[name] for r in results if name in r.baselines]
        baselines_summary[name] = _agg(metrics_list)

    best_name: str | None = None
    best_error = float("inf")
    for name, summary in baselines_summary.items():
        err = summary["median_relative_error"]
        if err < best_error:
            best_error = err
            best_name = name

    aletheia_error = aletheia_summary["median_relative_error"]
    improvement = (
        (best_error - aletheia_error) / best_error if best_error > 0 else 0.0
    )

    return {
        "n_threads": len(results),
        "aletheia": aletheia_summary,
        "baselines": baselines_summary,
        "best_baseline": best_name,
        "improvement_over_best_baseline": improvement,
    }


def make_pilot_batch(
    n_threads: int = 30,
    n_voices: int = 100,
    n_positions: int = 5,
    paraphrase_level: str = "medium",
    noise_ratio: float = 0.0,
    seed_base: int = 1000,
) -> list[SyntheticThread]:
    """Convenience wrapper to build a standard Stage 1 pilot batch.

    The defaults are the ones specified for Stage 1 in ``PROTOCOL.md``.
    Callers can override any parameter.
    """
    return generate_batch(
        n_threads=n_threads,
        n_voices=n_voices,
        n_positions=n_positions,
        paraphrase_level=paraphrase_level,  # type: ignore[arg-type]
        noise_ratio=noise_ratio,
        seed_base=seed_base,
    )