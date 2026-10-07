"""Tests for evaluation orchestration."""

from __future__ import annotations

from aletheia.baselines import (
    JaccardKeywordsBaseline,
    UniqueAccountsBaseline,
    UniqueCommentsBaseline,
)
from aletheia.evaluation import (
    ThreadResult,
    evaluate_batch,
    evaluate_thread,
    make_pilot_batch,
    summarize_results,
)
from aletheia.llm import MockClaimNormalizer
from aletheia.metrics import ClusteringMetrics
from aletheia.synthetic import generate_thread

# ---------------------------------------------------------------------------
# evaluate_thread
# ---------------------------------------------------------------------------


def test_evaluate_thread_returns_result() -> None:
    """evaluate_thread returns a ThreadResult with the right fields."""
    thread = generate_thread(n_voices=30, n_positions=3, seed=42)
    normalizer = MockClaimNormalizer()
    baselines = [UniqueAccountsBaseline(), UniqueCommentsBaseline()]

    result = evaluate_thread(thread, normalizer, baselines)

    assert isinstance(result, ThreadResult)
    assert result.seed == 42
    assert result.n_voices == 30
    assert result.n_positions == 3
    assert result.true_inflation == 10.0
    assert isinstance(result.aletheia, ClusteringMetrics)
    assert "unique_accounts" in result.baselines
    assert "unique_comments" in result.baselines


def test_evaluate_thread_metrics_are_in_range() -> None:
    """All metrics fall in their expected ranges."""
    thread = generate_thread(n_voices=50, n_positions=5, seed=0)
    result = evaluate_thread(thread, MockClaimNormalizer(), [])

    assert -1.0 <= result.aletheia.ari <= 1.0
    assert 0.0 <= result.aletheia.nmi <= 1.0
    assert 0.0 <= result.aletheia.purity <= 1.0
    assert result.aletheia.relative_error_inflation >= 0.0


# ---------------------------------------------------------------------------
# evaluate_batch
# ---------------------------------------------------------------------------


def test_evaluate_batch_default_normalizer_is_mock(monkeypatch) -> None:
    """evaluate_batch uses the mock normalizer by default."""
    monkeypatch.setenv("ALETHEIA_NORMALIZER", "mock")
    threads = [
        generate_thread(n_voices=20, n_positions=3, seed=i) for i in range(3)
    ]
    results = evaluate_batch(threads, baselines=[UniqueAccountsBaseline()])
    assert len(results) == 3
    assert all(isinstance(r, ThreadResult) for r in results)


def test_evaluate_batch_uses_all_four_baselines_by_default() -> None:
    """Default baselines are the four from get_baselines."""
    threads = [generate_thread(n_voices=20, n_positions=3, seed=0)]
    results = evaluate_batch(threads, normalizer=MockClaimNormalizer())
    assert len(results) == 1
    assert len(results[0].baselines) == 4


def test_evaluate_batch_empty_threads() -> None:
    """Empty batch returns empty list."""
    results = evaluate_batch([], normalizer=MockClaimNormalizer(), baselines=[])
    assert results == []


# ---------------------------------------------------------------------------
# summarize_results
# ---------------------------------------------------------------------------


def test_summarize_empty() -> None:
    """Empty input yields a well-formed zero summary."""
    summary = summarize_results([])
    assert summary["n_threads"] == 0
    assert summary["best_baseline"] is None
    assert summary["improvement_over_best_baseline"] == 0.0


def test_summarize_single_thread() -> None:
    """Summary of one thread has the expected keys."""
    thread = generate_thread(n_voices=30, n_positions=3, seed=0)
    results = evaluate_batch(
        [thread],
        normalizer=MockClaimNormalizer(),
        baselines=[UniqueAccountsBaseline(), UniqueCommentsBaseline()],
    )
    summary = summarize_results(results)

    assert summary["n_threads"] == 1
    assert isinstance(summary["aletheia"], dict)
    assert "median_relative_error" in summary["aletheia"]
    assert isinstance(summary["baselines"], dict)
    assert "unique_accounts" in summary["baselines"]
    assert "unique_comments" in summary["baselines"]
    assert summary["best_baseline"] in {"unique_accounts", "unique_comments"}


def test_summarize_best_baseline_is_lowest_error() -> None:
    """The best baseline is the one with the lowest median relative error."""
    thread = generate_thread(n_voices=50, n_positions=5, seed=0)
    results = evaluate_batch(
        [thread],
        normalizer=MockClaimNormalizer(),
        baselines=[
            UniqueAccountsBaseline(),
            UniqueCommentsBaseline(),
            JaccardKeywordsBaseline(),
        ],
    )
    summary = summarize_results(results)

    baselines = summary["baselines"]
    assert isinstance(baselines, dict)
    errors = {name: s["median_relative_error"] for name, s in baselines.items()}
    expected_best = min(errors, key=lambda k: errors[k])
    assert summary["best_baseline"] == expected_best


def test_summarize_improvement_computed_correctly() -> None:
    """improvement_over_best_baseline = (best_err - aletheia_err) / best_err."""
    thread = generate_thread(n_voices=30, n_positions=3, seed=0)
    results = evaluate_batch(
        [thread],
        normalizer=MockClaimNormalizer(),
        baselines=[UniqueAccountsBaseline(), UniqueCommentsBaseline()],
    )
    summary = summarize_results(results)

    baselines = summary["baselines"]
    aletheia = summary["aletheia"]
    assert isinstance(baselines, dict)
    assert isinstance(aletheia, dict)

    best_err = min(s["median_relative_error"] for s in baselines.values())
    al_err = aletheia["median_relative_error"]
    expected = (best_err - al_err) / best_err if best_err > 0 else 0.0
    assert summary["improvement_over_best_baseline"] == expected


# ---------------------------------------------------------------------------
# make_pilot_batch
# ---------------------------------------------------------------------------


def test_make_pilot_batch_defaults() -> None:
    """Default pilot batch has the PROTOCOL parameters."""
    batch = make_pilot_batch()
    assert len(batch) == 30
    assert all(t.n_voices == 100 for t in batch)
    assert all(t.n_positions == 5 for t in batch)
    assert all(t.paraphrase_level == "medium" for t in batch)


def test_make_pilot_batch_custom() -> None:
    """Custom parameters override defaults."""
    batch = make_pilot_batch(
        n_threads=5, n_voices=50, n_positions=3, paraphrase_level="high"
    )
    assert len(batch) == 5
    assert all(t.n_voices == 50 for t in batch)
    assert all(t.n_positions == 3 for t in batch)
    assert all(t.paraphrase_level == "high" for t in batch)


def test_make_pilot_batch_seeds_are_distinct() -> None:
    """Each thread in the batch has a distinct seed."""
    batch = make_pilot_batch(n_threads=5)
    seeds = [t.seed for t in batch]
    assert len(seeds) == len(set(seeds))