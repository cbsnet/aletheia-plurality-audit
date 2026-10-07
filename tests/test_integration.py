"""End-to-end integration test.

Runs the complete Aletheia pipeline (synthetic generator -> normalizer
-> metrics -> evaluation) with the deterministic mock normalizer, on a
controlled dataset where the expected behavior is known.

The point of this test is not to assert a specific numeric outcome, but
to verify that the pieces fit together without crashing and that the
aggregate metrics are within their expected ranges. It is the test that
a reviewer would run first after a fresh install.
"""

from __future__ import annotations

from aletheia.baselines import get_baselines
from aletheia.evaluation import evaluate_batch, summarize_results
from aletheia.llm import MockClaimNormalizer
from aletheia.synthetic import generate_batch


def test_end_to_end_pipeline_runs() -> None:
    """Full pipeline on a small batch completes without error."""
    threads = generate_batch(
        n_threads=5,
        n_voices=30,
        n_positions=5,
        paraphrase_level="medium",
        noise_ratio=0.0,
        seed_base=42,
    )
    normalizer = MockClaimNormalizer()
    baselines = get_baselines()
    results = evaluate_batch(threads, normalizer=normalizer, baselines=baselines)

    assert len(results) == 5
    for r in results:
        assert r.n_voices == 30
        assert r.n_positions == 5
        assert len(r.baselines) == 4


def test_end_to_end_summary_has_expected_keys() -> None:
    """Aggregate summary has the keys the notebooks and scripts rely on."""
    threads = generate_batch(
        n_threads=3,
        n_voices=20,
        n_positions=4,
        seed_base=7,
    )
    results = evaluate_batch(
        threads,
        normalizer=MockClaimNormalizer(),
        baselines=get_baselines(),
    )
    summary = summarize_results(results)

    assert summary["n_threads"] == 3
    assert "aletheia" in summary
    assert "baselines" in summary
    assert "median_relative_error" in summary["aletheia"]
    assert "best_baseline" in summary
    assert "improvement_over_best_baseline" in summary
    assert isinstance(summary["improvement_over_best_baseline"], float)


def test_end_to_end_is_deterministic() -> None:
    """Same seeds -> same summary, twice in a row."""
    def run_once():
        threads = generate_batch(
            n_threads=3, n_voices=20, n_positions=4, seed_base=99
        )
        results = evaluate_batch(
            threads,
            normalizer=MockClaimNormalizer(),
            baselines=get_baselines(),
        )
        return summarize_results(results)

    a = run_once()
    b = run_once()
    assert a["aletheia"] == b["aletheia"]
    assert a["best_baseline"] == b["best_baseline"]
    assert a["improvement_over_best_baseline"] == b["improvement_over_best_baseline"]


def test_end_to_end_with_noise() -> None:
    """Pipeline handles noise comments without crashing or NaN."""
    threads = generate_batch(
        n_threads=3,
        n_voices=30,
        n_positions=3,
        noise_ratio=0.1,
        seed_base=1,
    )
    results = evaluate_batch(
        threads,
        normalizer=MockClaimNormalizer(),
        baselines=get_baselines(),
    )
    for r in results:
        assert r.aletheia.relative_error_inflation >= 0.0
        assert r.aletheia.relative_error_inflation != float("inf")
        assert r.aletheia.relative_error_inflation == r.aletheia.relative_error_inflation


def test_end_to_end_all_baselines_produce_metrics() -> None:
    """Every baseline produces a ClusteringMetrics object."""
    threads = generate_batch(n_threads=2, n_voices=20, n_positions=4, seed_base=0)
    results = evaluate_batch(
        threads,
        normalizer=MockClaimNormalizer(),
        baselines=get_baselines(),
    )
    expected_names = {
        "unique_accounts",
        "unique_comments",
        "raw_embeddings",
        "jaccard_keywords",
    }
    for r in results:
        assert set(r.baselines.keys()) == expected_names
        for metrics in r.baselines.values():
            assert -1.0 <= metrics.ari <= 1.0
            assert 0.0 <= metrics.nmi <= 1.0
            assert 0.0 <= metrics.purity <= 1.0