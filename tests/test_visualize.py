"""Tests for visualization functions."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from aletheia.baselines import UniqueAccountsBaseline, UniqueCommentsBaseline
from aletheia.evaluation import evaluate_batch
from aletheia.llm import MockClaimNormalizer
from aletheia.synthetic import generate_thread
from aletheia.visualize import (
    figure_clustering_quality,
    figure_improvement_gauge,
    figure_inflation_scatter,
    figure_median_error_comparison,
    figure_per_thread_errors,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_results():
    """A small batch of evaluated threads for figure tests."""
    threads = [
        generate_thread(n_voices=30, n_positions=3, seed=i) for i in range(5)
    ]
    return evaluate_batch(
        threads,
        normalizer=MockClaimNormalizer(),
        baselines=[UniqueAccountsBaseline(), UniqueCommentsBaseline()],
    )


@pytest.fixture
def empty_results():
    """An empty result list."""
    return []


# ---------------------------------------------------------------------------
# figure_median_error_comparison
# ---------------------------------------------------------------------------


def test_median_error_comparison_returns_figure(small_results) -> None:
    """Returns a go.Figure."""
    fig = figure_median_error_comparison(small_results)
    assert isinstance(fig, go.Figure)


def test_median_error_comparison_has_bars(small_results) -> None:
    """Figure contains at least one bar trace with Aletheia and baselines."""
    fig = figure_median_error_comparison(small_results)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)
    # One bar per method: Aletheia + 2 baselines
    assert len(fig.data[0].x) == 3
    assert "Aletheia" in list(fig.data[0].x)


def test_median_error_comparison_handles_empty(empty_results) -> None:
    """Empty results do not crash and still produce a figure."""
    fig = figure_median_error_comparison(empty_results)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# figure_clustering_quality
# ---------------------------------------------------------------------------


def test_clustering_quality_returns_figure(small_results) -> None:
    """Returns a go.Figure."""
    fig = figure_clustering_quality(small_results)
    assert isinstance(fig, go.Figure)


def test_clustering_quality_has_two_bar_traces(small_results) -> None:
    """Two traces: ARI and NMI."""
    fig = figure_clustering_quality(small_results)
    assert len(fig.data) == 2
    names = {trace.name for trace in fig.data}
    assert names == {"ARI", "NMI"}
    for trace in fig.data:
        assert isinstance(trace, go.Bar)


def test_clustering_quality_handles_empty(empty_results) -> None:
    """Empty results still produce a valid figure."""
    fig = figure_clustering_quality(empty_results)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# figure_per_thread_errors
# ---------------------------------------------------------------------------


def test_per_thread_errors_returns_figure(small_results) -> None:
    """Returns a go.Figure."""
    fig = figure_per_thread_errors(small_results)
    assert isinstance(fig, go.Figure)


def test_per_thread_errors_has_one_box_per_method(small_results) -> None:
    """One box trace per method (Aletheia + baselines)."""
    fig = figure_per_thread_errors(small_results)
    assert len(fig.data) == 3
    for trace in fig.data:
        assert isinstance(trace, go.Box)
    names = [t.name for t in fig.data]
    assert "Aletheia" in names


def test_per_thread_errors_handles_empty(empty_results) -> None:
    """Empty results still produce a valid figure."""
    fig = figure_per_thread_errors(empty_results)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# figure_inflation_scatter
# ---------------------------------------------------------------------------


def test_inflation_scatter_returns_figure(small_results) -> None:
    """Returns a go.Figure."""
    fig = figure_inflation_scatter(small_results)
    assert isinstance(fig, go.Figure)


def test_inflation_scatter_has_diagonal(small_results) -> None:
    """The identity line is present and named 'Perfect estimate'."""
    fig = figure_inflation_scatter(small_results)
    names = [t.name for t in fig.data]
    assert "Perfect estimate" in names
    assert "Aletheia" in names


def test_inflation_scatter_handles_empty(empty_results) -> None:
    """Empty results: no diagonal, but still a valid figure."""
    fig = figure_inflation_scatter(empty_results)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# figure_improvement_gauge
# ---------------------------------------------------------------------------


def test_improvement_gauge_returns_figure(small_results) -> None:
    """Returns a go.Figure with one bar."""
    fig = figure_improvement_gauge(small_results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)


def test_improvement_gauge_handles_empty(empty_results) -> None:
    """Empty results produce a valid figure with a zero value."""
    fig = figure_improvement_gauge(empty_results)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


# ---------------------------------------------------------------------------
# figure_genealogy_pyvis (requires optional dependency)
# ---------------------------------------------------------------------------


@pytest.mark.requires_viz
def test_genealogy_pyvis_writes_html(tmp_path) -> None:
    """pyvis graph is written to HTML and the path is returned."""
    pytest.importorskip("pyvis")
    from aletheia.visualize import figure_genealogy_pyvis

    comments = ["alpha beta", "gamma delta", "alpha beta"]
    labels = [0, 1, 0]
    output = tmp_path / "graph.html"
    result_path = figure_genealogy_pyvis(
        comments, labels, output_html=str(output)
    )
    assert result_path == str(output)
    assert output.exists()
    assert output.stat().st_size > 0