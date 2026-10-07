"""Generate static figures and interactive HTML from saved results.

Reads the Stage 1 JSON files under ``data/results/`` and produces:

- PNG and SVG versions of the Plotly figures, under ``docs/figures/``
- Interactive HTML versions, under ``docs/figures/html/``

Requires ``kaleido`` for PNG/SVG export (install with ``pip install -e ".[viz]"``).

Usage:
    python scripts/generate_figures.py
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from aletheia.evaluation import ThreadResult
from aletheia.metrics import ClusteringMetrics
from aletheia.visualize import (
    figure_clustering_quality,
    figure_improvement_gauge,
    figure_inflation_scatter,
    figure_median_error_comparison,
    figure_per_thread_errors,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESULTS_DIR = _REPO_ROOT / "data" / "results"
_FIGURES_DIR = _REPO_ROOT / "docs" / "figures"
_HTML_DIR = _FIGURES_DIR / "html"


def _clustering_metrics_from_dict(d: dict) -> ClusteringMetrics:
    """Reconstruct a ClusteringMetrics from a JSON dict."""
    allowed = {f.name for f in fields(ClusteringMetrics)}
    return ClusteringMetrics(**{k: v for k, v in d.items() if k in allowed})


def _thread_results_from_json(path: Path) -> list[ThreadResult]:
    """Load ThreadResult objects from a stage JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for entry in payload["per_thread"]:
        baselines = {
            name: _clustering_metrics_from_dict(m)
            for name, m in entry["baselines"].items()
        }
        results.append(
            ThreadResult(
                seed=entry["seed"],
                n_voices=entry["n_voices"],
                n_positions=entry["n_positions"],
                true_inflation=entry["true_inflation"],
                aletheia=_clustering_metrics_from_dict(entry["aletheia"]),
                baselines=baselines,
            )
        )
    return results


def _export(fig, stem: str, include_svg: bool = True) -> None:
    """Write PNG, SVG, and HTML versions of a figure."""
    png_path = _FIGURES_DIR / f"{stem}.png"
    html_path = _HTML_DIR / f"{stem}.html"

    fig.write_image(str(png_path), scale=2)
    if include_svg:
        svg_path = _FIGURES_DIR / f"{stem}.svg"
        fig.write_image(str(svg_path))
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    print(f"    wrote {png_path.name}, {html_path.name}")


def _generate_for(stem: str, results: list[ThreadResult]) -> None:
    print(f"\n>>> {stem} ({len(results)} threads)")
    _export(figure_median_error_comparison(results), f"{stem}_median_error")
    _export(figure_clustering_quality(results), f"{stem}_quality")
    _export(figure_per_thread_errors(results), f"{stem}_per_thread")
    _export(figure_inflation_scatter(results), f"{stem}_inflation")
    _export(figure_improvement_gauge(results), f"{stem}_improvement")


def main() -> None:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _HTML_DIR.mkdir(parents=True, exist_ok=True)

    files_to_plot = [
        ("stage1_openai", _RESULTS_DIR / "stage1_openai.json"),
        ("stage1_npos2", _RESULTS_DIR / "stage1_npos2.json"),
        ("stage1_npos5", _RESULTS_DIR / "stage1_npos5.json"),
        ("stage1_npos10", _RESULTS_DIR / "stage1_npos10.json"),
    ]

    for stem, path in files_to_plot:
        if not path.exists():
            print(f"skip (not found): {path}")
            continue
        results = _thread_results_from_json(path)
        _generate_for(stem, results)

    # Print a summary table for the sweep.
    sweep_path = _RESULTS_DIR / "stage1_sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        print("\n=== Sweep summary ===")
        for row in sweep:
            print(
                f"n_pos={row['n_positions']:>2}  "
                f"I_true={row['true_inflation']:>5.1f}  "
                f"improv={row['improvement']:+.2%}  "
                f"ARI={row['aletheia_ari']:.4f}"
            )

    print(f"\nAll figures written to: {_FIGURES_DIR}")


if __name__ == "__main__":
    main()