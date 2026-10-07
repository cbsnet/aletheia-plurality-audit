"""Generate notebooks/02_confirmatory.ipynb.

This is the primary notebook in the reviewer path. It reports the
Stage 2 confirmatory results against the frozen hypotheses in
``PROTOCOL.md``.

Usage:
    python scripts/build_notebook_02.py
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT = _REPO_ROOT / "notebooks" / "02_confirmatory.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS = [
    md("""
# Aletheia — Stage 2 confirmatory analysis

This notebook reports the **confirmatory** results of the Aletheia
plurality-audit pipeline, tested against hypotheses frozen in
`PROTOCOL.md` (Amendments 2 and 3) **before** the Stage 2 data was
generated.

If you are reading only one notebook, read this one. The exploratory
work that motivated the hypotheses is in `01_pilot_exploratory.ipynb`.

## Summary of findings

| Regime | `I` | H1 (vs best baseline) | H2 (vs raw embeddings) |
|--------|-----|-----------------------|------------------------|
| Easy   | 3   | **SUPPORTED** (+75.76%, threshold 15%) | **SUPPORTED** (82.64%, threshold 20%) |
| Medium | 6   | **SUPPORTED** (+35.71%, threshold 15%) | **SUPPORTED** (53.57%, threshold 20%) |
| Hard   | 15  | *expected null* (−20.00%) | *not met* (0.00%) |

The negative result on the hard regime is documented in `PROTOCOL.md`,
Amendment 3, and reported here with the same prominence as the positive
results.
"""),
    md("""
## 1. Setup
"""),
    code("""
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

RESULTS_DIR = Path("..") / "data" / "results"
"""),
    md("""
## 2. Frozen thresholds

From `PROTOCOL.md`, the Stage 2 thresholds were calibrated at 50% of
the Stage 1 effect sizes:

- **H1 threshold:** relative improvement on median `I` error >= 15%
  over the best baseline, for `I` in {3, 6}.
- **H2 threshold:** relative reduction in error >= 20% relative to
  the raw-embeddings baseline, for `I` in {3, 6}.

For `I = 15`, the protocol predicts a null result due to
over-fragmentation (Amendment 3). A positive result there would be
reported as "unexpected win".
"""),
    code("""
H1_POSITIVE_MIN = 0.15
H2_MIN = 0.20

def load_stage2(n_pos):
    path = RESULTS_DIR / f"stage2_npos{n_pos}.json"
    return json.loads(path.read_text(encoding="utf-8"))
"""),
    md("""
## 3. Confirmatory results

The table below is the frozen-hypothesis check. It is identical in
structure to the one printed by `scripts/run_confirmatory.py`.
"""),
    code("""
rows = []
for n_pos in (2, 5, 10):
    payload = load_stage2(n_pos)
    s = payload["summary"]
    best = s["best_baseline"]
    best_err = s["baselines"][best]["median_relative_error"]
    al_err = s["aletheia"]["median_relative_error"]
    raw_err = s["baselines"]["raw_embeddings"]["median_relative_error"]
    improv = s["improvement_over_best_baseline"]
    h2_reduction = (raw_err - al_err) / raw_err if raw_err > 0 else 0.0
    I = payload["config"]["n_voices"] / n_pos
    if I <= 6:
        h1 = "SUPPORTED" if improv >= H1_POSITIVE_MIN else "not met"
    else:
        h1 = "expected" if improv < H1_POSITIVE_MIN else "UNEXPECTED WIN"
    h2 = "SUPPORTED" if h2_reduction >= H2_MIN else "not met"
    rows.append((n_pos, I, improv, al_err, raw_err, h2_reduction, h1, h2))

print(f"{'n_pos':>6} {'I':>5} {'Improv':>10} {'Aletheia err':>14} "
      f"{'raw_emb err':>14} {'H2 red.':>10} {'H1':>15} {'H2':>12}")
print("-" * 96)
for n_pos, I, improv, al_err, raw_err, h2_red, h1, h2 in rows:
    print(f"{n_pos:>6} {I:>5.1f} {improv:>9.2%} {al_err:>14.4f} "
          f"{raw_err:>14.4f} {h2_red:>9.2%} {h1:>15} {h2:>12}")
"""),
    md("""
## 4. Interpretation

### 4.1 H1 — supported in the positive regime

For `I` in {3, 6}, Aletheia reduces median relative error on `I` by
**35.71%** and **75.76%** relative to the best baseline. Both exceed
the frozen threshold of 15%. H1 is supported.

### 4.2 H2 — supported in the positive regime

For the same regimes, LLM normalization reduces error by **53.57%** and
**82.64%** relative to raw embeddings alone. Both exceed the frozen
threshold of 20%. H2 is supported.

### 4.3 Null result in the hard regime, as expected

For `I = 15`, Aletheia is **worse** than the best baseline by 20%, and
does **not** reduce error relative to raw embeddings (0.00% reduction).
This matches the prediction in Amendment 3: at high inflation, the LLM
normalizer over-fragments paraphrases of the same position, and its
advantage disappears.

This negative result is reported as a **documented scope limitation**,
not as a failure to be hidden. The scientific contribution of the
project is bounded accordingly: Aletheia is useful in the regime where
the task is to *distinguish* many positions, and degraded where the
task is to *merge* paraphrases of few positions.
"""),
    md("""
## 5. Figures
"""),
    code("""
def _metrics_from_dict(d):
    allowed = {f.name for f in fields(ClusteringMetrics)}
    return ClusteringMetrics(**{k: v for k, v in d.items() if k in allowed})

def _thread_results_from_json(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for e in payload["per_thread"]:
        baselines = {n: _metrics_from_dict(m) for n, m in e["baselines"].items()}
        results.append(ThreadResult(
            seed=e["seed"], n_voices=e["n_voices"], n_positions=e["n_positions"],
            true_inflation=e["true_inflation"],
            aletheia=_metrics_from_dict(e["aletheia"]),
            baselines=baselines,
        ))
    return results

def stage2_results(n_pos):
    return _thread_results_from_json(RESULTS_DIR / f"stage2_npos{n_pos}.json")
"""),
    md("### 5.1 Median error comparison"),
    code("""
for n_pos in (2, 5, 10):
    figure_median_error_comparison(stage2_results(n_pos)).show()
"""),
    md("### 5.2 Clustering quality (ARI / NMI)"),
    code("""
for n_pos in (2, 5, 10):
    figure_clustering_quality(stage2_results(n_pos)).show()
"""),
    md("### 5.3 Per-thread error distribution"),
    code("""
for n_pos in (2, 5, 10):
    figure_per_thread_errors(stage2_results(n_pos)).show()
"""),
    md("### 5.4 Predicted vs true inflation"),
    code("""
for n_pos in (2, 5, 10):
    figure_inflation_scatter(stage2_results(n_pos)).show()
"""),
    md("### 5.5 Improvement over best baseline"),
    code("""
for n_pos in (2, 5, 10):
    figure_improvement_gauge(stage2_results(n_pos)).show()
"""),
    md("""
## 6. Reproducibility

To reproduce these results, install the project with the api and viz
extras, set `OPENAI_API_KEY` in a `.env` file at the repository root,
and run `python scripts/run_confirmatory.py`. All responses are cached
under `.cache/llm/`, so re-running with the same seeds costs zero API
credits after the first run.

Results are committed under `data/results/stage2_*.json` so that a
reviewer can inspect the raw numbers without executing anything.
"""),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Wrote {_OUTPUT}")


if __name__ == "__main__":
    main()