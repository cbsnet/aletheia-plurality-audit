"""Stage 2 confirmatory run.

Tests the frozen hypotheses from ``PROTOCOL.md`` (Amendments 2 and 3)
on fresh seeds, distinct from those used in Stage 1.

Frozen hypotheses:

- H1 (positive regime): Aletheia beats the best baseline by >= 15%
  relative on I in {3, 6}.
- H1 (negative regime): Aletheia is expected NOT to beat the best
  baseline on I = 15 (over-fragmentation, documented in Amendment 3).
- H2 (ablation): LLM normalization reduces error by >= 20% relative to
  raw embeddings, on I in {3, 6}.

Results are written to ``data/results/stage2_*.json`` and the hypothesis
checks are printed to stdout.

Usage:
    python scripts/run_confirmatory.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from aletheia.evaluation import evaluate_batch, make_pilot_batch, summarize_results
from aletheia.llm import get_normalizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_DIR = _REPO_ROOT / "data" / "results"

# Frozen thresholds from PROTOCOL.md.
H1_POSITIVE_MIN_IMPROVEMENT = 0.15
H2_MIN_RELATIVE_REDUCTION = 0.20


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 confirmatory.")
    parser.add_argument("--n-threads", type=int, default=50)
    parser.add_argument("--n-voices", type=int, default=30)
    parser.add_argument(
        "--n-positions",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="Difficulty sweep (same as Stage 1).",
    )
    parser.add_argument(
        "--paraphrase-level",
        choices=["low", "medium", "high"],
        default="medium",
    )
    parser.add_argument("--noise-ratio", type=float, default=0.0)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=2000,
        help="Frozen Stage 2 seed base, distinct from Stage 1's 1000.",
    )
    parser.add_argument(
        "--normalizer",
        choices=["mock", "openai", "gemini"],
        default="openai",
    )
    return parser.parse_args()


def _serialize(results) -> list[dict]:
    out = []
    for r in results:
        out.append(
            {
                "seed": r.seed,
                "n_voices": r.n_voices,
                "n_positions": r.n_positions,
                "true_inflation": r.true_inflation,
                "aletheia": asdict(r.aletheia),
                "baselines": {name: asdict(m) for name, m in r.baselines.items()},
            }
        )
    return out


def _print_hypothesis_table(rows: list[dict]) -> None:
    print()
    print("=" * 100)
    print("Stage 2 — hypothesis checks")
    print("=" * 100)
    header = (
        f"{'n_pos':>6} {'I_true':>8} "
        f"{'Improv':>10} {'Aletheia err':>14} "
        f"{'raw_emb err':>14} {'H2 reduction':>14} "
        f"{'H1':>14} {'H2':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        n_pos = row["n_positions"]
        inflation = row["true_inflation"]
        improv = row["improvement"]
        al_err = row["aletheia_err"]
        raw_err = row["raw_embeddings_err"]

        # H2: relative reduction from raw_embeddings to Aletheia.
        h2_reduction = (
            (raw_err - al_err) / raw_err if raw_err > 0 else 0.0
        )

        # H1 evaluation depends on the regime.
        if inflation <= 6:
            h1_status = (
                "SUPPORTED"
                if improv >= H1_POSITIVE_MIN_IMPROVEMENT
                else "not met"
            )
        else:
            h1_status = (
                "expected"
                if improv < H1_POSITIVE_MIN_IMPROVEMENT
                else "UNEXPECTED WIN"
            )

        h2_status = (
            "SUPPORTED"
            if h2_reduction >= H2_MIN_RELATIVE_REDUCTION
            else "not met"
        )

        print(
            f"{n_pos:>6} {inflation:>8.2f} "
            f"{improv:>9.2%} "
            f"{al_err:>14.4f} "
            f"{raw_err:>14.4f} "
            f"{h2_reduction:>13.2%} "
            f"{h1_status:>14} {h2_status:>10}"
        )
    print()


def main() -> None:
    args = _parse_args()
    normalizer = get_normalizer(args.normalizer)
    output_dir = _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for n_pos in args.n_positions:
        print(f"\n>>> Stage 2: n_positions = {n_pos} ...")
        threads = make_pilot_batch(
            n_threads=args.n_threads,
            n_voices=args.n_voices,
            n_positions=n_pos,
            paraphrase_level=args.paraphrase_level,
            noise_ratio=args.noise_ratio,
            seed_base=args.seed_base,
        )
        results = evaluate_batch(threads, normalizer=normalizer)
        summary = summarize_results(results)

        out_path = output_dir / f"stage2_npos{n_pos}.json"
        payload = {
            "config": {
                "stage": 2,
                "n_threads": args.n_threads,
                "n_voices": args.n_voices,
                "n_positions": n_pos,
                "paraphrase_level": args.paraphrase_level,
                "noise_ratio": args.noise_ratio,
                "seed_base": args.seed_base,
                "normalizer": args.normalizer,
            },
            "summary": summary,
            "per_thread": _serialize(results),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"    saved: {out_path}")

        best_name = summary["best_baseline"]
        rows.append(
            {
                "n_positions": n_pos,
                "true_inflation": args.n_voices / n_pos,
                "aletheia_err": summary["aletheia"]["median_relative_error"],
                "best_baseline_err": summary["baselines"][best_name][
                    "median_relative_error"
                ],
                "best_baseline_name": best_name,
                "improvement": summary["improvement_over_best_baseline"],
                "raw_embeddings_err": summary["baselines"]["raw_embeddings"][
                    "median_relative_error"
                ],
                "aletheia_ari": summary["aletheia"]["median_ari"],
            }
        )

    _print_hypothesis_table(rows)

    combined_path = output_dir / "stage2_summary.json"
    combined_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Combined summary: {combined_path}")


if __name__ == "__main__":
    main()