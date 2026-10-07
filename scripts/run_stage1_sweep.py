"""Stage 1 sweep across problem difficulties.

Runs the Stage 1 pilot at multiple values of ``n_positions`` (the number
of conceptually distinct positions in each thread), to check whether
Aletheia's advantage over the baselines is stable as the problem
becomes harder.

Each configuration writes its own JSON file under ``data/results/``.
The script prints a final comparison table.

Usage:
    python scripts/run_stage1_sweep.py
    python scripts/run_stage1_sweep.py --n-positions 2 5 10 --n-threads 30
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 1 sweep.")
    parser.add_argument("--n-threads", type=int, default=30)
    parser.add_argument("--n-voices", type=int, default=30)
    parser.add_argument(
        "--n-positions",
        type=int,
        nargs="+",
        default=[2, 5, 10],
        help="Values of n_positions to sweep.",
    )
    parser.add_argument(
        "--paraphrase-level",
        choices=["low", "medium", "high"],
        default="medium",
    )
    parser.add_argument("--noise-ratio", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--normalizer",
        choices=["mock", "openai", "gemini"],
        default="openai",
    )
    return parser.parse_args()


def _serialize(results) -> list[dict]:
    out = []
    for r in results:
        entry = {
            "seed": r.seed,
            "n_voices": r.n_voices,
            "n_positions": r.n_positions,
            "true_inflation": r.true_inflation,
            "aletheia": asdict(r.aletheia),
            "baselines": {
                name: asdict(metrics) for name, metrics in r.baselines.items()
            },
        }
        out.append(entry)
    return out


def _print_sweep_table(rows: list[dict]) -> None:
    print()
    print("=" * 92)
    print("Stage 1 sweep — summary")
    print("=" * 92)
    header = (
        f"{'n_pos':>6} {'I_true':>8} "
        f"{'Aletheia err':>14} {'Best base err':>15} {'Improv':>10} "
        f"{'Aletheia ARI':>14} {'Best base ARI':>15}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['n_positions']:>6} "
            f"{row['true_inflation']:>8.2f} "
            f"{row['aletheia_err']:>14.4f} "
            f"{row['best_baseline_err']:>15.4f} "
            f"{row['improvement']:>9.2%} "
            f"{row['aletheia_ari']:>14.4f} "
            f"{row['best_baseline_ari']:>15.4f}"
        )
    print()


def main() -> None:
    args = _parse_args()
    normalizer = get_normalizer(args.normalizer)
    output_dir = _OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for n_pos in args.n_positions:
        print(f"\n>>> Running n_positions = {n_pos} ...")
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

        # Save per-config JSON.
        out_path = output_dir / f"stage1_npos{n_pos}.json"
        payload = {
            "config": {
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

        # Collect row for final table.
        best_name = summary["best_baseline"]
        best_err = summary["baselines"][best_name]["median_relative_error"]
        best_ari = summary["baselines"][best_name]["median_ari"]
        rows.append(
            {
                "n_positions": n_pos,
                "true_inflation": args.n_voices / n_pos,
                "aletheia_err": summary["aletheia"]["median_relative_error"],
                "best_baseline_err": best_err,
                "best_baseline_name": best_name,
                "improvement": summary["improvement_over_best_baseline"],
                "aletheia_ari": summary["aletheia"]["median_ari"],
                "best_baseline_ari": best_ari,
            }
        )
        print(f"    saved: {out_path}")

    _print_sweep_table(rows)

    combined_path = output_dir / "stage1_sweep.json"
    combined_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Combined summary: {combined_path}")


if __name__ == "__main__":
    main()