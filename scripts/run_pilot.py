"""Stage 1 exploratory pilot.

Generates a batch of synthetic threads, evaluates Aletheia against the
four naive baselines, prints a comparative table, and writes the raw
results to ``data/results/stage1.json``.

This script is the execution of the Stage 1 described in
``PROTOCOL.md``. It uses OpenAI (gpt-4o-mini) by default. Pass
``--normalizer mock`` to run without network access.

Usage:
    python scripts/run_pilot.py
    python scripts/run_pilot.py --n-threads 50 --n-voices 30
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
    parser = argparse.ArgumentParser(description="Run Stage 1 pilot.")
    parser.add_argument("--n-threads", type=int, default=30)
    parser.add_argument("--n-voices", type=int, default=30)
    parser.add_argument("--n-positions", type=int, default=5)
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
        help="Normalizer backend. Default: openai (gpt-4o-mini).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_OUTPUT_DIR / "stage1.json"),
    )
    return parser.parse_args()


def _print_table(summary: dict) -> None:
    """Print a compact comparative table to stdout."""
    print()
    print("=" * 72)
    print("Stage 1 pilot — summary")
    print("=" * 72)
    print(f"Threads evaluated: {summary['n_threads']}")
    print()

    header = f"{'Method':<24} {'Med. rel. err':>15} {'Med. ARI':>12} {'Med. NMI':>12}"
    print(header)
    print("-" * len(header))

    al = summary["aletheia"]
    print(
        f"{'Aletheia':<24} "
        f"{al['median_relative_error']:>15.4f} "
        f"{al['median_ari']:>12.4f} "
        f"{al['median_nmi']:>12.4f}"
    )

    for name in sorted(summary["baselines"].keys()):
        b = summary["baselines"][name]
        print(
            f"{name:<24} "
            f"{b['median_relative_error']:>15.4f} "
            f"{b['median_ari']:>12.4f} "
            f"{b['median_nmi']:>12.4f}"
        )

    print()
    print(f"Best baseline: {summary['best_baseline']}")
    improvement = summary["improvement_over_best_baseline"]
    sign = "+" if improvement >= 0 else ""
    print(f"Aletheia improvement over best baseline: {sign}{improvement:.2%}")
    print()


def _serialize(results) -> list[dict]:
    """Convert ThreadResult objects into JSON-serializable dicts."""
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


def main() -> None:
    args = _parse_args()

    print(f"Generating {args.n_threads} threads...")
    threads = make_pilot_batch(
        n_threads=args.n_threads,
        n_voices=args.n_voices,
        n_positions=args.n_positions,
        paraphrase_level=args.paraphrase_level,
        noise_ratio=args.noise_ratio,
        seed_base=args.seed_base,
    )

    print(f"Running evaluation with normalizer: {args.normalizer}")
    normalizer = get_normalizer(args.normalizer)
    results = evaluate_batch(threads, normalizer=normalizer)
    summary = summarize_results(results)

    _print_table(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "n_threads": args.n_threads,
            "n_voices": args.n_voices,
            "n_positions": args.n_positions,
            "paraphrase_level": args.paraphrase_level,
            "noise_ratio": args.noise_ratio,
            "seed_base": args.seed_base,
            "normalizer": args.normalizer,
        },
        "summary": summary,
        "per_thread": _serialize(results),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()