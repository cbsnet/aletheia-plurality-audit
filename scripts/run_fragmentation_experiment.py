"""Fragmentation experiment.

Tests whether an aggressively merge-oriented prompt reduces the
over-fragmentation observed at ``n_positions=2`` in the Stage 1 sweep.

Runs both the baseline prompt and the merge-aware prompt on the same
threads, saves both result sets to JSON, and prints a comparison table.

Usage:
    python scripts/run_fragmentation_experiment.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from aletheia.config import get_openai_key
from aletheia.evaluation import evaluate_batch, make_pilot_batch, summarize_results
from aletheia.experiments import OpenAIMergeAwareNormalizer
from aletheia.llm import OpenAIClaimNormalizer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_DIR = _REPO_ROOT / "data" / "results"


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


def main() -> None:
    key = get_openai_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    n_threads = 30
    n_voices = 30
    n_positions = 2
    seed_base = 1000

    print(f"Generating {n_threads} threads (n_positions={n_positions}, "
          f"n_voices={n_voices}, seed_base={seed_base})...")
    threads = make_pilot_batch(
        n_threads=n_threads,
        n_voices=n_voices,
        n_positions=n_positions,
        seed_base=seed_base,
    )

    variants = [
        ("baseline_prompt", OpenAIClaimNormalizer(api_key=key)),
        ("merge_prompt", OpenAIMergeAwareNormalizer(api_key=key)),
    ]

    collected: dict[str, dict] = {}

    for name, normalizer in variants:
        print(f"\n>>> Running variant: {name}")
        results = evaluate_batch(threads, normalizer=normalizer)
        summary = summarize_results(results)
        collected[name] = {"summary": summary, "results": results}

        out_path = _OUTPUT_DIR / f"fragmentation_{name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "variant": name,
                    "n_positions": n_positions,
                    "n_voices": n_voices,
                    "n_threads": n_threads,
                    "seed_base": seed_base,
                    "summary": summary,
                    "per_thread": _serialize(results),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"    saved: {out_path}")

    # Comparison table
    print()
    print("=" * 100)
    print("Fragmentation experiment — comparison")
    print("=" * 100)
    header = (
        f"{'Variant':<18} {'Err':>10} {'ARI':>10} {'NMI':>10} "
        f"{'Purity':>10} {'n_pred':>10} {'n_true':>10}"
    )
    print(header)
    print("-" * len(header))

    for name, data in collected.items():
        results = data["results"]
        s = data["summary"]
        n = len(results)
        mean_pred = sum(r.aletheia.n_predicted for r in results) / n
        mean_true = sum(r.aletheia.n_true for r in results) / n
        mean_purity = sum(r.aletheia.purity for r in results) / n
        print(
            f"{name:<18} "
            f"{s['aletheia']['median_relative_error']:>10.4f} "
            f"{s['aletheia']['median_ari']:>10.4f} "
            f"{s['aletheia']['median_nmi']:>10.4f} "
            f"{mean_purity:>10.4f} "
            f"{mean_pred:>10.2f} "
            f"{mean_true:>10.2f}"
        )

    print()
    print(
        f"Ground truth: {n_positions} positions per thread, "
        f"I_true = {n_voices / n_positions:.2f}"
    )
    print()


if __name__ == "__main__":
    main()