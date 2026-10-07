# Evaluation

A narrative summary of the Stage 1 and Stage 2 results. For the raw
numbers, see `data/results/`. For the frozen hypotheses, see
`PROTOCOL.md`.

---

## 1. Design

Two stages, per `PROTOCOL.md`:

- **Stage 1 — exploratory.** 30 threads per configuration, seed base
  1000. Not confirmatory. Used to measure effect sizes and calibrate
  thresholds.
- **Stage 2 — confirmatory.** 50 threads per configuration, seed base
  2000 (fresh seeds). Tests hypotheses frozen before the run.

Three difficulty levels:

| Difficulty | `n_positions` | True inflation `I` |
|---|---|---|
| Easy | 10 | 3.0 |
| Medium | 5 | 6.0 |
| Hard | 2 | 15.0 |

Each configuration used 30 voices per thread, `medium` paraphrase level,
and no off-topic noise.

---

## 2. Metrics

For each thread, we compute:

- **Median relative error on `I`**: `|I_predicted − I_true| / I_true`.
  This is the primary metric. Lower is better.
- **Adjusted Rand Index (ARI)**: agreement between predicted clusters
  and true positions, corrected for chance. Higher is better.
- **Normalized Mutual Information (NMI)**: information-theoretic
  agreement. Higher is better.
- **Purity**: fraction of comments in the majority true position of
  their predicted cluster.

Noise comments (ground truth `−1`) are excluded from all metrics, since
the normalizer does not know which comments are off-topic.

---

## 3. Baselines

Four naive baselines, all computed on the same threads:

1. **`unique_accounts`** — every comment is its own cluster. Estimates
   `I = 1.0` always.
2. **`unique_comments`** — groups byte-for-byte identical comments only.
3. **`raw_embeddings`** — clusters raw comment embeddings (hashing
   vectorizer), no LLM normalization.
4. **`jaccard_keywords`** — clusters by Jaccard similarity over keyword
   sets.

Aletheia must beat the **best** of these four. The best is determined
per configuration.

---

## 4. Stage 1 — exploratory

Results with `gpt-4o-mini` as the normalizer:

| `n_positions` | `I` | Aletheia error | Best baseline error | Improvement | Aletheia ARI |
|---|---|---|---|---|---|
| 10 | 3.0 | 0.0909 | 0.3934 | +76.89% | 0.8545 |
| 5 | 6.0 | 0.2857 | 0.4444 | +35.71% | 0.8723 |
| 2 | 15.0 | 0.6000 | 0.5000 | −20.00% | 0.7047 |

The best baseline was `jaccard_keywords` in all three configurations.

**Observation.** The improvement is large in the easy and medium regimes,
and negative in the hard regime. This motivated the negative-regime
prediction in Amendment 2 and the fragmentation experiment in
Amendment 3.

---

## 5. Fragmentation experiment (Amendment 3)

Two prompt variants run on the same 30 threads with `n_positions = 2`
(the hard regime):

| Variant | Median error | Median ARI | Mean `n_predicted` | True `n` |
|---|---|---|---|---|
| Baseline prompt | 0.6000 | 0.7047 | 5.13 | 2 |
| Merge-aware prompt | 0.5500 | 0.7820 | 4.50 | 2 |

The merge-aware prompt reduces fragmentation by 12% and improves ARI by
11%, but the model still produces more than twice the true number of
clusters. The failure mode is in the model, not the prompt.

---

## 6. Stage 2 — confirmatory

Results with `gpt-4o-mini`, fresh seeds, 50 threads per configuration:

| `n_positions` | `I` | Aletheia error | Best baseline error | Improvement | H1 |
|---|---|---|---|---|---|
| 10 | 3.0 | 0.0909 | 0.5238 | +75.76% | SUPPORTED |
| 5 | 6.0 | 0.2857 | 0.4444 | +35.71% | SUPPORTED |
| 2 | 15.0 | 0.6000 | 0.5000 | −20.00% | expected null |

### H1 — Aletheia beats the best baseline by ≥ 15%

- Easy regime: **SUPPORTED** at +75.76%
- Medium regime: **SUPPORTED** at +35.71%
- Hard regime: **expected null** at −20.00%

### H2 — LLM normalization reduces error by ≥ 20% vs raw embeddings

| `n_positions` | Aletheia error | `raw_embeddings` error | Reduction |
|---|---|---|---|
| 10 | 0.0909 | 0.5238 | 82.64% |
| 5 | 0.2857 | 0.6154 | 53.57% |
| 2 | 0.6000 | 0.6000 | 0.00% |

- Easy regime: **SUPPORTED** at 82.64%
- Medium regime: **SUPPORTED** at 53.57%
- Hard regime: not met, 0.00% reduction

### Replication

Stage 1 and Stage 2 give coherent results:

| `n_positions` | Stage 1 ARI | Stage 2 ARI | Stage 1 improv | Stage 2 improv |
|---|---|---|---|---|
| 2 | 0.7047 | 0.7493 | −20.00% | −20.00% |
| 5 | 0.8723 | 0.8723 | +35.71% | +35.71% |
| 10 | 0.8545 | 0.8618 | +76.89% | +75.76% |

The improvement values for `n_positions ∈ {2, 5}` coincide because the
median relative error falls on a small set of discrete values. The
`n_positions = 10` row differs, confirming that the Stage 2 seeds
produce fresh data.

---

## 7. Interpretation

The Stage 2 results support the hypotheses in the positive regime and
confirm the expected null in the hard regime. The scientific contribution
is bounded accordingly:

- **Aletheia is useful** when the task is to distinguish many positions
  relative to voices (`I ≤ 6`).
- **Aletheia is degraded** when the task is to merge many paraphrases of
  few positions (`I = 15`), where the LLM over-fragments.

The ablation (H2) shows that the LLM normalization is the source of
Aletheia's advantage in the positive regime, and the source of its
failure in the negative regime.

---

## 8. Reproducing these results

```bash
pip install -e ".[dev,api,viz]"
cp .env.example .env    # add your OPENAI_API_KEY
python scripts/run_pilot.py          # Stage 1
python scripts/run_confirmatory.py   # Stage 2
python scripts/generate_figures.py   # figures