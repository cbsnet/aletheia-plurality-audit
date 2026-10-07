# Changelog

The sequence of decisions and milestones of the Aletheia project, in
chronological order. This is not a release changelog; it is a record of
what was decided and when, so that the reasoning behind the current state
of the repository is auditable.

Dates are in ISO 8601 format.

---

## 2026-10-06 — Project scaffolding

- Created the repository with `README.md`, `PROTOCOL.md`, `pyproject.toml`,
  and a minimal CI workflow.
- Chose the `src/` layout with the package at `src/aletheia/`.
- Configured `ruff` for linting and `pytest` for testing, with a 70%
  coverage threshold.
- The initial `PROTOCOL.md` declared the two-stage design, the four
  baselines, and the kill criteria for H1 and H2.

## 2026-10-06 — Synthetic thread generator

- Implemented `src/aletheia/synthetic.py` with ten canonical positions
  and three paraphrase levels.
- The generator is deterministic given a seed and produces exact ground
  truth by construction.
- Added 12 tests.

## 2026-10-06 — Claim normalizer

- Implemented `src/aletheia/llm.py` with three backends:
  `MockClaimNormalizer` (keyword-based, no network),
  `OpenAIClaimNormalizer`, and `GeminiClaimNormalizer`.
- Both LLM backends cache responses on disk under `.cache/llm/`.
- Added 21 tests, none of which call the network.

## 2026-10-06 — Evaluation metrics

- Implemented `src/aletheia/metrics.py` with ARI, NMI, purity, and
  relative error on the inflation factor `I`.
- Noise comments (ground truth `-1`) are excluded from all metrics.
- Added 16 tests.

## 2026-10-06 — Embedders

- Implemented `src/aletheia/embedders.py` with three backends:
  `HashingEmbedder` (deterministic, no download),
  `SentenceTransformerEmbedder` (local model), and `OpenAIEmbedder`
  (API-backed).
- All embeddings are L2-normalized.
- Added 14 tests.

## 2026-10-06 — Clustering

- Implemented `src/aletheia/cluster.py` using agglomerative clustering
  with average linkage and cosine distance.
- The number of clusters is determined by a similarity threshold, not
  fixed a priori.
- Added 12 tests.

## 2026-10-06 — Baselines

- Implemented the four naive baselines in `src/aletheia/baselines.py`:
  `unique_accounts`, `unique_comments`, `raw_embeddings`, and
  `jaccard_keywords`.
- Added 19 tests.

## 2026-10-06 — Evaluation orchestration

- Implemented `src/aletheia/evaluation.py` to run Aletheia and all
  baselines on a batch of threads and aggregate the results.
- The primary output is `improvement_over_best_baseline`, which is
  positive when Aletheia beats the best baseline and negative otherwise.
- Added 12 tests.

## 2026-10-06 — Visualization

- Implemented `src/aletheia/visualize.py` with five Plotly figures and
  one pyvis graph.
- All figures are testable without rendering.
- Added 14 tests plus 1 marked `requires_viz`.

## 2026-10-07 — Stage 1 pilot (mock)

- Ran `scripts/run_pilot.py` with the mock normalizer.
- This run was a pipeline smoke test, not a scientific result, because
  the mock normalizer is itself one of the baselines.

## 2026-10-07 — Amendment 1: OpenAI as default normalizer

- Gemini `2.0-flash` was retired by Google and `3.8-flash` returned
  repeated 503 errors during the experiment window.
- OpenAI `gpt-4o-mini` was selected as the default normalizer for the
  remainder of the project.
- Gemini remains implemented as an alternative backend.

## 2026-10-07 — Stage 1 pilot (OpenAI)

- Ran `scripts/run_pilot.py --normalizer openai` with `n_positions=5`.
- Result: Aletheia reduced median relative error by 35.71% versus the
  best baseline (`jaccard_keywords`), and improved median ARI from 0.51
  to 0.87.

## 2026-10-07 — Stage 1 sweep

- Ran `scripts/run_stage1_sweep.py` across `n_positions ∈ {2, 5, 10}`.
- Result: Aletheia wins for `I ∈ {3, 6}` (+75.76% and +35.71%) and loses
  for `I = 15` (−20.00%).
- This finding narrowed the claim: Aletheia's advantage is conditional on
  the inflation regime.
- Documented in `PROTOCOL.md`, Amendment 2.

## 2026-10-07 — Fragmentation experiment

- Implemented `src/aletheia/experiments.py` with a merge-aware prompt
  variant.
- Ran `scripts/run_fragmentation_experiment.py` on the hard regime.
- Result: the merge-aware prompt reduced mean predicted clusters from
  5.13 to 4.50 (true value: 2), improving ARI from 0.70 to 0.78 but not
  eliminating the fragmentation.
- Concluded that the failure mode is in the model, not the prompt.
- Documented in `PROTOCOL.md`, Amendment 3.

## 2026-10-07 — Stage 2 confirmatory run

- Ran `scripts/run_confirmatory.py` with `n_threads=50`,
  `seed_base=2000` (fresh seeds), and frozen thresholds.
- Result: H1 and H2 are supported for `I ∈ {3, 6}`; H1 is a confirmed
  null for `I = 15`.
- The Stage 1 and Stage 2 results are coherent.

## 2026-10-07 — Figures and notebooks

- Implemented `scripts/generate_figures.py` to export PNG, SVG, and
  interactive HTML versions of all figures to `docs/figures/`.
- Implemented `scripts/build_notebook_01.py` and
  `scripts/build_notebook_02.py` to generate the two notebooks
  programmatically.
- Added `notebooks/01_pilot_exploratory.ipynb` and
  `notebooks/02_confirmatory.ipynb`.

## 2026-10-07 — Documentation closeout

- Added `docs/limitations.md`, `docs/evaluation.md`, and
  `docs/methodology.md`.
- Removed the placeholder link to Hugging Face Spaces, which was never
  deployed. The `docs/figures/` directory provides the same visual
  content, hosted on GitHub.
- Declared the scientific work complete in the README.

---

## What is intentionally missing

- **Streamlit deployment.** The dashboard exists in `app/streamlit_app.py`
  but is not deployed. The figures in `docs/figures/` serve the same
  purpose without a deployment step.
- **Cross-model replication.** Only `gpt-4o-mini` was used for the
  confirmatory run. Stated as a limitation.
- **Human user study.** Not conducted. Stated as a limitation.
- **Adversarial robustness.** Not addressed. Stated as a limitation.

Each of these is a legitimate extension. None of them is necessary for
the claim the project makes.