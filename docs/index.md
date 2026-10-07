# Aletheia — Documentation

Project: **Aletheia — Plurality Audit for Synthetic Governance Threads**

This directory contains the narrative documentation for the project. It
complements the README (which is a framing document) and `PROTOCOL.md`
(which is the preregistered analysis plan).

---

## Contents

- [`methodology.md`](methodology.md) — the epistemological grounding,
  and the reasons for the design choices.
- [`evaluation.md`](evaluation.md) — a narrative summary of the Stage 1
  and Stage 2 results.
- [`limitations.md`](limitations.md) — the full declaration of what the
  project does and does not claim.
- [`figures/`](figures/) — exported figures (PNG, SVG, interactive HTML).

---

## Reading order

For a reviewer with limited time:

1. The [README](../README.md), section "For reviewers".
2. [`PROTOCOL.md`](../PROTOCOL.md), paying attention to the commit
   timestamps relative to the result files.
3. [`notebooks/02_confirmatory.ipynb`](../notebooks/02_confirmatory.ipynb).
4. The three documents in this directory, in the order listed above.

---

## In one paragraph

Aletheia measures the gap between the apparent plurality of voices in a
governance thread and the real plurality of conceptually distinct
positions expressed. The metric is the inflation factor `I = P_apparent
/ P_real`. The method uses an LLM to normalize claims and semantic
clustering to group them. On synthetic threads with `I ≤ 6`, it reduces
median relative error by 36–77% versus the best of four naive baselines.
On synthetic threads with `I = 15`, it is worse than the best baseline,
due to LLM over-fragmentation. Both the positive and the negative results
are reported with the same prominence.