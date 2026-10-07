# Methodology

The epistemological grounding of the project, and the reasons for the
design choices that a reviewer might otherwise question.

This document is not a tutorial. It assumes the reader has read the
README and `PROTOCOL.md`.

---

## 1. The problem, stated precisely

Advanced LLMs allow automated agents to flood a digital forum with
coherent, context-aware, stylistically varied comments. Traditional
defenses operate at the level of the *agent*: bot detection, Sybil
resistance, rate limiting, cryptographic identity. All of them share a
common assumption: that a sufficiently sophisticated agent is detectable.

**The assumption is fragile.** An LLM-driven agent produces text that is
statistically indistinguishable from human text on the dimensions that
bot detectors use. It can operate at human-plausible rates. It can
produce surface-level stylistic variation at zero marginal cost. As the
capability of the underlying model increases, the detection task
degrades.

Aletheia does not contest this. It accepts that the agent may be
undetectable, and asks a different question: **what does the human
reviewer need in order to evaluate the thread honestly, even in the
presence of undetectable synthetic agents?**

The answer is not "detect the bots". The answer is "measure the
structure of the deliberation, not the identity of the participants".

---

## 2. Why the inflation factor `I`

Given a governance thread, define:

- `P_apparent` = number of distinct voices (accounts or comments)
- `P_real` = number of conceptually distinct positions expressed

The **inflation factor** is `I = P_apparent / P_real`.

**Why a ratio, not a difference.** A difference (`P_apparent − P_real`)
scales with thread size. A ratio is comparable across threads of
different lengths. A thread with 10 apparent voices and 2 real positions
has the same structural inflation as a thread with 500 apparent voices
and 100 real positions, and the ratio captures this.

**Why a scalar, not a vector.** Plurality could be characterized by many
features: distribution of positions, overlap between positions, entropy
of the position histogram. Each of these is informative; none of them is
a single number that a reviewer can read at a glance. `I` is chosen for
communicability, not for completeness.

**What `I` is not.** It is not a measure of manipulation. A thread with
`I = 1` can be entirely honest (every voice expresses a different
position). A thread with `I = 20` can be entirely organic (a popular
proposal attracts many paraphrases). `I` measures *structural redundancy*,
not intent.

---

## 3. Why the reviewer, not the agent

The threat model is deliberately **epistemic**, not adversarial.

Consider a human reviewer reading a governance thread. The reviewer has
limited attention and a well-documented tendency to use the number of
apparent voices as a heuristic for the level of support. This heuristic
is rational in a world where each voice costs something to produce. It
becomes exploitable when the cost of producing a voice approaches zero.

Aletheia does not attempt to change the incentive structure (that is
outside its scope) or to identify which voices are synthetic (that is
infeasible at the relevant capability level). It intervenes at the point
of *cognition*: it presents the reviewer with a summary of the
argumentative structure, so that the heuristic has something honest to
operate on.

This is why the project targets the *epistemic vulnerability* of the
reviewer, not the *technical fingerprint* of the agent. It is a
statement about where the leverage is, not a limitation of ambition.

---

## 4. Why two stages

The failure mode that the two-stage design prevents is **post-hoc
threshold selection**. If you run one experiment and then choose the
threshold that your result happens to clear, the threshold carries no
information. It is a rationalization, not a test.

The standard remedy in experimental science is to freeze the analysis
plan before observing the data. In a single-researcher project without a
preregistration registry, the closest achievable approximation is:

1. Run an **exploratory** stage on a first set of seeds, without any
   hypothesis testing.
2. Derive thresholds from the observed effect sizes, with a **50% safety
   margin** (take half of the observed effect as the threshold).
3. Freeze the thresholds in the protocol, with a public commit
   timestamp.
4. Run a **confirmatory** stage on fresh seeds, distinct from the first
   set, and report the results against the frozen thresholds.

The commit history of this repository is the audit trail for steps 1–4.
A reviewer can verify that the protocol's threshold amendments predate
the Stage 2 result files.

---

## 5. Why these four baselines

A baseline is useful only if it represents a plausible alternative that a
reviewer might have used instead of the proposed method. The four
baselines here correspond to four natural review strategies:

- **`unique_accounts`**: count distinct accounts. The naive heuristic
  that the project argues is exploitable.
- **`unique_comments`**: count distinct comments verbatim. A slightly
  more careful review that catches obvious copy-paste but not paraphrase.
- **`raw_embeddings`**: embed comments and cluster them. A "similarity
  search" alternative that does no semantic normalization.
- **`jaccard_keywords`**: cluster by keyword overlap. A lexical
  alternative that requires no embedding model.

The four span the space from trivial to non-trivial, and each is
implementable in under a hundred lines. If Aletheia cannot beat the best
of them, the contribution does not justify its complexity.

**The `raw_embeddings` baseline is the most informative.** It shares
everything with Aletheia except the LLM normalizer. The difference
between the two isolates the contribution of the normalization step
(H2 in `PROTOCOL.md`).

---

## 6. Why synthetic data

The alternative — real governance threads with manually annotated
positions — was rejected for three reasons.

**Cost.** Annotating positions in a real thread requires a reader to
identify the underlying conceptual positions, which is the same task the
LLM is being asked to perform. Inter-annotator agreement on this task is
likely low, which would make the ground truth noisy.

**Control.** On synthetic data, the true number of positions, the
paraphrase level, the noise ratio, and the inflation factor are all
known exactly and can be varied independently. On real data, none of
these is controllable, and the effect of each on the metric is
confounded with the others.

**Reproducibility.** The synthetic generator is deterministic and
committed. Any reviewer can regenerate the exact dataset. A manually
annotated real corpus would need to be shipped with the repository, and
its provenance and licensing would raise separate questions.

The trade-off is acknowledged: the results demonstrate that the method
works on a class of controlled inputs, not on any particular real
distribution. This is stated in `docs/limitations.md`.

---

## 7. What is deliberately not done

For completeness, the things we considered and rejected:

- **Fine-tuning an LLM.** The point is to characterize an existing,
  general-purpose model as a normalizer. Fine-tuning would confound the
  claim "this works out of the box" with "this works after training".
- **A human user study.** Cost and time. Declared as a limitation.
- **Adversarial robustness.** Out of scope. Declared as a limitation.
- **Multiple inflation metrics.** One is enough for a portfolio.
- **Cross-lingual evaluation.** English only.

Each of these is a legitimate extension of the project. None of them is
necessary for the claim the project makes.

---

## 8. The methodological claim

The scientific contribution of Aletheia is small. The methodological
contribution is the point: **that a modest project can be structured so
that its claims are auditable, its limits are declared, and its
conclusions are testable by a third party without access to the author.**

This is what the repository is designed to demonstrate. The results are
the medium; the process is the message.