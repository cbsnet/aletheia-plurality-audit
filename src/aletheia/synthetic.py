"""Synthetic thread generator with controlled ground truth.

This module generates synthetic governance threads where the number of
apparent voices (comments) and the number of conceptually distinct
positions are controlled by the caller. The ground-truth assignment of
each comment to a position is returned alongside the generated text.

The generator has two intended uses:

1. **Testing** — the template-based backend (default) is fully
   deterministic, requires no network access, and produces the same
   output given the same seed. It is what the CI runs.
2. **Stage 2 of the evaluation** — the same interface can be backed by
   an LLM for more realistic paraphrases. That backend is optional and
   lives outside this module.

The design goal is that the *ground truth* is exact and the *surface
form* varies in a controlled way. Clustering on raw embeddings should
degrade as the paraphrase level increases; LLM-based claim normalization
should recover the underlying positions.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

ParaphraseLevel = Literal["low", "medium", "high"]

# ---------------------------------------------------------------------------
# Canonical positions
# ---------------------------------------------------------------------------
# Ten positions on a single hypothetical governance topic (a reform
# proposal). They are semantically distinct but plausibly part of the
# same debate. The first N are used when a thread requests N positions.

_POSITION_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "support_as_is": {
        "low": [
            "I support the reform as-is.",
            "The reform as-is has my support.",
            "I am in favor of the reform as-is.",
        ],
        "medium": [
            "I back the proposal in its current form.",
            "The proposal in its current form gets my backing.",
            "My support goes to the proposal in its current form.",
        ],
        "high": [
            "Honestly, after thinking about it a lot, I think we should just go with it.",
            "There is no reason to change anything here; the current version works.",
            "I have read all the alternatives and none beats what we already have.",
        ],
    },
    "reject": {
        "low": [
            "I reject the reform.",
            "The reform should be rejected.",
            "My position is to reject the reform.",
        ],
        "medium": [
            "I am against the proposal.",
            "The proposal ought to be turned down.",
            "My stance is opposition to the proposal.",
        ],
        "high": [
            "We should not be doing this at all, and I mean it.",
            "This whole thing is a mistake and we should stop it now.",
            "Count me firmly on the no side, for reasons I will not repeat again.",
        ],
    },
    "postpone": {
        "low": [
            "We should postpone the decision.",
            "The decision should be postponed.",
            "Postponing the decision is what we need.",
        ],
        "medium": [
            "We ought to defer this for now.",
            "Deferring this for now seems wiser.",
            "My preference is to defer this for now.",
        ],
        "high": [
            "Why rush? Let us wait and see how things develop.",
            "I think the wise move here is patience, not action.",
            "Let us not decide today; there is no cost to waiting.",
        ],
    },
    "amend_specific": {
        "low": [
            "I support the reform with amendments.",
            "The reform with amendments has my support.",
            "Supporting the reform with amendments is right.",
        ],
        "medium": [
            "I back the proposal, but it needs changes.",
            "The proposal gets my backing if it is revised.",
            "My support goes to a revised version of the proposal.",
        ],
        "high": [
            "The direction is right but the details are wrong; fix them and I am in.",
            "Yes, but not like this. Change a few things and I will vote for it.",
            "I would support this if the drafting were cleaned up in specific places.",
        ],
    },
    "alternative": {
        "low": [
            "We should adopt an alternative reform.",
            "An alternative reform is what we should adopt.",
            "Adopting an alternative reform makes sense.",
        ],
        "medium": [
            "There is a better approach than the one proposed.",
            "We ought to consider a different route entirely.",
            "My proposal is to take a different approach altogether.",
        ],
        "high": [
            "Forget this draft. Here is what we should actually be doing.",
            "The proposal is fine, but it is solving the wrong problem. Try this instead.",
            "We are arguing about the wrong document; the real fix is elsewhere.",
        ],
    },
    "process_legitimacy": {
        "low": [
            "The process itself is not legitimate.",
            "I question the legitimacy of the process.",
            "The legitimacy of this process is questionable.",
        ],
        "medium": [
            "The way this was put together is questionable.",
            "I do not accept the procedure that led here.",
            "The procedure behind this is what concerns me.",
        ],
        "high": [
            "Why are we even voting on this? Nobody agreed to this process.",
            "The real problem is not the text, it is who decided we would discuss it.",
            "This thread exists because of a decision that was never properly made.",
        ],
    },
    "conditional": {
        "low": [
            "I support the reform if conditions are met.",
            "The reform has my support under conditions.",
            "Supporting the reform under conditions is right.",
        ],
        "medium": [
            "I back the proposal only if certain things happen first.",
            "The proposal gets my backing provided that some conditions hold.",
            "My support is conditional on specific requirements.",
        ],
        "high": [
            "I am in, but only if we get guarantees in writing first.",
            "Yes, provided that the funding question is answered beforehand.",
            "You have my vote the moment two specific things are addressed.",
        ],
    },
    "not_far_enough": {
        "low": [
            "The reform does not go far enough.",
            "The reform should go further.",
            "Going further than the reform is needed.",
        ],
        "medium": [
            "The proposal is too timid.",
            "The proposal is not ambitious enough.",
            "A bolder version of the proposal is what we need.",
        ],
        "high": [
            "This is a half-measure. We should be aiming much higher.",
            "If we are doing this at all, let us do it properly and go all the way.",
            "The draft is fine as a first step, but it stops where the real work begins.",
        ],
    },
    "technically_unfeasible": {
        "low": [
            "The reform is technically unfeasible.",
            "The reform cannot be implemented technically.",
            "Technical implementation of the reform is not possible.",
        ],
        "medium": [
            "The proposal cannot be built as described.",
            "The proposal is not implementable in practice.",
            "Implementing the proposal as written is not realistic.",
        ],
        "high": [
            "This looks nice on paper but it will not survive contact with reality.",
            "Nobody who has actually built one of these would propose this.",
            "The engineering alone makes this a non-starter.",
        ],
    },
    "clarification": {
        "low": [
            "I need clarification on the reform.",
            "Clarification on the reform is needed.",
            "Clarifying the reform is what I need.",
        ],
        "medium": [
            "I would like more details about the proposal.",
            "More details about the proposal are needed.",
            "My request is for more details about the proposal.",
        ],
        "high": [
            "Before I say anything, can someone explain what this actually changes?",
            "I am not voting until I understand what the practical effect would be.",
            "Could we get a plain-language summary of what this does?",
        ],
    },
}

_POSITION_ORDER: list[str] = list(_POSITION_TEMPLATES.keys())

# ---------------------------------------------------------------------------
# Off-topic noise comments
# ---------------------------------------------------------------------------

_NOISE_COMMENTS: list[str] = [
    "Anyone tried the new cafe that opened downtown?",
    "The weather this week has been unusually warm.",
    "I just finished a great book about medieval history.",
    "My cat learned a new trick yesterday, very proud.",
    "Has anyone seen the new season of that show?",
    "I am thinking of changing my desktop wallpaper.",
    "The traffic this morning was surprisingly light.",
    "Just came back from a long walk in the park.",
    "I have been trying to learn to cook Thai food.",
    "Does anyone have recommendations for a good podcast?",
]


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticThread:
    """A synthetic governance thread with exact ground truth.

    Attributes:
        comments: The generated comment strings, in shuffled order.
        true_positions: For each comment, the index of the position it
            belongs to, or ``-1`` if the comment is off-topic noise.
        n_voices: Number of comments (= ``len(comments)``).
        n_positions: Number of distinct conceptual positions.
        true_inflation: ``n_voices / n_positions``. The quantity that
            Aletheia is designed to estimate.
        paraphrase_level: The level used to generate the thread.
        noise_ratio: The fraction of comments that are off-topic.
        seed: The seed used to generate the thread.
    """

    comments: list[str]
    true_positions: list[int]
    n_voices: int
    n_positions: int
    true_inflation: float
    paraphrase_level: ParaphraseLevel
    noise_ratio: float
    seed: int


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_thread(
    n_voices: int,
    n_positions: int,
    paraphrase_level: ParaphraseLevel = "medium",
    noise_ratio: float = 0.0,
    seed: int = 0,
) -> SyntheticThread:
    """Generate a synthetic governance thread with exact ground truth.

    Args:
        n_voices: Total number of comments to generate. Must be >= 1.
        n_positions: Number of distinct positions. Must be in [1, 10].
        paraphrase_level: One of ``"low"``, ``"medium"``, ``"high"``.
        noise_ratio: Fraction of comments that are off-topic, in [0, 1).
        seed: Random seed for reproducibility.

    Returns:
        A :class:`SyntheticThread` with comments and ground truth.

    Raises:
        ValueError: If any argument is out of range.
    """
    if n_voices < 1:
        raise ValueError(f"n_voices must be >= 1, got {n_voices}")
    if not 1 <= n_positions <= len(_POSITION_ORDER):
        raise ValueError(
            f"n_positions must be in [1, {len(_POSITION_ORDER)}], got {n_positions}"
        )
    if paraphrase_level not in ("low", "medium", "high"):
        raise ValueError(f"invalid paraphrase_level: {paraphrase_level!r}")
    if not 0.0 <= noise_ratio < 1.0:
        raise ValueError(f"noise_ratio must be in [0, 1), got {noise_ratio}")

    rng = random.Random(seed)

    n_noise = round(n_voices * noise_ratio)
    n_clean = n_voices - n_noise

    position_names = _POSITION_ORDER[:n_positions]

    # Distribute clean comments across positions as evenly as possible,
    # with any remainder assigned to the first positions.
    base = n_clean // n_positions
    remainder = n_clean % n_positions
    per_position = [base + (1 if i < remainder else 0) for i in range(n_positions)]

    comments: list[str] = []
    true_positions: list[int] = []

    for pos_idx, count in enumerate(per_position):
        templates = _POSITION_TEMPLATES[position_names[pos_idx]][paraphrase_level]
        for _ in range(count):
            comments.append(rng.choice(templates))
            true_positions.append(pos_idx)

    for _ in range(n_noise):
        comments.append(rng.choice(_NOISE_COMMENTS))
        true_positions.append(-1)

    # Shuffle comments and ground truth together, so clusters are not
    # contiguous in the output.
    order = list(range(len(comments)))
    rng.shuffle(order)
    comments = [comments[i] for i in order]
    true_positions = [true_positions[i] for i in order]

    return SyntheticThread(
        comments=comments,
        true_positions=true_positions,
        n_voices=n_voices,
        n_positions=n_positions,
        true_inflation=n_voices / n_positions,
        paraphrase_level=paraphrase_level,
        noise_ratio=noise_ratio,
        seed=seed,
    )


def generate_batch(
    n_threads: int,
    n_voices: int,
    n_positions: int,
    paraphrase_level: ParaphraseLevel = "medium",
    noise_ratio: float = 0.0,
    seed_base: int = 0,
) -> list[SyntheticThread]:
    """Generate a batch of threads with distinct, derived seeds.

    The seed of thread ``i`` is ``seed_base + i``. This makes the batch
    fully reproducible while giving each thread an independent draw.

    Args:
        n_threads: Number of threads to generate.
        n_voices: Comments per thread.
        n_positions: Positions per thread.
        paraphrase_level: Paraphrase level for all threads.
        noise_ratio: Noise ratio for all threads.
        seed_base: Base seed; thread ``i`` uses ``seed_base + i``.

    Returns:
        A list of :class:`SyntheticThread`.
    """
    return [
        generate_thread(
            n_voices=n_voices,
            n_positions=n_positions,
            paraphrase_level=paraphrase_level,
            noise_ratio=noise_ratio,
            seed=seed_base + i,
        )
        for i in range(n_threads)
    ]