"""Experimental variants for the fragmentation investigation.

Kept separate from the core pipeline to make clear that these variants
are exploratory, not part of the main Aletheia method.
"""

from __future__ import annotations

from aletheia.llm import OpenAIClaimNormalizer

_MERGE_PROMPT_TEMPLATE = """You are analyzing a governance thread. Below are {n} \
comments, each prefixed by its index in brackets.

Your task: group the comments into clusters of *conceptually equivalent*
positions.

CRITICAL: Be aggressive about merging. In this thread, many comments are
paraphrases of the same underlying position — different wording, different
length, different framing, but the same claim. Do NOT create a new cluster
just because the wording differs.

Two comments belong to the same cluster if a neutral reader, after reading
both, would say "these two people are making the same point." If you are
unsure whether to merge two clusters, MERGE THEM.

A comment that is off-topic or does not express a position on the main
topic should go into its own singleton cluster.

Return your answer as a JSON object with a single key "clusters" whose
value is a list of lists of comment indices. Every comment index from 0
to {last} must appear in exactly one cluster. Do not repeat any index
and do not repeat any cluster.

Example output format:
{{"clusters": [[0, 5, 12], [1, 3], [2], [4, 7, 8, 9, 10, 11]]}}

Comments:
{comments}

Your JSON answer:"""


class OpenAIMergeAwareNormalizer(OpenAIClaimNormalizer):
    """OpenAI normalizer with an aggressively merge-oriented prompt.

    Designed to test whether the over-fragmentation observed at
    ``n_positions=2`` in the Stage 1 sweep can be mitigated by prompt
    engineering alone.
    """

    def _build_prompt(self, comments: list[str]) -> str:
        numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(comments))
        return _MERGE_PROMPT_TEMPLATE.format(
            n=len(comments),
            last=len(comments) - 1,
            comments=numbered,
        )