"""Tests for naive baselines."""

from __future__ import annotations

from aletheia.baselines import (
    JaccardKeywordsBaseline,
    RawEmbeddingsBaseline,
    UniqueAccountsBaseline,
    UniqueCommentsBaseline,
    get_baselines,
)
from aletheia.embedders import HashingEmbedder

# ---------------------------------------------------------------------------
# UniqueAccountsBaseline
# ---------------------------------------------------------------------------


def test_unique_accounts_empty() -> None:
    """Empty input -> empty labels."""
    assert UniqueAccountsBaseline().normalize([]) == []


def test_unique_accounts_every_comment_own_cluster() -> None:
    """Every comment gets its own unique label."""
    labels = UniqueAccountsBaseline().normalize(["a", "b", "c"])
    assert labels == [0, 1, 2]


def test_unique_accounts_identical_texts_still_separate() -> None:
    """Even identical texts get separate labels."""
    labels = UniqueAccountsBaseline().normalize(["a", "a", "a"])
    assert labels == [0, 1, 2]


def test_unique_accounts_name() -> None:
    """The name attribute matches the protocol."""
    assert UniqueAccountsBaseline().name == "unique_accounts"


# ---------------------------------------------------------------------------
# UniqueCommentsBaseline
# ---------------------------------------------------------------------------


def test_unique_comments_empty() -> None:
    """Empty input -> empty labels."""
    assert UniqueCommentsBaseline().normalize([]) == []


def test_unique_comments_groups_exact_duplicates() -> None:
    """Identical texts share a label, distinct texts do not."""
    labels = UniqueCommentsBaseline().normalize(["a", "b", "a", "c", "b"])
    assert labels == [0, 1, 0, 2, 1]


def test_unique_comments_paraphrases_stay_separate() -> None:
    """Different wording -> different labels, even if semantically same."""
    labels = UniqueCommentsBaseline().normalize(
        ["I support the reform", "The reform has my support"]
    )
    assert labels == [0, 1]


def test_unique_comments_name() -> None:
    """The name attribute matches the protocol."""
    assert UniqueCommentsBaseline().name == "unique_comments"


# ---------------------------------------------------------------------------
# RawEmbeddingsBaseline
# ---------------------------------------------------------------------------


def test_raw_embeddings_empty() -> None:
    """Empty input -> empty labels."""
    emb = RawEmbeddingsBaseline(embedder=HashingEmbedder(n_features=64))
    assert emb.normalize([]) == []


def test_raw_embeddings_groups_identical_comments() -> None:
    """Identical comments share a cluster."""
    emb = RawEmbeddingsBaseline(embedder=HashingEmbedder(n_features=128))
    labels = emb.normalize(["alpha beta gamma", "alpha beta gamma", "alpha beta gamma"])
    assert len(set(labels)) == 1


def test_raw_embeddings_separates_unrelated_comments() -> None:
    """Unrelated comments end up in different clusters."""
    emb = RawEmbeddingsBaseline(
        embedder=HashingEmbedder(n_features=512), similarity_threshold=0.9
    )
    labels = emb.normalize(
        [
            "completely unrelated sentence about cats",
            "quarterly financial report for the fiscal year",
            "the weather forecast predicts heavy rain",
        ]
    )
    assert len(set(labels)) == 3


def test_raw_embeddings_deterministic() -> None:
    """Same input -> same output."""
    emb_a = RawEmbeddingsBaseline(embedder=HashingEmbedder(n_features=128))
    emb_b = RawEmbeddingsBaseline(embedder=HashingEmbedder(n_features=128))
    texts = ["alpha beta", "gamma delta", "alpha beta"]
    assert emb_a.normalize(texts) == emb_b.normalize(texts)


def test_raw_embeddings_name() -> None:
    """The name attribute matches the protocol."""
    assert RawEmbeddingsBaseline().name == "raw_embeddings"


# ---------------------------------------------------------------------------
# JaccardKeywordsBaseline
# ---------------------------------------------------------------------------


def test_jaccard_empty() -> None:
    """Empty input -> empty labels."""
    assert JaccardKeywordsBaseline().normalize([]) == []


def test_jaccard_groups_similar_comments() -> None:
    """Comments sharing keywords are grouped."""
    labels = JaccardKeywordsBaseline(threshold=0.3).normalize(
        [
            "the reform addresses our concerns",
            "our concerns are addressed by the reform",
            "I reject the entire proposal",
        ]
    )
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_jaccard_name() -> None:
    """The name attribute matches the protocol."""
    assert JaccardKeywordsBaseline().name == "jaccard_keywords"


# ---------------------------------------------------------------------------
# get_baselines registry
# ---------------------------------------------------------------------------


def test_get_baselines_returns_four() -> None:
    """The registry returns exactly four baselines."""
    baselines = get_baselines()
    assert len(baselines) == 4


def test_get_baselines_names_are_distinct() -> None:
    """All baseline names are distinct."""
    baselines = get_baselines()
    names = [b.name for b in baselines]
    assert len(names) == len(set(names))


def test_get_baselines_all_have_normalize() -> None:
    """Every baseline exposes a callable normalize method."""
    for b in get_baselines():
        assert callable(b.normalize)
        result = b.normalize(["alpha beta", "gamma delta"])
        assert isinstance(result, list)
        assert len(result) == 2