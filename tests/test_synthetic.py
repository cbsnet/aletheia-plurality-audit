"""Tests for the synthetic thread generator."""

from __future__ import annotations

import pytest

from aletheia.synthetic import generate_batch, generate_thread


def test_generate_thread_returns_correct_shape() -> None:
    """Basic structural invariants of a generated thread."""
    thread = generate_thread(
        n_voices=50, n_positions=5, paraphrase_level="medium", seed=0
    )
    assert len(thread.comments) == 50
    assert len(thread.true_positions) == 50
    assert thread.n_voices == 50
    assert thread.n_positions == 5
    assert thread.true_inflation == 10.0


def test_generate_thread_is_reproducible() -> None:
    """Same seed -> same output, exactly."""
    a = generate_thread(n_voices=30, n_positions=3, seed=42)
    b = generate_thread(n_voices=30, n_positions=3, seed=42)
    assert a.comments == b.comments
    assert a.true_positions == b.true_positions


def test_generate_thread_different_seeds_differ() -> None:
    """Different seeds -> different outputs (with overwhelming probability)."""
    a = generate_thread(n_voices=30, n_positions=3, seed=1)
    b = generate_thread(n_voices=30, n_positions=3, seed=2)
    assert a.comments != b.comments


def test_true_positions_cover_all_positions() -> None:
    """Every position appears at least once when n_voices >= n_positions."""
    thread = generate_thread(n_voices=50, n_positions=5, seed=0)
    unique_positions = set(thread.true_positions)
    unique_positions.discard(-1)
    assert unique_positions == {0, 1, 2, 3, 4}


def test_noise_ratio_produces_expected_count() -> None:
    """The number of noise comments matches round(n_voices * noise_ratio)."""
    thread = generate_thread(
        n_voices=100, n_positions=5, noise_ratio=0.1, seed=0
    )
    n_noise = sum(1 for p in thread.true_positions if p == -1)
    assert n_noise == 10


def test_noise_ratio_zero_produces_no_noise() -> None:
    """With noise_ratio=0.0, no comment is off-topic."""
    thread = generate_thread(n_voices=50, n_positions=3, noise_ratio=0.0, seed=0)
    assert all(p != -1 for p in thread.true_positions)


def test_paraphrase_levels_are_distinct() -> None:
    """Different paraphrase levels produce different surface forms."""
    low = generate_thread(n_voices=30, n_positions=3, paraphrase_level="low", seed=0)
    high = generate_thread(n_voices=30, n_positions=3, paraphrase_level="high", seed=0)
    # Same ground truth positions, different text
    assert sorted(low.true_positions) == sorted(high.true_positions)
    assert low.comments != high.comments


def test_generate_batch_returns_independent_threads() -> None:
    """A batch of N threads has N distinct seeds and reproducible content."""
    batch = generate_batch(n_threads=5, n_voices=30, n_positions=3, seed_base=100)
    assert len(batch) == 5
    assert [t.seed for t in batch] == [100, 101, 102, 103, 104]
    # Regenerating with the same seed_base gives identical content
    batch2 = generate_batch(n_threads=5, n_voices=30, n_positions=3, seed_base=100)
    assert [t.comments for t in batch] == [t.comments for t in batch2]


def test_invalid_n_voices_raises() -> None:
    """n_voices < 1 is rejected."""
    with pytest.raises(ValueError, match="n_voices must be >= 1"):
        generate_thread(n_voices=0, n_positions=3)


def test_invalid_n_positions_raises() -> None:
    """n_positions outside [1, 10] is rejected."""
    with pytest.raises(ValueError, match="n_positions must be in"):
        generate_thread(n_voices=30, n_positions=0)
    with pytest.raises(ValueError, match="n_positions must be in"):
        generate_thread(n_voices=30, n_positions=11)


def test_invalid_paraphrase_level_raises() -> None:
    """Unknown paraphrase level is rejected."""
    with pytest.raises(ValueError, match="invalid paraphrase_level"):
        generate_thread(n_voices=30, n_positions=3, paraphrase_level="extreme")  # type: ignore[arg-type]


def test_invalid_noise_ratio_raises() -> None:
    """noise_ratio outside [0, 1) is rejected."""
    with pytest.raises(ValueError, match="noise_ratio must be in"):
        generate_thread(n_voices=30, n_positions=3, noise_ratio=-0.1)
    with pytest.raises(ValueError, match="noise_ratio must be in"):
        generate_thread(n_voices=30, n_positions=3, noise_ratio=1.0)