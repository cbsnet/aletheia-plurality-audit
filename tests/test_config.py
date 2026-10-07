"""Tests for the configuration loader."""

from __future__ import annotations

from aletheia.config import (
    get_default_normalizer,
    get_gemini_key,
    get_openai_key,
)


def test_get_openai_key_returns_none_when_unset(monkeypatch) -> None:
    """get_openai_key returns None if the environment variable is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert get_openai_key() is None


def test_get_openai_key_returns_value_when_set(monkeypatch) -> None:
    """get_openai_key returns the value when the variable is present."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-value")
    assert get_openai_key() == "sk-test-value"


def test_get_gemini_key_returns_none_when_unset(monkeypatch) -> None:
    """get_gemini_key returns None if the environment variable is missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_gemini_key() is None


def test_get_gemini_key_returns_value_when_set(monkeypatch) -> None:
    """get_gemini_key returns the value when the variable is present."""
    monkeypatch.setenv("GEMINI_API_KEY", "AQ-test-value")
    assert get_gemini_key() == "AQ-test-value"


def test_get_default_normalizer_defaults_to_mock(monkeypatch) -> None:
    """get_default_normalizer returns 'mock' when the variable is unset."""
    monkeypatch.delenv("ALETHEIA_NORMALIZER", raising=False)
    assert get_default_normalizer() == "mock"


def test_get_default_normalizer_returns_custom_value(monkeypatch) -> None:
    """get_default_normalizer returns the custom value when set."""
    monkeypatch.setenv("ALETHEIA_NORMALIZER", "gemini")
    assert get_default_normalizer() == "gemini"