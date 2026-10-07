"""Configuration loader.

Reads a local ``.env`` file if present, then falls back to system
environment variables. This lets the project run on any machine
without requiring OS-level configuration, while remaining compatible
with users who prefer system environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


def get_openai_key() -> str | None:
    """Return the OpenAI API key, or None if not configured."""
    return os.getenv("OPENAI_API_KEY")


def get_gemini_key() -> str | None:
    """Return the Gemini API key, or None if not configured."""
    return os.getenv("GEMINI_API_KEY")


def get_default_normalizer() -> str:
    """Return the default normalizer backend name.

    One of: ``openai``, ``gemini``, ``mock``. Defaults to ``mock``
    so that tests and CI run without any API keys.
    """
    return os.getenv("ALETHEIA_NORMALIZER", "mock")