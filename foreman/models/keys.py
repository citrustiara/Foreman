"""API key management — dotenv loading and validation."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("foreman.models.keys")


# Known API key environment variables
KNOWN_KEYS = {
    "GEMINI_API_KEY": "Google Gemini",
    "OPENAI_API_KEY": "OpenAI",
    "DEEPSEEK_API_KEY": "DeepSeek",
    "ANTHROPIC_API_KEY": "Anthropic",
    "OPENROUTER_API_KEY": "OpenRouter",
}


def load_env(repo_root: Path | None = None) -> None:
    """Load .env files from repo root and home directory."""
    # Home directory .env
    home_env = Path.home() / ".env"
    if home_env.exists():
        load_dotenv(home_env, override=False)
        logger.debug("Loaded .env from %s", home_env)

    # Repo root .env
    if repo_root:
        repo_env = repo_root / ".env"
        if repo_env.exists():
            load_dotenv(repo_env, override=False)
            logger.debug("Loaded .env from %s", repo_env)


def get_api_key(env_var: str) -> str | None:
    """Get an API key from environment, returning None if not set."""
    return os.environ.get(env_var)


def validate_keys(required_keys: list[str]) -> list[str]:
    """Validate that required API keys exist. Returns list of missing key names."""
    missing = []
    for key in required_keys:
        if not os.environ.get(key):
            missing.append(key)
    return missing


def available_providers() -> dict[str, bool]:
    """Check which providers have API keys configured."""
    return {name: bool(os.environ.get(key)) for key, name in KNOWN_KEYS.items()}


def key_status() -> dict[str, str]:
    """Get status string for each known key."""
    result = {}
    for key, provider in KNOWN_KEYS.items():
        val = os.environ.get(key)
        if val:
            result[provider] = f"set ({val[:4]}...{val[-4:]})"
        else:
            result[provider] = "not set"
    return result
