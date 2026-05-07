"""Model profile definitions for primary and secondary models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelProfile:
    """Configuration for a single LLM model."""
    name: str                    # Display name, e.g. "Primary"
    litellm_model: str           # LiteLLM model string, e.g. "gemini/gemini-2.5-pro"
    api_key_env: str             # Env var name for API key, e.g. "GEMINI_API_KEY"
    max_context: int             # Context window size
    encoding: str = "o200k_base" # tiktoken encoding
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0

def resolve_profile(model_string: str) -> ModelProfile:
    """Resolve a model string to a ModelProfile.

    Try to find it in the global OpenRouter cache.
    Finally, fallback to generic profile with reasonable defaults.
    """
    from foreman.models.fetcher import load_models
    
    # Try to find in OpenRouter cache
    cache = load_models()
    for m in cache:
        prefixed = f"openrouter/{m['id']}"
        if m["id"] == model_string or prefixed == model_string or model_string == m.get("name"):
            return ModelProfile(
                name=m["name"],
                litellm_model=prefixed if not model_string.startswith("openrouter/") else model_string,
                api_key_env="OPENROUTER_API_KEY",
                max_context=m["context_window"],
                cost_per_1m_input=m.get("cost_per_1m_input", 0.0),
                cost_per_1m_output=m.get("cost_per_1m_output", 0.0),
            )

    # Generic fallback
    return ModelProfile(
        name=model_string,
        litellm_model=model_string,
        api_key_env="OPENROUTER_API_KEY" if model_string.startswith("openrouter/") else "UNKNOWN_API_KEY",
        max_context=128_000,
    )