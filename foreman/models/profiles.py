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

    @property
    def is_openai(self) -> bool:
        return self.litellm_model.startswith("gpt-") or self.litellm_model.startswith("o")

    @property
    def is_gemini(self) -> bool:
        return "gemini" in self.litellm_model.lower()

    @property
    def is_anthropic(self) -> bool:
        return "claude" in self.litellm_model.lower()

    @property
    def is_deepseek(self) -> bool:
        return "deepseek" in self.litellm_model.lower()


# Well-known model presets
PRESETS: dict[str, ModelProfile] = {
    "gemini/gemini-2.5-pro": ModelProfile(
        name="Gemini 2.5 Pro",
        litellm_model="gemini/gemini-2.5-pro",
        api_key_env="GEMINI_API_KEY",
        max_context=1_000_000,
        encoding="o200k_base",
        cost_per_1m_input=1.25,
        cost_per_1m_output=10.0,
    ),
    "gemini/gemini-2.5-flash": ModelProfile(
        name="Gemini 2.5 Flash",
        litellm_model="gemini/gemini-2.5-flash",
        api_key_env="GEMINI_API_KEY",
        max_context=1_000_000,
        encoding="o200k_base",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
    ),
    "gpt-4o": ModelProfile(
        name="GPT-4o",
        litellm_model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        max_context=128_000,
        encoding="o200k_base",
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.0,
    ),
    "gpt-4o-mini": ModelProfile(
        name="GPT-4o Mini",
        litellm_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        max_context=128_000,
        encoding="o200k_base",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
    ),
    "deepseek/deepseek-chat": ModelProfile(
        name="DeepSeek V3",
        litellm_model="deepseek/deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
        max_context=131_072,
        encoding="o200k_base",
        cost_per_1m_input=0.27,
        cost_per_1m_output=1.10,
    ),
    "anthropic/claude-sonnet-4-20250514": ModelProfile(
        name="Claude Sonnet 4",
        litellm_model="anthropic/claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        max_context=200_000,
        encoding="o200k_base",
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0,
    ),
    # OpenRouter models (prefix with openrouter/)
    "openrouter/anthropic/claude-sonnet-4": ModelProfile(
        name="Claude Sonnet 4 (OR)",
        litellm_model="openrouter/anthropic/claude-sonnet-4",
        api_key_env="OPENROUTER_API_KEY",
        max_context=200_000,
        encoding="o200k_base",
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0,
    ),
    "openrouter/google/gemini-2.5-pro": ModelProfile(
        name="Gemini 2.5 Pro (OR)",
        litellm_model="openrouter/google/gemini-2.5-pro",
        api_key_env="OPENROUTER_API_KEY",
        max_context=1_000_000,
        encoding="o200k_base",
        cost_per_1m_input=1.25,
        cost_per_1m_output=10.0,
    ),
    "openrouter/deepseek/deepseek-chat-v3": ModelProfile(
        name="DeepSeek V3 (OR)",
        litellm_model="openrouter/deepseek/deepseek-chat-v3",
        api_key_env="OPENROUTER_API_KEY",
        max_context=131_072,
        encoding="o200k_base",
        cost_per_1m_input=0.27,
        cost_per_1m_output=1.10,
    ),
    "openrouter/xiaomi/mimo-v2.5-pro": ModelProfile(
        name="MiMo V2.5 Pro",
        litellm_model="openrouter/xiaomi/mimo-v2.5-pro",
        api_key_env="OPENROUTER_API_KEY",
        max_context=1_048_576,
        encoding="o200k_base",
        cost_per_1m_input=1.0,
        cost_per_1m_output=3.0,
    ),
    "openrouter/xiaomi/mimo-v2.5": ModelProfile(
        name="MiMo V2.5",
        litellm_model="openrouter/xiaomi/mimo-v2.5",
        api_key_env="OPENROUTER_API_KEY",
        max_context=1_048_576,
        encoding="o200k_base",
        cost_per_1m_input=0.4,
        cost_per_1m_output=2.0,
    ),
}


def resolve_profile(model_string: str) -> ModelProfile:
    """Resolve a model string to a ModelProfile.

    If the string matches a preset key exactly, return that profile.
    Otherwise, create a generic profile with reasonable defaults.
    """
    if model_string in PRESETS:
        return PRESETS[model_string]

    # Try to infer provider from prefix
    api_key_env = "UNKNOWN_API_KEY"
    max_context = 128_000
    if model_string.startswith("openrouter/"):
        api_key_env = "OPENROUTER_API_KEY"
        max_context = 200_000
    elif model_string.startswith("gemini/"):
        api_key_env = "GEMINI_API_KEY"
        max_context = 1_000_000
    elif model_string.startswith("deepseek/"):
        api_key_env = "DEEPSEEK_API_KEY"
        max_context = 131_072
    elif model_string.startswith("anthropic/"):
        api_key_env = "ANTHROPIC_API_KEY"
        max_context = 200_000
    elif model_string.startswith("gpt-"):
        api_key_env = "OPENAI_API_KEY"
        max_context = 128_000

    return ModelProfile(
        name=model_string,
        litellm_model=model_string,
        api_key_env=api_key_env,
        max_context=max_context,
    )