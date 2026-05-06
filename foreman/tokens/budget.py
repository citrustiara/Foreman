"""Token budget management — tracking and warnings."""

from __future__ import annotations

from dataclasses import dataclass, field


# Default context windows for known models
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "gemini/gemini-2.5-pro": 1_000_000,
    "gemini/gemini-2.5-flash": 1_000_000,
    "deepseek/deepseek-chat": 131_072,
    "claude-sonnet-4-20250514": 200_000,
    "anthropic/claude-sonnet-4-20250514": 200_000,
}


def get_context_window(model: str, fallback: int = 128_000) -> int:
    """Get context window for a model, with fallback.

    Checks hardcoded presets first, then OpenRouter cache.
    """
    # Direct match
    if model in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[model]
    # Partial match (e.g., "gemini/gemini-2.5-pro" matches "gemini-2.5-pro")
    model_lower = model.lower()
    for key, window in MODEL_CONTEXT_WINDOWS.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return window
    # Check OpenRouter cache
    try:
        from foreman.models.fetcher import load_models
        from pathlib import Path
        import os
        foreman_dir = Path.cwd() / ".foreman"
        for m in load_models(foreman_dir):
            if m["id"] == model or f"openrouter/{m['id']}" == model:
                return m.get("context_window", fallback)
    except Exception:
        pass
    return fallback


@dataclass
class BudgetWarning:
    """A token budget threshold warning."""
    threshold: float  # 0.0-1.0
    message: str
    level: str = "warning"  # "info", "warning", "critical"


@dataclass
class TokenBudget:
    """Tracks token allocation across context budget regions."""

    model: str = "gpt-4o"
    context_window: int = 128_000
    system_prompt_tokens: int = 0
    mermaid_tokens: int = 0
    session_tokens: int = 0
    reserved_output_tokens: int = 16_384
    warnings: list[BudgetWarning] = field(default_factory=lambda: [
        BudgetWarning(0.70, "Approaching context limit (70%)", "info"),
        BudgetWarning(0.85, "Context usage high (85%) — compact recommended", "warning"),
        BudgetWarning(0.95, "Context critical (95%) — compact required", "critical"),
    ])

    @classmethod
    def from_model(cls, model: str, output_reserve: int = 16_384) -> "TokenBudget":
        """Create a budget for a specific model."""
        ctx = get_context_window(model)
        return cls(model=model, context_window=ctx, reserved_output_tokens=output_reserve)

    @property
    def system_reserve(self) -> int:
        """Tokens reserved for system prompt + mermaid architecture."""
        return self.system_prompt_tokens + self.mermaid_tokens

    @property
    def total_reserved(self) -> int:
        """All tokens not available for session history."""
        return self.system_reserve + self.reserved_output_tokens

    @property
    def available_for_session(self) -> int:
        """Tokens available for conversation history."""
        return max(0, self.context_window - self.total_reserved)

    @property
    def used(self) -> int:
        """Total tokens currently in use."""
        return self.system_reserve + self.session_tokens

    @property
    def usage_ratio(self) -> float:
        """Current usage as a fraction of context window."""
        if self.context_window == 0:
            return 0.0
        return self.used / self.context_window

    @property
    def session_usage_ratio(self) -> float:
        """Session history usage as a fraction of available budget."""
        avail = self.available_for_session
        if avail == 0:
            return 1.0
        return min(1.0, self.session_tokens / avail)

    def check_warnings(self) -> list[BudgetWarning]:
        """Return warnings that are currently triggered."""
        ratio = self.usage_ratio
        return [w for w in self.warnings if ratio >= w.threshold]

    def should_compact(self, threshold: float = 0.70) -> bool:
        """Whether session usage exceeds the compact threshold."""
        return self.session_usage_ratio >= threshold

    def format_bar(self, width: int = 20) -> str:
        """Format an ASCII progress bar showing session usage."""
        ratio = self.session_usage_ratio
        filled = int(ratio * width)
        empty = width - filled
        bar = "#" * filled + "." * empty
        pct = int(ratio * 100)
        used_k = self.session_tokens // 1000
        total_k = self.available_for_session // 1000
        return f"{bar} {pct}% ({used_k}K/{total_k}K)"