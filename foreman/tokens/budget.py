"""Token budget management — tracking and warnings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    import asyncio
    from foreman.brain.session import Session


def get_context_window(model: str, fallback: int = 128_000) -> int:
    """Get context window for a model, with fallback.

    Checks OpenRouter cache first, then falls back to provider heuristics.
    """
    # Check OpenRouter cache
    try:
        from foreman.models.fetcher import load_models
        for m in load_models():
            if m["id"] == model or f"openrouter/{m['id']}" == model or model == m.get("name"):
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

    model: str
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
        """Tokens reserved for system prompt + project context metadata."""
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
        ratio = self.usage_ratio
        filled = int(ratio * width)
        if self.used > 0 and filled == 0:
            filled = 1
        empty = width - filled
        bar = "#" * filled + "." * empty
        pct = int(ratio * 100)
        used_k = self.used // 1000
        total_k = self.context_window // 1000
        return f"{bar} {pct}% ({used_k}K/{total_k}K)"
