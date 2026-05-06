"""Compact monitor — background token monitoring that triggers compaction."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from foreman.brain.session import Session
from foreman.tokens.counter import TokenCounter
from foreman.tokens.budget import TokenBudget

logger = logging.getLogger("foreman.compact.monitor")


@dataclass
class CompactEvent:
    """Event fired when a compaction occurs."""
    tokens_before: int
    tokens_after: int
    messages_archived: int
    summary_preview: str

    @property
    def reduction_pct(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return (1.0 - self.tokens_after / self.tokens_before) * 100

    def format(self) -> str:
        return (
            f"Context Compacted: {self.tokens_before:,} → {self.tokens_after:,} tokens "
            f"({self.reduction_pct:.0f}% reduction) | "
            f"{self.messages_archived} messages archived"
        )


class CompactMonitor:
    """Monitors token usage and signals when compaction should occur."""

    def __init__(
        self,
        token_counter: TokenCounter,
        budget: TokenBudget,
        threshold: float = 0.70,
        check_interval: float = 5.0,
    ):
        self.token_counter = token_counter
        self.budget = budget
        self.threshold = threshold
        self.check_interval = check_interval
        self._should_compact = False
        self._running = False
        self._task: asyncio.Task | None = None

    def check(self, session: Session, model: str = "gpt-4o") -> bool:
        """Check if compaction should occur based on current session tokens."""
        messages = [{"role": m.role, "content": m.content} for m in session.messages]
        token_count = self.token_counter.count_message_tokens(messages, model)

        # Update budget tracking
        self.budget.session_tokens = token_count

        if self.budget.session_usage_ratio >= self.threshold:
            logger.info(
                "Compact threshold reached: %.1f%% (threshold: %.0f%%)",
                self.budget.session_usage_ratio * 100,
                self.threshold * 100,
            )
            self._should_compact = True
            return True
        return False

    @property
    def should_compact(self) -> bool:
        return self._should_compact

    def reset(self) -> None:
        """Reset the compact flag after compaction is complete."""
        self._should_compact = False

    async def start_monitoring(self, session_getter, model: str = "gpt-4o") -> None:
        """Start background monitoring loop.

        session_getter: async callable that returns the current session
        """
        self._running = True
        while self._running:
            try:
                session = await session_getter()
                self.check(session, model)
            except Exception as e:
                logger.error("Monitor check failed: %s", e)
            await asyncio.sleep(self.check_interval)

    def stop_monitoring(self) -> None:
        """Stop the background monitoring loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    def start_background(self, session_getter, model: str = "gpt-4o") -> asyncio.Task:
        """Start monitoring as a background asyncio task."""
        self._task = asyncio.create_task(
            self.start_monitoring(session_getter, model)
        )
        return self._task