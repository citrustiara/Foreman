"""Cost tracking — records token usage and estimates spend per session."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

logger = logging.getLogger("foreman.tokens.cost")


@dataclass
class CostEntry:
    """A single LLM call cost record."""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


@dataclass
class SessionCost:
    """Accumulated cost for a session."""
    session_id: str
    entries: list[CostEntry] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_per_1m_input: float,
        cost_per_1m_output: float,
    ) -> CostEntry:
        """Record a single LLM call and return the entry."""
        input_cost = (input_tokens / 1_000_000) * cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * cost_per_1m_output
        total = input_cost + output_cost

        entry = CostEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total,
        )

        self.entries.append(entry)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += total

        logger.info(
            "Cost: $%.4f (%d in / %d out) [%s]",
            total, input_tokens, output_tokens, model,
        )
        return entry

    def format_summary(self) -> str:
        """Format a human-readable cost summary."""
        return (
            f"Total: ${self.total_cost:.4f} | "
            f"Input: {self.total_input_tokens:,} tokens | "
            f"Output: {self.total_output_tokens:,} tokens"
        )

    def format_detailed(self) -> str:
        """Format detailed cost breakdown."""
        lines = [
            f"[bold]Session Cost[/]",
            f"  Total spend: ${self.total_cost:.4f}",
            f"  Input tokens: {self.total_input_tokens:,}",
            f"  Output tokens: {self.total_output_tokens:,}",
            f"  Calls: {len(self.entries)}",
        ]
        if self.entries:
            lines.append("")
            lines.append("  [dim]Recent calls:[/]")
            for entry in self.entries[-5:]:
                lines.append(
                    f"    {entry.model}: {entry.input_tokens:,} in / {entry.output_tokens:,} out = ${entry.total_cost:.4f}"
                )
        return "\n".join(lines)


class CostTracker:
    """Manages cost tracking across sessions."""

    def __init__(self, costs_dir: Path):
        self.costs_dir = costs_dir
        self.costs_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[str, SessionCost] = {}

    def get_or_create(self, session_id: str) -> SessionCost:
        """Get or create a SessionCost for a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionCost(session_id=session_id)
        return self._sessions[session_id]

    async def save(self, session_id: str) -> None:
        """Persist cost data to disk."""
        cost = self._sessions.get(session_id)
        if not cost:
            return

        path = self.costs_dir / f"{session_id}.json"
        data = {
            "session_id": cost.session_id,
            "total_input_tokens": cost.total_input_tokens,
            "total_output_tokens": cost.total_output_tokens,
            "total_cost": cost.total_cost,
            "entries": [asdict(e) for e in cost.entries],
        }

        tmp_path = path.with_suffix(".json.tmp")
        async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, indent=2))
        await aiofiles.os.replace(str(tmp_path), str(path))

    async def load(self, session_id: str) -> SessionCost:
        """Load cost data from disk."""
        path = self.costs_dir / f"{session_id}.json"
        if not path.exists():
            return self.get_or_create(session_id)

        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())

        cost = SessionCost(
            session_id=data["session_id"],
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
        )
        for entry_data in data.get("entries", []):
            cost.entries.append(CostEntry(**entry_data))

        self._sessions[session_id] = cost
        return cost
