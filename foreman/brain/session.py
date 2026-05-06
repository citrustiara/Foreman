"""Session management — CRUD for conversation logs."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_session_id() -> str:
    return str(uuid.uuid4())


class Message:
    """A single message in a session."""

    __slots__ = ("role", "content", "timestamp", "token_count", "model", "tool_calls")

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: str | None = None,
        token_count: int = 0,
        model: str | None = None,
        tool_calls: list[dict] | None = None,
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or _utcnow()
        self.token_count = token_count
        self.model = model
        self.tool_calls = tool_calls or []

    def to_dict(self) -> dict[str, Any]:
        d = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
        }
        if self.model:
            d["model"] = self.model
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp"),
            token_count=data.get("token_count", 0),
            model=data.get("model"),
            tool_calls=data.get("tool_calls", []),
        )


class Session:
    """A conversation session with full message history."""

    def __init__(
        self,
        session_id: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        model_profile: str = "primary",
        messages: list[Message] | None = None,
        total_tokens: int = 0,
        compaction_count: int = 0,
    ):
        self.session_id = session_id or _new_session_id()
        self.created_at = created_at or _utcnow()
        self.updated_at = updated_at or _utcnow()
        self.model_profile = model_profile
        self.messages: list[Message] = messages or []
        self.total_tokens = total_tokens
        self.compaction_count = compaction_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model_profile": self.model_profile,
            "messages": [m.to_dict() for m in self.messages],
            "total_tokens": self.total_tokens,
            "compaction_count": self.compaction_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            model_profile=data.get("model_profile", "primary"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            total_tokens=data.get("total_tokens", 0),
            compaction_count=data.get("compaction_count", 0),
        )


class SessionStore:
    """Async file-backed session storage."""

    def __init__(self, sessions_dir: Path):
        self._dir = sessions_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json"

    def _tmp_path(self, session_id: str) -> Path:
        return self._dir / f"{session_id}.json.tmp"

    async def create(self, model_profile: str = "primary") -> Session:
        session = Session(model_profile=model_profile)
        await self.save(session)
        return session

    async def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            raw = await f.read()
        return Session.from_dict(json.loads(raw))

    async def save(self, session: Session) -> None:
        session.updated_at = _utcnow()
        data = json.dumps(session.to_dict(), indent=2, ensure_ascii=False)
        tmp = self._tmp_path(session.session_id)
        final = self._path(session.session_id)
        # Atomic write: tmp -> rename
        async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
            await f.write(data)
        # os.replace is atomic on most filesystems
        await aiofiles.os.replace(str(tmp), str(final))

    async def append_message(self, session: Session, message: Message) -> None:
        session.messages.append(message)
        session.total_tokens += message.token_count
        await self.save(session)

    async def get_recent(self, session: Session, count: int = 6) -> list[Message]:
        return session.messages[-count:] if session.messages else []

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with minimal metadata."""
        sessions = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    raw = await f.read()
                data = json.loads(raw)
                sessions.append({
                    "session_id": data["session_id"],
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": len(data.get("messages", [])),
                    "total_tokens": data.get("total_tokens", 0),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions