"""Context injector — the main compaction flow that replaces old messages with a summary."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import aiofiles.os

from foreman.brain.session import Session, Message, SessionStore
from foreman.compact.monitor import CompactEvent
from foreman.compact.summarizer import Summarizer
from foreman.models.profiles import ModelProfile, resolve_profile
from foreman.tokens.counter import TokenCounter

logger = logging.getLogger("foreman.compact.injector")


async def compact_session(
    session: Session,
    store: SessionStore,
    summarizer: Summarizer,
    token_counter: TokenCounter,
    model: str,
    recent_to_keep: int = 6,
    summary_model: ModelProfile | None = None,
    compactions_dir: Path | None = None,
) -> CompactEvent:
    """Full compact flow:

    1. Keep system prompt (index 0) untouched
    2. Keep the most recent `recent_to_keep` exchanges
    3. Replace all older messages with a single summary message
    4. Archive original messages to compactions/
    5. Update session totals
    6. Return a CompactEvent for TUI notification
    """
    if summary_model is None:
        summary_model = resolve_profile("gemini/gemini-2.5-flash")

    tokens_before = session.total_tokens

    # Separate messages
    system_msgs = [m for m in session.messages if m.role == "system"]
    non_system = [m for m in session.messages if m.role != "system"]

    if len(non_system) <= recent_to_keep:
        logger.info("Not enough messages to compact (%d non-system)", len(non_system))
        return CompactEvent(
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            messages_archived=0,
            summary_preview="No compaction needed — not enough messages.",
        )

    # Split: messages to compact vs messages to keep
    to_compact = non_system[:-recent_to_keep]
    to_keep = non_system[-recent_to_keep:]

    logger.info(
        "Compacting %d messages, keeping %d recent",
        len(to_compact),
        len(to_keep),
    )

    # Generate summary
    messages_for_summary = [
        {"role": m.role, "content": m.content} for m in to_compact
    ]
    summary_text = await summarizer.summarize_session(messages_for_summary, summary_model)

    # Create summary message
    summary_msg = Message(
        role="summary",
        content=summary_text,
        token_count=token_counter.count_tokens(summary_text, model),
        model=summary_model.litellm_model,
    )

    # Archive original messages
    if compactions_dir:
        await _archive_compaction(
            compactions_dir, session.session_id, to_compact, session.compaction_count
        )

    # Rebuild session messages
    session.messages = system_msgs + [summary_msg] + to_keep
    session.total_tokens = sum(m.token_count for m in session.messages)
    session.compaction_count += 1

    # Save updated session
    await store.save(session)

    tokens_after = session.total_tokens

    return CompactEvent(
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        messages_archived=len(to_compact),
        summary_preview=summary_text[:200] + ("..." if len(summary_text) > 200 else ""),
    )


async def compact_session_self(
    session: Session,
    store: SessionStore,
    summarizer: Summarizer,
    token_counter: TokenCounter,
    primary_model: ModelProfile,
    model: str,
    recent_to_keep: int = 6,
    compactions_dir: Path | None = None,
) -> CompactEvent:
    """Self-compact flow: the primary model writes its own summary.

    Same as compact_session but uses the primary model for summarization
    instead of the secondary. Better quality summaries after implement+test
    because the model that did the work compresses its own context.
    """
    tokens_before = session.total_tokens

    system_msgs = [m for m in session.messages if m.role == "system"]
    non_system = [m for m in session.messages if m.role != "system"]

    if len(non_system) <= recent_to_keep:
        logger.info("Not enough messages to compact (%d non-system)", len(non_system))
        return CompactEvent(
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            messages_archived=0,
            summary_preview="No compaction needed — not enough messages.",
        )

    to_compact = non_system[:-recent_to_keep]
    to_keep = non_system[-recent_to_keep:]

    logger.info(
        "Self-compacting %d messages, keeping %d recent",
        len(to_compact),
        len(to_keep),
    )

    messages_for_summary = [
        {"role": m.role, "content": m.content} for m in to_compact
    ]
    summary_text = await summarizer.self_compress(messages_for_summary, primary_model)

    summary_msg = Message(
        role="summary",
        content=summary_text,
        token_count=token_counter.count_tokens(summary_text, model),
        model=primary_model.litellm_model,
    )

    if compactions_dir:
        await _archive_compaction(
            compactions_dir, session.session_id, to_compact, session.compaction_count
        )

    session.messages = system_msgs + [summary_msg] + to_keep
    session.total_tokens = sum(m.token_count for m in session.messages)
    session.compaction_count += 1

    await store.save(session)

    tokens_after = session.total_tokens

    return CompactEvent(
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        messages_archived=len(to_compact),
        summary_preview=summary_text[:200] + ("..." if len(summary_text) > 200 else ""),
    )


async def _archive_compaction(
    compactions_dir: Path,
    session_id: str,
    messages: list[Message],
    compaction_number: int,
) -> None:
    """Archive compacted messages to a JSON file."""
    compactions_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{session_id}_{compaction_number}.json"
    path = compactions_dir / filename

    data = {
        "session_id": session_id,
        "compaction_number": compaction_number,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "messages": [m.to_dict() for m in messages],
    }

    # Atomic write
    tmp_path = path.with_suffix(".json.tmp")
    async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, indent=2, ensure_ascii=False))
    await aiofiles.os.replace(str(tmp_path), str(path))

    logger.info("Archived %d messages to %s", len(messages), path)
