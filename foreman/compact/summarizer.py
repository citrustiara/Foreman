"""Summarizer — uses the secondary model to compact conversation history."""

from __future__ import annotations

import logging

from foreman.models.profiles import ModelProfile, resolve_profile
from foreman.models.router import ModelRouter

logger = logging.getLogger("foreman.compact.summarizer")

SUMMARIZE_PROMPT = """You are a context summarizer for a coding assistant.
Compress the following conversation into a structured summary.
Preserve:
1. All decisions made (architecture, naming, approach)
2. Files being actively edited (full paths)
3. Current bugs, errors, or unresolved questions
4. Goals the user is working toward
5. Any constraints or requirements stated

Output format:
## Decisions
- ...

## Active Files
- path/to/file.py: description of changes

## Current State
- Bug/Error: ...
- Goal: ...

## Constraints
- ...
"""


SELF_COMPRESS_PROMPT = """You have been coding and testing in this conversation. Now compress the entire conversation history into a structured handoff summary that preserves everything a fresh instance of yourself would need to continue working.

Be thorough. Include:
1. Every architectural/naming/approach decision made
2. All files modified or created — full paths and what changed
3. Every bug encountered, its root cause, and how it was fixed
4. Current state: what works, what's broken, what's pending
5. Exact test results or error messages that are still relevant
6. Any constraints, preferences, or requirements the user stated
7. The user's communication style and level of detail preference

Write the summary as if you're leaving notes for yourself to resume exactly where you left off."""


class Summarizer:
    """Uses a model to summarize conversation history."""

    def __init__(self, router: ModelRouter, max_summary_tokens: int = 2000):
        self.router = router
        self.max_summary_tokens = max_summary_tokens

    async def summarize_session(
        self,
        messages: list[dict[str, str]],
        summary_model: ModelProfile | None = None,
    ) -> str:
        """Summarize a list of conversation messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            summary_model: The model profile to use for summarization.
                          Defaults to gemini/gemini-2.5-flash.

        Returns:
            The summary text.
        """
        if summary_model is None:
            summary_model = resolve_profile("gemini/gemini-2.5-flash")

        # Build the summarization prompt
        conversation_text = self._format_messages_for_summary(messages)

        full_prompt = f"{SUMMARIZE_PROMPT}\n\n---\n\n{conversation_text}"

        logger.info(
            "Summarizing %d messages with %s",
            len(messages),
            summary_model.litellm_model,
        )

        summary = await self.router.generate(
            profile=summary_model,
            messages=[
                {"role": "system", "content": "You are a concise, accurate summarizer."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.3,
            max_tokens=self.max_summary_tokens,
        )

        logger.info("Generated summary: %d chars", len(summary))
        return summary

    async def self_compress(
        self,
        messages: list[dict[str, str]],
        primary_model: ModelProfile,
    ) -> str:
        """The primary model writes its own handoff summary.

        Used after implement+test cycles when the model has full context
        and can write a more informed summary than the secondary model.
        """
        conversation_text = self._format_messages_for_summary(messages)
        full_prompt = f"{SELF_COMPRESS_PROMPT}\n\n---\n\n{conversation_text}"

        logger.info(
            "Self-compressing %d messages with %s",
            len(messages),
            primary_model.litellm_model,
        )

        summary = await self.router.generate(
            profile=primary_model,
            messages=[
                {"role": "system", "content": "You are compressing your own conversation history into a handoff summary."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.2,
            max_tokens=self.max_summary_tokens,
        )

        logger.info("Self-compressed summary: %d chars", len(summary))
        return summary

    @staticmethod
    def _format_messages_for_summary(messages: list[dict[str, str]]) -> str:
        """Format messages into a readable string for the summarizer."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                continue  # Skip system prompts
            parts.append(f"[{role.upper()}]\n{content}")
        return "\n\n".join(parts)