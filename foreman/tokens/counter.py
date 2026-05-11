"""Token counting via tiktoken."""

from __future__ import annotations

from typing import Any

import tiktoken

# Per-message overhead for ChatML format
MESSAGE_OVERHEAD = 4  # <|start|>role\n + content + <|end|>\n
PRIMING_TOKENS = 3  # <|start|>assistant<|message|>


class TokenCounter:
    """Caches tiktoken encodings per model for fast repeated counting."""

    def __init__(self) -> None:
        self._encodings: dict[str, tiktoken.Encoding] = {}

    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        if model not in self._encodings:
            normalized = model.split("/", 1)[-1]
            try:
                self._encodings[model] = tiktoken.encoding_for_model(normalized)
            except KeyError:
                enc_name = self._resolve_encoding(model)
                self._encodings[model] = tiktoken.get_encoding(enc_name)
        return self._encodings[model]

    @staticmethod
    def _resolve_encoding(model: str) -> str:
        """Map model name to tiktoken encoding name."""
        model_lower = model.lower()
        # Newer models often use o200k
        if any(x in model_lower for x in ["o200k", "o1-", "o3-", "gpt-4o", "gpt-5", "claude-4"]):
            return "o200k_base"
        
        # Default to cl100k_base
        return "cl100k_base"

    @staticmethod
    def _iter_strings(value: Any):
        """Yield all nested string values in dict/list payloads."""
        if isinstance(value, str):
            yield value
            return
        if isinstance(value, list):
            for item in value:
                yield from TokenCounter._iter_strings(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from TokenCounter._iter_strings(item)

    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in a string for a given model."""
        enc = self._get_encoding(model)
        return len(enc.encode(text))

    def count_message_tokens(self, messages: list[dict[str, Any]], model: str) -> int:
        """Count tokens for a full message list, including per-message overhead."""
        enc = self._get_encoding(model)
        total = 0
        for msg in messages:
            total += MESSAGE_OVERHEAD
            for field in ("role", "content", "name", "tool_call_id", "function_call", "tool_calls"):
                if field in msg:
                    for text in self._iter_strings(msg[field]):
                        total += len(enc.encode(text))
        total += PRIMING_TOKENS
        return total

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        model: str,
        system_prompt: str = "",
    ) -> int:
        """Count tokens for a chat including system prompt."""
        total = 0
        if system_prompt:
            total += self.count_message_tokens(
                [{"role": "system", "content": system_prompt}], model
            )
        total += self.count_message_tokens(messages, model)
        return total

    def fit_to_budget(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        model: str,
    ) -> tuple[list[dict[str, Any]], int]:
        """Drop oldest messages until under budget, preserving system prompt.

        Assistant messages that contain ``tool_calls`` are bundled with their
        subsequent tool-result messages and dropped as an atomic group.  Splitting
        them would leave orphaned tool results (no matching tool_call_id on the
        assistant side) and trigger a 400 from every provider.

        Returns (fitted_messages, dropped_count).
        """
        # Separate system messages from conversation messages
        system_msgs: list[dict[str, Any]] = []
        other_msgs: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)

        # Group conversation messages so that each assistant+tool_calls block is
        # bundled with all its tool-result messages.
        groups: list[list[dict[str, Any]]] = []
        i = 0
        while i < len(other_msgs):
            msg = other_msgs[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Collect the tool results that follow
                group = [msg]
                tc_ids = {tc["id"] for tc in msg["tool_calls"] if isinstance(tc, dict) and "id" in tc}
                j = i + 1
                while j < len(other_msgs) and other_msgs[j].get("role") == "tool":
                    group.append(other_msgs[j])
                    tc_ids.discard(other_msgs[j].get("tool_call_id", ""))
                    j += 1
                groups.append(group)
                i = j
            else:
                groups.append([msg])
                i += 1

        current = system_msgs + [m for g in groups for m in g]
        tokens = self.count_message_tokens(current, model)

        dropped = 0
        while tokens > max_tokens and groups:
            removed_group = groups.pop(0)
            dropped += len(removed_group)
            current = system_msgs + [m for g in groups for m in g]
            tokens = self.count_message_tokens(current, model)

        return current, dropped
