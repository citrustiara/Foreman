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
            enc_name = self._resolve_encoding(model)
            self._encodings[model] = tiktoken.get_encoding(enc_name)
        return self._encodings[model]

    @staticmethod
    def _resolve_encoding(model: str) -> str:
        """Map model name to tiktoken encoding name."""
        model_lower = model.lower()
        # Newer models often use o200k
        if any(x in model_lower for x in ["o200k", "o1-", "o3-"]):
            return "o200k_base"
        
        # Default to cl100k_base
        return "cl100k_base"

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
            for value in msg.values():
                if isinstance(value, str):
                    total += len(enc.encode(value))
                elif isinstance(value, list):  # tool_calls
                    for item in value:
                        if isinstance(item, dict):
                            for v in item.values():
                                if isinstance(v, str):
                                    total += len(enc.encode(v))
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