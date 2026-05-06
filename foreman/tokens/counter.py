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
        if "gpt-4o" in model_lower or "gpt-5" in model_lower or "o200k" in model_lower:
            return "o200k_base"
        if "gpt-4" in model_lower or "gpt-3.5" in model_lower:
            return "cl100k_base"
        # Safe fallback for Gemini, DeepSeek, Claude, etc.
        return "cl100k_base"

    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        """Count tokens in a string for a given model."""
        enc = self._get_encoding(model)
        return len(enc.encode(text))

    def count_message_tokens(self, messages: list[dict[str, Any]], model: str = "gpt-4o") -> int:
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
        model: str = "gpt-4o",
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
        model: str = "gpt-4o",
    ) -> tuple[list[dict[str, Any]], int]:
        """Drop oldest messages until under budget, preserving system prompt.

        Returns (fitted_messages, dropped_count).
        """
        # Separate system prompt if present
        system_msgs: list[dict[str, Any]] = []
        other_msgs: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                other_msgs.append(msg)

        current = system_msgs + other_msgs
        tokens = self.count_message_tokens(current, model)

        dropped = 0
        while tokens > max_tokens and other_msgs:
            removed = other_msgs.pop(0)
            dropped += 1
            current = system_msgs + other_msgs
            tokens = self.count_message_tokens(current, model)

        return current, dropped