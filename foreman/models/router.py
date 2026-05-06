"""LiteLLM async router — multi-provider LLM call management."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import litellm

from foreman.models.profiles import ModelProfile

logger = logging.getLogger("foreman.models.router")


class ModelRouter:
    """Wraps litellm.acompletion for async LLM calls with fallback support."""

    def __init__(self) -> None:
        # Suppress litellm's verbose logging by default
        litellm.suppress_debug_info = True
        litellm.logging = False

    async def generate(
        self,
        profile: ModelProfile,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a completion from the model. Returns the text response."""
        logger.info("Generating with %s (%d messages)", profile.litellm_model, len(messages))
        try:
            response = await litellm.acompletion(
                model=profile.litellm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content or ""
            logger.info("Generated %d chars from %s", len(content), profile.litellm_model)
            return content
        except Exception as e:
            logger.error("Generation failed with %s: %s", profile.litellm_model, e)
            raise

    async def generate_with_tools(
        self,
        profile: ModelProfile,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> litellm.ModelResponse:
        """Generate a completion with tool support. Returns the raw response object.

        Use this for agentic loops where the model may call tools.
        Check response.choices[0].message.tool_calls to see if tools were called.
        """
        logger.info("Generating with tools (%s, %d messages)", profile.litellm_model, len(messages))
        try:
            response = await litellm.acompletion(
                model=profile.litellm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **kwargs,
            )
            return response
        except Exception as e:
            logger.error("Generation with tools failed for %s: %s", profile.litellm_model, e)
            raise

    async def generate_stream(
        self,
        profile: ModelProfile,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from the model. Yields text chunks.

        Note: Streaming mode does not support tool calls. Use generate_with_tools() for agentic loops.
        """
        logger.info("Streaming with %s (%d messages)", profile.litellm_model, len(messages))
        try:
            response = await litellm.acompletion(
                model=profile.litellm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
        except Exception as e:
            logger.error("Streaming failed with %s: %s", profile.litellm_model, e)
            raise

    async def generate_with_fallback(
        self,
        primary: ModelProfile,
        fallback: ModelProfile,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Try primary model, fall back to secondary on failure."""
        try:
            return await self.generate(primary, messages, **kwargs)
        except Exception as primary_error:
            logger.warning(
                "Primary model %s failed, falling back to %s: %s",
                primary.litellm_model,
                fallback.litellm_model,
                primary_error,
            )
            try:
                return await self.generate(fallback, messages, **kwargs)
            except Exception as fallback_error:
                logger.error("Fallback model %s also failed: %s", fallback.litellm_model, fallback_error)
                raise fallback_error from primary_error

    async def generate_stream_with_fallback(
        self,
        primary: ModelProfile,
        fallback: ModelProfile,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Try streaming from primary, fall back to secondary on failure."""
        try:
            async for chunk in self.generate_stream(primary, messages, **kwargs):
                yield chunk
        except Exception as primary_error:
            logger.warning(
                "Primary model %s failed, falling back to %s",
                primary.litellm_model,
                fallback.litellm_model,
            )
            try:
                async for chunk in self.generate_stream(fallback, messages, **kwargs):
                    yield chunk
            except Exception as fallback_error:
                logger.error("Fallback model %s also failed", fallback.litellm_model)
                raise fallback_error from primary_error
