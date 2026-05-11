"""LiteLLM async router — multi-provider LLM call management."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, Callable

import litellm

from foreman.models.profiles import ModelProfile

logger = logging.getLogger("foreman.models.router")


class ModelRouter:
    """Wraps litellm.acompletion for async LLM calls with fallback support."""

    def __init__(self) -> None:
        # Suppress litellm's verbose logging by default
        litellm.suppress_debug_info = True
        litellm.logging = False

    @staticmethod
    def _supports_reasoning_retry(exception: Exception, kwargs: dict[str, Any]) -> bool:
        if "reasoning_effort" not in kwargs:
            return False
        message = str(exception).lower()
        return "reason" in message or "effort" in message or "invalid parameter" in message

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
            if self._supports_reasoning_retry(e, kwargs):
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("reasoning_effort", None)
                response = await litellm.acompletion(
                    model=profile.litellm_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **retry_kwargs,
                )
                return response.choices[0].message.content or ""
            logger.error("Generation failed with %s: %s", profile.litellm_model, e)
            raise

    async def generate_with_tools(
        self,
        profile: ModelProfile,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        delta_callback: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> litellm.ModelResponse:
        """Generate a completion with tool support. Returns the raw response object.

        If delta_callback is provided, it will stream the response and call the callback
        with text chunks as they arrive.
        """
        logger.info("Generating with tools (%s, %d messages, stream=%s)", 
                    profile.litellm_model, len(messages), delta_callback is not None)
        try:
            if delta_callback:
                # Use streaming mode
                try:
                    response_stream = await litellm.acompletion(
                        model=profile.litellm_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        stream=True,
                        stream_options={"include_usage": True},
                        **kwargs,
                    )
                except Exception as e:
                    if not self._supports_reasoning_retry(e, kwargs):
                        raise
                    retry_kwargs = dict(kwargs)
                    retry_kwargs.pop("reasoning_effort", None)
                    response_stream = await litellm.acompletion(
                        model=profile.litellm_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        stream=True,
                        stream_options={"include_usage": True},
                        **retry_kwargs,
                    )
                
                chunks = []
                async for chunk in response_stream:
                    chunks.append(chunk)
                    delta = chunk.choices[0].delta
                    if delta.content:
                        delta_callback(delta.content)
                
                # Reassemble into a full response object
                return litellm.stream_chunk_builder(chunks)
            else:
                # Normal non-streaming mode
                try:
                    response = await litellm.acompletion(
                        model=profile.litellm_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        **kwargs,
                    )
                except Exception as e:
                    if not self._supports_reasoning_retry(e, kwargs):
                        raise
                    retry_kwargs = dict(kwargs)
                    retry_kwargs.pop("reasoning_effort", None)
                    response = await litellm.acompletion(
                        model=profile.litellm_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        **retry_kwargs,
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
