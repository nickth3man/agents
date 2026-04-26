"""Async LLM calling utilities with streaming support."""

import os
from typing import AsyncIterator

from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

from utils.observability import get_tracer, logger as log

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = "https://openrouter.ai/api/v1"
model = os.getenv("OPENROUTER_MODEL")
app_name = os.getenv("APP_NAME", "PocketFlow-Debate-Chat")

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
LLM_TIMEOUT = 60.0


def _make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=LLM_TIMEOUT,
        default_headers={
            "X-Title": app_name,
            "HTTP-Referer": os.getenv("SITE_URL", ""),
        },
    )


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503):
        return True
    if isinstance(exc, (TimeoutError, APITimeoutError)):
        return True
    return False


async def call_llm_async(message: str, system_prompt: str | None = None) -> str:
    """Call LLM asynchronously (non-streaming). Returns full response."""
    tracer = get_tracer()
    if not model:
        raise ValueError(
            "OPENROUTER_MODEL environment variable must be set. "
            "Example: anthropic/claude-3.5-sonnet"
        )

    prompt_len = len(message) + (len(system_prompt) if system_prompt else 0)
    log.info("llm_call_start", model=model, prompt_chars=prompt_len)

    client = _make_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            import time
            with tracer.start_as_current_span("llm_call") as span:
                span.set_attribute("llm.model", model)
                span.set_attribute("llm.prompt_chars", prompt_len)
                t0 = time.perf_counter()
                response = await client.chat.completions.create(
                    model=model, messages=messages
                )
                result = response.choices[0].message.content or ""
                elapsed = time.perf_counter() - t0
                span.set_attribute("llm.latency_ms", elapsed * 1000)
                span.set_attribute("llm.response_chars", len(result))
                log.info(
                    "llm_call_done",
                    model=model,
                    latency_ms=round(elapsed * 1000, 2),
                    response_chars=len(result),
                )
                return result
        except Exception as exc:
            if not _is_retryable(exc) or attempt == MAX_RETRIES:
                log.error("llm_call_failed", error=type(exc).__name__, attempt=attempt)
                raise
            wait = RETRY_BACKOFF ** attempt
            log.warning(
                "llm_call_retry",
                attempt=attempt,
                max_retries=MAX_RETRIES,
                error=type(exc).__name__,
                wait_seconds=round(wait, 1),
            )
            import asyncio
            await asyncio.sleep(wait)


async def call_llm_stream_async(
    message: str,
    system_prompt: str | None = None,
    token_queue=None,
) -> str:
    """Call LLM with async streaming.

    Tokens are put into token_queue in real-time.
    Returns the full accumulated response string.
    """
    tracer = get_tracer()
    if not model:
        raise ValueError(
            "OPENROUTER_MODEL environment variable must be set. "
            "Example: anthropic/claude-3.5-sonnet"
        )

    prompt_len = len(message) + (len(system_prompt) if system_prompt else 0)
    log.info("llm_stream_start", model=model, prompt_chars=prompt_len)

    client = _make_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            import time
            with tracer.start_as_current_span("llm_stream") as span:
                span.set_attribute("llm.model", model)
                span.set_attribute("llm.prompt_chars", prompt_len)
                t0 = time.perf_counter()
                log.info("llm_stream_connecting", model=model, base_url=base_url)
                stream = await client.chat.completions.create(
                    model=model, messages=messages, stream=True
                )
                log.info("llm_stream_connected")

                accumulated: list[str] = []
                token_count = 0
                async for chunk in stream:
                    try:
                        if (
                            chunk.choices
                            and hasattr(chunk.choices[0].delta, "content")
                            and chunk.choices[0].delta.content is not None
                        ):
                            text = chunk.choices[0].delta.content
                            accumulated.append(text)
                            token_count += 1
                            if token_queue is not None:
                                token_queue.put(text)
                    except (IndexError, AttributeError):
                        pass

                elapsed = time.perf_counter() - t0
                result = "".join(accumulated)
                span.set_attribute("llm.latency_ms", elapsed * 1000)
                span.set_attribute("llm.tokens", token_count)
                span.set_attribute("llm.response_chars", len(result))
                log.info(
                    "llm_stream_done",
                    model=model,
                    latency_ms=round(elapsed * 1000, 2),
                    tokens=token_count,
                    response_chars=len(result),
                )
                return result
        except Exception as exc:
            if not _is_retryable(exc) or attempt == MAX_RETRIES:
                log.error("llm_stream_failed", error=type(exc).__name__, attempt=attempt)
                raise
            wait = RETRY_BACKOFF ** attempt
            log.warning(
                "llm_stream_retry",
                attempt=attempt,
                max_retries=MAX_RETRIES,
                error=type(exc).__name__,
                wait_seconds=round(wait, 1),
            )
            import asyncio
            await asyncio.sleep(wait)
