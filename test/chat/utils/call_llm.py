import logging
import os
import time
from queue import Queue

from openai import OpenAI

logging.basicConfig(level=logging.INFO)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

logger = logging.getLogger(__name__)

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = "https://openrouter.ai/api/v1"
model = os.getenv("OPENROUTER_MODEL")
app_name = os.getenv("APP_NAME", "PocketFlow-Debate-Chat")


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "X-Title": app_name,
            "HTTP-Referer": os.getenv("SITE_URL", ""),
        },
    )


def parse_chunk(chunk: ChatCompletionChunk) -> str | None:
    """Extract text content from an OpenAI streaming chunk."""
    if (
        hasattr(chunk.choices[0].delta, "content")
        and chunk.choices[0].delta.content is not None
    ):
        return chunk.choices[0].delta.content
    return None


def call_llm(message: str, system_prompt: str | None = None) -> str:
    """Call LLM synchronously (non-streaming). Returns full response."""
    if not model:
        raise ValueError(
            "OPENROUTER_MODEL environment variable must be set. "
            "Example: anthropic/claude-3.5-sonnet"
        )

    prompt_len = len(message) + (len(system_prompt) if system_prompt else 0)
    logger.info("llm_call_start model=%s prompt_chars=%d", model, prompt_len)
    t0 = time.perf_counter()

    client = _make_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})
    response: ChatCompletion = client.chat.completions.create(
        model=model, messages=messages
    )
    result = response.choices[0].message.content or ""

    elapsed = time.perf_counter() - t0
    logger.info(
        "llm_call_done model=%s latency=%.2fs response_chars=%d",
        model,
        elapsed,
        len(result),
    )
    return result


def call_llm_stream(
    message: str,
    system_prompt: str | None = None,
    stream_queues: list[Queue] | None = None,
) -> str:
    """Call LLM with streaming enabled.

    Tokens are broadcast to all queues in stream_queues in real-time.
    Returns the full accumulated response string.
    """
    if not model:
        raise ValueError(
            "OPENROUTER_MODEL environment variable must be set. "
            "Example: anthropic/claude-3.5-sonnet"
        )

    prompt_len = len(message) + (len(system_prompt) if system_prompt else 0)
    logger.info("llm_stream_start model=%s prompt_chars=%d", model, prompt_len)
    t0 = time.perf_counter()

    client = _make_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})

    logger.info("llm_stream connecting model=%s base_url=%s", model, base_url)
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    logger.info("llm_stream connected, awaiting chunks...")

    accumulated: list[str] = []
    token_count = 0
    for chunk in stream:
        text = parse_chunk(chunk)
        if text:
            accumulated.append(text)
            token_count += 1
            if stream_queues:
                for q in stream_queues:
                    q.put(text)

    elapsed = time.perf_counter() - t0
    result = "".join(accumulated)
    logger.info(
        "llm_stream_done model=%s latency=%.2fs tokens=%d response_chars=%d",
        model,
        elapsed,
        token_count,
        len(result),
    )
    return result
