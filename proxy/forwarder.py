from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import litellm

logger = logging.getLogger(__name__)

_PASSTHROUGH_FIELDS = (
    "temperature",
    "max_tokens",
    "top_p",
    "stop",
    "tools",
    "tool_choice",
    "frequency_penalty",
    "presence_penalty",
    "n",
    "user",
    "stream_options",
)


def _build_kwargs(model: str, messages: list[dict], body: dict) -> dict[str, Any]:
    """Build the kwargs dict for a litellm.acompletion call."""
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    for field in _PASSTHROUGH_FIELDS:
        if field in body:
            kwargs[field] = body[field]
    return kwargs


def _get_cost_rates(model: str) -> tuple[float, float]:
    """Return (input_cost_per_token, output_cost_per_token). Falls back to (0, 0)."""
    try:
        info = litellm.get_model_info(model)
        return (
            info.get("input_cost_per_token", 0.0) or 0.0,
            info.get("output_cost_per_token", 0.0) or 0.0,
        )
    except Exception:
        return (0.0, 0.0)


async def call_non_streaming(
    model: str, messages: list[dict], body: dict
) -> tuple[dict, int, float]:
    """Forward a non-streaming request via LiteLLM.

    Returns (response_dict, output_tokens, cost_usd).
    cost_usd covers both input and output tokens.
    """
    kwargs = _build_kwargs(model, messages, body)

    # Fix E+F: use native async acompletion — no thread, no queue, no blocking I/O
    response = await litellm.acompletion(**kwargs)

    output_tokens: int = 0
    if response.usage and response.usage.completion_tokens:
        output_tokens = response.usage.completion_tokens

    input_tokens_used: int = 0
    if response.usage and response.usage.prompt_tokens:
        input_tokens_used = response.usage.prompt_tokens

    input_rate, output_rate = _get_cost_rates(model)
    cost_usd = (input_tokens_used * input_rate) + (output_tokens * output_rate)

    response_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    return response_dict, output_tokens, cost_usd


async def stream_completion(
    model: str, messages: list[dict], body: dict
) -> AsyncIterator[tuple[str, int, float]]:
    """Async generator that yields SSE strings.

    Each yielded tuple is (sse_string, output_tokens, cost_usd).
    - For intermediate chunks: output_tokens=0, cost_usd=0.
    - For the final "data: [DONE]\\n\\n" item: real output_tokens and cost_usd.

    Fix E+F: replaced the thread+queue+stop_event pattern with native
    litellm.acompletion async streaming.  Client disconnect cancels the
    async generator directly — no OS thread left running in the background,
    no unbounded queue accumulating the full response in memory.
    """
    kwargs = _build_kwargs(model, messages, body)
    kwargs["stream"] = True

    accumulated_text = ""

    async for chunk in await litellm.acompletion(**kwargs):
        chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
        sse_line = f"data: {json.dumps(chunk_dict)}\n\n"
        for choice in chunk_dict.get("choices", []):
            delta = choice.get("delta", {})
            accumulated_text += delta.get("content") or ""
        yield sse_line, 0, 0.0

    # All chunks consumed — compute cost and emit DONE
    output_tokens = 0
    try:
        output_tokens = litellm.token_counter(model=model, text=accumulated_text)
    except Exception:
        output_tokens = 0

    input_rate, output_rate = _get_cost_rates(model)
    try:
        input_tokens_est = litellm.token_counter(model=model, messages=messages)
    except Exception:
        input_tokens_est = 0
    cost_usd = (input_tokens_est * input_rate) + (output_tokens * output_rate)

    yield "data: [DONE]\n\n", output_tokens, cost_usd
