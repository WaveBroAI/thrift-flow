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


def _compute_streaming_cost(
    model: str,
    accumulated_text: str,
    provider_usage: dict,
    messages: list[dict],
) -> tuple[int, float]:
    """Return (output_tokens, cost_usd) preferring provider-reported usage.

    When the provider sends a usage chunk (stream_options.include_usage=true),
    those authoritative counts are used directly.  Otherwise falls back to
    litellm.token_counter estimates from the accumulated text.
    """
    input_rate, output_rate = _get_cost_rates(model)
    if provider_usage:
        output_tokens = int(provider_usage.get("completion_tokens") or 0)
        input_tokens = int(provider_usage.get("prompt_tokens") or 0)
    else:
        output_tokens = 0
        try:
            output_tokens = litellm.token_counter(model=model, text=accumulated_text)
        except Exception:
            pass
        input_tokens = 0
        try:
            input_tokens = litellm.token_counter(model=model, messages=messages)
        except Exception:
            pass
    cost_usd = (input_tokens * input_rate) + (output_tokens * output_rate)
    return output_tokens, cost_usd


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
    Fix #1: accumulates tool_calls[*].function.arguments tokens (not just content).
    Fix #2: uses provider-reported usage chunk when stream_options.include_usage=true.
    Fix #3: on mid-stream error, yields partial DONE with accumulated cost before
            propagating the exception so callers always see a [DONE] sentinel.
    Fix #5: stores the acompletion object and calls aclose() in finally for explicit
            HTTP transport cleanup even on client disconnect or error.
    """
    kwargs = _build_kwargs(model, messages, body)
    kwargs["stream"] = True

    accumulated_text = ""
    provider_usage: dict = {}

    # Fix #5: store separately so we can aclose() in finally
    _acomp = await litellm.acompletion(**kwargs)
    try:
        try:
            async for chunk in _acomp:
                chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                sse_line = f"data: {json.dumps(chunk_dict)}\n\n"
                for choice in chunk_dict.get("choices", []):
                    delta = choice.get("delta", {})
                    # Fix #1: accumulate content and tool-call argument tokens
                    accumulated_text += delta.get("content") or ""
                    for tc in delta.get("tool_calls") or []:
                        accumulated_text += (tc.get("function") or {}).get("arguments") or ""
                # Fix #2: capture provider-reported usage (stream_options.include_usage)
                if chunk_dict.get("usage"):
                    provider_usage = chunk_dict["usage"]
                yield sse_line, 0, 0.0
        except Exception:
            # Fix #3: yield partial DONE with accumulated cost before propagating
            output_tokens, cost_usd = _compute_streaming_cost(
                model, accumulated_text, provider_usage, messages
            )
            yield "data: [DONE]\n\n", output_tokens, cost_usd
            raise
    finally:
        # Fix #5: explicit close to ensure HTTP transport is released
        if hasattr(_acomp, "aclose"):
            try:
                await _acomp.aclose()
            except Exception:
                pass

    # Happy path: all chunks consumed — emit DONE with final cost
    output_tokens, cost_usd = _compute_streaming_cost(
        model, accumulated_text, provider_usage, messages
    )
    yield "data: [DONE]\n\n", output_tokens, cost_usd
