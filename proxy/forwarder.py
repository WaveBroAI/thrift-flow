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
    those authoritative counts are used directly per side.  A missing or
    non-numeric field falls back to litellm.token_counter for that side only.
    When no provider usage is available, both sides are estimated.
    """

    def _to_int(val: object) -> int | None:
        """Return int(val), or None if val is absent or not parseable."""
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    input_rate, output_rate = _get_cost_rates(model)
    output_tokens = 0
    input_tokens = 0

    if provider_usage:
        # Prefer provider-reported values; fall back per-side to token_counter
        # when a field is absent (None) or contains an unparseable value.
        ct = _to_int(provider_usage.get("completion_tokens"))
        pt = _to_int(provider_usage.get("prompt_tokens"))
        if ct is not None:
            output_tokens = ct
        else:
            try:
                output_tokens = litellm.token_counter(model=model, text=accumulated_text)
            except Exception:
                pass
        if pt is not None:
            input_tokens = pt
        else:
            try:
                input_tokens = litellm.token_counter(model=model, messages=messages)
            except Exception:
                pass
    else:
        try:
            output_tokens = litellm.token_counter(model=model, text=accumulated_text)
        except Exception:
            pass
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
    Fix #1: accumulates tool_calls[*].function.name and .arguments tokens.
    Fix #2: uses provider-reported usage chunk when stream_options.include_usage=true.
    Fix #3: on mid-stream provider error, yields [DONE] with accumulated cost then
            re-raises.  Note: GeneratorExit (client disconnect) bypasses the except
            handler — callers only see a forwarder-level [DONE] for provider errors,
            not for client-side disconnects.
    Fix #5: stores the acompletion object and calls aclose() in finally for explicit
            HTTP transport cleanup even on client disconnect or error.
    """
    kwargs = _build_kwargs(model, messages, body)
    kwargs["stream"] = True

    accumulated_text = ""
    provider_usage: dict = {}
    _stream_error: Exception | None = None

    # Fix #5: store separately so we can aclose() in finally
    _acomp = await litellm.acompletion(**kwargs)
    try:
        async for chunk in _acomp:
            chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
            sse_line = f"data: {json.dumps(chunk_dict)}\n\n"
            for choice in chunk_dict.get("choices", []):
                delta = choice.get("delta", {})
                # Fix #1: accumulate content and tool-call name+argument tokens
                accumulated_text += delta.get("content") or ""
                for tc in delta.get("tool_calls") or []:
                    if tc is not None:  # guard against null placeholders in tool_calls array
                        func = tc.get("function") or {}
                        accumulated_text += func.get("name") or ""
                        accumulated_text += func.get("arguments") or ""
            # Fix #2: capture provider-reported usage (stream_options.include_usage)
            if chunk_dict.get("usage"):
                provider_usage = chunk_dict["usage"]
            yield sse_line, 0, 0.0
    except Exception as exc:
        # Fix #3: capture error; [DONE] with partial cost is emitted below
        _stream_error = exc
    finally:
        # Fix #5: explicit close to ensure HTTP transport is released.
        # BaseException (not just Exception) is suppressed here so that
        # CancelledError during cleanup does not mask the original error.
        if hasattr(_acomp, "aclose"):
            try:
                await _acomp.aclose()
            except BaseException:
                pass

    # Emit [DONE] with final cost for both success and provider-error paths.
    # GeneratorExit (client disconnect) terminates the generator in finally above
    # and never reaches this point — the caller handles that case independently.
    output_tokens, cost_usd = _compute_streaming_cost(
        model, accumulated_text, provider_usage, messages
    )
    yield "data: [DONE]\n\n", output_tokens, cost_usd

    if _stream_error is not None:
        raise _stream_error
