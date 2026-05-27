"""
Forwarder unit tests — mocks litellm.acompletion at the litellm layer.
Tests stream_completion directly to cover fixes #1, #2, and #3.
Uses asyncio.run() so no pytest-asyncio dependency is needed.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from proxy.forwarder import stream_completion


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_chunk(content=None, tool_calls=None, usage=None):
    """Build a mock litellm streaming chunk."""
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls

    chunk_dict: dict = {
        "id": "test-chunk",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    if usage is not None:
        chunk_dict["usage"] = usage

    mock = MagicMock()
    mock.model_dump.return_value = chunk_dict
    return mock


class _AsyncIter:
    """Async iterator that yields from a list, then either stops or raises."""

    def __init__(self, items: list, raise_after: bool = False):
        self._items = iter(items)
        self._raise_after = raise_after

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            if self._raise_after:
                raise RuntimeError("provider exploded")
            raise StopAsyncIteration

    async def aclose(self):
        pass


async def _collect(model, messages, body):
    """Drain stream_completion and return list of (sse_str, tok, cost) tuples."""
    results = []
    async for item in stream_completion(model, messages, body):
        results.append(item)
    return results


_MODEL = "openrouter/test/model"
_MESSAGES = [{"role": "user", "content": "Hi"}]


# ── Fix #1: tool-call argument tokens ────────────────────────────────────────

def test_tool_call_arguments_counted_in_output_tokens():
    """Fix #1: tokens in tool_calls[*].function.arguments must be counted."""
    chunks = [
        _make_chunk(tool_calls=[{"function": {"arguments": '{"city": "Taipei"}'}}]),
    ]
    acomp = _AsyncIter(chunks)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
            patch("litellm.token_counter", return_value=5),
            patch("litellm.get_model_info", return_value={}),
        ):
            return await _collect(_MODEL, _MESSAGES, {})

    results = asyncio.run(_run())

    # Intermediate chunk + DONE
    assert len(results) == 2
    done_sse, output_tokens, _ = results[-1]
    assert done_sse == "data: [DONE]\n\n"
    assert output_tokens == 5, "token_counter should be called on tool_call arguments"


def test_content_only_chunk_still_counted():
    """Baseline: plain content chunks are counted (regression guard for #1 change)."""
    chunks = [_make_chunk(content="Hello world")]
    acomp = _AsyncIter(chunks)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
            patch("litellm.token_counter", return_value=2),
            patch("litellm.get_model_info", return_value={}),
        ):
            return await _collect(_MODEL, _MESSAGES, {})

    results = asyncio.run(_run())
    _, output_tokens, _ = results[-1]
    assert output_tokens == 2


# ── Fix #2: provider usage chunk overrides token_counter ─────────────────────

def test_provider_usage_chunk_overrides_token_counter():
    """Fix #2: when a usage chunk arrives, use provider's counts, not token_counter."""
    chunks = [
        _make_chunk(content="Hello"),
        _make_chunk(usage={"prompt_tokens": 42, "completion_tokens": 99, "total_tokens": 141}),
    ]
    acomp = _AsyncIter(chunks)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
            patch("litellm.token_counter", return_value=999),  # must NOT be used
            patch("litellm.get_model_info", return_value={}),
        ):
            return await _collect(
                _MODEL,
                _MESSAGES,
                {"stream_options": {"include_usage": True}},
            )

    results = asyncio.run(_run())

    done_sse, output_tokens, _ = results[-1]
    assert done_sse == "data: [DONE]\n\n"
    assert output_tokens == 99, (
        "output_tokens must come from provider usage chunk (99), not token_counter (999)"
    )


def test_no_usage_chunk_falls_back_to_token_counter():
    """When no usage chunk arrives, token_counter estimate is used (regression guard)."""
    chunks = [_make_chunk(content="Hi")]
    acomp = _AsyncIter(chunks)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
            patch("litellm.token_counter", return_value=7),
            patch("litellm.get_model_info", return_value={}),
        ):
            return await _collect(_MODEL, _MESSAGES, {})

    results = asyncio.run(_run())
    _, output_tokens, _ = results[-1]
    assert output_tokens == 7


# ── Fix #3: mid-stream exception yields DONE before propagating ───────────────

def test_mid_stream_exception_yields_done_with_partial_tokens():
    """Fix #3: on provider error, stream_completion yields [DONE] before raising."""
    chunks = [_make_chunk(content="Hel")]  # partial content before crash
    acomp = _AsyncIter(chunks, raise_after=True)  # raises RuntimeError after chunks

    collected = []
    raised = None

    async def _run():
        nonlocal raised
        try:
            async for item in stream_completion(_MODEL, _MESSAGES, {}):
                collected.append(item)
        except RuntimeError as exc:
            raised = exc

    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
        patch("litellm.token_counter", return_value=3),
        patch("litellm.get_model_info", return_value={}),
    ):
        asyncio.run(_run())

    assert raised is not None, "RuntimeError must propagate out of stream_completion"
    done_items = [item for item in collected if item[0] == "data: [DONE]\n\n"]
    assert len(done_items) == 1, "Exactly one [DONE] must be yielded even on error"
    assert done_items[0][1] > 0, "output_tokens in partial DONE must be > 0"


def test_mid_stream_exception_does_not_emit_duplicate_done():
    """Fix #3 + server fix: [DONE] is not emitted twice when error occurs after content."""
    chunks = [_make_chunk(content="Hello")]
    acomp = _AsyncIter(chunks, raise_after=True)

    collected = []

    async def _run():
        try:
            async for item in stream_completion(_MODEL, _MESSAGES, {}):
                collected.append(item)
        except RuntimeError:
            pass

    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
        patch("litellm.token_counter", return_value=1),
        patch("litellm.get_model_info", return_value={}),
    ):
        asyncio.run(_run())

    done_count = sum(1 for item in collected if item[0] == "data: [DONE]\n\n")
    assert done_count == 1, f"Expected exactly 1 [DONE], got {done_count}"
