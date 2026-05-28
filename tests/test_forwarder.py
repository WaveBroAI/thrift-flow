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

from proxy.forwarder import call_non_streaming, stream_completion


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


# ── Regression: null placeholder in tool_calls array (CONFIRMED bug #1) ──────

def test_null_entry_in_tool_calls_does_not_abort_stream():
    """A None placeholder in tool_calls must be skipped, not crash the stream."""
    chunks = [
        # tool_calls list with a null entry followed by a real entry
        _make_chunk(tool_calls=[None, {"function": {"name": "get_weather", "arguments": '{"city":"SF"}'}}]),
        _make_chunk(content="Done"),
    ]
    acomp = _AsyncIter(chunks)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
            patch("litellm.token_counter", return_value=4),
            patch("litellm.get_model_info", return_value={}),
        ):
            return await _collect(_MODEL, _MESSAGES, {})

    results = asyncio.run(_run())

    # 2 intermediate chunks + 1 DONE — stream must NOT have aborted early
    assert len(results) == 3, (
        f"Expected 3 items (2 chunks + DONE), got {len(results)} — "
        "null tool_calls entry likely caused an early abort"
    )
    done_sse, _, _ = results[-1]
    assert done_sse == "data: [DONE]\n\n"


def test_null_only_tool_calls_list_does_not_abort_stream():
    """A tool_calls list containing only None entries must not crash the stream."""
    chunks = [_make_chunk(tool_calls=[None, None])]
    acomp = _AsyncIter(chunks)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
            patch("litellm.token_counter", return_value=0),
            patch("litellm.get_model_info", return_value={}),
        ):
            return await _collect(_MODEL, _MESSAGES, {})

    results = asyncio.run(_run())
    assert results[-1][0] == "data: [DONE]\n\n"


# ── Regression: partial provider_usage (CONFIRMED bug #2) ────────────────────

def test_partial_provider_usage_missing_completion_tokens_falls_back():
    """When provider_usage lacks completion_tokens, token_counter is used for output side."""
    chunks = [
        _make_chunk(usage={"prompt_tokens": 10}),  # no completion_tokens
    ]
    acomp = _AsyncIter(chunks)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
            patch("litellm.token_counter", return_value=7),  # fallback value
            patch("litellm.get_model_info", return_value={}),
        ):
            return await _collect(_MODEL, _MESSAGES, {})

    results = asyncio.run(_run())

    _, output_tokens, _ = results[-1]
    assert output_tokens == 7, (
        "output_tokens must fall back to token_counter (7) when completion_tokens "
        "is absent from provider_usage, not silently zero"
    )


def test_partial_provider_usage_missing_prompt_tokens_falls_back():
    """When provider_usage lacks prompt_tokens, cost calculation uses token_counter for input."""
    chunks = [
        _make_chunk(usage={"completion_tokens": 20}),  # no prompt_tokens
    ]
    acomp = _AsyncIter(chunks)

    cost_captured = []

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
            patch("litellm.token_counter", return_value=5),
            # Use a model with known rates so we can verify cost > 0
            patch("litellm.get_model_info", return_value={
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            }),
        ):
            async for sse, tok, cost in stream_completion(_MODEL, _MESSAGES, {}):
                if sse == "data: [DONE]\n\n":
                    cost_captured.append((tok, cost))

    asyncio.run(_run())

    assert cost_captured, "No [DONE] tuple captured"
    output_tokens, cost_usd = cost_captured[0]
    assert output_tokens == 20, "output_tokens should come from provider (20)"
    # cost = (fallback_input=5 * 0.001) + (provider_output=20 * 0.002) = 0.045
    assert cost_usd > 0, (
        "cost_usd must be > 0 when prompt_tokens falls back to token_counter"
    )


def test_non_numeric_provider_usage_falls_back_to_token_counter():
    """Non-numeric completion_tokens must fall back gracefully, not raise ValueError."""
    chunks = [
        _make_chunk(usage={"prompt_tokens": "bad", "completion_tokens": "abc"}),
    ]
    acomp = _AsyncIter(chunks)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=acomp)),
            patch("litellm.token_counter", return_value=6),
            patch("litellm.get_model_info", return_value={}),
        ):
            return await _collect(_MODEL, _MESSAGES, {})

    # Must not raise ValueError — should complete and return token_counter fallback
    results = asyncio.run(_run())
    assert results[-1][0] == "data: [DONE]\n\n"
    _, output_tokens, _ = results[-1]
    assert output_tokens == 6, (
        "Non-numeric provider usage must fall back to token_counter (6), not crash"
    )


# ── call_non_streaming ────────────────────────────────────────────────────────

def _make_ns_response(completion_tokens=10, prompt_tokens=5, has_usage=True):
    """Build a mock LiteLLM non-streaming response."""
    resp = MagicMock()
    if has_usage:
        resp.usage = MagicMock()
        resp.usage.completion_tokens = completion_tokens
        resp.usage.prompt_tokens = prompt_tokens
    else:
        resp.usage = None
    resp.model_dump.return_value = {
        "id": "ns-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return resp


def test_call_non_streaming_returns_response_dict_and_tokens():
    """Happy path: returns correct (response_dict, output_tokens, cost_usd) tuple."""
    resp = _make_ns_response(completion_tokens=7, prompt_tokens=3)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=resp)),
            patch("litellm.get_model_info", return_value={}),
        ):
            return await call_non_streaming(_MODEL, _MESSAGES, {})

    response_dict, output_tokens, cost_usd = asyncio.run(_run())
    assert output_tokens == 7
    assert cost_usd == 0.0  # no rates configured → zero cost
    assert response_dict["id"] == "ns-test"
    assert response_dict["object"] == "chat.completion"


def test_call_non_streaming_cost_calculation_with_rates():
    """cost_usd = (prompt_tokens * input_rate) + (completion_tokens * output_rate)."""
    resp = _make_ns_response(completion_tokens=20, prompt_tokens=10)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=resp)),
            patch("litellm.get_model_info", return_value={
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            }),
        ):
            return await call_non_streaming(_MODEL, _MESSAGES, {})

    _, output_tokens, cost_usd = asyncio.run(_run())
    assert output_tokens == 20
    # (10 * 0.001) + (20 * 0.002) = 0.010 + 0.040 = 0.050
    assert abs(cost_usd - 0.050) < 1e-9


def test_call_non_streaming_no_usage_returns_zero_tokens_and_cost():
    """When response.usage is None, output_tokens=0 and cost_usd=0.0."""
    resp = _make_ns_response(has_usage=False)

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=resp)),
            patch("litellm.get_model_info", return_value={
                "input_cost_per_token": 0.001,
                "output_cost_per_token": 0.002,
            }),
        ):
            return await call_non_streaming(_MODEL, _MESSAGES, {})

    _, output_tokens, cost_usd = asyncio.run(_run())
    assert output_tokens == 0
    assert cost_usd == 0.0


def test_call_non_streaming_passthrough_fields_forwarded():
    """temperature, max_tokens etc. must appear in kwargs; unknown fields must not."""
    resp = _make_ns_response()
    captured: dict = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return resp

    async def _run():
        with (
            patch("litellm.acompletion", new=_fake_acompletion),
            patch("litellm.get_model_info", return_value={}),
        ):
            return await call_non_streaming(
                _MODEL,
                _MESSAGES,
                {"temperature": 0.7, "max_tokens": 256, "unknown_field": "ignored"},
            )

    asyncio.run(_run())
    assert captured["model"] == _MODEL
    assert captured["messages"] == _MESSAGES
    assert captured["temperature"] == 0.7
    assert captured["max_tokens"] == 256
    assert "unknown_field" not in captured
    assert "stream" not in captured  # non-streaming path must NOT set stream=True


def test_call_non_streaming_exception_propagates():
    """If litellm.acompletion raises, the exception must propagate unchanged."""
    async def _run():
        with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("provider down"))):
            await call_non_streaming(_MODEL, _MESSAGES, {})

    with pytest.raises(RuntimeError, match="provider down"):
        asyncio.run(_run())


def test_call_non_streaming_dict_fallback_when_no_model_dump():
    """When response lacks model_dump, dict(response) is used as response_dict."""

    class _LegacyResponse:
        """Mapping-protocol object — dict() works, model_dump attribute absent."""
        def __init__(self):
            self.usage = None
            self._data = {"id": "legacy-resp", "object": "chat.completion"}

        def keys(self):
            return self._data.keys()

        def __getitem__(self, key):
            return self._data[key]

    resp = _LegacyResponse()

    async def _run():
        with (
            patch("litellm.acompletion", new=AsyncMock(return_value=resp)),
            patch("litellm.get_model_info", return_value={}),
        ):
            return await call_non_streaming(_MODEL, _MESSAGES, {})

    response_dict, output_tokens, cost_usd = asyncio.run(_run())
    assert response_dict["id"] == "legacy-resp"
    assert output_tokens == 0
    assert cost_usd == 0.0
