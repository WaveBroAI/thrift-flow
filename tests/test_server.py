"""
Server endpoint tests — no real LLM calls, no API keys needed.
Uses FastAPI TestClient + mock patches on the forwarder functions.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from proxy.config import ModelConfig, ProxyConfig, RoutingConfig, ServerConfig, TrackingConfig
from proxy.server import create_app
from proxy.tracker import RequestTracker


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return ProxyConfig(
        server=ServerConfig(host="127.0.0.1", port=8888),
        models=ModelConfig(
            aliases={
                "cheap": "openrouter/test/cheap-model",
                "strong": "openrouter/test/strong-model",
            },
            default="cheap",
        ),
        tracking=TrackingConfig(db=":memory:", enabled=True),
    )


@pytest.fixture
def tracker(tmp_path):
    return RequestTracker(str(tmp_path / "test.db"))


@pytest.fixture
def client(config, tracker):
    app = create_app(config, tracker)
    return TestClient(app)


# ── helper mock objects ───────────────────────────────────────────────────────

_MOCK_RESPONSE = {
    "id": "test-123",
    "object": "chat.completion",
    "model": "openrouter/test/cheap-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


async def _mock_stream_gen():
    """Async generator that mimics stream_completion output."""
    chunk = json.dumps({
        "id": "test-123",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "Hello!"}, "finish_reason": None}],
    })
    yield f"data: {chunk}\n\n", 0, 0.0
    # Final item carries real output_tokens and cost
    yield "data: [DONE]\n\n", 5, 0.0001


# ── utility endpoints ─────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_list_models(client, config):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    model_ids = {m["id"] for m in data["data"]}
    assert model_ids == set(config.models.aliases)


# ── non-streaming ─────────────────────────────────────────────────────────────

def test_non_streaming_returns_llm_response(client):
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0001)),
        patch("litellm.token_counter", return_value=10),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "Hi"}]},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == "Hello!"


def test_alias_is_resolved_before_forwarding(client):
    """'cheap' alias must be resolved to the real model name before calling forwarder."""
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0)) as mock_fwd,
        patch("litellm.token_counter", return_value=5),
    ):
        client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "Hi"}]},
        )

    resolved = mock_fwd.call_args[0][0]
    assert resolved == "openrouter/test/cheap-model"


def test_unknown_model_passes_through_unchanged(client):
    """A full model name not in aliases should be forwarded as-is."""
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0)) as mock_fwd,
        patch("litellm.token_counter", return_value=5),
    ):
        client.post(
            "/v1/chat/completions",
            json={
                "model": "openrouter/some/custom-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

    resolved = mock_fwd.call_args[0][0]
    assert resolved == "openrouter/some/custom-model"


def test_non_streaming_logs_to_tracker(client, tracker):
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0001)),
        patch("litellm.token_counter", return_value=10),
    ):
        client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "Hi"}]},
        )

    summary = tracker.get_summary()
    assert summary["total_requests"] == 1
    assert summary["total_input_tokens"] == 10
    assert summary["total_output_tokens"] == 5

    recent = tracker.get_recent(limit=1)
    assert recent[0]["model_requested"] == "cheap"
    assert recent[0]["model_resolved"] == "openrouter/test/cheap-model"
    assert recent[0]["streaming"] == 0


def test_non_streaming_logs_error_on_forwarder_failure(client, tracker):
    with (
        patch("proxy.server.call_non_streaming", side_effect=RuntimeError("provider down")),
        patch("litellm.token_counter", return_value=5),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "Hi"}]},
        )

    assert resp.status_code == 500
    recent = tracker.get_recent(limit=1)
    assert recent[0]["status"] == 500
    assert "provider down" in (recent[0]["error"] or "")


def test_client_id_and_session_key_are_tracked(client, tracker):
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0)),
        patch("litellm.token_counter", return_value=5),
    ):
        client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"X-Client-ID": "crab-bot", "X-Session-Key": "rocketchat:wave"},
        )

    recent = tracker.get_recent(limit=1)
    assert recent[0]["client_id"] == "crab-bot"
    assert recent[0]["session_key"] == "rocketchat:wave"


# ── streaming ─────────────────────────────────────────────────────────────────

def test_streaming_content_type(client):
    with (
        patch("proxy.server.stream_completion", return_value=_mock_stream_gen()),
        patch("litellm.token_counter", return_value=8),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cheap",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


def test_streaming_numeric_one_triggers_sse(client):
    """Fix C: {"stream": 1} must be treated as streaming (not just {"stream": True})."""
    with (
        patch("proxy.server.stream_completion", return_value=_mock_stream_gen()),
        patch("litellm.token_counter", return_value=8),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cheap",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": 1,
            },
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data: [DONE]" in resp.text


def test_streaming_body_contains_done_sentinel(client):
    with (
        patch("proxy.server.stream_completion", return_value=_mock_stream_gen()),
        patch("litellm.token_counter", return_value=8),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cheap",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert "data: [DONE]" in resp.text
    assert "Hello!" in resp.text


def test_streaming_logs_to_tracker(client, tracker):
    with (
        patch("proxy.server.stream_completion", return_value=_mock_stream_gen()),
        patch("litellm.token_counter", return_value=8),
    ):
        client.post(
            "/v1/chat/completions",
            json={
                "model": "cheap",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    recent = tracker.get_recent(limit=1)
    assert len(recent) == 1
    assert recent[0]["streaming"] == 1
    assert recent[0]["output_tokens"] == 5   # from final item of _mock_stream_gen
    assert recent[0]["input_tokens"] == 8    # from mocked token_counter


async def _mock_stream_gen_error_after_done():
    """Yields content + [DONE] with token counts, then raises — simulating Fix #3 path."""
    chunk = json.dumps({
        "id": "test-456",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "Partial"}, "finish_reason": None}],
    })
    yield f"data: {chunk}\n\n", 0, 0.0
    yield "data: [DONE]\n\n", 5, 0.0002
    raise RuntimeError("provider exploded after DONE")


def test_streaming_error_after_done_logs_status_500_with_tokens(client, tracker):
    """Fix #3: when stream_completion yields [DONE] then raises, tracker records
    status=500, correct output_tokens (from [DONE] payload), and the error message.
    The client must receive exactly one [DONE] in the response body."""
    with (
        patch("proxy.server.stream_completion", return_value=_mock_stream_gen_error_after_done()),
        patch("litellm.token_counter", return_value=8),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "cheap",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

    assert resp.status_code == 200  # HTTP envelope is always 200 for SSE
    # Exactly one [DONE] sentinel — server must NOT emit a second one
    done_count = resp.text.count("data: [DONE]")
    assert done_count == 1, f"Expected exactly 1 [DONE], got {done_count}"

    recent = tracker.get_recent(limit=1)
    assert len(recent) == 1
    rec = recent[0]
    assert rec["status"] == 500, "Provider error after [DONE] must be logged as status 500"
    assert rec["output_tokens"] == 5, "output_tokens must be captured from the [DONE] payload"
    assert rec["error"] is not None and "provider exploded" in rec["error"]


# ── usage endpoints ───────────────────────────────────────────────────────────

def test_usage_summary_endpoint(client, tracker):
    tracker.log_request("cheap", "openrouter/test/cheap-model", 10, 5, 0.001)
    tracker.log_request("strong", "openrouter/test/strong-model", 20, 10, 0.005)

    resp = client.get("/v1/usage")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] == 2
    assert data["total_input_tokens"] == 30
    assert data["total_output_tokens"] == 15


def test_usage_by_model_endpoint(client, tracker):
    tracker.log_request("cheap", "openrouter/test/cheap-model", 10, 5, 0.001)
    tracker.log_request("cheap", "openrouter/test/cheap-model", 15, 8, 0.002)
    tracker.log_request("strong", "openrouter/test/strong-model", 20, 10, 0.005)

    resp = client.get("/v1/usage/by-model")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    by_model = {row["model_resolved"]: row for row in data}
    assert by_model["openrouter/test/cheap-model"]["total_requests"] == 2
    assert by_model["openrouter/test/strong-model"]["total_requests"] == 1


def test_usage_recent_endpoint(client, tracker):
    for i in range(5):
        tracker.log_request("cheap", "openrouter/test/cheap-model", i, i, 0.0)

    resp = client.get("/v1/usage/recent?limit=3")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3


# ── adaptive routing integration ───────────────────────────────────────────────

_ROUTING_ALIASES = {
    "cheap":  "openrouter/test/cheap-model",
    "medium": "openrouter/test/medium-model",
    "strong": "openrouter/test/strong-model",
    "auto":   "openrouter/test/cheap-model",   # fallback placeholder
}


@pytest.fixture
def routing_client(tmp_path):
    """TestClient with adaptive routing enabled (categorizer mocked in each test)."""
    cfg = ProxyConfig(
        server=ServerConfig(host="127.0.0.1", port=8888),
        models=ModelConfig(aliases=_ROUTING_ALIASES, default="cheap"),
        tracking=TrackingConfig(db=":memory:", enabled=True),
        routing=RoutingConfig(
            enabled=True,
            db=str(tmp_path / "routing.db"),
            categorizer_model="groq/llama-3.1-8b-instant",
        ),
    )
    app = create_app(cfg, None)
    return TestClient(app)


def _cat_mock(category: str, confidence: float = 0.95) -> MagicMock:
    """Build a mock litellm response shaped like a categorizer reply."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(
        {"category": category, "confidence": confidence}
    )
    return resp


def test_auto_model_routes_coding_to_strong(routing_client):
    """model='auto', 'coding' category → strong model forwarded to forwarder."""
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0)) as mock_fwd,
        patch("proxy.router.litellm.acompletion", new_callable=AsyncMock,
              return_value=_cat_mock("coding", 0.97)),
        patch("litellm.token_counter", return_value=5),
    ):
        resp = routing_client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "fix my code"}]},
        )
    assert resp.status_code == 200
    assert mock_fwd.call_args[0][0] == "openrouter/test/strong-model"


def test_auto_model_routes_casual_to_cheap(routing_client):
    """model='auto', 'casual' category → cheap model forwarded."""
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0)) as mock_fwd,
        patch("proxy.router.litellm.acompletion", new_callable=AsyncMock,
              return_value=_cat_mock("casual", 0.98)),
        patch("litellm.token_counter", return_value=5),
    ):
        resp = routing_client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "good morning!"}]},
        )
    assert resp.status_code == 200
    assert mock_fwd.call_args[0][0] == "openrouter/test/cheap-model"


def test_non_auto_model_with_routing_enabled_skips_router(routing_client):
    """model='cheap' with routing enabled: categorizer NOT called, alias used."""
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0)) as mock_fwd,
        patch("proxy.router.litellm.acompletion", new_callable=AsyncMock) as mock_cat,
        patch("litellm.token_counter", return_value=5),
    ):
        resp = routing_client.post(
            "/v1/chat/completions",
            json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    mock_cat.assert_not_called()
    assert mock_fwd.call_args[0][0] == "openrouter/test/cheap-model"


def test_routing_disabled_auto_model_uses_alias(client):
    """Routing disabled: model='auto' falls through to alias resolution.
    The base config has no 'auto' alias → 'auto' is forwarded unchanged."""
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0)) as mock_fwd,
        patch("proxy.router.litellm.acompletion", new_callable=AsyncMock) as mock_cat,
        patch("litellm.token_counter", return_value=5),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    mock_cat.assert_not_called()
    assert mock_fwd.call_args[0][0] == "auto"   # no alias → passed through unchanged


def test_auto_model_no_user_messages_routes_as_casual(routing_client):
    """No user-role messages: last_user_msg='' → LLMCategorizer short-circuits
    to 'casual' without an LLM call → cheap model forwarded."""
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0)) as mock_fwd,
        patch("proxy.router.litellm.acompletion", new_callable=AsyncMock) as mock_cat,
        patch("litellm.token_counter", return_value=0),
    ):
        resp = routing_client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "system", "content": "you are helpful"}],
            },
        )
    assert resp.status_code == 200
    mock_cat.assert_not_called()   # empty text → casual without LLM call
    assert mock_fwd.call_args[0][0] == "openrouter/test/cheap-model"


def test_auto_model_routing_streaming_uses_correct_model(routing_client):
    """model='auto' + stream=True: routing resolves model before streaming."""
    with (
        patch("proxy.server.stream_completion", return_value=_mock_stream_gen()) as mock_stream,
        patch("proxy.router.litellm.acompletion", new_callable=AsyncMock,
              return_value=_cat_mock("reasoning", 0.88)),
        patch("litellm.token_counter", return_value=5),
    ):
        resp = routing_client.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "should I use microservices?"}],
                "stream": True,
            },
        )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert mock_stream.call_args[0][0] == "openrouter/test/strong-model"  # reasoning → strong


def test_auto_model_multimodal_content_does_not_crash(routing_client):
    """Fix #1+#2: multimodal content (list) must not crash via .strip() or .encode()."""
    multimodal_msg = [{"type": "text", "text": "fix my code"}, {"type": "image_url", "url": "..."}]
    with (
        patch("proxy.server.call_non_streaming", return_value=(_MOCK_RESPONSE, 5, 0.0)) as mock_fwd,
        patch("proxy.router.litellm.acompletion", new_callable=AsyncMock) as mock_cat,
        patch("litellm.token_counter", return_value=5),
    ):
        resp = routing_client.post(
            "/v1/chat/completions",
            json={"model": "auto", "messages": [{"role": "user", "content": multimodal_msg}]},
        )
    # Must not 500 — content list handled gracefully (categorize returns casual)
    assert resp.status_code == 200
    mock_cat.assert_not_called()   # list → non-string → casual short-circuit, no LLM call
    assert mock_fwd.call_args[0][0] == "openrouter/test/cheap-model"  # casual → cheap
