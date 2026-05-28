"""
Unit tests for proxy/router.py — LLMCategorizer, RoutingLogger, ModelRouter.

No real LLM calls or API keys needed: litellm.acompletion is mocked throughout.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from proxy.config import RoutingConfig
from proxy.router import (
    CATEGORY_TIER_MAP,
    VALID_CATEGORIES,
    VALID_SOURCES,
    LLMCategorizer,
    ModelRouter,
    RoutingLogger,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_response(content: str) -> MagicMock:
    """Build a MagicMock shaped like a litellm completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _categorizer(model: str = "groq/llama-3.1-8b-instant", **kw) -> LLMCategorizer:
    return LLMCategorizer(model=model, **kw)


# ── constants sanity checks ───────────────────────────────────────────────────

def test_valid_categories_has_eight_entries():
    assert len(VALID_CATEGORIES) == 8


def test_every_category_has_a_tier():
    assert set(CATEGORY_TIER_MAP.keys()) == VALID_CATEGORIES


def test_tier_values_are_cheap_medium_strong():
    assert set(CATEGORY_TIER_MAP.values()) <= {"cheap", "medium", "strong"}


def test_valid_sources_contains_expected_values():
    assert "llm_categorizer" in VALID_SOURCES
    assert "default" in VALID_SOURCES
    assert "continuation" in VALID_SOURCES


# ── LLMCategorizer.categorize — empty / whitespace ────────────────────────────

@pytest.mark.anyio
async def test_empty_text_returns_casual_without_llm_call():
    cat = _categorizer()
    with patch("proxy.router.litellm.acompletion", new_callable=AsyncMock) as mock_ac:
        result = await cat.categorize("")
    assert result == ("casual", 1.0, None)
    mock_ac.assert_not_called()


@pytest.mark.anyio
async def test_whitespace_only_returns_casual_without_llm_call():
    cat = _categorizer()
    with patch("proxy.router.litellm.acompletion", new_callable=AsyncMock) as mock_ac:
        result = await cat.categorize("   \t\n  ")
    assert result == ("casual", 1.0, None)
    mock_ac.assert_not_called()


# ── LLMCategorizer.categorize — happy path ────────────────────────────────────

@pytest.mark.anyio
async def test_happy_path_returns_correct_category_and_confidence():
    cat = _categorizer()
    payload = json.dumps({"category": "coding", "confidence": 0.97})
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_mock_response(payload),
    ):
        category, confidence, latency_ms = await cat.categorize("fix this bug")

    assert category == "coding"
    assert confidence == pytest.approx(0.97)
    assert isinstance(latency_ms, float)
    assert latency_ms >= 0.0


@pytest.mark.anyio
async def test_latency_ms_is_positive_float():
    cat = _categorizer()
    payload = json.dumps({"category": "casual", "confidence": 0.9})
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_mock_response(payload),
    ):
        _, _, latency_ms = await cat.categorize("hi there")

    assert isinstance(latency_ms, float)
    assert latency_ms >= 0.0


# ── LLMCategorizer.categorize — context forwarding ───────────────────────────

@pytest.mark.anyio
async def test_context_is_included_as_separate_user_message():
    cat = _categorizer()
    payload = json.dumps({"category": "coding", "confidence": 0.85})
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_mock_response(payload),
    ) as mock_ac:
        await cat.categorize("still broken", context="[Previous routing: coding, 5s ago]")

    call_messages = mock_ac.call_args.kwargs["messages"]
    # system + context + user = 3 messages
    assert len(call_messages) == 3
    assert call_messages[1]["role"] == "user"
    assert "[Previous routing:" in call_messages[1]["content"]
    assert call_messages[2]["content"] == "still broken"


@pytest.mark.anyio
async def test_no_context_sends_two_messages():
    cat = _categorizer()
    payload = json.dumps({"category": "casual", "confidence": 0.9})
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_mock_response(payload),
    ) as mock_ac:
        await cat.categorize("hello")

    call_messages = mock_ac.call_args.kwargs["messages"]
    assert len(call_messages) == 2  # system + user only


# ── LLMCategorizer.categorize — text truncation ──────────────────────────────

@pytest.mark.anyio
async def test_long_text_is_truncated_to_2000_chars():
    cat = _categorizer()
    long_text = "x" * 5000
    payload = json.dumps({"category": "casual", "confidence": 0.5})
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_mock_response(payload),
    ) as mock_ac:
        await cat.categorize(long_text)

    call_messages = mock_ac.call_args.kwargs["messages"]
    user_content = call_messages[-1]["content"]
    assert len(user_content) == 2000


@pytest.mark.anyio
async def test_long_context_is_truncated_to_2000_chars():
    cat = _categorizer()
    long_ctx = "c" * 5000
    payload = json.dumps({"category": "casual", "confidence": 0.5})
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_mock_response(payload),
    ) as mock_ac:
        await cat.categorize("hi", context=long_ctx)

    call_messages = mock_ac.call_args.kwargs["messages"]
    ctx_content = call_messages[1]["content"]
    assert len(ctx_content) == 2000


# ── LLMCategorizer.categorize — LLM failure ──────────────────────────────────

@pytest.mark.anyio
async def test_acompletion_raises_returns_unknown():
    cat = _categorizer()
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        side_effect=RuntimeError("provider down"),
    ):
        result = await cat.categorize("hello")

    assert result == ("unknown", 0.0, None)


# ── LLMCategorizer.categorize — kwargs forwarding ────────────────────────────

@pytest.mark.anyio
async def test_api_base_and_api_key_forwarded_to_acompletion():
    cat = _categorizer(api_base="https://custom.endpoint", api_key="sk-test")
    payload = json.dumps({"category": "casual", "confidence": 0.9})
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_mock_response(payload),
    ) as mock_ac:
        await cat.categorize("hi")

    kwargs = mock_ac.call_args.kwargs
    assert kwargs["api_base"] == "https://custom.endpoint"
    assert kwargs["api_key"] == "sk-test"


@pytest.mark.anyio
async def test_timeout_forwarded_to_acompletion():
    cat = _categorizer(timeout=5.0)
    payload = json.dumps({"category": "casual", "confidence": 0.9})
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_mock_response(payload),
    ) as mock_ac:
        await cat.categorize("hi")

    assert mock_ac.call_args.kwargs["timeout"] == 5.0


@pytest.mark.anyio
async def test_no_api_base_not_forwarded():
    """When api_base is None, it must NOT appear in kwargs (avoids sending 'null')."""
    cat = _categorizer()  # default: api_base=None
    payload = json.dumps({"category": "casual", "confidence": 0.9})
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_mock_response(payload),
    ) as mock_ac:
        await cat.categorize("hi")

    assert "api_base" not in mock_ac.call_args.kwargs
    assert "api_key" not in mock_ac.call_args.kwargs


# ── LLMCategorizer._parse_response ───────────────────────────────────────────

def test_parse_response_empty_string_returns_unknown():
    cat = _categorizer()
    assert cat._parse_response("") == ("unknown", 0.0)


def test_parse_response_valid_json():
    cat = _categorizer()
    raw = json.dumps({"category": "reasoning", "confidence": 0.88})
    assert cat._parse_response(raw) == ("reasoning", 0.88)


def test_parse_response_strips_json_fences():
    cat = _categorizer()
    raw = "```json\n{\"category\": \"creative\", \"confidence\": 0.96}\n```"
    category, confidence = cat._parse_response(raw)
    assert category == "creative"
    assert confidence == pytest.approx(0.96)


def test_parse_response_handles_prose_around_json():
    cat = _categorizer()
    raw = 'Sure! Here is my answer: {"category": "analysis", "confidence": 0.92} Hope that helps.'
    category, confidence = cat._parse_response(raw)
    assert category == "analysis"
    assert confidence == pytest.approx(0.92)


def test_parse_response_no_json_object_returns_unknown():
    cat = _categorizer()
    assert cat._parse_response("I cannot categorize this.") == ("unknown", 0.0)


def test_parse_response_malformed_json_returns_unknown():
    cat = _categorizer()
    assert cat._parse_response("{category: broken}") == ("unknown", 0.0)


def test_parse_response_unknown_category_falls_back():
    cat = _categorizer()
    raw = json.dumps({"category": "totally_made_up", "confidence": 0.99})
    category, _ = cat._parse_response(raw)
    assert category == "unknown"


def test_parse_response_confidence_string_coerced():
    """Confidence given as a JSON string (e.g. "0.8") should be coerced to float."""
    cat = _categorizer()
    raw = '{"category": "casual", "confidence": "0.8"}'
    _, confidence = cat._parse_response(raw)
    assert confidence == pytest.approx(0.8)


def test_parse_response_confidence_none_falls_back_to_zero():
    cat = _categorizer()
    raw = '{"category": "casual", "confidence": null}'
    _, confidence = cat._parse_response(raw)
    assert confidence == pytest.approx(0.0)


def test_parse_response_confidence_non_numeric_falls_back_to_zero():
    cat = _categorizer()
    raw = '{"category": "casual", "confidence": "high"}'
    _, confidence = cat._parse_response(raw)
    assert confidence == pytest.approx(0.0)


def test_parse_response_confidence_clamped_above_one():
    cat = _categorizer()
    raw = json.dumps({"category": "casual", "confidence": 1.5})
    _, confidence = cat._parse_response(raw)
    assert confidence == pytest.approx(1.0)


def test_parse_response_confidence_clamped_below_zero():
    cat = _categorizer()
    raw = json.dumps({"category": "casual", "confidence": -0.3})
    _, confidence = cat._parse_response(raw)
    assert confidence == pytest.approx(0.0)


def test_parse_response_missing_confidence_falls_back_to_zero():
    cat = _categorizer()
    raw = '{"category": "casual"}'
    _, confidence = cat._parse_response(raw)
    assert confidence == pytest.approx(0.0)


# ── RoutingLogger — init ──────────────────────────────────────────────────────

def test_routing_logger_creates_table(tmp_path):
    db = str(tmp_path / "test.db")
    RoutingLogger(db)
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "routing_log" in tables


def test_routing_logger_init_is_idempotent(tmp_path):
    """Creating RoutingLogger twice against the same DB must not raise."""
    db = str(tmp_path / "test.db")
    RoutingLogger(db)
    RoutingLogger(db)  # second init — CREATE TABLE IF NOT EXISTS


def test_routing_logger_bad_db_path_does_not_raise():
    """A path that cannot be created must be swallowed, not raised."""
    logger = RoutingLogger("/nonexistent/path/to/db.sqlite")
    # Should not raise — _init_db catches the error


# ── RoutingLogger — log ───────────────────────────────────────────────────────

def test_routing_logger_writes_one_row(tmp_path):
    db = str(tmp_path / "test.db")
    rl = RoutingLogger(db)
    rl.log(
        category="coding",
        selected_tier="strong",
        tier_mapping_version="v1",
        model_used="groq/llama-3.1-8b-instant",
        router_version="phase1_llm",
        source="llm_categorizer",
        category_confidence=0.97,
        pool_eligible=False,
        prompt="fix this bug",
        latency_ms=220.5,
    )
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT * FROM routing_log").fetchall()
    assert len(rows) == 1


def test_routing_logger_fields_are_correct(tmp_path):
    db = str(tmp_path / "test.db")
    rl = RoutingLogger(db)
    rl.log(
        category="coding",
        selected_tier="strong",
        tier_mapping_version="v1",
        model_used="groq/llama-3.1-8b-instant",
        router_version="phase1_llm",
        source="llm_categorizer",
        category_confidence=0.97,
        pool_eligible=True,
        prompt="fix this bug",
        latency_ms=220.5,
    )
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM routing_log").fetchone()

    assert row["category"] == "coding"
    assert row["selected_tier"] == "strong"
    assert row["tier_mapping_version"] == "v1"
    assert row["model_used"] == "groq/llama-3.1-8b-instant"
    assert row["router_version"] == "phase1_llm"
    assert row["source"] == "llm_categorizer"
    assert row["category_confidence"] == pytest.approx(0.97)
    assert row["pool_eligible"] == 1
    assert row["prompt"] == "fix this bug"
    assert row["latency_ms"] == pytest.approx(220.5)


def test_routing_logger_prompt_hash_is_sha256(tmp_path):
    db = str(tmp_path / "test.db")
    rl = RoutingLogger(db)
    prompt_text = "fix this bug"
    expected_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
    rl.log(
        category="coding",
        selected_tier="strong",
        tier_mapping_version="v1",
        model_used="groq/llama-3.1-8b-instant",
        router_version="phase1_llm",
        source="llm_categorizer",
        prompt=prompt_text,
    )
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT prompt_hash FROM routing_log").fetchone()
    assert row["prompt_hash"] == expected_hash


def test_routing_logger_null_prompt_gives_null_hash(tmp_path):
    db = str(tmp_path / "test.db")
    rl = RoutingLogger(db)
    rl.log(
        category="casual",
        selected_tier="cheap",
        tier_mapping_version="v1",
        model_used="openrouter/minimax/minimax-m2.5",
        router_version="phase1_llm",
        source="default",
        prompt=None,
    )
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT prompt_hash, prompt FROM routing_log").fetchone()
    assert row["prompt_hash"] is None
    assert row["prompt"] is None


def test_routing_logger_unknown_source_still_writes(tmp_path):
    """Unknown source should warn but still write the row."""
    db = str(tmp_path / "test.db")
    rl = RoutingLogger(db)
    rl.log(
        category="casual",
        selected_tier="cheap",
        tier_mapping_version="v1",
        model_used="groq/llama-3.1-8b-instant",
        router_version="phase1_llm",
        source="skill_match",  # not in VALID_SOURCES for proxy
    )
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT source FROM routing_log").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "skill_match"


def test_routing_logger_bad_db_path_log_does_not_raise():
    """log() against an uninitialised (bad-path) logger must not raise."""
    rl = RoutingLogger("/nonexistent/path/to/db.sqlite")
    # _init_db already failed silently; log() must also fail silently
    rl.log(
        category="casual",
        selected_tier="cheap",
        tier_mapping_version="v1",
        model_used="groq/llama-3.1-8b-instant",
        router_version="phase1_llm",
        source="default",
    )


def test_routing_logger_multiple_rows(tmp_path):
    db = str(tmp_path / "test.db")
    rl = RoutingLogger(db)
    for _ in range(3):
        rl.log(
            category="casual",
            selected_tier="cheap",
            tier_mapping_version="v1",
            model_used="groq/llama-3.1-8b-instant",
            router_version="phase1_llm",
            source="llm_categorizer",
        )
    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM routing_log").fetchone()[0]
    assert count == 3


# ── ModelRouter helpers ───────────────────────────────────────────────────────

_DEFAULT_ALIASES = {
    "cheap":  "groq/llama-3.1-8b-instant",
    "medium": "openrouter/qwen/qwen3-235b-a22b",
    "strong": "anthropic/claude-sonnet-4-5",
}


def _make_routing_cfg(tmp_path, **overrides) -> RoutingConfig:
    defaults: dict = dict(
        db=str(tmp_path / "routing.db"),
        confidence_threshold=0.7,
        min_prompt_length_for_pool=10,
        session_ttl_seconds=1800,
        categorizer_timeout=5.0,
    )
    defaults.update(overrides)
    return RoutingConfig(**defaults)


def _make_router(tmp_path, aliases=None, **cfg_overrides) -> tuple[ModelRouter, str]:
    cfg = _make_routing_cfg(tmp_path, **cfg_overrides)
    router = ModelRouter(aliases=aliases or _DEFAULT_ALIASES, routing_config=cfg)
    return router, cfg.db


def _cat_response(category: str, confidence: float = 0.9) -> MagicMock:
    return _mock_response(json.dumps({"category": category, "confidence": confidence}))


# ── ModelRouter — construction ────────────────────────────────────────────────

def test_model_router_constructs_without_api_key_env(tmp_path):
    """Constructing with no categorizer_api_key_env must not raise."""
    router, _ = _make_router(tmp_path)
    assert router is not None


def test_model_router_api_key_from_env(tmp_path, monkeypatch):
    """When env var is set, the categorizer receives the resolved key."""
    monkeypatch.setenv("TEST_ROUTER_API_KEY", "sk-test-key-abc")
    router, _ = _make_router(tmp_path, categorizer_api_key_env="TEST_ROUTER_API_KEY")
    assert router._categorizer._api_key == "sk-test-key-abc"


def test_model_router_missing_api_key_env_warns(tmp_path, caplog):
    """When env var is declared but not set, a WARNING is logged."""
    import logging
    with caplog.at_level(logging.WARNING, logger="proxy.router"):
        _make_router(tmp_path, categorizer_api_key_env="NONEXISTENT_ROUTER_KEY_XYZ")
    assert "NONEXISTENT_ROUTER_KEY_XYZ" in caplog.text


def test_model_router_categorizer_falls_back_to_cheap_alias(tmp_path):
    """When categorizer_model is None, the cheap alias is used."""
    router, _ = _make_router(tmp_path)  # categorizer_model=None by default
    assert router._categorizer._model == _DEFAULT_ALIASES["cheap"]


# ── ModelRouter — happy paths ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_model_router_coding_returns_strong(tmp_path):
    router, _ = _make_router(tmp_path)
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_cat_response("coding", 0.97),
    ):
        tier, model = await router.route(
            "fix this bug please",
            [{"role": "user", "content": "fix this bug please"}],
        )
    assert tier == "strong"
    assert model == _DEFAULT_ALIASES["strong"]


@pytest.mark.anyio
async def test_model_router_casual_returns_cheap(tmp_path):
    router, _ = _make_router(tmp_path)
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_cat_response("casual", 0.95),
    ):
        tier, model = await router.route(
            "good morning!",
            [{"role": "user", "content": "good morning!"}],
        )
    assert tier == "cheap"
    assert model == _DEFAULT_ALIASES["cheap"]


@pytest.mark.anyio
async def test_model_router_unknown_category_returns_medium(tmp_path):
    router, _ = _make_router(tmp_path)
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_cat_response("unknown", 0.4),
    ):
        tier, model = await router.route(
            "xyzzy plugh",
            [{"role": "user", "content": "xyzzy plugh"}],
        )
    assert tier == "medium"
    assert model == _DEFAULT_ALIASES["medium"]


# ── ModelRouter — session caching / continuation ──────────────────────────────

@pytest.mark.anyio
async def test_model_router_no_session_key_no_cache_stored(tmp_path):
    """session_key=None — categorizer is called and nothing is cached."""
    router, _ = _make_router(tmp_path)
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_cat_response("casual", 0.9),
    ) as mock_ac:
        await router.route("hi", [{"role": "user", "content": "hi"}])
    mock_ac.assert_called_once()
    assert not router._conv_context


@pytest.mark.anyio
async def test_model_router_cache_hit_last_role_user_calls_categorizer(tmp_path):
    """Cache hit + last role='user' → NOT a continuation → categorizer called."""
    router, _ = _make_router(tmp_path)
    router._conv_context["s1"] = {
        "category": "coding", "tier": "strong",
        "model": _DEFAULT_ALIASES["strong"],
        "last_user_msg": "first question",
        "timestamp": time.time(),
    }
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_cat_response("coding", 0.91),
    ) as mock_ac:
        await router.route(
            "follow-up",
            [{"role": "user", "content": "follow-up"}],
            session_key="s1",
        )
    mock_ac.assert_called_once()


@pytest.mark.anyio
async def test_model_router_cache_hit_last_role_tool_is_continuation(tmp_path):
    """Cache hit + last role='tool' → continuation → categorizer skipped."""
    router, _ = _make_router(tmp_path)
    router._conv_context["s2"] = {
        "category": "coding", "tier": "strong",
        "model": _DEFAULT_ALIASES["strong"],
        "last_user_msg": "run my code",
        "timestamp": time.time(),
    }
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
    ) as mock_ac:
        tier, model = await router.route(
            "tool_result",
            [{"role": "tool", "content": "tool output"}],
            session_key="s2",
        )
    mock_ac.assert_not_called()
    assert tier == "strong"
    assert model == _DEFAULT_ALIASES["strong"]


@pytest.mark.anyio
async def test_model_router_cache_hit_last_role_assistant_is_continuation(tmp_path):
    """Cache hit + last role='assistant' → continuation → categorizer skipped."""
    router, _ = _make_router(tmp_path)
    router._conv_context["s3"] = {
        "category": "reasoning", "tier": "strong",
        "model": _DEFAULT_ALIASES["strong"],
        "last_user_msg": "what's the plan?",
        "timestamp": time.time(),
    }
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
    ) as mock_ac:
        tier, model = await router.route(
            "",
            [{"role": "assistant", "content": "here is the plan..."}],
            session_key="s3",
        )
    mock_ac.assert_not_called()
    assert tier == "strong"


@pytest.mark.anyio
async def test_model_router_no_cache_last_role_tool_calls_categorizer(tmp_path):
    """No cache + last role='tool' → _is_continuation returns False → categorizer called."""
    router, _ = _make_router(tmp_path)
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_cat_response("coding", 0.9),
    ) as mock_ac:
        await router.route(
            "tool result text",
            [{"role": "tool", "content": "result"}],
            session_key="s-fresh",
        )
    mock_ac.assert_called_once()


@pytest.mark.anyio
async def test_model_router_cache_expired_calls_categorizer(tmp_path):
    """Expired cache + last role='tool' → not a continuation → categorizer called."""
    router, _ = _make_router(tmp_path, session_ttl_seconds=1)
    router._conv_context["s4"] = {
        "category": "coding", "tier": "strong",
        "model": _DEFAULT_ALIASES["strong"],
        "last_user_msg": "old message",
        "timestamp": time.time() - 9999,     # far in the past
    }
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_cat_response("analysis", 0.85),
    ) as mock_ac:
        tier, model = await router.route(
            "new message here",
            [{"role": "tool", "content": "tool result"}],
            session_key="s4",
        )
    mock_ac.assert_called_once()
    assert tier == "medium"   # "analysis" maps to medium


@pytest.mark.anyio
async def test_model_router_empty_messages_calls_categorizer(tmp_path):
    """Empty message list → _is_continuation returns False → categorizer called."""
    router, _ = _make_router(tmp_path)
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_cat_response("casual", 0.9),
    ) as mock_ac:
        await router.route("hi", [], session_key="s5")
    mock_ac.assert_called_once()


@pytest.mark.anyio
async def test_model_router_session_context_updated_after_route(tmp_path):
    """After a normal route(), session cache is populated with new category/tier."""
    router, _ = _make_router(tmp_path)
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_cat_response("creative", 0.88),
    ):
        await router.route(
            "write me a poem",
            [{"role": "user", "content": "write me a poem"}],
            session_key="s6",
        )
    assert "s6" in router._conv_context
    assert router._conv_context["s6"]["category"] == "creative"
    assert router._conv_context["s6"]["tier"] == "medium"


@pytest.mark.anyio
async def test_model_router_build_context_returns_none_no_cache(tmp_path):
    """_build_context returns None when session has no prior cache entry."""
    router, _ = _make_router(tmp_path)
    result = router._build_context("no-such-session", "some text")
    assert result is None


@pytest.mark.anyio
async def test_model_router_build_context_deletes_expired_entry(tmp_path):
    """_build_context removes the expired entry and returns None."""
    router, _ = _make_router(tmp_path, session_ttl_seconds=1)
    router._conv_context["s7"] = {
        "category": "coding", "tier": "strong",
        "model": _DEFAULT_ALIASES["strong"],
        "last_user_msg": "old",
        "timestamp": time.time() - 9999,
    }
    result = router._build_context("s7", "new")
    assert result is None
    assert "s7" not in router._conv_context


# ── ModelRouter — routing_log entries ────────────────────────────────────────

@pytest.mark.anyio
async def test_model_router_logs_llm_categorizer_source(tmp_path):
    router, db = _make_router(tmp_path)
    with patch(
        "proxy.router.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=_cat_response("coding", 0.97),
    ):
        await router.route(
            "fix this bug",
            [{"role": "user", "content": "fix this bug"}],
            prompt_for_hash="fix this bug",
            session_key="s8",
        )
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM routing_log").fetchone()
    assert row["source"] == "llm_categorizer"
    assert row["category"] == "coding"
    assert row["selected_tier"] == "strong"
    assert row["router_version"] == "phase2_llm"


@pytest.mark.anyio
async def test_model_router_logs_continuation_source(tmp_path):
    router, db = _make_router(tmp_path)
    router._conv_context["s9"] = {
        "category": "coding", "tier": "strong",
        "model": _DEFAULT_ALIASES["strong"],
        "last_user_msg": "fix code",
        "timestamp": time.time(),
    }
    with patch("proxy.router.litellm.acompletion", new_callable=AsyncMock) as mock_ac:
        await router.route(
            "",
            [{"role": "tool", "content": "output"}],
            session_key="s9",
        )
    mock_ac.assert_not_called()
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT source, category FROM routing_log").fetchone()
    assert row["source"] == "continuation"
    assert row["category"] == "coding"


# ── ModelRouter — _is_pool_eligible ──────────────────────────────────────────

def test_pool_eligible_all_conditions_met(tmp_path):
    router, _ = _make_router(tmp_path, confidence_threshold=0.7, min_prompt_length_for_pool=5)
    assert router._is_pool_eligible("coding", 0.8, "fix my bug") is True


def test_pool_eligible_unknown_category_is_false(tmp_path):
    router, _ = _make_router(tmp_path)
    assert router._is_pool_eligible("unknown", 0.9, "some long text here") is False


def test_pool_eligible_low_confidence_is_false(tmp_path):
    router, _ = _make_router(tmp_path, confidence_threshold=0.7)
    assert router._is_pool_eligible("coding", 0.5, "fix my bug here today") is False


def test_pool_eligible_short_text_is_false(tmp_path):
    router, _ = _make_router(tmp_path, min_prompt_length_for_pool=10)
    assert router._is_pool_eligible("coding", 0.9, "hi") is False
