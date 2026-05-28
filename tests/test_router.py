"""
Unit tests for proxy/router.py — LLMCategorizer and RoutingLogger.

No real LLM calls or API keys needed: litellm.acompletion is mocked throughout.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from proxy.router import (
    CATEGORY_TIER_MAP,
    VALID_CATEGORIES,
    VALID_SOURCES,
    LLMCategorizer,
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
