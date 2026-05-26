"""Tests for RequestTracker — no LiteLLM calls involved."""
import sqlite3
import os

import pytest

from proxy.tracker import RequestTracker


def test_db_init_creates_schema(tmp_path):
    """Initializing RequestTracker creates the request_log table and indexes."""
    db_path = str(tmp_path / "test.db")
    RequestTracker(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='request_log'"
    )
    assert cursor.fetchone() is not None, "request_log table should exist"

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_ts'"
    )
    assert cursor.fetchone() is not None, "idx_ts index should exist"

    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_client'"
    )
    assert cursor.fetchone() is not None, "idx_client index should exist"
    conn.close()


def test_log_request_inserts_row(tmp_path):
    """log_request inserts a row and returns its id."""
    db_path = str(tmp_path / "test.db")
    tracker = RequestTracker(db_path)

    row_id = tracker.log_request(
        model_requested="cheap",
        model_resolved="openrouter/minimax/minimax-m2.5",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_ms=250.0,
        streaming=False,
        status=200,
    )
    assert isinstance(row_id, int)
    assert row_id >= 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM request_log WHERE id = ?", (row_id,)).fetchone()
    assert row is not None
    assert row["model_requested"] == "cheap"
    assert row["model_resolved"] == "openrouter/minimax/minimax-m2.5"
    assert row["input_tokens"] == 100
    assert row["output_tokens"] == 50
    assert row["total_tokens"] == 150
    assert abs(row["cost_usd"] - 0.001) < 1e-9
    assert row["streaming"] == 0
    assert row["status"] == 200
    conn.close()


def test_get_summary_returns_correct_aggregate(tmp_path):
    """get_summary aggregates stats correctly across multiple rows."""
    db_path = str(tmp_path / "test.db")
    tracker = RequestTracker(db_path)

    tracker.log_request(
        model_requested="cheap",
        model_resolved="openrouter/minimax/minimax-m2.5",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        latency_ms=100.0,
    )
    tracker.log_request(
        model_requested="strong",
        model_resolved="openrouter/minimax/minimax-m2.5",
        input_tokens=200,
        output_tokens=80,
        cost_usd=0.002,
        latency_ms=300.0,
    )

    summary = tracker.get_summary()
    assert summary["total_requests"] == 2
    assert summary["total_input_tokens"] == 300
    assert summary["total_output_tokens"] == 130
    assert summary["total_tokens"] == 430
    assert abs(summary["total_cost_usd"] - 0.003) < 1e-9
    assert abs(summary["avg_latency_ms"] - 200.0) < 1e-6


def test_get_by_model_returns_per_model_breakdown(tmp_path):
    """get_by_model returns separate rows per resolved model."""
    db_path = str(tmp_path / "test.db")
    tracker = RequestTracker(db_path)

    tracker.log_request(
        model_requested="cheap",
        model_resolved="model-a",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
    )
    tracker.log_request(
        model_requested="cheap",
        model_resolved="model-a",
        input_tokens=50,
        output_tokens=25,
        cost_usd=0.0005,
    )
    tracker.log_request(
        model_requested="strong",
        model_resolved="model-b",
        input_tokens=200,
        output_tokens=100,
        cost_usd=0.01,
    )

    by_model = tracker.get_by_model()
    assert len(by_model) == 2

    model_map = {r["model_resolved"]: r for r in by_model}
    assert "model-a" in model_map
    assert "model-b" in model_map

    assert model_map["model-a"]["total_requests"] == 2
    assert model_map["model-a"]["total_input_tokens"] == 150
    assert model_map["model-a"]["total_output_tokens"] == 75
    assert model_map["model-b"]["total_requests"] == 1
    assert model_map["model-b"]["total_input_tokens"] == 200
