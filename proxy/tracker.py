from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


class RequestTracker:
    """Thread-safe SQLite request logger. Opens a new connection per operation."""

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS request_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            client_id TEXT,
            session_key TEXT,
            model_requested TEXT NOT NULL,
            model_resolved TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL NOT NULL DEFAULT 0,
            latency_ms REAL,
            streaming INTEGER NOT NULL DEFAULT 0,
            status INTEGER NOT NULL DEFAULT 200,
            error TEXT
        );
    """
    _CREATE_INDEX_TS = "CREATE INDEX IF NOT EXISTS idx_ts ON request_log(ts);"
    _CREATE_INDEX_CLIENT = "CREATE INDEX IF NOT EXISTS idx_client ON request_log(client_id);"

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                # Enable WAL journal mode for better concurrent reader/writer
                # throughput. WAL is a persistent DB-level setting — all
                # subsequent connections (including RoutingLogger's) inherit it
                # automatically regardless of which class initialises the DB first.
                # Best-effort: a read-only DB or WAL-unsupported filesystem must
                # not prevent the table from being created or crash the server.
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                except Exception as wal_exc:
                    logger.warning(
                        f"[RequestTracker] WAL mode unavailable at {self._db_path}: {wal_exc}"
                    )
                conn.execute(self._CREATE_TABLE)
                conn.execute(self._CREATE_INDEX_TS)
                conn.execute(self._CREATE_INDEX_CLIENT)

    def log_request(
        self,
        model_requested: str,
        model_resolved: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: Optional[float] = None,
        streaming: bool = False,
        status: int = 200,
        error: Optional[str] = None,
        client_id: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> int:
        """Insert a request log row. Returns the inserted row id."""
        total_tokens = input_tokens + output_tokens
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO request_log (
                        client_id, session_key, model_requested, model_resolved,
                        input_tokens, output_tokens, total_tokens, cost_usd,
                        latency_ms, streaming, status, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        client_id,
                        session_key,
                        model_requested,
                        model_resolved,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        cost_usd,
                        latency_ms,
                        1 if streaming else 0,
                        status,
                        error,
                    ),
                )
                return cursor.lastrowid  # type: ignore[return-value]

    def get_summary(self) -> dict[str, Any]:
        """Return aggregate stats across all logged requests."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total_requests,
                        COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
                        COALESCE(SUM(total_tokens), 0) AS total_tokens,
                        COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                        COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms
                    FROM request_log
                    """
                ).fetchone()
                return dict(row)

    def get_by_model(self) -> list[dict[str, Any]]:
        """Return per-model breakdown sorted by total requests descending."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        model_resolved,
                        COUNT(*) AS total_requests,
                        COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
                        COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
                        COALESCE(SUM(total_tokens), 0) AS total_tokens,
                        COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                        COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms
                    FROM request_log
                    GROUP BY model_resolved
                    ORDER BY total_requests DESC
                    """
                ).fetchall()
                return [dict(r) for r in rows]

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent request log rows."""
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM request_log
                    ORDER BY ts DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
