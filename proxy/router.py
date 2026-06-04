"""
Adaptive Model Router — Phase 1: LLM-based task categorizer.

Classifies incoming prompts into task categories, which are mapped to model
tiers (cheap / medium / strong).  All routing decisions are persisted to a
SQLite routing_log table (separate from request_log, same DB file).

Classes:
  LLMCategorizer   — async classify a prompt via a cheap LLM (e.g. Groq)
  RoutingLogger    — persist routing decisions to routing_log table
  EmbeddingRouter  — k-NN embedding lookup (shadow/live mode)
  ModelRouter      — orchestrates categorizer + logger + embedding router

Phase 2: EmbeddingRouter — k-NN embedding lookup (shadow/live mode).
  shadow: k-NN runs alongside LLM categorizer, result logged but not used for routing.
  live:   k-NN result used when pool is large enough; LLM categorizer as fallback.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from collections import OrderedDict
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# ── Background task GC guard ──────────────────────────────────────────────────
# asyncio._all_tasks is a WeakSet — a Task with no other reference can be GC'd
# before it runs. This module-level set holds strong references until done.
_background_tasks: set[asyncio.Task] = set()


def _schedule_background(coro) -> None:
    """Schedule *coro* as a fire-and-forget task, holding a strong reference.

    Without this guard, a Task created by asyncio.create_task() can be
    garbage-collected before the event loop runs it — the Python docs explicitly
    warn to "save a reference to avoid a task disappearing mid-execution."
    The done-callback removes the strong reference once the task completes.
    """
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ── Category taxonomy ─────────────────────────────────────────────────────────

VALID_CATEGORIES = frozenset({
    "casual",
    "simple_lookup",
    "creative",
    "analysis",
    "coding",
    "reasoning",
    "unknown",
})

# category → tier mapping  (tier_mapping_version: v2)
CATEGORY_TIER_MAP: dict[str, str] = {
    "casual":        "cheap",
    "simple_lookup": "cheap",
    "creative":      "medium",
    "analysis":      "medium",   # covers both explaining/summarising AND research/expert-domain
    "coding":        "strong",
    "reasoning":     "strong",
    "unknown":       "medium",   # conservative default
}

# ── Categorizer system prompt ─────────────────────────────────────────────────

_CATEGORIZER_SYSTEM_PROMPT = """\
You are a task categorizer. Classify the user message into exactly one category.

Categories:
- casual: Greetings, small talk, thanks, emotional support ("good morning", \
"how are you", "thank you", "haha")
- simple_lookup: Direct factual questions with short, self-contained answers \
(definitions, translations, "what is X", "who is Y", "when did Z happen")
- creative: Generating original content (stories, poems, marketing copy, \
slogans, character names, scripts)
- analysis: Explaining, summarising, comparing, or analysing any topic — whether \
content already provided by the user OR questions requiring synthesis of \
external/expert-domain information (regulations, medical, financial, legal, \
current events, "latest policy on X", "compare A and B")
- coding: Writing code, debugging, code review, explaining code, software or \
system architecture design
- reasoning: Multi-step logic, planning, decision-making with tradeoffs, math \
word problems, "should I do X or Y given constraints Z"
- unknown: Cannot be confidently assigned to any of the above categories

Examples:
- "morning!" → {"category": "casual", "confidence": 0.98}
- "what does 'ephemeral' mean?" → {"category": "simple_lookup", "confidence": 0.95}
- "what are the current GDPR requirements for cookie consent?" → \
{"category": "analysis", "confidence": 0.90}
- "write me a birthday poem for my cat" → {"category": "creative", "confidence": 0.96}
- "summarize this article for me" → {"category": "analysis", "confidence": 0.92}
- "fix this Python TypeError: list index out of range" → \
{"category": "coding", "confidence": 0.97}
- "should I use microservices or a monolith for my startup?" → \
{"category": "reasoning", "confidence": 0.88}

If the message is preceded by context lines in this format:
  [Previous routing: <category>, <N>s ago]
  [Previous message:] <prior user message>
  [Current message:] <message to classify>
Use the previous routing context to inform your decision. Follow-up refinements,
corrections, or additions within a coding/analysis/reasoning/creative
conversation should remain in the same category unless the current message clearly
shifts topic.

Respond with valid JSON only, no other text:
{"category": "<one of the 7 categories>", "confidence": <float 0.0–1.0>}
"""


# ── LLMCategorizer ────────────────────────────────────────────────────────────

class LLMCategorizer:
    """Async classify a prompt into a task category using a cheap LLM.

    Usage:
        category, confidence, latency_ms = await categorizer.categorize(text)

    Returns:
        (category, confidence, latency_ms): category is one of VALID_CATEGORIES,
        confidence is a float in [0.0, 1.0], latency_ms is float ms for the LLM
        call (None when no LLM call was made, e.g. empty input).

    Never raises — any failure returns ("unknown", 0.0, None).
    """

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 15.0,
    ):
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._timeout = timeout

    async def categorize(
        self,
        text: str,
        context: str | None = None,
    ) -> tuple[str, float, float | None]:
        """Classify text into a routing category.

        Args:
            text:    The user message to classify. Truncated to 2000 chars.
            context: Optional prior-routing context string (built by
                     ModelRouter._build_context). When provided, context is sent
                     as one user message and text is sent as a second user message
                     prefixed with "[Current message:]" — matching the three-line
                     format described in _CATEGORIZER_SYSTEM_PROMPT.
                     Truncated to 2000 chars.

        Returns:
            (category, confidence, latency_ms).
            latency_ms is None when no LLM call was made (empty input).
        """
        # Guard: non-string input (e.g. multimodal content list) → treat as casual.
        if not isinstance(text, str) or not text or not text.strip():
            return "casual", 1.0, None

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _CATEGORIZER_SYSTEM_PROMPT},
        ]
        if context:
            # Send prior context then the current message with its label so the
            # LLM receives the exact three-line format the system prompt describes:
            #   [Previous routing: <cat>, <N>s ago]
            #   [Previous message:] <prior user message>
            #   [Current message:] <message to classify>
            messages.append({"role": "user", "content": context[:2000]})
            messages.append({"role": "user", "content": f"[Current message:] {text[:2000]}"})
        else:
            messages.append({"role": "user", "content": text[:2000]})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "timeout": self._timeout,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key

        t0 = time.perf_counter()
        try:
            response = await litellm.acompletion(**kwargs)
            raw: str = response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning(f"[Categorizer] LLM call failed: {exc}")
            return "unknown", 0.0, None
        latency_ms = (time.perf_counter() - t0) * 1000

        category, confidence = self._parse_response(raw)
        logger.info(
            f"[Categorizer] category={category} confidence={confidence:.2f} "
            f"latency={latency_ms:.0f}ms"
        )
        return category, confidence, latency_ms

    def _parse_response(self, raw: str) -> tuple[str, float]:
        """Parse LLM output into (category, confidence). Never raises."""
        if not raw:
            logger.warning("[Categorizer] Empty response from LLM")
            return "unknown", 0.0

        # Strip markdown code fences if model wrapped response in ```json ... ```
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        # Try direct parse first (well-formed response with no surrounding prose).
        # Fall back to raw_decode for responses with extra prose around the JSON.
        # raw_decode is string-aware and correctly handles `{`/`}` inside string values.
        data: Any = None
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            start = clean.find("{")
            if start == -1:
                logger.warning(f"[Categorizer] No JSON object found in response: {raw[:100]}")
                return "unknown", 0.0
            try:
                data, _ = json.JSONDecoder().raw_decode(clean, start)
            except json.JSONDecodeError as exc:
                logger.warning(f"[Categorizer] JSON parse error: {exc} | raw={raw[:100]}")
                return "unknown", 0.0

        # Guard: LLM may return valid but non-object JSON (list, string, int, null).
        if not isinstance(data, dict):
            logger.warning(
                f"[Categorizer] Expected JSON object, got {type(data).__name__}: {raw[:100]}"
            )
            return "unknown", 0.0

        category = str(data.get("category", "unknown"))
        if category not in VALID_CATEGORIES:
            logger.warning(
                f"[Categorizer] Unrecognised category '{category}', falling back to 'unknown'"
            )
            category = "unknown"

        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0.0, 1.0]

        return category, confidence


# ── RoutingLogger ─────────────────────────────────────────────────────────────

_CREATE_ROUTING_LOG_SQL = """
CREATE TABLE IF NOT EXISTS routing_log (
    id                             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp                      DATETIME DEFAULT CURRENT_TIMESTAMP,
    prompt_hash                    TEXT,
    prompt                         TEXT,
    category                       TEXT    NOT NULL,
    category_confidence            REAL,
    selected_tier                  TEXT    NOT NULL,
    tier_mapping_version           TEXT    NOT NULL,
    model_used                     TEXT    NOT NULL,
    router_version                 TEXT    NOT NULL,
    source                         TEXT    NOT NULL,
    pool_eligible                  INTEGER NOT NULL DEFAULT 0,
    latency_ms                     REAL,
    embedding                      BLOB,
    embedding_predicted_category   TEXT,
    embedding_predicted_confidence REAL
);
"""

# Migration list: (column_name, sql_type) for columns added after initial schema.
# Applied at init time so existing DBs are upgraded automatically.
_ROUTING_LOG_MIGRATIONS = [
    ("embedding",                      "BLOB"),
    ("embedding_predicted_category",   "TEXT"),
    ("embedding_predicted_confidence", "REAL"),
]

# Valid source values — enforced at write time (warn, not reject)
VALID_SOURCES = frozenset({"llm_categorizer", "default", "continuation", "embedding_lookup"})


class RoutingLogger:
    """Persist routing decisions to a SQLite routing_log table.

    Each log() call opens a short-lived connection — safe for fire-and-forget
    via asyncio.to_thread without shared-connection locking.

    Never raises: all errors are logged and swallowed so a logging failure
    never affects the user-facing response.

    Can share the same SQLite file as RequestTracker — each class owns a
    separate table.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create routing_log table if it does not already exist.

        WAL journal mode is enabled as a best-effort step *after* the table is
        created, so a read-only or WAL-unsupported filesystem only loses the
        concurrency improvement — the table itself is always present if the DB
        is writable at all.

        Schema migrations for new columns are applied after table creation so
        existing DBs are upgraded automatically without data loss.
        """
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                # Create table first — this is the critical operation.
                conn.execute(_CREATE_ROUTING_LOG_SQL)
                conn.commit()
                # WAL improves concurrent reader/writer throughput; best-effort:
                # silently skip on read-only DBs or filesystems without WAL support.
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                except Exception as wal_exc:
                    logger.warning(
                        f"[RoutingLogger] WAL mode unavailable at {self._db_path}: {wal_exc}"
                    )
                # Apply schema migrations for existing DBs that predate new columns.
                existing_cols = {
                    row[1]
                    for row in conn.execute("PRAGMA table_info(routing_log)").fetchall()
                }
                for col_name, col_type in _ROUTING_LOG_MIGRATIONS:
                    if col_name not in existing_cols:
                        try:
                            conn.execute(
                                f"ALTER TABLE routing_log ADD COLUMN {col_name} {col_type}"
                            )
                            conn.commit()
                            logger.info(
                                f"[RoutingLogger] Migrated: added column '{col_name}' ({col_type})"
                            )
                        except Exception as mig_exc:
                            logger.warning(
                                f"[RoutingLogger] Migration failed for column '{col_name}': {mig_exc}"
                            )
            finally:
                conn.close()
        except Exception as exc:
            logger.error(
                f"[RoutingLogger] Failed to initialise DB at {self._db_path}: {exc}"
            )

    def log(
        self,
        *,
        category: str,
        selected_tier: str,
        tier_mapping_version: str,
        model_used: str,
        router_version: str,
        source: str,
        category_confidence: float | None = None,
        pool_eligible: bool = False,
        prompt: str | None = None,
        latency_ms: float | None = None,
        embedding: bytes | None = None,
        embedding_predicted_category: str | None = None,
        embedding_predicted_confidence: float | None = None,
    ) -> None:
        """Write one routing decision row. Never raises.

        Args:
            category:                      task category (one of VALID_CATEGORIES)
            selected_tier:                 resolved tier (cheap / medium / strong)
            tier_mapping_version:          tag for the category→tier mapping version
            model_used:                    LiteLLM model string
            router_version:                which routing path was active
            source:                        one of VALID_SOURCES
            category_confidence:           float [0, 1] or None for non-LLM paths
            pool_eligible:                 True if quality gate passed for embedding pool
            prompt:                        raw prompt text — SHA-256 hash also derived
            latency_ms:                    time for the LLM categorizer call (ms)
            embedding:                     passage embedding as raw float32 bytes (or None)
            embedding_predicted_category:  k-NN predicted category (or None)
            embedding_predicted_confidence: k-NN prediction confidence (or None)
        """
        import hashlib

        if source not in VALID_SOURCES:
            logger.warning(f"[RoutingLogger] Unknown source '{source}', storing anyway")

        try:
            # Only hash and store string prompts — discard non-string values (e.g.
            # multimodal content lists) as NULL rather than storing a mangled repr.
            prompt_hash: str | None = None
            if isinstance(prompt, str):
                prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            else:
                prompt = None  # non-string → NULL in DB (no-op if already None)

            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO routing_log (
                        prompt_hash, prompt, category, category_confidence,
                        selected_tier, tier_mapping_version, model_used,
                        router_version, source, pool_eligible, latency_ms,
                        embedding, embedding_predicted_category,
                        embedding_predicted_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prompt_hash,
                        prompt,
                        category,
                        category_confidence,
                        selected_tier,
                        tier_mapping_version,
                        model_used,
                        router_version,
                        source,
                        1 if pool_eligible else 0,
                        latency_ms,
                        embedding,
                        embedding_predicted_category,
                        embedding_predicted_confidence,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
            logger.debug(
                f"[RoutingLogger] logged source={source} category={category} "
                f"tier={selected_tier} latency_ms={latency_ms} pool_eligible={pool_eligible}"
            )
        except Exception as exc:
            logger.error(f"[RoutingLogger] Failed to write routing_log: {exc}")


# ── EmbeddingRouter ────────────────────────────────────────────────────────────

class EmbeddingRouter:
    """Phase 2: multilingual-e5-small k-NN routing.

    Uses intfloat/multilingual-e5-small (384-dim) to embed prompts and find the
    k nearest neighbours in the routing pool to predict the task category.

    Prefix convention required by the e5 model family:
      Incoming routing queries  → embed with "query: " prefix
      Pool entries (stored)     → embed with "passage: " prefix

    The model is lazy-loaded on first use via asyncio.to_thread so the event loop
    is never blocked. If sentence-transformers is not installed or the model fails
    to load, all methods degrade silently — embed_query/embed_passage return None,
    lookup returns ("unknown", 0.0).

    Pool is loaded from SQLite and cached in memory (TTL = pool_cache_ttl seconds).
    """

    EMBEDDING_DIM = 384  # intfloat/multilingual-e5-small output dimension

    def __init__(
        self,
        db_path: str,
        model_name: str = "intfloat/multilingual-e5-small",
        k: int = 5,
        min_pool_size: int = 20,
        pool_cache_ttl: float = 300.0,
    ) -> None:
        self._db_path = db_path
        self._model_name = model_name
        self._k = k
        self._min_pool_size = min_pool_size
        self._pool_cache_ttl = pool_cache_ttl
        self._model = None
        self._available: bool | None = None   # None = not yet attempted
        self._pool_cache: dict[int, tuple[str, Any]] = {}
        self._pool_loaded_at: float = 0.0

    # ── Model lifecycle ───────────────────────────────────────────────────────

    def _ensure_model(self) -> bool:
        """Load the embedding model if not already loaded. Sync — call via to_thread.

        Returns True if the model is available, False if it could not be loaded.
        Subsequent calls short-circuit using the cached _available flag.
        """
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._model = SentenceTransformer(self._model_name)
            test = self._model.encode(["test"], normalize_embeddings=True)
            dim = test.shape[1]
            if dim != self.EMBEDDING_DIM:
                logger.error(
                    f"[EmbeddingRouter] Expected {self.EMBEDDING_DIM}-dim embedding, "
                    f"got {dim} — model '{self._model_name}' may be wrong"
                )
                self._available = False
                return False
            self._available = True
            logger.info(f"[EmbeddingRouter] Loaded '{self._model_name}' ({self.EMBEDDING_DIM}-dim)")
            return True
        except Exception as e:
            logger.error(
                f"[EmbeddingRouter] Failed to load model '{self._model_name}': {e}. "
                "Install sentence-transformers to enable embedding routing."
            )
            self._available = False
            return False

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_query(self, text: str) -> "Any | None":
        """Embed an incoming prompt with 'query: ' prefix. Sync — call via to_thread.

        Returns a float32 numpy array of shape (384,), or None if unavailable.
        """
        if not self._ensure_model():
            return None
        try:
            import numpy as np
            result = self._model.encode([f"query: {text}"], normalize_embeddings=True)
            return result[0].astype(np.float32)
        except Exception as e:
            logger.warning(f"[EmbeddingRouter] embed_query failed: {e}")
            return None

    def embed_passage(self, text: str) -> "Any | None":
        """Embed a pool entry with 'passage: ' prefix. Sync — call via to_thread.

        Returns a float32 numpy array of shape (384,), or None if unavailable.
        """
        if not self._ensure_model():
            return None
        try:
            import numpy as np
            result = self._model.encode([f"passage: {text}"], normalize_embeddings=True)
            return result[0].astype(np.float32)
        except Exception as e:
            logger.warning(f"[EmbeddingRouter] embed_passage failed: {e}")
            return None

    # ── Pool management ───────────────────────────────────────────────────────

    def _load_pool(self) -> "dict[int, tuple[str, Any]]":
        """Load pool entries from DB with TTL-based in-memory caching. Sync — call via to_thread.

        Only entries with pool_eligible=1 AND embedding IS NOT NULL are loaded.
        Embeddings with unexpected dimensions are silently skipped.
        """
        now = time.monotonic()
        if self._pool_cache and (now - self._pool_loaded_at) < self._pool_cache_ttl:
            return self._pool_cache

        pool: dict[int, tuple[str, Any]] = {}
        try:
            import numpy as np
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    "SELECT id, category, embedding FROM routing_log "
                    "WHERE pool_eligible=1 AND embedding IS NOT NULL"
                ).fetchall()
            for row_id, category, emb_bytes in rows:
                if not emb_bytes:
                    continue
                emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
                if emb.shape[0] != self.EMBEDDING_DIM:
                    continue  # skip malformed entries silently
                pool[row_id] = (category, emb)
            logger.debug(f"[EmbeddingRouter] Loaded {len(pool)} pool entries from DB")
        except Exception as e:
            logger.warning(f"[EmbeddingRouter] Failed to load pool from DB: {e}")

        self._pool_cache = pool
        self._pool_loaded_at = now
        return pool

    def invalidate_pool_cache(self) -> None:
        """Force a pool reload on the next lookup call."""
        self._pool_cache.clear()
        self._pool_loaded_at = 0.0

    # ── k-NN lookup ───────────────────────────────────────────────────────────────

    def lookup(self, query_emb: "Any") -> tuple[str, float]:
        """k-NN lookup against the embedding pool. Sync — call via to_thread.

        Returns (category, confidence) where confidence is the fraction of the k
        nearest neighbours that agree on the winning category.
        Returns ("unknown", 0.0) when pool is too small or on any error.
        """
        pool = self._load_pool()
        if len(pool) < self._min_pool_size:
            logger.debug(
                f"[EmbeddingRouter] Pool too small ({len(pool)} < {self._min_pool_size}), "
                "skipping k-NN lookup"
            )
            return "unknown", 0.0

        try:
            import numpy as np
            from collections import Counter
            ids = list(pool.keys())
            categories = [pool[i][0] for i in ids]
            matrix = np.stack([pool[i][1] for i in ids])   # (N, 384)
            sims = matrix @ query_emb                        # cosine similarity (L2-normalised)
            top_k_indices = np.argsort(sims)[-self._k:][::-1]
            top_cats = [categories[i] for i in top_k_indices]
            counter = Counter(top_cats)
            winner, count = counter.most_common(1)[0]
            confidence = float(count) / self._k
            logger.debug(
                f"[EmbeddingRouter] k-NN winner={winner} conf={confidence:.2f} "
                f"top_cats={top_cats}"
            )
            return winner, confidence
        except Exception as e:
            logger.warning(f"[EmbeddingRouter] lookup failed: {e}")
            return "unknown", 0.0


# ── ModelRouter ───────────────────────────────────────────────────────────────

class ModelRouter:
    """Adaptive model router — Phase 2.

    Wraps LLMCategorizer + RoutingLogger and adds session-level caching to skip
    re-categorization during tool-loop continuations (when the last message role
    is ``tool`` or ``assistant`` and the session cache is still fresh).

    Usage:
        router = ModelRouter(aliases=config.models.aliases,
                             routing_config=config.routing)
        tier, model = await router.route(text, messages, session_key="abc")

    Returns:
        (tier, model): tier ∈ {cheap, medium, strong},
                       model is the LiteLLM model string for that tier.

    Never raises — degrades to (medium, medium_model) on any unhandled error.
    """

    def __init__(
        self,
        aliases: dict[str, str],
        routing_config: Any,          # RoutingConfig — avoid circular import at module level
    ) -> None:
        self._aliases = aliases
        self._cfg = routing_config
        self._embedding_enabled = routing_config.embedding_enabled
        # Session cache size comes from config (max_session_cache_size).
        # OrderedDict preserves insertion order for FIFO eviction.
        self._max_session_cache: int = routing_config.max_session_cache_size
        self._conv_context: OrderedDict[str, dict] = OrderedDict()

        # Warn at construction time if any tier has no alias — easier to catch
        # misconfiguration at startup than to silently forward an empty model string.
        missing_tiers = set(CATEGORY_TIER_MAP.values()) - set(aliases)
        if missing_tiers:
            logger.warning(
                f"[ModelRouter] No alias defined for tier(s) {missing_tiers} — "
                "requests routing to these tiers will fall back to any available alias"
            )

        # Resolve API key from environment
        api_key: str | None = None
        if routing_config.categorizer_api_key_env:
            api_key = os.environ.get(routing_config.categorizer_api_key_env)
            if not api_key:
                logger.warning(
                    f"[ModelRouter] Env var '{routing_config.categorizer_api_key_env}' "
                    "is not set — categorizer calls may fail"
                )

        # Categorizer model falls back to cheap alias when not explicitly set
        cat_model = routing_config.categorizer_model or aliases.get("cheap", "")

        self._categorizer = LLMCategorizer(
            model=cat_model,
            api_base=routing_config.categorizer_api_base,
            api_key=api_key,
            timeout=routing_config.categorizer_timeout,
        )
        self._routing_logger = RoutingLogger(routing_config.db)

        # EmbeddingRouter: only created when embedding is enabled (shadow or live)
        self._embedder: EmbeddingRouter | None = (
            EmbeddingRouter(
                db_path=routing_config.db,
                model_name=routing_config.embedding_model,
                k=routing_config.embedding_k,
                min_pool_size=routing_config.embedding_min_pool_size,
                pool_cache_ttl=routing_config.embedding_pool_cache_ttl,
            ) if routing_config.embedding_enabled else None
        )

    @property
    def _ROUTER_VERSION(self) -> str:
        if self._embedding_enabled == "shadow":
            return "phase2_shadow"
        elif self._embedding_enabled:
            return "phase2_embedding"
        return "phase1_llm"

    async def route(
        self,
        text: str,
        messages: list[dict],
        prompt_for_hash: str | None = None,
        session_key: str | None = None,
    ) -> tuple[str, str]:
        """Classify text and return (tier, model_name).

        Skips the categorizer and returns the cached (tier, model) when the last
        message is a tool-loop continuation within the session TTL.

        Args:
            text:            Last user message to classify.
            messages:        Full conversation message list (roles checked for
                             tool-loop detection).
            prompt_for_hash: Raw prompt stored in routing_log (may differ from
                             text when caller passes the full conversation).
            session_key:     Opaque key identifying the conversation session.
                             No caching when None.

        Returns:
            (tier, model): e.g. ("strong", "anthropic/claude-sonnet-4-5")
        """
        # ── tool-loop continuation: skip categorizer, reuse cached tier ────────
        if session_key and self._is_continuation(messages, session_key):
            cached = self._conv_context[session_key]
            tier = cached["tier"]
            model = cached["model"]
            # Fire-and-forget: log write is analytics-only; don't block the
            # response waiting for SQLite I/O.
            _schedule_background(asyncio.to_thread(
                self._routing_logger.log,
                category=cached["category"],
                selected_tier=tier,
                tier_mapping_version=self._cfg.tier_mapping_version,
                model_used=model,
                router_version=self._ROUTER_VERSION,
                source="continuation",
                prompt=prompt_for_hash,
            ))
            return tier, model

        # ── normal path: call categorizer ─────────────────────────────────────
        context = self._build_context(session_key) if session_key else None

        category, confidence, latency_ms = await self._categorizer.categorize(
            text, context=context
        )

        tier = CATEGORY_TIER_MAP.get(category, "medium")
        model = self._aliases.get(tier, self._aliases.get("cheap", ""))
        if not model:
            # Last-resort fallback: pick any available alias rather than forward "".
            model = next(iter(self._aliases.values()), "")
            logger.warning(
                f"[ModelRouter] No alias for tier '{tier}' or 'cheap' — "
                f"falling back to '{model}'"
            )

        pool_eligible = self._is_pool_eligible(category, confidence, text)

        if session_key:
            self._update_context(session_key, category, tier, model, text)

        # ── Phase 2: embedding shadow/live ────────────────────────────────────
        emb_pred_category: str | None = None
        emb_pred_confidence: float | None = None

        if self._embedder is not None:
            # Embed query and do k-NN lookup (both CPU-bound — run in thread)
            query_emb = await asyncio.to_thread(self._embedder.embed_query, text)
            if query_emb is not None:
                pred_cat, pred_conf = await asyncio.to_thread(
                    self._embedder.lookup, query_emb
                )
                if pred_cat != "unknown":
                    emb_pred_category = pred_cat
                    emb_pred_confidence = pred_conf
                    logger.info(
                        f"[Router] embedding_knn={pred_cat} conf={pred_conf:.2f} "
                        f"(llm={category})"
                    )
                    # In live mode, override tier/model with embedding prediction.
                    # Shadow mode (embedding_enabled == "shadow") logs only — no override.
                    if self._embedding_enabled is True:
                        category = pred_cat
                        tier = CATEGORY_TIER_MAP.get(category, "medium")
                        model = self._aliases.get(tier, self._aliases.get("cheap", ""))
                        if not model:
                            model = next(iter(self._aliases.values()), "")

        # Fire-and-forget: embed passage + write log in a single background coroutine
        # so the passage embedding is stored atomically with the routing row.
        _schedule_background(self._embed_and_log(
            text=text,
            category=category,
            tier=tier,
            model=model,
            confidence=confidence,
            pool_eligible=pool_eligible,
            prompt_for_hash=prompt_for_hash,
            latency_ms=latency_ms,
            emb_pred_category=emb_pred_category,
            emb_pred_confidence=emb_pred_confidence,
        ))

        return tier, model

    async def _embed_and_log(
        self,
        *,
        text: str,
        category: str,
        tier: str,
        model: str,
        confidence: float,
        pool_eligible: bool,
        prompt_for_hash: str | None,
        latency_ms: float | None,
        emb_pred_category: str | None,
        emb_pred_confidence: float | None,
    ) -> None:
        """Compute passage embedding (if pool-eligible) and write routing_log row.

        Runs as a fire-and-forget background task so it never blocks the response.
        """
        embedding_bytes: bytes | None = None
        if pool_eligible and self._embedder is not None and isinstance(text, str) and text:
            passage_emb = await asyncio.to_thread(self._embedder.embed_passage, text)
            if passage_emb is not None:
                embedding_bytes = passage_emb.tobytes()

        await asyncio.to_thread(
            self._routing_logger.log,
            category=category,
            selected_tier=tier,
            tier_mapping_version=self._cfg.tier_mapping_version,
            model_used=model,
            router_version=self._ROUTER_VERSION,
            source="llm_categorizer",
            category_confidence=confidence,
            pool_eligible=pool_eligible,
            prompt=prompt_for_hash,
            latency_ms=latency_ms,
            embedding=embedding_bytes,
            embedding_predicted_category=emb_pred_category,
            embedding_predicted_confidence=emb_pred_confidence,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _is_continuation(self, messages: list[dict], session_key: str) -> bool:
        """True if last message is a tool/assistant turn and cache is still fresh."""
        if not messages:
            return False
        last_role = messages[-1].get("role", "")
        if last_role not in ("tool", "assistant"):
            return False
        cached = self._conv_context.get(session_key)
        if not cached:
            return False
        return (time.time() - cached["timestamp"]) <= self._cfg.session_ttl_seconds

    def _build_context(self, session_key: str) -> str | None:
        """Return prior-context hint string, or None if cache missing / expired.

        Returned string contains the first two lines of the three-line context
        format described in _CATEGORIZER_SYSTEM_PROMPT. The third line
        ([Current message:]) is added by LLMCategorizer.categorize() when it
        assembles the messages list.
        """
        cached = self._conv_context.get(session_key)
        if not cached:
            return None
        age = time.time() - cached["timestamp"]
        if age > self._cfg.session_ttl_seconds:
            del self._conv_context[session_key]
            return None
        return (
            f"[Previous routing: {cached['category']}, {int(age)}s ago]\n"
            f"[Previous message:] {cached['last_user_msg'][:200]}"
        )

    def _update_context(
        self,
        session_key: str,
        category: str,
        tier: str,
        model: str,
        text: str,
    ) -> None:
        # Move existing key to end so re-used sessions don't get prematurely evicted.
        self._conv_context.pop(session_key, None)
        self._conv_context[session_key] = {
            "category": category,
            "tier":     tier,
            "model":    model,
            "last_user_msg": text,
            "timestamp": time.time(),
        }
        # Evict oldest entry if cap exceeded. One call adds at most one entry
        # (existing key was popped first), so a single check suffices.
        if len(self._conv_context) > self._max_session_cache:
            self._conv_context.popitem(last=False)

    def _is_pool_eligible(
        self, category: str, confidence: float, text: str
    ) -> bool:
        """Quality gate: True when this prompt is worth adding to the embedding pool."""
        return (
            category != "unknown"
            and confidence >= self._cfg.confidence_threshold
            and isinstance(text, str)
            and len(text) >= self._cfg.min_prompt_length_for_pool
        )
