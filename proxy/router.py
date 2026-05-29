"""
Adaptive Model Router — Phase 1: LLM-based task categorizer.

Classifies incoming prompts into task categories, which are mapped to model
tiers (cheap / medium / strong).  All routing decisions are persisted to a
SQLite routing_log table (separate from request_log, same DB file).

Classes:
  LLMCategorizer  — async classify a prompt via a cheap LLM (e.g. Groq)
  RoutingLogger   — persist routing decisions to routing_log table

Phase 2 (EmbeddingRouter + ModelRouter) is implemented in a later PR.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any

import litellm

logger = logging.getLogger(__name__)

# ── Category taxonomy ─────────────────────────────────────────────────────────

VALID_CATEGORIES = frozenset({
    "casual",
    "simple_lookup",
    "research_lookup",
    "creative",
    "analysis",
    "coding",
    "reasoning",
    "unknown",
})

# category → tier mapping  (tier_mapping_version: v1)
CATEGORY_TIER_MAP: dict[str, str] = {
    "casual":           "cheap",
    "simple_lookup":    "cheap",
    "research_lookup":  "medium",
    "creative":         "medium",
    "analysis":         "medium",
    "coding":           "strong",
    "reasoning":        "strong",
    "unknown":          "medium",   # conservative default
}

# ── Categorizer system prompt ─────────────────────────────────────────────────

_CATEGORIZER_SYSTEM_PROMPT = """\
You are a task categorizer. Classify the user message into exactly one category.

Categories:
- casual: Greetings, small talk, thanks, emotional support ("good morning", \
"how are you", "thank you", "haha")
- simple_lookup: Direct factual questions with short, self-contained answers \
(definitions, translations, "what is X", "who is Y", "when did Z happen")
- research_lookup: Questions requiring synthesis of current, complex, or \
expert-domain information (regulations, medical, financial, legal, \
"latest policy on X", "compare regulatory options for Y")
- creative: Generating original content (stories, poems, marketing copy, \
slogans, character names, scripts)
- analysis: Explaining concepts, summarizing text, comparing items, analyzing \
content already provided by the user
- coding: Writing code, debugging, code review, explaining code, software or \
system architecture design
- reasoning: Multi-step logic, planning, decision-making with tradeoffs, math \
word problems, "should I do X or Y given constraints Z"
- unknown: Cannot be confidently assigned to any of the above categories

Examples:
- "morning!" → {"category": "casual", "confidence": 0.98}
- "what does 'ephemeral' mean?" → {"category": "simple_lookup", "confidence": 0.95}
- "what are the current GDPR requirements for cookie consent?" → \
{"category": "research_lookup", "confidence": 0.90}
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
corrections, or additions within a coding/analysis/reasoning/creative/research_lookup
conversation should remain in the same category unless the current message clearly
shifts topic.

Respond with valid JSON only, no other text:
{"category": "<one of the 8 categories>", "confidence": <float 0.0–1.0>}
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
            context: Optional context string prepended as a separate user message.
                     Truncated to 2000 chars. Used for conversation continuity.

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
            messages.append({"role": "user", "content": context[:2000]})
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
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            DATETIME DEFAULT CURRENT_TIMESTAMP,
    prompt_hash          TEXT,
    prompt               TEXT,
    category             TEXT    NOT NULL,
    category_confidence  REAL,
    selected_tier        TEXT    NOT NULL,
    tier_mapping_version TEXT    NOT NULL,
    model_used           TEXT    NOT NULL,
    router_version       TEXT    NOT NULL,
    source               TEXT    NOT NULL,
    pool_eligible        INTEGER NOT NULL DEFAULT 0,
    latency_ms           REAL
);
"""

# Valid source values — enforced at write time (warn, not reject)
VALID_SOURCES = frozenset({"llm_categorizer", "default", "continuation"})


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
        """Create routing_log table if it does not already exist."""
        import sqlite3
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(_CREATE_ROUTING_LOG_SQL)
                conn.commit()
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
    ) -> None:
        """Write one routing decision row. Never raises.

        Args:
            category:             task category (one of VALID_CATEGORIES)
            selected_tier:        resolved tier (cheap / medium / strong)
            tier_mapping_version: tag for the category→tier mapping version
            model_used:           LiteLLM model string
            router_version:       which routing path was active
            source:               one of VALID_SOURCES
            category_confidence:  float [0, 1] or None for non-LLM paths
            pool_eligible:        True if quality gate passed for embedding pool
            prompt:               raw prompt text — SHA-256 hash also derived
            latency_ms:           time for the LLM categorizer call (ms)
        """
        import hashlib
        import sqlite3

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
                        router_version, source, pool_eligible, latency_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    _ROUTER_VERSION = "phase2_llm"

    def __init__(
        self,
        aliases: dict[str, str],
        routing_config: Any,          # RoutingConfig — avoid circular import at module level
    ) -> None:
        self._aliases = aliases
        self._cfg = routing_config
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
            await asyncio.to_thread(
                self._routing_logger.log,
                category=cached["category"],
                selected_tier=tier,
                tier_mapping_version=self._cfg.tier_mapping_version,
                model_used=model,
                router_version=self._ROUTER_VERSION,
                source="continuation",
                prompt=prompt_for_hash,
            )
            return tier, model

        # ── normal path: call categorizer ─────────────────────────────────────
        context = self._build_context(session_key, text) if session_key else None

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
        )

        return tier, model

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

    def _build_context(self, session_key: str, current_text: str) -> str | None:
        """Return prior-context hint string, or None if cache missing / expired."""
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
