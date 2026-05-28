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

import json
import logging
import re
import time
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
        if not text or not text.strip():
            # Empty / whitespace → treat as casual; no LLM call needed.
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

        # Extract first JSON object — handles extra prose around the JSON
        match = re.search(r"\{[^}]+\}", clean)
        if not match:
            logger.warning(f"[Categorizer] No JSON object found in response: {raw[:100]}")
            return "unknown", 0.0

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError as exc:
            logger.warning(f"[Categorizer] JSON parse error: {exc} | raw={raw[:100]}")
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
VALID_SOURCES = frozenset({"llm_categorizer", "default"})


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
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(_CREATE_ROUTING_LOG_SQL)
                conn.commit()
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

        prompt_hash: str | None = None
        if prompt is not None:
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()

        try:
            with sqlite3.connect(self._db_path) as conn:
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
            logger.debug(
                f"[RoutingLogger] logged source={source} category={category} "
                f"tier={selected_tier} latency_ms={latency_ms} pool_eligible={pool_eligible}"
            )
        except Exception as exc:
            logger.error(f"[RoutingLogger] Failed to write routing_log: {exc}")
