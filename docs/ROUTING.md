# Adaptive Model Routing

This document explains how thrift-flow's `model: "auto"` routing works — from the moment a request arrives to the moment a model is selected.

---

## Overview

Routing is a two-layer system. The fast layer (k-NN embedding lookup) runs first; the slower but more general layer (LLM categorizer) only runs when the fast layer isn't confident enough.

```
Incoming prompt
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Layer 1: k-NN Embedding Router  (<20ms, local CPU) │
│                                                      │
│  embed prompt → find k nearest neighbours in pool   │
│  majority-vote category → confidence = k_agree / k  │
│                                                      │
│  confidence >= threshold (default 0.85)?             │
│       YES → use k-NN result, SKIP Layer 2            │
│       NO  → fall through to Layer 2                  │
└─────────────────────────────────────────────────────┘
      │ (low confidence or shadow mode)
      ▼
┌─────────────────────────────────────────────────────┐
│  Layer 2: LLM Categorizer  (~220ms, Groq API)       │
│                                                      │
│  send prompt to Groq llama-3.1-8b-instant           │
│  parse JSON response: {category, confidence}        │
│  result is authoritative                             │
└─────────────────────────────────────────────────────┘
      │
      ▼
category → tier → model alias → forward request
```

Both layers log their decision to `routing_log` in `tracking.db`.

---

## Category Taxonomy

Every prompt is classified into one of 7 categories. The mapping from category to model tier is fixed in `CATEGORY_TIER_MAP` and versioned via `tier_mapping_version` in the config.

| Category | Description | Tier |
|---|---|---|
| `casual` | Greetings, small talk, thanks, emotional support | **cheap** |
| `simple_lookup` | Direct factual questions with short, self-contained answers (definitions, translations, "what is X") | **cheap** |
| `creative` | Generating original content — stories, poems, marketing copy, scripts | **medium** |
| `analysis` | Explaining, summarising, comparing, or analysing any topic — including questions that require synthesising external or expert-domain information (regulations, medical, financial, legal, current events) | **medium** |
| `coding` | Writing code, debugging, code review, explaining code, system architecture | **strong** |
| `reasoning` | Multi-step logic, planning, decision-making with tradeoffs, math problems | **strong** |
| `unknown` | Cannot be confidently assigned to any of the above | **medium** (conservative default) |

> **Note on `analysis`:** This category intentionally merges what might be called "research_lookup" — empirically, embedding models cannot distinguish "explain this concept" from "research this regulation" in vector space, and both map to the same tier, so keeping them separate adds noise without benefit.

Tier → model alias is configured in `config.yaml` under `models.aliases`:

```yaml
models:
  aliases:
    cheap:  "groq/llama-3.1-8b-instant"
    medium: "openrouter/qwen/qwen3-235b-a22b"
    strong: "anthropic/claude-sonnet-4-5"
```

---

## LLM Categorizer

The LLM categorizer is the authoritative fallback when the embedding router isn't confident enough. It uses a small, fast LLM (Groq's `llama-3.1-8b-instant` by default, ~220ms) to classify prompts.

### How it works

1. The last user message is extracted from the request (truncated to 2000 chars).
2. A system prompt describing the 7 categories is prepended.
3. If a session context exists (from a prior turn), a three-line context block is injected as a separate user message:
   ```
   [Previous routing: coding, 12s ago]
   [Previous message:] no, need dedup
   [Current message:] narrow down to full-time employees
   ```
4. The model returns a JSON response: `{"category": "coding", "confidence": 0.97}`
5. The category is validated against `VALID_CATEGORIES`; unrecognised values fall back to `unknown`.
6. Confidence is clamped to `[0.0, 1.0]`.

### Reliability

The categorizer is designed to never raise. Any failure — timeout, bad JSON, provider error — returns `("unknown", 0.0, None)` and routes to medium tier. The `routing_log` records every decision including failures, making it easy to audit degraded periods.

### Quality gate for the embedding pool

Only high-quality LLM categorizer results are added to the embedding pool:

- `category != "unknown"`
- `confidence >= confidence_threshold` (default 0.7)
- `len(prompt) >= min_prompt_length_for_pool` (default 10 chars)

---

## Embedding Router

The embedding router provides fast, local classification by finding similar prompts in a pool of past routing decisions. It uses `intfloat/multilingual-e5-small` — a 384-dimension multilingual model — running on local CPU.

### Prefix convention

The e5 model family requires asymmetric prefixes:

| Use | Prefix |
|---|---|
| Query (new incoming prompt) | `"query: "` + text |
| Pool entry (stored at routing time) | `"passage: "` + text |

Using the wrong prefix produces incorrect cosine similarities. The prefixes are applied automatically inside `embed_query()` and `embed_passage()`.

### Pool management

Every request that passes the quality gate gets its passage embedding stored in `routing_log` as a `BLOB` (float32, 384 dimensions). The embedding router loads this pool into memory on first use (cached with a 300-second TTL by default).

Pool size matters: with fewer than `embedding_min_pool_size` entries (default 20), k-NN is skipped entirely and the LLM categorizer runs for every request.

### k-NN lookup

For each incoming prompt:

1. `embed_query(text)` → 384-dim float32 vector
2. Compute cosine similarity against all pool entries: `sims = matrix @ query_emb`
3. Take the top-k most similar entries (default k=5)
4. Majority-vote the category: the winning category and its vote fraction are returned
5. `confidence = votes_for_winner / k` — so 5/5 = 1.0, 3/5 = 0.6, etc.

### Shadow mode vs live mode

| `embedding_enabled` | Behaviour |
|---|---|
| `false` | Embedding router disabled. LLM categorizer runs for every request. |
| `"shadow"` | k-NN runs alongside the LLM categorizer. k-NN result is logged but **never affects routing**. Use this to build and validate the pool before going live. |
| `true` | Confidence-gated live mode (see below). |

### Confidence-gated live mode

When `embedding_enabled: true`, the router uses a threshold to decide whether k-NN is confident enough to skip Groq:

```
k-NN confidence >= embedding_live_confidence_threshold (default 0.85)
    → use k-NN result, skip LLM categorizer entirely
    → source logged as "embedding_lookup"

k-NN confidence < threshold (or pool too small, or shadow mode)
    → fall through to LLM categorizer
    → source logged as "llm_categorizer"
    → Groq result + embedding stored in pool (learning)
```

This creates a self-improving system: prompts the k-NN isn't sure about are handled by Groq, which labels and embeds them, growing the pool. Over time, k-NN becomes confident on more and more prompt types and the Groq call rate naturally declines.

### Pool portability

Embeddings are portable across deployments as long as:

1. The same model (`intfloat/multilingual-e5-small`) is used
2. The same prefix convention (`passage: ` for stored entries) is preserved

To verify compatibility between two instances, embed the same string in both and check that the cosine similarity is ~1.0.

---

## Context-Aware Routing

Multi-turn conversations can mislead the categorizer. A follow-up message like "no, need dedup" makes no sense in isolation — it needs the prior routing context to be correctly classified as `coding` rather than `simple_lookup`.

### How it works

Pass an `X-Session-Key` header (an opaque identifier for the conversation) with each request. The router maintains an in-process cache (`_conv_context`) keyed by session key:

```
Request 1: "write a script to aggregate employee data"
  → categorized as coding
  → stored in _conv_context[session_key] = {category: "coding", ...}

Request 2 (same session): "no, need dedup"
  → context injected: "[Previous routing: coding, 12s ago]\n[Previous message:] ..."
  → categorized as coding (follow-up correctly classified)
```

### Tool-loop continuation

When the last message in `messages` has `role: "tool"` or `role: "assistant"` (i.e., the model is mid-tool-loop), the router skips categorization entirely and reuses the cached tier from the previous turn. This avoids unnecessary Groq calls during agentic tool chains.

### Session cache limits

- TTL: `session_ttl_seconds` (default 1800 — 30 minutes)
- Max entries: `max_session_cache_size` (default 1000, FIFO eviction)

---

## The Self-Improving Flywheel

The system improves passively as traffic flows through it:

```
New prompt type arrives
      │
      ▼ k-NN confidence < 0.85
LLM categorizer (Groq) classifies it
      │
      ▼ if pool_eligible
Passage embedding stored in routing_log pool
      │
      ▼ next time a similar prompt arrives
k-NN finds this entry as a neighbour
      │
      ▼ after enough similar prompts accumulate
k-NN confidence crosses 0.85 threshold
      │
      ▼
Groq no longer called for this prompt type ✓
```

The pool grows organically from real traffic. No manual labelling is required. The only intervention needed is setting `embedding_enabled: true` and choosing a confidence threshold that balances quality against coverage.

---

## Model Drift and Behavioral Consistency

Adaptive routing is a cost optimization — but it comes with a behavioral trade-off that is easy to overlook.

### The problem

Different models have different "dialects". A model trained by one lab develops implicit shorthand in its outputs: particular phrasing, formatting habits, reasoning patterns. When an agent accumulates memory or long-running context written in that dialect, switching to a model from a different lab can cause **behavioral drift** — the new model misreads the context, ignores implicit cues, or responds in a noticeably different style.

> *"The memory contained a cue like 'launch sequence #8'. The original model read that and knew exactly what to do. Others said '…launch what now?'"*
> — [A2H Labs: Memory Speaks Only One Dialect](https://guide.a2hlabs.com/ch2-memory/#finding-04-memory-speaks-only-one-dialect)

This becomes a real problem when `model: "auto"` routes some turns to cheap (e.g. GPT-4o-mini) and others to strong (e.g. Claude Sonnet). If the agent's system prompt or memory contains patterns optimised for one model's dialect, the other tier may behave inconsistently.

### Recommendations

**1. Use models from the same vendor across tiers.**

Assign `cheap`, `medium`, and `strong` to models from the same family when behavioral consistency matters. Models within one vendor's lineup share training lineage and tend to interpret context more uniformly:

```yaml
# Consistent: all from the same vendor
models:
  aliases:
    cheap:  "openai/gpt-4o-mini"
    medium: "openai/gpt-4o"
    strong: "openai/o1"

# Riskier for agents with accumulated memory or complex personas:
models:
  aliases:
    cheap:  "groq/llama-3.1-8b-instant"
    medium: "openai/gpt-4o"
    strong: "anthropic/claude-sonnet-4-5"
```

**2. Write agent memory and system prompts in normalized, model-agnostic language.**

Avoid implicit shorthand that only one model would recognise. Prefer explicit, self-contained instructions:

```
# Fragile (model-specific shorthand):
"Use mode 7 for analysis tasks."

# Robust (self-contained):
"When asked to summarise or compare content, use a structured bullet-point
format with a brief conclusion at the end."
```

If you need to migrate an agent from one model to another, have the current model rewrite its accumulated memory in explicit plain language before switching — treat it like an employee handover document, not personal notes.

**3. Use routing for stateless tasks first.**

Routing works best for tasks that are largely self-contained per turn: casual questions, one-shot lookups, standalone coding problems. For agents with persistent personas, long-running sessions, or memory-dependent workflows, validate behavioral consistency across tiers before enabling `model: "auto"`.

---

## Configuration Reference

All routing settings live under `routing:` in `config.yaml`.

| Field | Default | Description |
|---|---|---|
| `enabled` | `false` | Master switch. Set `true` to activate `model: "auto"` routing. |
| `categorizer_model` | `null` | LiteLLM model string for the LLM categorizer (e.g. `"groq/llama-3.1-8b-instant"`). Falls back to the `cheap` alias if not set. |
| `categorizer_api_base` | `null` | Custom endpoint override for the categorizer (e.g. to point at a self-hosted model). |
| `categorizer_api_key_env` | `null` | Name of the env var holding the categorizer's API key (e.g. `"GROQ_API_KEY"`). |
| `db` | `"tracking.db"` | SQLite file for `routing_log`. Shared with the request tracker. |
| `tier_mapping_version` | `"v2"` | Tag stored in every `routing_log` row. Bump when changing `CATEGORY_TIER_MAP`. |
| `confidence_threshold` | `0.7` | Minimum LLM categorizer confidence for a prompt to be pool-eligible. |
| `min_prompt_length_for_pool` | `10` | Minimum prompt length (chars) for pool eligibility. |
| `session_ttl_seconds` | `1800` | Context cache TTL in seconds. |
| `categorizer_timeout` | `5.0` | Timeout for the LLM categorizer call (seconds). |
| `max_session_cache_size` | `1000` | Maximum in-process session cache entries (FIFO eviction). |
| `embedding_enabled` | `false` | `false` / `"shadow"` / `true` — controls the embedding router mode. |
| `embedding_model` | `"intfloat/multilingual-e5-small"` | Sentence-transformers model name. |
| `embedding_k` | `5` | Number of nearest neighbours for k-NN voting. |
| `embedding_min_pool_size` | `20` | Minimum pool entries before k-NN is attempted. |
| `embedding_pool_cache_ttl` | `300.0` | Pool in-memory cache TTL in seconds. |
| `embedding_live_confidence_threshold` | `0.85` | Minimum k-NN confidence to skip the LLM categorizer in live mode. |
