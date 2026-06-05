# thrift-flow — Roadmap

> OpenAI-compatible LLM proxy with adaptive model routing, cost tracking, and embedding-based caching.
>
> Last updated: 2026-06-05

---

## End-state request flow

```
Client application
  │
  │  POST /v1/chat/completions
  ▼
thrift-flow proxy (localhost:8888)
  ├─ Token optimizer  (history truncation, prompt caching)
  ├─ Adaptive router  (categorize → select cheap / medium / strong)
  └─ Cost tracker     (log tokens + cost to SQLite)
  │
  │  forward to resolved model
  ▼
LLM provider (OpenRouter / Groq / Anthropic / ...)
```

---

## Phase 1 — Proxy Foundation ✅ Done

**Goal:** A transparent passthrough proxy with full token visibility and cost logging.

### What was built

- `POST /v1/chat/completions` — OpenAI-compatible endpoint (streaming + non-streaming)
- `GET /v1/models` — lists configured model aliases
- `GET /v1/usage` / `/v1/usage/by-model` / `/v1/usage/recent` — usage dashboard endpoints
- Model alias mapping: `cheap / medium / strong / auto` → real LiteLLM model names via `config.yaml`
- SQLite request tracker: logs every request with input tokens, output tokens, estimated cost, and latency
- LiteLLM forwarding via async streaming (non-blocking event loop)
- Unit tests: config + tracker

---

## Phase 2 — Adaptive Model Routing ✅ Done

**Goal:** Automatically select the right model tier for each request. Clients send `model: "auto"` and the proxy decides.

### What was built

**LLM categorizer** (`proxy/router.py`)
- Classifies prompts using Groq `llama-3.1-8b-instant` (~220ms latency)
- 7 categories: `casual`, `simple_lookup`, `creative`, `analysis`, `coding`, `reasoning`, `unknown`
- Category → tier mapping (v2):
  - cheap: casual, simple_lookup
  - medium: creative, analysis, unknown
  - strong: coding, reasoning

**Embedding router** (`EmbeddingRouter`)
- `intfloat/multilingual-e5-small` (384-dim), k-NN lookup against historical routing pool
- Shadow mode: runs alongside LLM categorizer for accuracy comparison without affecting routing
- Live mode: k-NN result used directly when pool is large enough
- Pool eligibility quality gate: category != unknown AND confidence >= threshold AND prompt length >= minimum

**Context-aware routing**
- `X-Session-Key` header enables per-session routing context
- Tool-loop continuations reuse the cached routing decision without re-categorizing
- Session cache with configurable TTL (default: 30 minutes) and FIFO eviction cap

**Routing log**
- All decisions persisted to `routing_log` table in SQLite
- Stores: category, confidence, tier, model, source, pool_eligible, latency_ms, embedding

---

## Phase 3 — Confidence-Gated Live Mode ✅ Done

**Goal:** Reduce LLM categorizer calls by using the embedding k-NN result directly when its confidence is high enough.

### What was built

- `embedding_live_confidence_threshold` config parameter (default: 0.85)
- When `embedding_enabled: true` and k-NN confidence >= threshold, the Groq categorizer call is skipped entirely
- Shadow mode (`embedding_enabled: "shadow"`) continues to run both paths and log both results for accuracy analysis
- Routing source field distinguishes `embedding_lookup` vs `llm_categorizer` in the log

---

## Phase 4 — Token Optimization 🔲 Planned

**Goal:** Reduce token spend by compressing or truncating requests before forwarding.

### What to build

| Feature | Method | Expected savings |
|---|---|---|
| **History truncation** | Drop oldest non-system messages when total exceeds threshold | High, low complexity |
| **Prompt caching hints** | Add `cache_control` headers for Anthropic models | Medium — cuts repeated system prompt cost |
| **Conversation summarization** | Summarize old turns into a single summary message | High, higher complexity |

### Planned config additions

```yaml
optimization:
  history_truncation:
    enabled: true
    max_input_tokens: 8000
    keep_system_prompt: true
    keep_last_n_turns: 4
  prompt_caching:
    enabled: false    # Anthropic cache_control header injection
```

### Acceptance criteria

- Requests exceeding `max_input_tokens` are automatically truncated before forwarding
- System prompt is always preserved
- Token savings are visible in the `/v1/usage` dashboard

---

## Phase 5 — Multi-Client Auth + Remote Deploy 🔲 Future

**Goal:** Harden the proxy for multi-tenant and remote deployments.

### What to build

- Per-client API keys (`X-API-Key` header) for access control
- Per-client usage quotas and rate limits
- `GET /v1/usage?client_id=...` — per-client cost breakdown
- HTTPS support (reverse proxy via nginx or built-in TLS)
- Remote deployment guide

---

## Phase Summary

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Proxy foundation + cost tracking | ✅ Done |
| Phase 2 | Adaptive routing — LLM categorizer + embedding router | ✅ Done |
| Phase 3 | Confidence-gated live mode | ✅ Done |
| Phase 4 | Token optimization (truncation, prompt caching) | 🔲 Planned |
| Phase 5 | Multi-client auth + remote deploy | 🔲 Future |
