# thrift-flow — Phases Plan

> OpenAI-compatible LLM proxy for the family agent infra (crab-bot, wavehammer, and future agents).
> Goal: transparent token optimization, cost visibility, and adaptive model routing — all in one place.
>
> Last updated: 2026-05-26

---

## Background

crab-bot's current architecture has routing logic (`model_router.py`) tightly coupled to the bot.
thrift-flow extracts this into a standalone proxy, so any agent can benefit from routing and
optimization without duplicating logic.

**End-state request flow (after all phases):**

```
Agent (crab-bot / wavehammer / ...)
  │
  │  POST /v1/chat/completions
  ▼
thrift-flow proxy (localhost:8888)
  ├─ Token optimizer (history truncation, compression)
  ├─ Adaptive router  (categorize → select cheap/medium/strong)
  └─ Cost tracker     (log tokens + cost to SQLite)
  │
  │  forward to real model
  ▼
LLM provider (OpenRouter, Groq, Anthropic, ...)
```

---

## Phase 1 — Proxy Foundation ✅ DONE

**Goal:** Get the proxy running as a transparent passthrough with token visibility.

### What was built

- `POST /v1/chat/completions` — OpenAI-compatible endpoint (streaming + non-streaming)
- `GET /v1/models` — lists configured model aliases
- `GET /v1/usage` / `/v1/usage/by-model` / `/v1/usage/recent` — cost dashboard endpoints
- Model alias mapping: `cheap / medium / strong / auto` → real LiteLLM model names via `config.yaml`
- SQLite tracker: logs every request with input tokens, output tokens, estimated cost, latency
- LiteLLM forwarding via `asyncio.Queue` streaming pattern (non-blocking event loop)
- 8 unit tests (config + tracker)

### Routing in Phase 1

Routing still lives in crab-bot (`agents/model_router.py`). The proxy is a passthrough — crab-bot
selects the model and passes it as the `model` field. thrift-flow logs and forwards.

### Client integration (when ready)

```bash
# crab-bot .env
AI_MODEL=openai/cheap
OPENAI_API_BASE=http://localhost:8888/v1
```

The proxy holds all provider API keys. Clients don't need them.

---

## Phase 2 — Adaptive Model Routing Migration

**Goal:** Move `model_router.py` from crab-bot into thrift-flow. Routing becomes a proxy concern,
invisible to all clients.

### What to build

1. **`proxy/router.py`** — port `ModelRouter` + `LLMCategorizer` + `EmbeddingRouter` from crab-bot
   - LLM categorizer (Groq `llama-3.1-8b-instant`, ~220ms)
   - 8 task categories: `casual / simple_lookup / research_lookup / creative / analysis / coding / reasoning / unknown`
   - Category → tier → model mapping
   - Context-aware routing via `X-Session-Key` header (replaces `conv_key` in crab-bot)

2. **Tool-loop continuation detection** — avoid re-categorizing mid-tool-loop calls
   - If `messages[-1].role` is `"tool"` or `"assistant"`: continuation → reuse cached routing decision
   - Cache keyed by `X-Session-Key` header, TTL 30 minutes

3. **`model: "auto"` support** — when client sends `model: "auto"`, proxy categorizes and picks tier
   - All other model names still work as before (passthrough or alias lookup)

4. **Feature flags in `config.yaml`**
   ```yaml
   routing:
     enabled: true
     llm_categorizer_enabled: true
     embedding_lookup_enabled: "shadow"   # false | true | "shadow"
     categorizer:
       model: "groq/llama-3.1-8b-instant"
       api_key_env: "GROQ_API_KEY"
       timeout_seconds: 3.0
     confidence_threshold: 0.7
     min_prompt_length: 10
     session_ttl_seconds: 1800
   ```

5. **Routing log** — extend `tracking.db` with routing columns
   - `category`, `category_confidence`, `router_version`, `source`, `pool_eligible`
   - Merge with existing `request_log` or new `routing_log` table (TBD)

### crab-bot changes (Phase 2)

- Set `AI_MODEL=openai/auto` in `.env`
- Set `OPENAI_API_BASE=http://localhost:8888/v1` in `.env`
- Remove routing call from `agents/agent.py` (no more `router.route()`)
- `agents/model_router.py` → deprecated / deleted
- Keep `GROQ_API_KEY` in crab-bot until categorizer is fully in proxy

### Acceptance criteria

- crab-bot sends `model: "auto"` to proxy
- Proxy categorizes and routes to the right model tier
- Context-aware routing works across multi-turn conversations (via `X-Session-Key`)
- All existing crab-bot behavior preserved
- `routing_log` in `tracking.db` queryable for accuracy analysis

---

## Phase 3 — Token Optimization

**Goal:** Reduce token spend by compressing or truncating requests before forwarding.

### Candidates (in priority order)

| Feature | Method | Expected savings |
|---|---|---|
| **History truncation** | Drop oldest non-system messages when total > threshold | High, easy |
| **Prompt caching hints** | Add `cache_control` headers for Anthropic models | Medium (cuts repeat system prompt cost) |
| **Conversation summarization** | Summarize old turns into a single summary message | High, complex |
| **Prompt compression** | LLMLingua-style semantic compression of long messages | Medium, adds latency |

### Config additions

```yaml
optimization:
  history_truncation:
    enabled: true
    max_input_tokens: 8000      # truncate if messages exceed this
    keep_system_prompt: true    # always keep system messages
    keep_last_n_turns: 4        # always keep the most recent N user/assistant pairs
  prompt_caching:
    enabled: false              # Anthropic prompt caching (cache_control header)
```

### Acceptance criteria

- Requests over `max_input_tokens` are automatically truncated before forwarding
- System prompt is always preserved
- Token savings are visible in `/v1/usage` dashboard

---

## Phase 4 — Multi-Client & Auth (Future)

**Goal:** Harden the proxy for use beyond localhost.

- Per-client API keys (`X-API-Key` header) for access control
- Per-client usage quotas and rate limits
- `GET /v1/usage?client_id=crab-bot` — per-client cost breakdown
- HTTPS support (reverse proxy via nginx or built-in TLS)
- Deploy to `bot.bugfamily.com` alongside crab-bot

---

## Phase Summary

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Proxy foundation + token logging | ✅ Done |
| Phase 2 | Adaptive routing migration from crab-bot | 🔲 Next |
| Phase 3 | Token optimization (truncation, caching) | 🔲 Planned |
| Phase 4 | Multi-client auth + remote deploy | 🔲 Future |
