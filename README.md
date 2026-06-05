# thrift-flow

An OpenAI-compatible LLM proxy with adaptive model routing, cost tracking, and embedding-based caching.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What it does

thrift-flow sits between your application and any LLM provider. Send `model: "auto"` and the proxy automatically picks the right model tier — cheap, medium, or strong — by classifying the prompt using an LLM categorizer and a k-NN embedding router. Every request is logged to SQLite with token counts and cost estimates. A built-in dashboard lets you inspect usage at any time.

---

## Architecture

```
Your app
  │
  │  POST /v1/chat/completions  (OpenAI-compatible)
  ▼
thrift-flow  (localhost:8888)
  ├─ Model router
  │    ├─ k-NN embedding lookup  (intfloat/multilingual-e5-small)
  │    │    └─ confidence >= 0.85 → skip LLM categorizer
  │    └─ LLM categorizer        (Groq llama-3.1-8b-instant, ~220ms)
  │         prompt → category → tier (cheap / medium / strong)
  ├─ Cost tracker  (SQLite)
  └─ Forwarder     (LiteLLM)
  │
  │  forward to resolved model
  ▼
LLM provider  (OpenRouter / Groq / Anthropic / OpenAI / ...)
```

---

## Quick start

**1. Clone and create the virtualenv**

```bash
git clone https://github.com/your-org/thrift-flow.git
cd thrift-flow
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure**

```bash
cp .env.example .env
# Edit .env — add at least one provider API key
```

Edit `config.yaml` to set your model aliases (see [Configuration](#configuration)).

**3. Run**

```bash
./start.sh
# or: python main.py
```

The proxy starts on `http://localhost:8888`.

**4. Point your app at it**

```bash
# In your client .env
OPENAI_API_BASE=http://localhost:8888/v1
OPENAI_API_KEY=not-used          # any non-empty string
```

Send `model: "auto"` to enable adaptive routing, or use `cheap` / `medium` / `strong` directly.

---

## Configuration

`config.yaml` has four sections:

### `server`
```yaml
server:
  host: "0.0.0.0"
  port: 8888
```

### `models`
Map tier names to real LiteLLM model strings. The model name prefix determines the provider.
```yaml
models:
  aliases:
    cheap:  "groq/llama-3.1-8b-instant"
    medium: "openrouter/qwen/qwen3-235b-a22b"
    strong: "anthropic/claude-sonnet-4-5"
    auto:   "openrouter/minimax/minimax-m2.5"   # fallback when routing is disabled
  default: "cheap"
```

Provider prefix → required API key in `.env`:

| Prefix | Env var |
|---|---|
| `openrouter/` | `OPENROUTER_API_KEY` |
| `groq/` | `GROQ_API_KEY` |
| `anthropic/` | `ANTHROPIC_API_KEY` |
| `openai/` | `OPENAI_API_KEY` |
| `gemini/` | `GEMINI_API_KEY` |
| `together_ai/` | `TOGETHERAI_API_KEY` |

### `tracking`
```yaml
tracking:
  db: "tracking.db"
  enabled: true
```

### `routing`
```yaml
routing:
  enabled: false                    # set true to activate model="auto" routing
  categorizer_model: "groq/llama-3.1-8b-instant"
  categorizer_api_key_env: "GROQ_API_KEY"
  confidence_threshold: 0.7         # min confidence for embedding pool eligibility
  session_ttl_seconds: 1800         # context cache TTL (30 min)
  embedding_enabled: false          # false | "shadow" | true
  embedding_model: "intfloat/multilingual-e5-small"
  embedding_live_confidence_threshold: 0.85   # k-NN conf required to skip LLM categorizer
```

Set `embedding_enabled: "shadow"` to run the k-NN router in parallel with the LLM categorizer (logs results but does not affect routing). Set to `true` to use k-NN results when confidence is high enough.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat endpoint (streaming supported) |
| `GET` | `/v1/models` | List configured model aliases |
| `GET` | `/v1/usage` | Aggregate token and cost summary |
| `GET` | `/v1/usage/by-model` | Usage broken down by model |
| `GET` | `/v1/usage/recent` | Recent requests (default: last 50) |
| `GET` | `/health` | Health check |

### Headers

| Header | Description |
|---|---|
| `X-Session-Key` | Opaque session identifier for context-aware routing (multi-turn continuity) |
| `X-Client-ID` | Optional client identifier logged with each request |

---

## How model="auto" routing works

When a request arrives with `model: "auto"` and `routing.enabled: true`:

1. **k-NN lookup** — the last user message is embedded with `intfloat/multilingual-e5-small` and compared against the pool of past routing decisions. If `embedding_enabled: true` and confidence >= `embedding_live_confidence_threshold` (default 0.85), the k-NN result is used directly and the LLM categorizer is skipped.

2. **LLM categorizer** — if k-NN is disabled, in shadow mode, or confidence is below threshold, Groq `llama-3.1-8b-instant` classifies the prompt into one of 7 categories (~220ms).

3. **Tier mapping** — category maps to a tier, tier maps to a model:

   | Category | Tier |
   |---|---|
   | `casual`, `simple_lookup` | cheap |
   | `creative`, `analysis` | medium |
   | `coding`, `reasoning` | strong |
   | `unknown` | medium (conservative default) |

4. **Context-aware routing** — if `X-Session-Key` is set, tool-loop continuations reuse the cached routing decision without re-categorizing.

Every routing decision is stored in `routing_log` alongside the request log.

---

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run with coverage
pytest --cov=proxy --cov-report=term-missing
```

Optional: install `sentence-transformers` to enable embedding routing locally:

```bash
pip install sentence-transformers
```

---

## License

MIT — see [LICENSE](LICENSE).
