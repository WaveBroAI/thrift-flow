from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import litellm
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from proxy.config import ProxyConfig
from proxy.forwarder import call_non_streaming, stream_completion
from proxy.router import ModelRouter
from proxy.tracker import RequestTracker

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"


def create_app(config: ProxyConfig, tracker: Optional[RequestTracker]) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="thrift-flow", version=_VERSION)

    # ── Adaptive model router (Phase 2) ───────────────────────────────────────
    # Created once per app instance; holds the per-session context cache.
    # Only active when config.routing.enabled=True.
    _model_router: ModelRouter | None = None
    if config.routing.enabled:
        _model_router = ModelRouter(
            aliases=config.models.aliases,
            routing_config=config.routing,
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": _VERSION}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        data = [
            {
                "id": alias,
                "object": "model",
                "created": 0,
                "owned_by": "thrift-flow",
            }
            for alias in config.models.aliases
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        client_id: Optional[str] = request.headers.get("X-Client-ID")
        session_key: Optional[str] = request.headers.get("X-Session-Key")

        try:
            body: dict = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        model_requested: str = body.get("model", config.models.default)
        messages: list[dict] = body.get("messages", [])

        # ── model resolution ──────────────────────────────────────────────────
        # model="auto" + routing enabled → adaptive router picks tier + model.
        # Any other model name → alias resolution (or pass-through if unknown).
        if model_requested == "auto" and _model_router is not None:
            # content can be a list for multimodal requests — use str only.
            _raw_content = next(
                (m.get("content") for m in reversed(messages)
                 if m.get("role") == "user"),
                None,
            )
            _last_user_msg = _raw_content if isinstance(_raw_content, str) else ""
            _, model_resolved = await _model_router.route(
                _last_user_msg,
                messages,
                # Pass None for empty text (multimodal/no-user-msg) so routing_log
                # stores NULL rather than sha256("") for all such requests.
                prompt_for_hash=_last_user_msg or None,
                session_key=session_key,
            )
        else:
            model_resolved = config.resolve_model(model_requested)

        _stream = body.get("stream")
        is_streaming: bool = _stream is True or _stream == 1  # Fix C: accept numeric 1

        # Count input tokens
        input_tokens = 0
        try:
            input_tokens = litellm.token_counter(
                model=model_resolved, messages=messages
            )
        except Exception:
            input_tokens = 0

        start_time = time.monotonic()

        if is_streaming:

            async def _generate():
                output_tokens = 0
                cost_usd = 0.0
                status = 200
                error_msg = None
                stream_done = False  # Fix 8: track normal stream completion

                try:
                    async for sse_str, tok, cost in stream_completion(
                        model_resolved, messages, body
                    ):
                        if sse_str.startswith("data: [DONE]"):  # Fix D: set before yield to avoid race
                            output_tokens = tok
                            cost_usd = cost
                            stream_done = True
                        yield sse_str.encode()
                except Exception as exc:
                    logger.exception("Streaming error")
                    status = 500
                    error_msg = str(exc)
                    if not stream_done:  # Fix #3: stream_completion may have already yielded [DONE]
                        yield b"data: [DONE]\n\n"
                finally:
                    # Fix 8: GeneratorExit (client disconnect) bypasses except-Exception above;
                    # stream_done=False and status=200 together identify a client disconnect.
                    if not stream_done and status == 200:
                        status = 499  # Client Closed Request
                    latency_ms = (time.monotonic() - start_time) * 1000
                    if tracker is not None:
                        try:
                            # Fix A: shield protects the log write from CancelledError on disconnect;
                            # except BaseException catches CancelledError (inherits BaseException, not Exception)
                            await asyncio.shield(asyncio.to_thread(
                                tracker.log_request,
                                model_requested=model_requested,
                                model_resolved=model_resolved,
                                input_tokens=input_tokens,
                                output_tokens=output_tokens,
                                cost_usd=cost_usd,
                                latency_ms=latency_ms,
                                streaming=True,
                                status=status,
                                error=error_msg,
                                client_id=client_id,
                                session_key=session_key,
                            ))
                        except asyncio.CancelledError:
                            pass  # expected on client disconnect; shielded tracker write continues
                        except BaseException:
                            logger.exception("Failed to log streaming request")

            return StreamingResponse(
                _generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            try:
                response_dict, output_tokens, cost_usd = await call_non_streaming(
                    model_resolved, messages, body
                )
                latency_ms = (time.monotonic() - start_time) * 1000

                if tracker is not None:
                    try:  # Fix 3: don't let a tracker failure destroy a successful response
                        await asyncio.to_thread(  # Fix 5: non-blocking
                            tracker.log_request,
                            model_requested=model_requested,
                            model_resolved=model_resolved,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cost_usd=cost_usd,
                            latency_ms=latency_ms,
                            streaming=False,
                            status=200,
                            error=None,
                            client_id=client_id,
                            session_key=session_key,
                        )
                    except Exception:
                        logger.exception("Failed to log request")

                return JSONResponse(content=response_dict)

            except HTTPException:
                raise
            except Exception as exc:
                latency_ms = (time.monotonic() - start_time) * 1000
                logger.exception("Non-streaming request failed")

                if tracker is not None:
                    try:
                        await asyncio.to_thread(  # Fix 5: non-blocking
                            tracker.log_request,
                            model_requested=model_requested,
                            model_resolved=model_resolved,
                            input_tokens=input_tokens,
                            output_tokens=0,
                            cost_usd=0.0,
                            latency_ms=latency_ms,
                            streaming=False,
                            status=500,
                            error=str(exc),
                            client_id=client_id,
                            session_key=session_key,
                        )
                    except Exception:
                        logger.exception("Failed to log failed request")

                raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/v1/usage")
    async def usage_summary() -> dict[str, Any]:
        if tracker is None:
            return {"tracking_enabled": False}
        return await asyncio.to_thread(tracker.get_summary)  # Fix B: don't block the event loop

    @app.get("/v1/usage/by-model")
    async def usage_by_model() -> list[dict[str, Any]]:
        if tracker is None:
            return []
        return await asyncio.to_thread(tracker.get_by_model)  # Fix B

    @app.get("/v1/usage/recent")
    async def usage_recent(limit: int = 50) -> list[dict[str, Any]]:
        if tracker is None:
            return []
        return await asyncio.to_thread(tracker.get_recent, limit)  # Fix B

    return app
