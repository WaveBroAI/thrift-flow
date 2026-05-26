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
from proxy.tracker import RequestTracker

logger = logging.getLogger(__name__)

_VERSION = "0.1.0"


def create_app(config: ProxyConfig, tracker: Optional[RequestTracker]) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(title="thrift-flow", version=_VERSION)

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
        model_resolved: str = config.resolve_model(model_requested)
        messages: list[dict] = body.get("messages", [])
        is_streaming: bool = body.get("stream") is True

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
                        yield sse_str.encode()
                        if sse_str.startswith("data: [DONE]"):  # Fix 9: reliable DONE detection
                            output_tokens = tok
                            cost_usd = cost
                            stream_done = True
                except Exception as exc:
                    logger.exception("Streaming error")
                    status = 500
                    error_msg = str(exc)
                    yield b"data: [DONE]\n\n"
                finally:
                    # Fix 8: GeneratorExit (client disconnect) bypasses except-Exception above;
                    # stream_done=False and status=200 together identify a client disconnect.
                    if not stream_done and status == 200:
                        status = 499  # Client Closed Request
                    latency_ms = (time.monotonic() - start_time) * 1000
                    if tracker is not None:
                        try:
                            await asyncio.to_thread(  # Fix 5: don't block the event loop
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
                            )
                        except Exception:
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
        return tracker.get_summary()

    @app.get("/v1/usage/by-model")
    async def usage_by_model() -> list[dict[str, Any]]:
        if tracker is None:
            return []
        return tracker.get_by_model()

    @app.get("/v1/usage/recent")
    async def usage_recent(limit: int = 50) -> list[dict[str, Any]]:
        if tracker is None:
            return []
        return tracker.get_recent(limit=limit)

    return app
