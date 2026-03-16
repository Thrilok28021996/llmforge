"""OpenAI-compatible API server using Starlette.

Exposes /v1/chat/completions and /v1/models endpoints so other apps
(VS Code extensions, web UIs, scripts) can use LLM Forge as a backend.

Usage:
    llmforge serve --port 8000
    # Then point any OpenAI-compatible client at http://localhost:8000/v1
"""

from __future__ import annotations

import json
import logging
import time
import uuid

logger = logging.getLogger(__name__)


def create_app(backend, config):
    """Create the Starlette ASGI app."""
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, StreamingResponse
        from starlette.routing import Route
    except ImportError:
        raise RuntimeError(
            "Server dependencies not installed. "
            "Run: pip install 'llmforge[server]'"
        )

    from llmforge.domain.models import (
        ChatMessage,
        GenerationParams,
        InferenceRequest,
    )

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def list_models(request: Request) -> JSONResponse:
        models = await backend.list_models()
        data = []
        for m in models:
            data.append({
                "id": m.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llmforge",
            })
        return JSONResponse({
            "object": "list",
            "data": data,
        })

    async def chat_completions(request: Request):
        body = await request.json()
        model_id = body.get("model", "")
        messages_raw = body.get("messages", [])
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        max_tokens = body.get("max_tokens", 2048)
        seed = body.get("seed")

        messages = []
        system_prompt = None
        for m in messages_raw:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_prompt = content
            else:
                messages.append(ChatMessage(role=role, content=content))

        params = GenerationParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )

        request_obj = InferenceRequest(
            model_id=model_id,
            messages=messages,
            params=params,
            system_prompt=system_prompt,
        )

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream:
            return StreamingResponse(
                _stream_response(backend, request_obj, request_id, model_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await _non_stream_response(
                backend, request_obj, request_id, model_id
            )

    async def _stream_response(backend, request, request_id, model_id):
        """SSE streaming response matching OpenAI format."""
        async for chunk in backend.generate(request):
            if chunk.text:
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk.text},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n"

            if chunk.is_final:
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }],
                }
                if chunk.prompt_tokens or chunk.completion_tokens:
                    data["usage"] = {
                        "prompt_tokens": chunk.prompt_tokens or 0,
                        "completion_tokens": chunk.completion_tokens or 0,
                        "total_tokens": (
                            (chunk.prompt_tokens or 0)
                            + (chunk.completion_tokens or 0)
                        ),
                    }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
                return

    async def _non_stream_response(
        backend, request, request_id, model_id
    ) -> JSONResponse:
        """Non-streaming response matching OpenAI format."""
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in backend.generate(request):
            if chunk.text:
                full_text += chunk.text
            if chunk.is_final:
                prompt_tokens = chunk.prompt_tokens or 0
                completion_tokens = chunk.completion_tokens or 0
                break

        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_text,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })

    routes = [
        Route("/health", health),
        Route("/v1/models", list_models),
        Route("/v1/chat/completions", chat_completions, methods=["POST"]),
    ]

    return Starlette(routes=routes)
