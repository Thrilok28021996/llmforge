"""OpenRouter backend — single API key unlocks 100+ models via OpenAI-compat SSE."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from llmforge.config import OpenRouterConfig
from llmforge.domain.models import (
    InferenceRequest,
    ModelDescriptor,
    TokenChunk,
)

logger = logging.getLogger(__name__)

API_BASE = "https://openrouter.ai/api/v1"


class OpenRouterBackend:
    """Connects to OpenRouter's OpenAI-compatible API with extra routing headers."""

    def __init__(self, config: OpenRouterConfig):
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "HTTP-Referer": "https://github.com/llmforge/llmforge",
                "X-Title": "LLM Forge",
            },
            timeout=httpx.Timeout(config.timeout_secs, connect=10.0),
        )
        self._cancelled = False
        self._active_response: httpx.Response | None = None

    @property
    def id(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return "OpenRouter"

    async def cancel(self) -> None:
        self._cancelled = True
        if self._active_response:
            await self._active_response.aclose()

    async def generate(self, request: InferenceRequest) -> AsyncIterator[TokenChunk]:
        """Stream tokens via OpenRouter's chat completions SSE endpoint."""
        self._cancelled = False

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        body: dict = {
            "model": request.model_id,
            "messages": messages,
            "stream": True,
            "temperature": request.params.temperature,
            "top_p": request.params.top_p,
            "max_tokens": request.params.max_tokens,
            "frequency_penalty": request.params.frequency_penalty,
            "presence_penalty": request.params.presence_penalty,
        }
        # Fall back to repeat_penalty conversion if no explicit frequency_penalty
        if request.params.frequency_penalty == 0 and request.params.repeat_penalty != 1.0:
            body["frequency_penalty"] = max(0, request.params.repeat_penalty - 1.0)
        if request.params.top_k > 0:
            body["top_k"] = request.params.top_k
        if request.params.seed is not None:
            body["seed"] = request.params.seed
        if request.params.stop_strings:
            body["stop"] = request.params.stop_strings
        if request.tools:
            body["tools"] = request.tools

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=body
            ) as resp:
                self._active_response = resp
                try:
                    if resp.status_code != 200:
                        error_text = ""
                        async for chunk in resp.aiter_text():
                            error_text += chunk
                        try:
                            err_data = json.loads(error_text)
                            err_msg = err_data.get("error", {}).get(
                                "message", error_text
                            )
                        except json.JSONDecodeError:
                            err_msg = error_text
                        yield TokenChunk(
                            text=f"OpenRouter error {resp.status_code}: {err_msg}",
                            is_final=True,
                        )
                        return

                    buffer = ""
                    async for raw_bytes in resp.aiter_bytes():
                        if self._cancelled:
                            return

                        buffer += raw_bytes.decode("utf-8", errors="replace")

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue

                            data_str = line[6:]
                            if data_str == "[DONE]":
                                yield TokenChunk.final()
                                return

                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                finish = choices[0].get("finish_reason")

                                if content:
                                    yield TokenChunk(text=content)

                                if finish:
                                    usage = data.get("usage", {})
                                    yield TokenChunk.final(
                                        prompt_tokens=usage.get("prompt_tokens"),
                                        completion_tokens=usage.get(
                                            "completion_tokens"
                                        ),
                                    )
                                    return

                    # Stream ended without [DONE]
                    yield TokenChunk.final()

                finally:
                    self._active_response = None

        except (httpx.StreamClosed, httpx.RemoteProtocolError):
            return
        except httpx.ConnectError:
            yield TokenChunk(
                text="Cannot connect to OpenRouter API. Check your API key.",
                is_final=True,
            )
        except httpx.ReadTimeout:
            yield TokenChunk(text="OpenRouter request timed out.", is_final=True)
        except Exception as e:
            if not self._cancelled:
                yield TokenChunk(text=f"OpenRouter error: {e}", is_final=True)

    async def list_models(self) -> list[ModelDescriptor]:
        """Fetch available models from OpenRouter."""
        if not self._config.api_key:
            return []
        try:
            resp = await self._client.get("/models")
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Failed to list OpenRouter models: %s", e)
            return []

        models = []
        for m in data.get("data", []):
            model_id = m.get("id", "")
            ctx = m.get("context_length")
            pricing = m.get("pricing", {})
            name = m.get("name", model_id)

            # Show pricing info in name for visibility
            prompt_cost = pricing.get("prompt", "")
            if prompt_cost and float(prompt_cost) == 0:
                name = f"{name} [free]"

            models.append(
                ModelDescriptor(
                    id=f"openrouter:{model_id}",
                    name=name,
                    backend="openrouter",
                    context_length=ctx,
                )
            )
        return models

    async def is_available(self) -> bool:
        if not self._config.api_key:
            return False
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()
