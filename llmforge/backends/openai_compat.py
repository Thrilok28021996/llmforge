"""OpenAI-compatible backend — works with LM Studio, vLLM, llama.cpp server, LocalAI."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from llmforge.domain.models import (
    InferenceRequest,
    ModelDescriptor,
    TokenChunk,
)

logger = logging.getLogger(__name__)


class OpenAICompatBackend:
    """Connects to any OpenAI-compatible API (LM Studio, vLLM, llama.cpp server, etc.)."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "not-needed",
        timeout: int = 120,
    ):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        self._cancelled = False
        self._active_response: httpx.Response | None = None

    @property
    def id(self) -> str:
        return "openai-compat"

    @property
    def display_name(self) -> str:
        return "OpenAI-Compatible"

    async def cancel(self) -> None:
        self._cancelled = True
        if self._active_response:
            await self._active_response.aclose()

    async def generate(self, request: InferenceRequest) -> AsyncIterator[TokenChunk]:
        """Stream tokens via the OpenAI chat completions SSE endpoint."""
        self._cancelled = False

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        body = {
            "model": request.model_id,
            "messages": messages,
            "stream": True,
            "temperature": request.params.temperature,
            "top_p": request.params.top_p,
            "max_tokens": request.params.max_tokens,
            "frequency_penalty": request.params.repeat_penalty - 1.0,
        }
        if request.params.seed is not None:
            body["seed"] = request.params.seed

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
                        err = f"API error {resp.status_code}: {error_text}"
                        yield TokenChunk(text=err, is_final=True)
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

                            data_str = line[6:]  # Strip "data: "
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
                finally:
                    self._active_response = None

        except (httpx.StreamClosed, httpx.RemoteProtocolError):
            return  # Expected when cancel() closes the stream
        except httpx.ConnectError:
            yield TokenChunk(
                text="Cannot connect to API server.",
                is_final=True,
            )
        except httpx.ReadTimeout:
            yield TokenChunk(text="API request timed out.", is_final=True)
        except Exception as e:
            yield TokenChunk(text=f"API error: {e}", is_final=True)

    async def list_models(self) -> list[ModelDescriptor]:
        """Fetch available models via /v1/models."""
        try:
            resp = await self._client.get("/models")
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Failed to list models: %s", e)
            return []

        models = []
        for m in data.get("data", []):
            models.append(
                ModelDescriptor(
                    id=f"openai:{m['id']}",
                    name=m["id"],
                    backend="openai-compat",
                )
            )
        return models

    async def is_available(self) -> bool:
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()
