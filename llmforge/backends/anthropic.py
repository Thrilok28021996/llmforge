"""Anthropic Claude backend — streams via the Messages API."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from llmforge.config import AnthropicConfig
from llmforge.domain.models import (
    InferenceRequest,
    ModelDescriptor,
    TokenChunk,
)

logger = logging.getLogger(__name__)

API_BASE = "https://api.anthropic.com"
API_VERSION = "2023-06-01"

# Known models (Anthropic has no list-models endpoint)
KNOWN_MODELS = [
    ("claude-opus-4-20250514", "Claude Opus 4"),
    ("claude-sonnet-4-20250514", "Claude Sonnet 4"),
    ("claude-haiku-4-20250414", "Claude Haiku 4"),
    ("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"),
    ("claude-3-5-haiku-20241022", "Claude 3.5 Haiku"),
]


class AnthropicBackend:
    """Connects to the Anthropic Messages API with SSE streaming."""

    def __init__(self, config: AnthropicConfig):
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(config.timeout_secs, connect=10.0),
        )
        self._cancelled = False
        self._active_response: httpx.Response | None = None

    @property
    def id(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return "Anthropic"

    async def cancel(self) -> None:
        self._cancelled = True
        if self._active_response:
            await self._active_response.aclose()

    async def generate(self, request: InferenceRequest) -> AsyncIterator[TokenChunk]:
        """Stream tokens via the Anthropic Messages SSE endpoint."""
        self._cancelled = False

        # Separate system from conversation messages
        system_text = request.system_prompt or ""
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                system_text = msg.content
            else:
                messages.append({"role": msg.role, "content": msg.content})

        body: dict = {
            "model": request.model_id,
            "messages": messages,
            "max_tokens": request.params.max_tokens,
            "stream": True,
            "temperature": request.params.temperature,
            "top_p": request.params.top_p,
        }
        if system_text:
            body["system"] = system_text
        if request.params.top_k > 0:
            body["top_k"] = request.params.top_k

        # Include tools if present
        if hasattr(request, "tools") and request.tools:
            body["tools"] = request.tools

        try:
            async with self._client.stream(
                "POST", "/v1/messages", json=body
            ) as resp:
                self._active_response = resp
                try:
                    if resp.status_code != 200:
                        error_text = ""
                        async for chunk in resp.aiter_text():
                            error_text += chunk
                        try:
                            err_data = json.loads(error_text)
                            err_msg = err_data.get("error", {}).get("message", error_text)
                        except json.JSONDecodeError:
                            err_msg = error_text
                        yield TokenChunk(
                            text=f"Anthropic error {resp.status_code}: {err_msg}",
                            is_final=True,
                        )
                        return

                    buffer = ""
                    input_tokens = 0
                    output_tokens = 0

                    async for raw_bytes in resp.aiter_bytes():
                        if self._cancelled:
                            return

                        buffer += raw_bytes.decode("utf-8", errors="replace")

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()

                            if not line:
                                continue

                            if line.startswith("event: "):
                                continue

                            if not line.startswith("data: "):
                                continue

                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            event_type = data.get("type", "")

                            if event_type == "message_start":
                                usage = data.get("message", {}).get("usage", {})
                                input_tokens = usage.get("input_tokens", 0)

                            elif event_type == "content_block_delta":
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")
                                    if text:
                                        yield TokenChunk(text=text)
                                elif delta.get("type") == "input_json_delta":
                                    # Tool use delta — yield as tool JSON
                                    partial = delta.get("partial_json", "")
                                    if partial:
                                        yield TokenChunk(text=partial)

                            elif event_type == "message_delta":
                                usage = data.get("usage", {})
                                output_tokens = usage.get("output_tokens", 0)

                            elif event_type == "message_stop":
                                yield TokenChunk.final(
                                    prompt_tokens=input_tokens,
                                    completion_tokens=output_tokens,
                                )
                                return
                finally:
                    self._active_response = None

        except (httpx.StreamClosed, httpx.RemoteProtocolError):
            return
        except httpx.ConnectError:
            yield TokenChunk(
                text="Cannot connect to Anthropic API. Check your API key.",
                is_final=True,
            )
        except httpx.ReadTimeout:
            yield TokenChunk(text="Anthropic request timed out.", is_final=True)
        except Exception as e:
            if not self._cancelled:
                yield TokenChunk(text=f"Anthropic error: {e}", is_final=True)

    async def list_models(self) -> list[ModelDescriptor]:
        return [
            ModelDescriptor(
                id=f"anthropic:{model_id}",
                name=display_name,
                backend="anthropic",
            )
            for model_id, display_name in KNOWN_MODELS
        ]

    async def is_available(self) -> bool:
        if not self._config.api_key:
            return False
        try:
            resp = await self._client.post(
                "/v1/messages",
                json={
                    "model": "claude-haiku-4-20250414",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return resp.status_code in (200, 400, 429)
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()
