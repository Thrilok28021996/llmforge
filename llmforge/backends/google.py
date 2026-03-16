"""Google Gemini backend — streams via the Generative Language API."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from llmforge.config import GoogleConfig
from llmforge.domain.models import (
    InferenceRequest,
    ModelDescriptor,
    TokenChunk,
)

logger = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com"


class GoogleBackend:
    """Connects to the Google Gemini API with SSE streaming."""

    def __init__(self, config: GoogleConfig):
        self._config = config
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_secs, connect=10.0),
        )
        self._cancelled = False
        self._active_response: httpx.Response | None = None

    @property
    def id(self) -> str:
        return "google"

    @property
    def display_name(self) -> str:
        return "Google Gemini"

    async def cancel(self) -> None:
        self._cancelled = True
        if self._active_response:
            await self._active_response.aclose()

    async def generate(self, request: InferenceRequest) -> AsyncIterator[TokenChunk]:
        """Stream tokens via the Gemini streamGenerateContent SSE endpoint."""
        self._cancelled = False

        model = request.model_id
        url = (
            f"{API_BASE}/v1beta/models/{model}:streamGenerateContent"
            f"?key={self._config.api_key}&alt=sse"
        )

        # Build Gemini message format
        contents = []
        system_instruction = None

        for msg in request.messages:
            if msg.role == "system":
                system_instruction = {"parts": [{"text": msg.content}]}
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}],
                })

        if request.system_prompt:
            system_instruction = {"parts": [{"text": request.system_prompt}]}

        body: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.params.temperature,
                "topP": request.params.top_p,
                "topK": request.params.top_k,
                "maxOutputTokens": request.params.max_tokens,
            },
        }
        if system_instruction:
            body["systemInstruction"] = system_instruction

        try:
            async with self._client.stream("POST", url, json=body) as resp:
                self._active_response = resp
                try:
                    if resp.status_code != 200:
                        error_text = ""
                        async for chunk in resp.aiter_text():
                            error_text += chunk
                        try:
                            err_data = json.loads(error_text)
                            err_msg = (
                                err_data.get("error", {}).get("message", error_text)
                            )
                        except json.JSONDecodeError:
                            err_msg = error_text
                        yield TokenChunk(
                            text=f"Gemini error {resp.status_code}: {err_msg}",
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
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            candidates = data.get("candidates", [])
                            if candidates:
                                content = candidates[0].get("content", {})
                                parts = content.get("parts", [])
                                for part in parts:
                                    text = part.get("text", "")
                                    if text:
                                        yield TokenChunk(text=text)

                                finish = candidates[0].get("finishReason")
                                if finish:
                                    usage = data.get("usageMetadata", {})
                                    yield TokenChunk.final(
                                        prompt_tokens=usage.get(
                                            "promptTokenCount"
                                        ),
                                        completion_tokens=usage.get(
                                            "candidatesTokenCount"
                                        ),
                                    )
                                    return

                            # usageMetadata may appear mid-stream; we capture it
                            # in the finishReason block above

                    # If stream ends without finishReason
                    yield TokenChunk.final()

                finally:
                    self._active_response = None

        except (httpx.StreamClosed, httpx.RemoteProtocolError):
            return
        except httpx.ConnectError:
            yield TokenChunk(
                text="Cannot connect to Google API. Check your API key.",
                is_final=True,
            )
        except httpx.ReadTimeout:
            yield TokenChunk(text="Gemini request timed out.", is_final=True)
        except Exception as e:
            if not self._cancelled:
                yield TokenChunk(text=f"Gemini error: {e}", is_final=True)

    async def list_models(self) -> list[ModelDescriptor]:
        """Fetch available Gemini models."""
        if not self._config.api_key:
            return []
        try:
            resp = await self._client.get(
                f"{API_BASE}/v1beta/models",
                params={"key": self._config.api_key},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Failed to list Gemini models: %s", e)
            return []

        models = []
        for m in data.get("models", []):
            supported = m.get("supportedGenerationMethods", [])
            if "generateContent" not in supported:
                continue
            model_id = m["name"].removeprefix("models/")
            models.append(
                ModelDescriptor(
                    id=f"google:{model_id}",
                    name=m.get("displayName", model_id),
                    backend="google",
                    context_length=m.get("inputTokenLimit"),
                )
            )
        return models

    async def is_available(self) -> bool:
        if not self._config.api_key:
            return False
        try:
            resp = await self._client.get(
                f"{API_BASE}/v1beta/models",
                params={"key": self._config.api_key},
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()
