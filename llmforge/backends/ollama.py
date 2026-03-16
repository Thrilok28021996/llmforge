"""Ollama backend — streams inference via the Ollama HTTP API."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from llmforge.config import OllamaConfig
from llmforge.domain.models import (
    InferenceRequest,
    ModelDescriptor,
    TokenChunk,
)

logger = logging.getLogger(__name__)


class OllamaBackend:
    """Connects to a running Ollama daemon via HTTP."""

    def __init__(self, config: OllamaConfig):
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout_secs, connect=10.0),
        )
        self._cancelled = False
        self._active_response: httpx.Response | None = None

    @property
    def id(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama"

    async def cancel(self) -> None:
        """Cancel in-progress inference by closing the active stream."""
        self._cancelled = True
        if self._active_response:
            await self._active_response.aclose()

    async def generate(
        self, request: InferenceRequest
    ) -> AsyncIterator[TokenChunk]:
        """Stream tokens from Ollama's /api/chat endpoint (NDJSON)."""
        self._cancelled = False

        messages = []
        if request.system_prompt:
            messages.append(
                {"role": "system", "content": request.system_prompt}
            )
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        body = {
            "model": request.model_id,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": request.params.temperature,
                "top_p": request.params.top_p,
                "top_k": request.params.top_k,
                "num_predict": request.params.max_tokens,
                "num_ctx": request.params.context_length,
                "repeat_penalty": request.params.repeat_penalty,
            },
        }
        if request.params.seed is not None:
            body["options"]["seed"] = request.params.seed

        try:
            async with self._client.stream(
                "POST", "/api/chat", json=body
            ) as resp:
                self._active_response = resp
                try:
                    if resp.status_code != 200:
                        error_text = ""
                        async for chunk in resp.aiter_text():
                            error_text += chunk
                        err = f"Ollama error {resp.status_code}: {error_text}"
                        yield TokenChunk(text=err, is_final=True)
                        return

                    buffer_parts: list[str] = []
                    async for raw_bytes in resp.aiter_bytes():
                        if self._cancelled:
                            return

                        buffer_parts.append(
                            raw_bytes.decode("utf-8", errors="replace")
                        )
                        buffer = "".join(buffer_parts)

                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            buffer_parts = [buffer]
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError as e:
                                logger.debug(
                                    "NDJSON parse error: %s (line: %s)",
                                    e,
                                    line[:100],
                                )
                                continue

                            msg = data.get("message", {})
                            content = msg.get("content", "")
                            done = data.get("done", False)

                            # Detect thinking blocks
                            if content:
                                yield TokenChunk(text=content)

                            if done:
                                yield TokenChunk.final(
                                    text="",
                                    prompt_tokens=data.get(
                                        "prompt_eval_count"
                                    ),
                                    completion_tokens=data.get("eval_count"),
                                )
                                return
                finally:
                    self._active_response = None

        except (httpx.StreamClosed, httpx.RemoteProtocolError):
            # Expected when cancel() closes the active stream
            return
        except httpx.ConnectError:
            yield TokenChunk(
                text=(
                    "Cannot connect to Ollama. Is it running? "
                    "Start with: ollama serve"
                ),
                is_final=True,
            )
        except httpx.ReadTimeout:
            yield TokenChunk(
                text="Ollama request timed out.", is_final=True
            )
        except Exception as e:
            if not self._cancelled:
                yield TokenChunk(text=f"Ollama error: {e}", is_final=True)

    async def list_models(self) -> list[ModelDescriptor]:
        """Fetch locally available models from Ollama."""
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Failed to list Ollama models: %s", e)
            return []

        models = []
        for m in data.get("models", []):
            details = m.get("details", {})
            param_count = None
            param_str = details.get("parameter_size", "")
            if param_str:
                param_count = _parse_param_size(param_str)

            models.append(
                ModelDescriptor(
                    id=f"ollama:{m['name']}",
                    name=m["name"],
                    backend="ollama",
                    size_bytes=m.get("size"),
                    parameter_count=param_count,
                    quantization=details.get("quantization_level"),
                    families=details.get("families", []),
                    modified_at=m.get("modified_at"),
                )
            )
        return models

    async def pull_model(self, name: str) -> AsyncIterator[dict]:
        """Pull a model, yielding progress dicts with status/completed/total."""
        body = {"name": name, "stream": True}
        async with self._client.stream(
            "POST", "/api/pull", json=body
        ) as resp:
            resp.raise_for_status()
            buffer = ""
            async for raw_bytes in resp.aiter_bytes():
                buffer += raw_bytes.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield data
                    except json.JSONDecodeError:
                        pass

    async def delete_model(self, name: str) -> None:
        """Delete a local model."""
        resp = await self._client.request("DELETE", "/api/delete", json={"name": name})
        resp.raise_for_status()

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            resp = await self._client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._client.aclose()


def _parse_param_size(s: str) -> int | None:
    """Parse '7B' -> 7_000_000_000, '3.2B' -> 3_200_000_000."""
    s = s.strip().upper()
    multipliers = {"B": 1_000_000_000, "M": 1_000_000, "K": 1_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return int(float(s[:-1]) * mult)
            except ValueError:
                return None
    return None
