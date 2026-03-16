"""Native llama.cpp backend — runs GGUF models in-process via llama-cpp-python.

No Ollama or external server required. Metal auto-enabled on Apple Silicon.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from llmforge.config import LlamaCppConfig
from llmforge.domain.models import (
    InferenceRequest,
    ModelDescriptor,
    TokenChunk,
)

logger = logging.getLogger(__name__)


class LlamaCppBackend:
    """Direct GGUF inference via llama-cpp-python. Zero external dependencies."""

    def __init__(self, config: LlamaCppConfig):
        self._config = config
        self._model = None
        self._model_path: str | None = None
        self._cancelled = False

    @property
    def id(self) -> str:
        return "llamacpp"

    @property
    def display_name(self) -> str:
        return "llama.cpp (native)"

    async def cancel(self) -> None:
        self._cancelled = True

    def _ensure_model(self, model_path: str) -> None:
        """Load or switch model (blocking — called from spawn_blocking)."""
        if self._model_path == model_path and self._model is not None:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. "
                "Run: pip install 'llmforge[llamacpp]'"
            )

        if self._model is not None:
            del self._model
            self._model = None

        logger.info("Loading GGUF model: %s", model_path)
        load_kwargs: dict = {
            "model_path": model_path,
            "n_ctx": self._config.context_length,
            "n_gpu_layers": self._config.n_gpu_layers,
            "verbose": False,
            "use_mmap": self._config.use_mmap,
            "use_mlock": self._config.use_mlock,
            "flash_attn": self._config.flash_attention,
            "n_batch": self._config.eval_batch_size,
            "type_k": 1 if self._config.use_fp16_kv else 0,  # 1=f16, 0=f32
            "type_v": 1 if self._config.use_fp16_kv else 0,
        }
        if self._config.rope_freq_base > 0:
            load_kwargs["rope_freq_base"] = self._config.rope_freq_base
        if self._config.rope_freq_scale > 0:
            load_kwargs["rope_freq_scale"] = self._config.rope_freq_scale
        if self._config.num_experts is not None:
            load_kwargs["n_experts"] = self._config.num_experts
        if self._config.cpu_threads > 0:
            load_kwargs["n_threads"] = self._config.cpu_threads

        # Speculative decoding
        if self._config.speculative == "prompt-lookup":
            try:
                from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                load_kwargs["draft_model"] = LlamaPromptLookupDecoding(
                    num_pred_tokens=self._config.speculative_num_tokens,
                )
                logger.info(
                    "Speculative decoding: prompt-lookup (%d tokens)",
                    self._config.speculative_num_tokens,
                )
            except ImportError:
                logger.warning(
                    "Prompt lookup decoding not available — "
                    "upgrade llama-cpp-python to >=0.2.38"
                )
        elif self._config.speculative == "draft-model":
            draft_path = self._config.speculative_draft_model
            if draft_path and Path(draft_path).expanduser().exists():
                try:
                    from llama_cpp.llama_speculative import LlamaPromptLookupDecoding  # noqa: F811

                    # Load draft model as a separate Llama instance
                    draft = Llama(
                        model_path=str(Path(draft_path).expanduser()),
                        n_ctx=self._config.context_length,
                        n_gpu_layers=self._config.n_gpu_layers,
                        verbose=False,
                    )
                    load_kwargs["draft_model"] = draft
                    logger.info("Speculative decoding: draft model %s", draft_path)
                except Exception as e:
                    logger.warning("Failed to load draft model: %s", e)
            else:
                logger.warning(
                    "Draft model path not found: %s", draft_path
                )

        self._model = Llama(**load_kwargs)
        self._model_path = model_path
        logger.info("Model loaded: %s", model_path)

    async def generate(
        self, request: InferenceRequest
    ) -> AsyncIterator[TokenChunk]:
        """Stream tokens from a GGUF model."""
        self._cancelled = False

        model_path = request.model_id
        # If it's a prefixed ID like "llamacpp:/path/to/model.gguf"
        if model_path.startswith("llamacpp:"):
            model_path = model_path[len("llamacpp:"):]

        # Resolve relative paths
        if not Path(model_path).is_absolute():
            # Search configured model directories
            for d in self._config.model_dirs:
                candidate = Path(d) / model_path
                if candidate.exists():
                    model_path = str(candidate)
                    break

        if not Path(model_path).exists():
            yield TokenChunk(
                text=f"Model file not found: {model_path}",
                is_final=True,
            )
            return

        # Load model in a thread (blocking operation)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, self._ensure_model, model_path
            )
        except Exception as e:
            yield TokenChunk(text=f"Failed to load model: {e}", is_final=True)
            return

        # Build chat messages
        messages = []
        if request.system_prompt:
            messages.append({
                "role": "system", "content": request.system_prompt
            })
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        # Run inference in thread pool
        prompt_tokens = 0
        completion_tokens = 0

        try:
            # Use create_chat_completion with streaming
            def _stream():
                gen_kwargs: dict = {
                    "messages": messages,
                    "stream": True,
                    "temperature": request.params.temperature,
                    "top_p": request.params.top_p,
                    "top_k": request.params.top_k,
                    "max_tokens": request.params.max_tokens,
                    "repeat_penalty": request.params.repeat_penalty,
                    "seed": request.params.seed or -1,
                }
                if request.params.min_p > 0:
                    gen_kwargs["min_p"] = request.params.min_p
                if request.params.frequency_penalty != 0:
                    gen_kwargs["frequency_penalty"] = request.params.frequency_penalty
                if request.params.presence_penalty != 0:
                    gen_kwargs["presence_penalty"] = request.params.presence_penalty
                if request.params.stop_strings:
                    gen_kwargs["stop"] = request.params.stop_strings
                return self._model.create_chat_completion(**gen_kwargs)

            stream = await loop.run_in_executor(None, _stream)

            # Iterate stream in thread, push chunks through queue
            queue: asyncio.Queue[TokenChunk | None] = asyncio.Queue()

            def _consume():
                nonlocal prompt_tokens, completion_tokens
                try:
                    for chunk_data in stream:
                        if self._cancelled:
                            break
                        choices = chunk_data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            finish = choices[0].get("finish_reason")
                            if content:
                                loop.call_soon_threadsafe(
                                    queue.put_nowait,
                                    TokenChunk(text=content),
                                )
                            if finish:
                                usage = chunk_data.get("usage", {})
                                prompt_tokens = usage.get(
                                    "prompt_tokens", 0
                                )
                                completion_tokens = usage.get(
                                    "completion_tokens", 0
                                )
                except Exception as e:
                    if not self._cancelled:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            TokenChunk(
                                text=f"\n\nInference error: {e}",
                            ),
                        )
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            asyncio.get_event_loop().run_in_executor(None, _consume)

            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

            yield TokenChunk.final(
                prompt_tokens=prompt_tokens or None,
                completion_tokens=completion_tokens or None,
            )

        except Exception as e:
            if not self._cancelled:
                yield TokenChunk(
                    text=f"llama.cpp error: {e}", is_final=True
                )

    async def list_models(self) -> list[ModelDescriptor]:
        """Scan configured directories for GGUF files."""
        models = []
        seen = set()

        for dir_path in self._config.model_dirs:
            d = Path(dir_path).expanduser()
            if not d.is_dir():
                continue
            for gguf_file in sorted(d.rglob("*.gguf")):
                if gguf_file.name in seen:
                    continue
                seen.add(gguf_file.name)

                size_bytes = gguf_file.stat().st_size
                # Try to extract quant from filename
                quant = _guess_quantization(gguf_file.name)
                params = _guess_params_from_name(gguf_file.name)

                models.append(
                    ModelDescriptor(
                        id=f"llamacpp:{gguf_file}",
                        name=gguf_file.stem,
                        backend="llamacpp",
                        size_bytes=size_bytes,
                        quantization=quant,
                        parameter_count=params,
                    )
                )
        return models

    async def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    async def close(self):
        if self._model is not None:
            del self._model
            self._model = None
            self._model_path = None


def _guess_quantization(filename: str) -> str | None:
    """Extract quantization level from GGUF filename."""
    upper = filename.upper()
    for q in [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
        "Q6_K", "Q8_0", "F16", "F32",
    ]:
        if q in upper:
            return q
    return None


def _guess_params_from_name(filename: str) -> int | None:
    """Try to guess parameter count from filename like 'llama-7b' or '3.2B'."""
    import re
    match = re.search(r"(\d+\.?\d*)[bB]", filename)
    if match:
        try:
            return int(float(match.group(1)) * 1_000_000_000)
        except ValueError:
            pass
    return None
