"""Model descriptors and memory estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelDescriptor:
    id: str  # "ollama:llama3.2:3b" or "gguf:/path/to/model.gguf"
    name: str
    backend: str  # "ollama" | "llamacpp" | "mlx"
    size_bytes: int | None = None
    parameter_count: int | None = None
    quantization: str | None = None
    context_length: int | None = None
    capabilities: dict[str, bool] = field(default_factory=dict)
    families: list[str] = field(default_factory=list)
    modified_at: str | None = None

    @property
    def size_gb(self) -> float | None:
        if self.size_bytes is not None:
            return self.size_bytes / (1024 ** 3)
        return None

    @property
    def param_billions(self) -> float | None:
        if self.parameter_count is not None:
            return self.parameter_count / 1e9
        return None


BACKEND_PREFIXES = ("ollama:", "openai-compat:", "anthropic:", "google:", "openrouter:")


def strip_backend_prefix(model_id: str) -> str:
    """Remove the backend prefix from a model ID, e.g. 'ollama:llama3.2:3b' → 'llama3.2:3b'."""
    for prefix in BACKEND_PREFIXES:
        if model_id.startswith(prefix):
            return model_id[len(prefix):]
    return model_id


def quant_bits(quant: str | None) -> int:
    """Estimate bits-per-weight for a quantization level."""
    if not quant:
        return 16
    q = quant.upper()
    if "Q2" in q:
        return 2
    if "Q3" in q:
        return 3
    if "Q4" in q:
        return 4
    if "Q5" in q:
        return 5
    if "Q6" in q:
        return 6
    if "Q8" in q:
        return 8
    if "F16" in q:
        return 16
    if "F32" in q:
        return 32
    return 4  # conservative default


def estimate_memory_bytes(model: ModelDescriptor) -> int:
    """Rough memory estimate for running a model."""
    params = model.parameter_count or 7_000_000_000
    bits = quant_bits(model.quantization)
    model_bytes = (params * bits) // 8
    # KV cache estimate
    ctx = model.context_length or 4096
    kv_bytes = ctx * 32 * 32 * 128 * 4
    return model_bytes + kv_bytes + 512_000_000  # +0.5GB overhead


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    model_id: str | None = None
    tool_call_id: str | None = None


@dataclass
class GenerationParams:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    context_length: int = 4096
    repeat_penalty: float = 1.1
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "context_length": self.context_length,
            "repeat_penalty": self.repeat_penalty,
            "seed": self.seed,
        }


@dataclass
class InferenceRequest:
    model_id: str
    messages: list[ChatMessage]
    params: GenerationParams = field(default_factory=GenerationParams)
    system_prompt: str | None = None
    tools: list[dict] | None = None


@dataclass
class TokenChunk:
    text: str
    is_final: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tool_calls: list[dict] | None = None

    @classmethod
    def final(cls, text: str = "", prompt_tokens: int | None = None,
              completion_tokens: int | None = None) -> TokenChunk:
        return cls(text=text, is_final=True, prompt_tokens=prompt_tokens,
                   completion_tokens=completion_tokens)
