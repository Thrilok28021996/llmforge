"""Configuration management with TOML + environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

# ── Paths ────────────────────────────────────────────────────────────────────

def data_dir() -> Path:
    """~/.llmforge — all persistent data lives here."""
    p = Path.home() / ".llmforge"
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_path() -> Path:
    return data_dir() / "config.toml"


def db_path() -> Path:
    return data_dir() / "data.sqlite"


# ── Config models ────────────────────────────────────────────────────────────

class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    timeout_secs: int = 120


class OpenAICompatConfig(BaseModel):
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "not-needed"
    timeout_secs: int = 120


class AnthropicConfig(BaseModel):
    api_key: str = ""
    default_model: str = "claude-sonnet-4-20250514"
    timeout_secs: int = 120


class GoogleConfig(BaseModel):
    api_key: str = ""
    timeout_secs: int = 120


class MCPServerConfig(BaseModel):
    name: str
    command: list[str]
    env: dict[str, str] = Field(default_factory=dict)


class MCPConfig(BaseModel):
    servers: list[MCPServerConfig] = Field(default_factory=list)


class OpenRouterConfig(BaseModel):
    api_key: str = ""
    timeout_secs: int = 120
    default_model: str = "meta-llama/llama-3.2-3b-instruct:free"


class WebSearchConfig(BaseModel):
    enabled: bool = False
    provider: str = "duckduckgo"  # "duckduckgo" | "searxng" | "tavily"
    searxng_url: str = "http://localhost:8080"
    tavily_api_key: str = ""
    max_results: int = 5


class RAGConfig(BaseModel):
    enabled: bool = False
    embedding_model: str = "nomic-embed-text"
    embedding_method: str = "auto"  # "auto" | "ollama" | "llamacpp" | "tfidf"
    chunk_size: int = 512
    overlap: int = 64
    top_k: int = 3
    rerank: bool = True
    rerank_model: str = "llama3.2:3b"
    watch_dirs: list[str] = Field(default_factory=list)  # auto-ingest folders


class LlamaCppConfig(BaseModel):
    model_dirs: list[str] = Field(default_factory=list)
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    context_length: int = 4096
    flash_attention: bool = False
    eval_batch_size: int = 512
    rope_freq_base: float = 0.0  # 0 = auto
    rope_freq_scale: float = 0.0  # 0 = auto
    use_mmap: bool = True
    use_mlock: bool = False
    use_fp16_kv: bool = True
    num_experts: int | None = None  # MoE models only
    cpu_threads: int = 0  # 0 = auto
    # Speculative decoding
    speculative: str = "off"  # "off" | "prompt-lookup" | "draft-model"
    speculative_num_tokens: int = 10  # tokens to predict ahead (10=gpu, 2=cpu)
    speculative_draft_model: str = ""  # path to smaller GGUF for draft-model mode


class ProfilerConfig(BaseModel):
    poll_interval_ms: int = 200
    sparkline_width: int = 60


class ScoringConfig(BaseModel):
    enabled: bool = True
    judge_model: str = "llama3.2:3b"  # Ollama model used for LLM-as-judge


class GenerationDefaults(BaseModel):
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    context_length: int = 4096
    repeat_penalty: float = 1.1
    seed: int | None = None
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_strings: list[str] = Field(default_factory=list)


class Config(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    openai_compat: OpenAICompatConfig = Field(default_factory=OpenAICompatConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    google: GoogleConfig = Field(default_factory=GoogleConfig)
    llamacpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    openrouter: OpenRouterConfig = Field(default_factory=OpenRouterConfig)
    backend: str = "ollama"  # "ollama" | "openai-compat" | "anthropic" | "google" | "openrouter"
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    generation: GenerationDefaults = Field(default_factory=GenerationDefaults)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    theme: str = "dark"

    @classmethod
    def load(cls) -> Config:
        """Load from TOML file, with env var overrides."""
        import logging

        path = config_path()
        data: dict[str, Any] = {}
        if path.exists():
            try:
                data = tomllib.loads(path.read_text())
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Invalid config at %s, using defaults: %s", path, e
                )

        # Env overrides
        if url := os.environ.get("LLMFORGE_OLLAMA_URL"):
            data.setdefault("ollama", {})["base_url"] = url
        if key := os.environ.get("LLMFORGE_ANTHROPIC_KEY"):
            data.setdefault("anthropic", {})["api_key"] = key
        if key := os.environ.get("LLMFORGE_GOOGLE_KEY"):
            data.setdefault("google", {})["api_key"] = key
        if key := os.environ.get("LLMFORGE_OPENROUTER_KEY"):
            data.setdefault("openrouter", {})["api_key"] = key
        if key := os.environ.get("LLMFORGE_TAVILY_KEY"):
            data.setdefault("web_search", {})["tavily_api_key"] = key

        return cls.model_validate(data)

    def save(self) -> None:
        """Write current config to TOML."""
        path = config_path()
        path.write_text(tomli_w.dumps(self.model_dump()))
