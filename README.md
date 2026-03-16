# LLM Forge

A fully self-contained terminal-based LLM developer toolkit. Run local models, profile performance, compare outputs, track experiments — all from the terminal. No Electron, no browser, no external apps required.

```
┌─────────────────────────────────┬────────────────────┐
│ Messages                        │ Live Profiler      │
│                                 │                    │
│ [you] explain GGUF quantization │ t/s  ████████ 42.3 │
│                                 │ TTFT 180ms         │
│ [model] GGUF (GPT-Generated    │                    │
│ Unified Format) stores...       │ GPU  ██████░░ 73%  │
│                                 │ RAM  █████░░░ 9GB  │
│                                 │                    │
│                                 │ ┌─ t/s over time ─┐│
│                                 │ │   ╭╮  ╭─╮       ││
│                                 │ │  ╭╯╰──╯ ╰─      ││
│                                 │ └─────────────────┘│
├─────────────────────────────────┴────────────────────┤
│ Model: qwen2.5:7b  temp:0.7  ctx:4096  AGENT  WEB   │
├──────────────────────────────────────────────────────┤
│ > _                                                  │
└──────────────────────────────────────────────────────┘
```

## Quick Start

**Zero-dependency mode** (native inference, no Ollama needed):

```bash
pip install -e '.[all]'
llmforge download --popular          # pick a model to download
llmforge -b llamacpp chat ~/.llmforge/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

**With Ollama** (if you already have it):

```bash
pip install -e .
llmforge chat llama3.2:3b
```

## Features

### Core
- **Interactive Chat** with live streaming Markdown rendering
- **6 Backends** — Ollama, OpenAI-compatible, Anthropic Claude, Google Gemini, OpenRouter (100+ cloud models), native llama.cpp
- **Native GGUF Inference** — run models directly via llama-cpp-python, Metal auto-enabled on Apple Silicon
- **Model Downloads** — download GGUF models from HuggingFace, no Ollama registry needed
- **OpenAI-Compatible API Server** — `llmforge serve` exposes your models to other apps

### Developer Tools
- **Live Hardware Profiler** — real-time GPU/CPU/RAM gauges, tokens/sec sparklines, TTFT display
- **A/B Model Comparison** — side-by-side streaming with per-column metrics
- **Parameter Sweep** — automated temperature sweep with scoring
- **Experiment Tracker** — every run logged with TTFT, t/s, latency, hardware snapshots
- **Quality Scoring** — BLEU, ROUGE-L, and LLM-as-judge (in-process, no sidecar)

### LM Studio-Style Controls
- **Parameter Tuning Panel** — Temperature, Top P, Top K, Max Tokens, Context Length, Repeat Penalty, Seed
- **Presets** — Creative (1.2), Balanced (0.7), Precise (0.2), Code (0.1)
- **Model Library** — browse, filter, pull/delete models, RAM fit indicators (green/yellow/red)

### RAG & Knowledge
- **Document RAG** — ingest files, chunk, embed, cosine similarity search
- **Built-in Embeddings** — auto-fallback: Ollama → llama.cpp → TF-IDF (always works)
- **Re-ranking** — BM25 keyword + LLM-based relevance scoring for better retrieval
- **Folder Watch** — point at directories, auto-ingest new/modified files (like GPT4All LocalDocs)
- **Web Search RAG** — DuckDuckGo, SearXNG, Tavily — inject live web results as context

### Agents & Tools
- **Agent Mode** — plan/execute loop with built-in tools + MCP
- **Code Interpreter** — 9 languages: Python, JavaScript, TypeScript, Bash, Ruby, Go, Rust, C, C++
- **MCP Tool Calling** — JSON-RPC 2.0 stdio protocol for external tool servers
- **Web Search Tool** — available to agents for live information retrieval

### Workflow
- **Prompt Template Library** — create, edit, version templates with `{{variable}}` substitution
- **Chat Branching** — fork conversations from any point
- **Session Management** — persist, resume, export chats as Markdown
- **Workspaces** — organize sessions and templates
- **@file References** — `@path/to/file` inlines file contents in prompts
- **Pipe Mode** — `echo "question" | llmforge chat model` for scripting/CI
- **Command Palette** — `Ctrl+P` for quick actions

## Commands

```
llmforge                                # Open model picker
llmforge chat <model>                   # Interactive chat with live profiler
llmforge chat <model> --temp 0.3        # Chat with custom temperature
llmforge compare <m1> <m2> [m3] [m4]   # Side-by-side model comparison
llmforge models                         # Browse model library
llmforge exp                            # View experiment history
llmforge sessions                       # Browse/resume past chats
llmforge sweep <model>                  # Parameter sweep across temperatures
llmforge score <model> "prompt"         # Score a response for quality
llmforge run <model> "prompt"           # Non-interactive single inference
llmforge serve --port 8000              # Start OpenAI-compatible API server
llmforge download --popular             # Download popular GGUF models
llmforge download --list                # List local GGUF models
llmforge download "llama 3.2"           # Search HuggingFace for models
llmforge ingest docs/ README.md         # Ingest documents for RAG
llmforge ingest --list                  # List ingested documents
llmforge config --show                  # View current configuration
llmforge config --set backend llamacpp  # Change settings
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Escape` | Cancel generation |
| `Ctrl+P` | Command palette |
| `Ctrl+M` | Switch model |
| `Ctrl+T` | Toggle parameter panel |
| `Ctrl+R` | Insert prompt template |
| `Ctrl+A` | Toggle agent mode |
| `Ctrl+W` | Toggle web search |
| `Ctrl+F` | Fork conversation |
| `Ctrl+E` | Export chat |
| `Ctrl+L` | Clear chat |
| `1-4` | Parameter presets (Creative/Balanced/Precise/Code) |

## Backends

| Backend | Config | What it connects to |
|---------|--------|---------------------|
| `llamacpp` | `-b llamacpp` | **Native GGUF inference** — nothing else needed |
| `ollama` | `-b ollama` (default) | Ollama daemon |
| `openai-compat` | `-b openai-compat` | LM Studio, vLLM, llama.cpp server |
| `anthropic` | `-b anthropic` | Claude API |
| `google` | `-b google` | Gemini API |
| `openrouter` | `-b openrouter` | 100+ cloud models via single API key |

## Installation

```bash
# Full install (all features)
pip install -e '.[all]'

# Minimal install (Ollama backend only)
pip install -e .

# Specific extras
pip install -e '.[llamacpp]'    # Native GGUF inference
pip install -e '.[scoring]'    # BLEU/ROUGE scoring
pip install -e '.[rag]'        # PDF ingestion for RAG
pip install -e '.[server]'     # API server mode
```

### Requirements

- Python 3.11+
- **No external apps required** with `llamacpp` backend
- [Ollama](https://ollama.com) only if using the `ollama` backend

## Configuration

Config file: `~/.llmforge/config.toml`

```toml
backend = "llamacpp"  # or "ollama", "anthropic", "google", "openrouter"

[llamacpp]
model_dirs = ["~/.llmforge/models", "~/Downloads"]
n_gpu_layers = -1  # -1 = all layers on GPU
context_length = 4096

[ollama]
base_url = "http://localhost:11434"

[anthropic]
api_key = ""  # or set LLMFORGE_ANTHROPIC_KEY env var

[google]
api_key = ""  # or set LLMFORGE_GOOGLE_KEY env var

[openrouter]
api_key = ""  # or set LLMFORGE_OPENROUTER_KEY env var

[generation]
temperature = 0.7
top_p = 0.9
top_k = 40
max_tokens = 2048
context_length = 4096
repeat_penalty = 1.1

[rag]
enabled = true
embedding_method = "auto"  # "auto" | "ollama" | "llamacpp" | "tfidf"
rerank = true
watch_dirs = ["~/Documents/notes"]  # auto-ingest folders
chunk_size = 512
top_k = 3

[web_search]
enabled = false
provider = "duckduckgo"  # "duckduckgo" | "searxng" | "tavily"

[mcp]
[[mcp.servers]]
name = "filesystem"
command = ["npx", "-y", "@anthropic/mcp-filesystem"]

[scoring]
enabled = true
judge_model = "llama3.2:3b"

[profiler]
poll_interval_ms = 200
sparkline_width = 60
```

### Environment Variables

```bash
export LLMFORGE_OLLAMA_URL="http://localhost:11434"
export LLMFORGE_ANTHROPIC_KEY="sk-ant-..."
export LLMFORGE_GOOGLE_KEY="AIza..."
export LLMFORGE_OPENROUTER_KEY="sk-or-..."
export LLMFORGE_TAVILY_KEY="tvly-..."
```

## Architecture

```
llmforge/
├── backends/          # 6 inference backends
│   ├── ollama.py      # Ollama NDJSON streaming
│   ├── llamacpp.py    # Native GGUF via llama-cpp-python
│   ├── openai_compat.py  # LM Studio, vLLM, etc.
│   ├── anthropic.py   # Claude SSE streaming
│   ├── google.py      # Gemini SSE streaming
│   └── openrouter.py  # 100+ cloud models
├── models/            # Model management
│   └── downloader.py  # HuggingFace GGUF downloads
├── rag/               # RAG pipeline
│   ├── chunker.py     # Document chunking
│   ├── embeddings.py  # Multi-backend embeddings (Ollama/llama.cpp/TF-IDF)
│   ├── store.py       # SQLite vector store
│   ├── reranker.py    # BM25 + LLM re-ranking
│   ├── watcher.py     # Folder auto-sync
│   ├── context.py     # Context builder
│   └── web_search.py  # DuckDuckGo/SearXNG/Tavily
├── tools/             # Agent tools
│   ├── agent.py       # Plan/execute loop
│   └── code_exec.py   # Multi-language sandbox (9 langs)
├── mcp/               # Model Context Protocol
│   ├── client.py      # JSON-RPC 2.0 stdio client
│   ├── types.py       # Tool definitions
│   └── tool_loop.py   # Iterative tool calling
├── server/            # API server
│   └── app.py         # OpenAI-compatible endpoints
├── ui/                # Textual TUI
│   ├── app.py         # Main app + command palette
│   ├── screens/       # Chat, Compare, Models, Experiments, Sessions, Templates, Sweep
│   └── widgets/       # Profiler, Parameter panel
├── domain/            # Core types
│   ├── models.py      # ModelDescriptor, InferenceRequest, TokenChunk
│   ├── profiler.py    # TTFT, t/s, context tracking
│   └── hardware.py    # CPU/RAM/GPU monitoring
├── storage/           # Persistence
│   └── db.py          # SQLite with schema migrations
├── scoring/           # Quality metrics
├── config.py          # TOML config + env overrides
└── cli.py             # Click CLI (12 commands)
```

## Data

All data stored in `~/.llmforge/`:

```
~/.llmforge/
├── config.toml        # Configuration
├── data.sqlite        # Sessions, runs, templates, experiments
├── models/            # Downloaded GGUF models
└── exports/           # Exported chat Markdown files
```

## License

**PolyForm Noncommercial 1.0.0** — Source available, free for everyone except commercial use.

- Personal use — free
- Education & research — free
- Non-profits — free
- **Companies / commercial use — not permitted**

See [LICENSE](LICENSE) for full terms.
