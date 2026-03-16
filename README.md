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
- **Interactive Chat** with live streaming Markdown rendering and multi-line input
- **6 Backends** — Ollama, OpenAI-compatible, Anthropic Claude, Google Gemini, OpenRouter (100+ cloud models), native llama.cpp
- **Native GGUF Inference** — run models directly via llama-cpp-python, Metal auto-enabled on Apple Silicon
- **Speculative Decoding** — prompt-lookup (free speedup) or draft-model mode for up to ~2x faster generation
- **Model Downloads** — download GGUF models from HuggingFace, no Ollama registry needed
- **OpenAI-Compatible API Server** — `llmforge serve` exposes your models to other apps
- **Code Block Copy** — click `[Copy]` on any code block to copy to clipboard

### Developer Tools
- **Live Hardware Profiler** — real-time GPU/CPU/RAM gauges, tokens/sec sparklines, TTFT display
- **A/B Model Comparison** — side-by-side streaming with per-column metrics
- **Parameter Sweep** — automated temperature sweep with scoring
- **Experiment Tracker** — every run logged with TTFT, t/s, latency, hardware snapshots
- **Quality Scoring** — BLEU, ROUGE-L, and LLM-as-judge (in-process, no sidecar)

### LM Studio-Style Controls
- **Full Parameter Panel** — Temperature, Top P, Top K, Min P, Max Tokens, Context Length, Repeat Penalty, Frequency Penalty, Presence Penalty, Stop Strings, Seed
- **Presets** — Creative (1.2), Balanced (0.7), Precise (0.2), Code (0.1)
- **Model Library** — browse, filter, pull/delete models, RAM fit indicators (green/yellow/red)
- **Per-Backend Parameters** — each backend only receives the parameters it supports

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
- **Multi-line Input** — paste entire functions; Enter sends, Shift+Enter for newline
- **Pipe Mode** — `echo "question" | llmforge chat model` for scripting/CI
- **Command Palette** — `Ctrl+P` for quick actions

---

## How to Access Each Feature

| Feature | How to activate | What you need |
|---------|----------------|---------------|
| **Chat** | `llmforge chat <model>` | A model (Ollama, GGUF, or cloud API key) |
| **Live Profiler** | Always visible in chat (right panel) | Nothing — built-in |
| **Parameter Panel** | `Ctrl+T` during chat | Nothing — built-in |
| **Presets** | Press `1` `2` `3` `4` during chat | Nothing — built-in |
| **RAG** | `llmforge ingest <files>`, then chat normally | `rag.enabled = true` in config |
| **Folder Watch** | Set `rag.watch_dirs` in config, then start chat | `watchfiles` package (included) |
| **Web Search** | `Ctrl+W` during chat | Nothing for DuckDuckGo; API key for Tavily |
| **Agent Mode** | `Ctrl+A` during chat | A model that follows tool-call instructions |
| **Code Interpreter** | Automatic in agent mode when model calls `run_code` | Language runtime installed (e.g. `python3`, `node`) |
| **MCP Tools** | Add `[[mcp.servers]]` to config, tools auto-load | MCP server package (e.g. `npx @anthropic/mcp-filesystem`) |
| **A/B Compare** | `llmforge compare model1 model2` | Two or more models |
| **Experiments** | `llmforge exp` | Runs are auto-logged from any chat/run |
| **Scoring** | `llmforge score <model> "prompt"` | `pip install 'llmforge[scoring]'` for BLEU/ROUGE |
| **Prompt Templates** | `Ctrl+R` during chat | Create templates in the template browser |
| **Chat Branching** | `Ctrl+F` during chat | An active conversation |
| **Session Resume** | `llmforge sessions` | Previous chat sessions (auto-saved) |
| **Model Library** | `llmforge models` | Ollama running, or GGUF files in model_dirs |
| **Model Download** | `llmforge download --popular` | Internet connection |
| **API Server** | `llmforge serve --port 8000` | `pip install 'llmforge[server]'` |
| **Export Chat** | `Ctrl+E` during chat | An active conversation |
| **@file References** | Type `@path/to/file` in chat input | File exists on disk |
| **Pipe Mode** | `echo "question" \| llmforge chat model` | Nothing — built-in |
| **Config** | `llmforge config --show` or `llmforge config --set key value` | Nothing — built-in |
| **Command Palette** | `Ctrl+P` during any TUI screen | Nothing — built-in |

### Quick Setup Recipes

**Minimal (just chat, no external dependencies):**
```bash
pip install -e '.[llamacpp]'
llmforge download --popular                    # download a model
llmforge -b llamacpp chat ~/.llmforge/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

**Chat + RAG over your docs:**
```bash
pip install -e '.[all]'
llmforge config --set rag.enabled true
llmforge ingest ~/Documents/notes/ ~/projects/docs/
llmforge chat llama3.2:3b
# Now ask questions about your documents
```

**Chat + Web Search:**
```bash
llmforge chat llama3.2:3b
# Press Ctrl+W to enable web search
# Ask about current events — results are fetched from DuckDuckGo
```

**Agent with Code Execution:**
```bash
llmforge chat llama3.2:3b
# Press Ctrl+A to enable agent mode
# Ask it to write and run code — it executes, sees output, iterates
```

**Agent + MCP (filesystem access):**
```bash
# Add to ~/.llmforge/config.toml:
# [[mcp.servers]]
# name = "filesystem"
# command = ["npx", "-y", "@anthropic/mcp-filesystem", "/path/to/dir"]

llmforge chat llama3.2:3b
# Press Ctrl+A for agent mode
# Ask it to read/write files — it uses the MCP filesystem tool
```

**API Server for external apps:**
```bash
pip install -e '.[server]'
llmforge serve --port 8000
# Point VS Code Continue, Open WebUI, or scripts at http://localhost:8000/v1
```

**A/B Model Comparison:**
```bash
llmforge compare llama3.2:3b qwen2.5:7b phi4:14b
# Same prompt sent to all models, streamed side-by-side with metrics
```

**Auto-ingest folders (like GPT4All LocalDocs):**
```bash
llmforge config --set rag.enabled true
llmforge config --set rag.watch_dirs '["~/Documents/notes", "~/projects/docs"]'
llmforge chat llama3.2:3b
# Files are auto-ingested on startup and when changed
```

---

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
llmforge ingest --delete 3              # Remove a document from RAG store
llmforge config --show                  # View current configuration
llmforge config --set backend llamacpp  # Change settings
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift+Enter` | Insert newline (multi-line input) |
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
| `Ctrl+U` | Clear input |
| `1-4` | Parameter presets (Creative/Balanced/Precise/Code) |

---

## RAG (Retrieval-Augmented Generation)

RAG lets the model answer questions about your documents. LLM Forge handles the entire pipeline: ingest files → chunk into passages → embed into vectors → store in SQLite → retrieve relevant chunks at query time → re-rank → inject into the prompt.

### How It Works

```
Your documents                          Your question
     │                                       │
     ▼                                       ▼
 ┌────────┐                            ┌──────────┐
 │ Ingest │                            │  Embed   │
 │ & Chunk│                            │  query   │
 └───┬────┘                            └────┬─────┘
     │                                      │
     ▼                                      ▼
 ┌────────┐    cosine similarity      ┌──────────┐
 │ Embed  │◄──────────────────────────│  Search  │
 │ chunks │                           │  store   │
 └───┬────┘                           └────┬─────┘
     │                                      │
     ▼                                      ▼
 ┌────────┐                           ┌──────────┐
 │ SQLite │                           │ Re-rank  │
 │ store  │                           │(BM25/LLM)│
 └────────┘                           └────┬─────┘
                                            │
                                            ▼
                                      ┌──────────┐
                                      │ Inject   │
                                      │ context  │
                                      │ into     │
                                      │ prompt   │
                                      └──────────┘
```

### Getting Started with RAG

**1. Enable RAG in config:**

```toml
[rag]
enabled = true
```

**2. Ingest your documents:**

```bash
# Ingest individual files
llmforge ingest README.md notes.txt api_docs.pdf

# Ingest entire directories (recursive)
llmforge ingest docs/ src/

# Check what's been ingested
llmforge ingest --list

# Remove a document
llmforge ingest --delete 3
```

**3. Chat — RAG context is automatically included:**

```bash
llmforge chat llama3.2:3b
> What does the API authentication flow look like?
# The model answers using content from your ingested docs
```

### Supported File Types

`.pdf`, `.md`, `.txt`, `.rst`, `.csv`, `.json`, `.yaml`, `.toml`, `.xml`, `.html`, `.log`, `.py`, `.js`, `.ts`, `.rs`, `.go`, `.java`, `.c`, `.cpp`, `.h`, `.rb`, `.php`, `.swift`, `.sh`, `.bash`, `.sql`, `.r`, `.tex`, `.org`

PDF support requires the `rag` extra: `pip install -e '.[rag]'`

### Embedding Backends

LLM Forge tries three embedding methods in order, so RAG always works regardless of your setup:

| Priority | Method | Requirements | Quality |
|----------|--------|-------------|---------|
| 1st | **Ollama** | Ollama running + `nomic-embed-text` model | Best |
| 2nd | **llama.cpp** | `pip install 'llmforge[llamacpp]'` + a GGUF embedding model | Good |
| 3rd | **TF-IDF** | Nothing (built-in) | Basic but functional |

Force a specific method:

```toml
[rag]
embedding_method = "tfidf"  # "auto" | "ollama" | "llamacpp" | "tfidf"
```

The TF-IDF fallback uses feature hashing to produce 384-dimensional vectors — no external model needed. It works well for keyword-heavy queries and ensures RAG is available even on machines with no GPU or Ollama.

### Re-ranking

Raw vector search returns chunks sorted by cosine similarity. Re-ranking improves precision by applying a second scoring pass:

| Method | How it works | When it's used |
|--------|-------------|----------------|
| **LLM re-ranking** | Asks a model to rate each chunk's relevance 0-10 | Default if Ollama is available |
| **BM25 keyword** | Classic term-frequency scoring blended with vector score | Fallback if no LLM available |

The final score is a weighted blend:
- LLM mode: 30% vector score + 70% LLM relevance score
- Keyword mode: 40% vector score + 60% BM25 score

Configure in `config.toml`:

```toml
[rag]
rerank = true                 # enable re-ranking (default: true)
rerank_model = "llama3.2:3b"  # model for LLM re-ranking
top_k = 3                     # number of chunks injected into prompt
chunk_size = 512              # tokens per chunk (smaller = more precise)
overlap = 64                  # overlap between chunks (preserves context)
```

### Folder Watch (Auto-Ingest)

Like GPT4All's LocalDocs — point at directories and LLM Forge auto-ingests new and modified files:

```toml
[rag]
watch_dirs = ["~/Documents/notes", "~/projects/docs"]
```

On startup, the watcher:
1. Scans all directories recursively
2. Ingests any files not already in the store
3. Watches for file changes (create, modify) via `watchfiles`
4. Auto-ingests changed files in the background

### Tuning RAG Quality

| Goal | Setting |
|------|---------|
| More precise retrieval | Decrease `chunk_size` to 256, increase `top_k` to 5 |
| Broader context | Increase `chunk_size` to 1024 |
| Faster (skip re-ranking) | Set `rerank = false` |
| Better quality (slower) | Use LLM re-ranking with a larger model |
| Zero-dependency RAG | Set `embedding_method = "tfidf"`, `rerank = false` |

---

## Web Search

Inject live web results as context for your prompts — the model can answer questions about current events, recent documentation, etc.

### How It Works

When web search is enabled, LLM Forge:
1. Takes your message as a search query
2. Fetches results from the configured provider
3. Formats them as context (title + URL + snippet)
4. Prepends the context to the model's system prompt
5. The model answers using both its training data and the live results

### Providers

| Provider | API Key? | Setup | Best for |
|----------|----------|-------|----------|
| **DuckDuckGo** | No | Nothing | Quick use, no signup |
| **SearXNG** | No | Self-host instance | Privacy, customization |
| **Tavily** | Yes | Get key at tavily.com | AI-optimized results |

### Usage

**Toggle in chat:** Press `Ctrl+W` to enable/disable web search. A `WEB` badge appears in the status bar when active.

**Configure provider:**

```toml
[web_search]
enabled = false             # toggle default state
provider = "duckduckgo"     # "duckduckgo" | "searxng" | "tavily"
max_results = 5             # number of results to fetch

# SearXNG (self-hosted)
# searxng_url = "http://localhost:8080"

# Tavily (API key required)
# tavily_api_key = "tvly-..."  # or set LLMFORGE_TAVILY_KEY env var
```

**Example session:**

```
[WEB enabled]
> What happened at WWDC 2025?

# LLM Forge searches DuckDuckGo, fetches 5 results,
# injects them as context, and the model summarizes the findings
```

Web search is also available as a tool in Agent Mode — the agent can decide to search the web when it needs current information.

---

## Agent Mode

Agent mode gives the model the ability to plan, execute tools, observe results, and iterate — turning a chat model into an autonomous agent.

### How It Works

```
You ask a question
       │
       ▼
┌─────────────┐
│ Model thinks │──► "I need to run some code to verify this"
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Tool call:  │──► run_code(language="python", code="print(2**32)")
│ run_code    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Tool result │──► "4294967296"
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Model uses  │──► "2^32 is 4,294,967,296"
│ result      │
└─────────────┘
```

The agent loop runs for up to 10 rounds. Each round: generate → detect tool calls → execute → feed results back.

### Built-in Tools

| Tool | What it does |
|------|-------------|
| **run_code** | Execute code in 9 languages with output capture |
| **web_search** | Search the web using the configured provider |

### Usage

Press `Ctrl+A` in chat to toggle agent mode. An `AGENT` badge appears in the status bar.

```
[AGENT enabled]
> Write a Python script that downloads the top 10 HackerNews stories and save it

# The agent:
# 1. Writes the Python code
# 2. Calls run_code to test it
# 3. Observes the output
# 4. Fixes any errors and re-runs
# 5. Presents the final working script
```

### Tool Call Format

The model emits tool calls as structured text:

```
<tool_call>{"name": "run_code", "arguments": {"language": "python", "code": "print('hello')"}}</tool_call>
```

LLM Forge detects these, executes the tool, and feeds the result back as a message. This works with any model that can follow the tool call format in its system prompt.

### Agent + MCP

When MCP servers are configured, the agent has access to both built-in tools AND external MCP tools. The model sees all available tools and can call any of them.

---

## Code Interpreter

The code interpreter executes code in a sandboxed temp directory with output capture, timeouts, and auto-language detection.

### Supported Languages

| Language | Runtime | Detection patterns |
|----------|---------|-------------------|
| Python | `python3` | `import`, `def`, `print(` |
| JavaScript | `node` | `console.log`, `const`, `require(` |
| TypeScript | `npx tsx` | `.ts` indicators, TypeScript syntax |
| Bash | `bash` | `#!/bin/bash`, shell commands |
| Ruby | `ruby` | `puts`, `end` |
| Go | `go run` | `package main`, `func main()` |
| Rust | `rustc` + run | `fn main()`, `println!`, `let mut` |
| C | `cc` + run | `#include`, `int main` |
| C++ | `c++` + run | `iostream`, `std::`, `cout` |

### How Language Detection Works

If you don't specify a language, LLM Forge auto-detects from the code content:

1. Check for shebang lines (`#!/usr/bin/env python`)
2. Check for Go patterns (`package main`)
3. Check for C/C++ patterns (`#include`, `iostream`)
4. Check for Rust patterns (`fn main()`, `println!`)
5. Check for JavaScript patterns (`console.log`)
6. Check for Ruby patterns (`puts`)
7. Default: Python

### Sandbox Features

- Runs in an isolated temporary directory
- Restricted `PATH` and `HOME` environment
- Configurable timeout (default 30 seconds)
- Output truncated at 10,000 characters
- Compiled languages (Go, Rust, C, C++) are compiled and run in one step

### Check Available Languages

```bash
# The agent automatically checks which runtimes are installed
# Only languages with available runtimes are offered to the model
```

---

## MCP (Model Context Protocol)

MCP lets external tool servers plug into LLM Forge. Any MCP-compatible server (filesystem access, database queries, API calls, etc.) can provide tools to your model.

### How It Works

```
LLM Forge                              MCP Server (subprocess)
    │                                        │
    │──── initialize (JSON-RPC 2.0) ────────►│
    │◄─── capabilities + protocol version ───│
    │                                        │
    │──── tools/list ───────────────────────►│
    │◄─── [list of available tools] ─────────│
    │                                        │
    │  ... during chat ...                   │
    │                                        │
    │──── tools/call {name, args} ──────────►│
    │◄─── {result} ─────────────────────────│
    │                                        │
    │──── shutdown ─────────────────────────►│
    └                                        └
```

LLM Forge launches MCP servers as subprocesses and communicates over stdio using JSON-RPC 2.0 (protocol version `2024-11-05`).

### Configuring MCP Servers

Add servers to `config.toml`:

```toml
[mcp]

# Filesystem access — read/write files, list directories
[[mcp.servers]]
name = "filesystem"
command = ["npx", "-y", "@anthropic/mcp-filesystem", "/home/user/projects"]

# GitHub — create issues, read PRs, search repos
[[mcp.servers]]
name = "github"
command = ["npx", "-y", "@modelcontextprotocol/server-github"]
[mcp.servers.env]
GITHUB_TOKEN = "ghp_..."

# SQLite — query databases
[[mcp.servers]]
name = "sqlite"
command = ["npx", "-y", "@modelcontextprotocol/server-sqlite", "mydb.sqlite"]

# Custom server — any MCP-compatible executable
[[mcp.servers]]
name = "my-tools"
command = ["python", "my_mcp_server.py"]
[mcp.servers.env]
API_KEY = "..."
```

### How Tools Are Used

Once configured, MCP tools are available in two ways:

1. **Agent mode** (`Ctrl+A`): The agent sees all MCP tools alongside built-in tools and can call any of them autonomously
2. **Tool-calling mode**: If MCP servers are connected but agent mode is off, LLM Forge runs a tool-calling loop — the model can request tool calls that get executed and fed back

**Example — filesystem MCP:**

```
> Read the file src/main.py and explain what it does

# Model calls MCP tool: read_file(path="src/main.py")
# MCP server reads the file and returns its contents
# Model explains the code
```

**Example — GitHub MCP:**

```
[AGENT enabled]
> Create a GitHub issue in my repo about the login bug we discussed

# Agent calls: create_issue(repo="user/repo", title="Login bug", body="...")
```

### Multiple Servers

You can connect multiple MCP servers simultaneously. Each server's tools are namespaced by the server name to avoid conflicts. The model sees all tools from all connected servers.

### Writing Your Own MCP Server

Any program that reads JSON-RPC 2.0 from stdin and writes responses to stdout can be an MCP server. See the [MCP specification](https://modelcontextprotocol.io) for the protocol details.

Minimal Python example:

```python
import json
import sys

def handle(request):
    if request["method"] == "initialize":
        return {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}}
    elif request["method"] == "tools/list":
        return {"tools": [{"name": "greet", "description": "Say hello", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}}}]}
    elif request["method"] == "tools/call":
        name = request["params"]["arguments"]["name"]
        return {"content": [{"type": "text", "text": f"Hello, {name}!"}]}

for line in sys.stdin:
    req = json.loads(line)
    result = handle(req)
    response = {"jsonrpc": "2.0", "id": req.get("id"), "result": result}
    print(json.dumps(response), flush=True)
```

---

## API Server

`llmforge serve` exposes an OpenAI-compatible REST API, so external apps (VS Code extensions, web UIs, scripts, other tools) can use LLM Forge as a backend.

### Starting the Server

```bash
# Default: localhost:8000
llmforge serve

# Custom host/port
llmforge serve --port 3000 --host 0.0.0.0

# With a specific backend
llmforge -b llamacpp serve --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat completion (streaming or non-streaming) |

### Usage Examples

**curl — non-streaming:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

**curl — streaming (SSE):**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**Python (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

**JavaScript (fetch):**

```javascript
const response = await fetch("http://localhost:8000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "llama3.2:3b",
    messages: [{ role: "user", content: "Hello!" }],
  }),
});
const data = await response.json();
console.log(data.choices[0].message.content);
```

### Supported Parameters

The `/v1/chat/completions` endpoint accepts:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model ID |
| `messages` | array | required | Chat messages (`role` + `content`) |
| `stream` | bool | false | Enable SSE streaming |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 0.9 | Nucleus sampling threshold |
| `max_tokens` | int | 2048 | Max response tokens |
| `seed` | int | null | Reproducible outputs |

### Use Cases

- **VS Code Continue extension** — point it at `http://localhost:8000/v1`
- **Open WebUI** — use as an OpenAI-compatible backend
- **Custom scripts** — automate with any HTTP client
- **CI/CD pipelines** — run model evaluations in automation

---

## Scoring & Experiments

Every inference run is automatically logged to the experiment tracker with performance metrics.

### Automatic Logging

Each run records:
- Model ID, prompt, response
- TTFT (time to first token), tokens/sec, total latency
- Hardware snapshot (CPU, GPU, RAM usage)
- Generation parameters used

### Quality Scoring

```bash
# Score a single response
llmforge score llama3.2:3b "Explain photosynthesis"

# Score with a reference answer (enables BLEU/ROUGE)
llmforge score llama3.2:3b "Explain photosynthesis" --reference "Photosynthesis is..."
```

| Metric | What it measures | When available |
|--------|-----------------|----------------|
| **BLEU** | N-gram overlap with reference | With `--reference` |
| **ROUGE-L** | Longest common subsequence with reference | With `--reference` |
| **LLM-as-judge** | Quality rating 0-10 from a judge model | Always (uses `scoring.judge_model`) |

### Experiment Browser

```bash
# TUI experiment browser
llmforge exp

# Shows: model, t/s, TTFT, tokens, date
# Select a run to see full details: prompt, response, hardware, scores, parameters
```

### Parameter Sweep

```bash
# Sweep temperature from 0.1 to 1.5 in steps of 0.2
llmforge sweep llama3.2:3b
```

Runs the same prompt at multiple temperature values and scores each output, helping find the sweet spot for your use case.

---

## Model Management

### Download GGUF Models

Download models directly from HuggingFace — no Ollama registry needed:

```bash
# Browse popular pre-selected models (Llama, Qwen, Phi, Gemma, Mistral, DeepSeek)
llmforge download --popular

# Search HuggingFace
llmforge download "llama 3.2"
llmforge download "qwen 7b gguf"

# List already-downloaded models
llmforge download --list
```

Models are saved to `~/.llmforge/models/`.

### Model Library Browser

```bash
llmforge models
```

Interactive TUI with:
- All models across all backends (Ollama, GGUF, cloud)
- Size, quantization, parameter count
- RAM fit indicators: green (fits), yellow (tight), red (won't fit)
- `p` to pull a new Ollama model
- `Delete` to remove a model

### @file References

In chat, prefix any file path with `@` to inline its contents:

```
> Explain this code: @src/main.py

# The file contents are expanded inline before sending to the model
# Works with any file type, truncated at 50k chars
```

---

## Backends

| Backend | Config | What it connects to |
|---------|--------|---------------------|
| `llamacpp` | `-b llamacpp` | **Native GGUF inference** — nothing else needed |
| `ollama` | `-b ollama` (default) | Ollama daemon |
| `openai-compat` | `-b openai-compat` | LM Studio, vLLM, llama.cpp server |
| `anthropic` | `-b anthropic` | Claude API |
| `google` | `-b google` | Gemini API |
| `openrouter` | `-b openrouter` | 100+ cloud models via single API key |

### Parameters by Backend

Not all parameters are supported by all backends. LLM Forge automatically sends only the parameters each backend understands:

| Parameter | Ollama | LlamaCpp | OpenAI-compat | Anthropic | Google | OpenRouter |
|-----------|:------:|:--------:|:-------------:|:---------:|:------:|:----------:|
| temperature | yes | yes | yes | yes | yes | yes |
| top_p | yes | yes | yes | yes | yes | yes |
| top_k | yes | yes | — | yes | yes | yes |
| min_p | yes | yes | — | — | — | — |
| max_tokens | yes | yes | yes | yes | yes | yes |
| repeat_penalty | yes | yes | — | — | — | — |
| frequency_penalty | yes | yes | yes | — | yes | yes |
| presence_penalty | yes | yes | yes | — | yes | yes |
| stop_strings | yes | yes | yes | yes | yes | yes |
| seed | yes | yes | yes | — | yes | yes |
| flash_attention | — | yes | — | — | — | — |
| speculative decoding | — | yes | — | — | — | — |
| rope_freq_base/scale | — | yes | — | — | — | — |
| eval_batch_size | — | yes | — | — | — | — |
| use_mmap / use_mlock | — | yes | — | — | — | — |
| fp16_kv_cache | — | yes | — | — | — | — |
| num_experts (MoE) | — | yes | — | — | — | — |
| cpu_threads | — | yes | — | — | — | — |

---

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

---

## Configuration

Config file: `~/.llmforge/config.toml`

```toml
backend = "llamacpp"  # or "ollama", "openai-compat", "anthropic", "google", "openrouter"

# ── Native GGUF Backend ─────────────────────────────────────────────────────

[llamacpp]
model_dirs = ["~/.llmforge/models", "~/Downloads"]
n_gpu_layers = -1           # -1 = all layers on GPU (Metal on macOS)
context_length = 4096

# Performance tuning
flash_attention = false     # enable for faster attention (supported models only)
eval_batch_size = 512       # tokens processed per batch (higher = faster prompt eval)
cpu_threads = 0             # 0 = auto-detect

# Memory & loading
use_mmap = true             # memory-mapped file access (faster loading)
use_mlock = false           # lock model in RAM (prevents swapping)
use_fp16_kv = true          # half-precision KV cache (saves ~50% KV memory)

# Context extension (RoPE)
rope_freq_base = 0.0        # 0 = use model default
rope_freq_scale = 0.0       # 0 = use model default

# Mixture of Experts
# num_experts = 2           # only for MoE models (e.g. Mixtral) — uncomment to set

# Speculative decoding — up to ~2x faster generation
speculative = "off"                   # "off" | "prompt-lookup" | "draft-model"
speculative_num_tokens = 10           # tokens to predict ahead (10 for GPU, 2 for CPU)
speculative_draft_model = ""          # path to smaller GGUF (draft-model mode only)
# Example:
# speculative = "prompt-lookup"       # free speedup, no extra model needed
# speculative = "draft-model"         # use a small model (e.g. 1B) to draft tokens
# speculative_draft_model = "~/.llmforge/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# ── Backend Connections ──────────────────────────────────────────────────────

[ollama]
base_url = "http://localhost:11434"

[openai-compat]
base_url = "http://localhost:1234/v1"  # LM Studio, vLLM, etc.
api_key = "not-needed"

[anthropic]
api_key = ""  # or set LLMFORGE_ANTHROPIC_KEY env var

[google]
api_key = ""  # or set LLMFORGE_GOOGLE_KEY env var

[openrouter]
api_key = ""  # or set LLMFORGE_OPENROUTER_KEY env var

# ── Generation Defaults ──────────────────────────────────────────────────────
# These can be overridden per-session via the parameter panel (Ctrl+T)

[generation]
temperature = 0.7           # randomness (0.0 = deterministic, 2.0 = very random)
top_p = 0.9                 # nucleus sampling threshold
top_k = 40                  # limit to top K most likely tokens
min_p = 0.0                 # minimum probability threshold (0 = disabled)
max_tokens = 2048           # maximum response length
context_length = 4096       # context window size
repeat_penalty = 1.1        # penalize repeated tokens (1.0 = off)
frequency_penalty = 0.0     # OpenAI-style: penalize by frequency of appearance
presence_penalty = 0.0      # OpenAI-style: penalize any repeated token equally
stop_strings = []           # stop generation when these strings appear
# seed = 42                 # uncomment for reproducible outputs

# ── RAG & Knowledge ──────────────────────────────────────────────────────────

[rag]
enabled = true
embedding_method = "auto"   # "auto" | "ollama" | "llamacpp" | "tfidf"
embedding_model = "nomic-embed-text"  # Ollama embedding model
rerank = true               # BM25 + LLM re-ranking for better retrieval
rerank_model = "llama3.2:3b"  # model for LLM-based re-ranking
chunk_size = 512            # tokens per chunk
overlap = 64                # overlap between chunks
top_k = 3                   # chunks injected into prompt
watch_dirs = []             # auto-ingest folders (e.g. ["~/Documents/notes"])

# ── Web Search ───────────────────────────────────────────────────────────────

[web_search]
enabled = false
provider = "duckduckgo"     # "duckduckgo" | "searxng" | "tavily"
max_results = 5
# searxng_url = "http://localhost:8080"
# tavily_api_key = ""       # or set LLMFORGE_TAVILY_KEY env var

# ── MCP Tool Servers ─────────────────────────────────────────────────────────

[mcp]
# Add MCP servers — each gets launched as a subprocess
# Tools from all servers are available to the model

# [[mcp.servers]]
# name = "filesystem"
# command = ["npx", "-y", "@anthropic/mcp-filesystem", "/path/to/dir"]

# [[mcp.servers]]
# name = "github"
# command = ["npx", "-y", "@modelcontextprotocol/server-github"]
# [mcp.servers.env]
# GITHUB_TOKEN = "ghp_..."

# ── Scoring ──────────────────────────────────────────────────────────────────

[scoring]
enabled = true
judge_model = "llama3.2:3b" # model used for LLM-as-judge scoring

# ── Profiler ─────────────────────────────────────────────────────────────────

[profiler]
poll_interval_ms = 200
sparkline_width = 60
```

### CLI Configuration

```bash
# View current config
llmforge config --show

# Change settings (supports dot notation for nested keys)
llmforge config --set backend llamacpp
llmforge config --set rag.enabled true
llmforge config --set rag.chunk_size 256
llmforge config --set anthropic.api_key sk-ant-...
llmforge config --set web_search.provider tavily
```

### Environment Variables

```bash
export LLMFORGE_OLLAMA_URL="http://localhost:11434"
export LLMFORGE_ANTHROPIC_KEY="sk-ant-..."
export LLMFORGE_GOOGLE_KEY="AIza..."
export LLMFORGE_OPENROUTER_KEY="sk-or-..."
export LLMFORGE_TAVILY_KEY="tvly-..."
```

### Speculative Decoding

Speculative decoding makes inference faster without changing output quality. Two modes are available for the `llamacpp` backend:

**Prompt Lookup** — free speedup, no extra model needed:
```toml
[llamacpp]
speculative = "prompt-lookup"
speculative_num_tokens = 10  # 10 for GPU, 2 for CPU
```
Predicts ahead by looking at tokens already in the prompt/context. Typically ~1.3-1.5x speedup.

**Draft Model** — uses a smaller GGUF as speculator:
```toml
[llamacpp]
speculative = "draft-model"
speculative_num_tokens = 10
speculative_draft_model = "~/.llmforge/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
```
Loads a small model (e.g. 1B) to draft candidate tokens, then the main model verifies them in a single forward pass. Up to ~1.8x speedup. Best results when draft and main models are from the same family (e.g. Llama 3.2 1B drafting for Llama 3.1 8B).

### Parameter Guide

| Parameter | Range | What it does |
|-----------|-------|--------------|
| **temperature** | 0.0 - 2.0 | Controls randomness. Lower = more deterministic, higher = more creative |
| **top_p** | 0.0 - 1.0 | Nucleus sampling — only consider tokens with cumulative probability up to this value |
| **top_k** | 1 - 100 | Only consider the K most likely next tokens |
| **min_p** | 0.0 - 1.0 | Discard tokens below this probability relative to the top token. Modern alternative to top_k |
| **max_tokens** | 64 - 32768 | Maximum number of tokens to generate in a response |
| **context_length** | 512 - 131072 | Size of the context window (prompt + response) |
| **repeat_penalty** | 1.0 - 2.0 | Penalize repeated tokens. 1.0 = off, >1.0 = less repetition. Used by Ollama and llama.cpp |
| **frequency_penalty** | 0.0 - 2.0 | OpenAI-style — penalty proportional to how often a token has appeared |
| **presence_penalty** | 0.0 - 2.0 | OpenAI-style — flat penalty for any repeated token regardless of count |
| **stop_strings** | list | Stop generation when any of these strings appear in the output |
| **seed** | integer | Fix the random seed for reproducible outputs |

> **repeat_penalty vs frequency/presence_penalty:** These serve similar purposes. `repeat_penalty` is the llama.cpp / Ollama convention (applied multiplicatively, default 1.1). `frequency_penalty` and `presence_penalty` are the OpenAI convention (applied additively, default 0). Use one style or the other — if you set `frequency_penalty`, the `repeat_penalty` is ignored for backends that use the OpenAI convention.

---

## Architecture

```
llmforge/
├── backends/          # 6 inference backends
│   ├── ollama.py      # Ollama NDJSON streaming
│   ├── llamacpp.py    # Native GGUF via llama-cpp-python + speculative decoding
│   ├── openai_compat.py  # LM Studio, vLLM, etc.
│   ├── anthropic.py   # Claude SSE streaming
│   ├── google.py      # Gemini SSE streaming
│   └── openrouter.py  # 100+ cloud models
├── models/            # Model management
│   └── downloader.py  # HuggingFace GGUF downloads
├── rag/               # RAG pipeline
│   ├── chunker.py     # Document chunking (paragraph-boundary, configurable size/overlap)
│   ├── embeddings.py  # Multi-backend embeddings (Ollama → llama.cpp → TF-IDF fallback)
│   ├── store.py       # SQLite vector store (cosine similarity search)
│   ├── reranker.py    # BM25 keyword + LLM-based re-ranking
│   ├── watcher.py     # Folder auto-sync (watchfiles-based)
│   ├── context.py     # Context builder (search → rerank → format)
│   └── web_search.py  # DuckDuckGo / SearXNG / Tavily
├── tools/             # Agent tools
│   ├── agent.py       # Plan/execute loop (10 rounds max, built-in + MCP tools)
│   └── code_exec.py   # Multi-language sandbox (9 langs, auto-detect, timeout)
├── mcp/               # Model Context Protocol
│   ├── client.py      # JSON-RPC 2.0 stdio client (subprocess lifecycle)
│   ├── types.py       # ToolDefinition, ToolCall, ToolResult
│   └── tool_loop.py   # Iterative tool calling loop (10 rounds max)
├── server/            # API server
│   └── app.py         # OpenAI-compatible endpoints (/v1/chat/completions, /v1/models)
├── ui/                # Textual TUI
│   ├── app.py         # Main app + command palette
│   ├── screens/       # Chat, Compare, Models, Experiments, Sessions, Templates, Sweep
│   └── widgets/       # Profiler, Parameter panel
├── domain/            # Core types
│   ├── models.py      # ModelDescriptor, InferenceRequest, TokenChunk, GenerationParams
│   ├── profiler.py    # TTFT, t/s, context tracking
│   └── hardware.py    # CPU/RAM/GPU monitoring
├── storage/           # Persistence
│   └── db.py          # SQLite with schema migrations (sessions, runs, templates, workspaces)
├── scoring/           # Quality metrics (BLEU, ROUGE-L, LLM-as-judge)
├── config.py          # TOML config + env overrides
└── cli.py             # Click CLI (12 commands)
```

## Data

All data stored in `~/.llmforge/`:

```
~/.llmforge/
├── config.toml        # Configuration
├── data.sqlite        # Sessions, runs, templates, experiments, RAG vectors
├── models/            # Downloaded GGUF models
└── exports/           # Exported chat Markdown files
```

## License

MIT — free for everyone, do whatever you want, just keep the copyright notice. See [LICENSE](LICENSE).
