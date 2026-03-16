"""CLI entry point — click subcommands dispatch to the TUI app."""

from __future__ import annotations

import asyncio
import sys

import click

from llmforge import __version__
from llmforge.config import Config


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="llmforge")
@click.option("--backend", "-b",
              type=click.Choice([
                  "ollama", "openai-compat", "anthropic",
                  "google", "openrouter", "llamacpp",
              ]),
              help="Backend to use (default: from config)")
@click.pass_context
def main(ctx: click.Context, backend: str | None):
    """LLM Forge — Terminal-based local LLM developer toolkit.

    Run without a subcommand to open the model picker.
    """
    config = Config.load()
    if backend:
        config.backend = backend
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    if ctx.invoked_subcommand is None:
        _run_app(config, mode="default")


@main.command()
@click.argument("model")
@click.option("--temp", type=click.FloatRange(0.0, 2.0), help="Temperature")
@click.option("--ctx", "context_length", type=click.IntRange(min=128),
              help="Context length")
@click.option("--max-tokens", type=click.IntRange(min=1),
              help="Max tokens to generate")
@click.option("--system", "-s", help="System prompt")
@click.pass_context
def chat(
    ctx: click.Context,
    model: str,
    temp: float | None,
    context_length: int | None,
    max_tokens: int | None,
    system: str | None,
):
    """Start an interactive chat with a model.

    Supports stdin piping: echo "question" | llmforge chat model

    \b
    Examples:
      llmforge chat llama3.2:3b
      llmforge chat qwen2.5:7b --temp 0.3
      echo "explain this" | llmforge chat llama3.2:3b
      cat error.log | llmforge chat llama3.2:3b
    """
    config = ctx.obj["config"]
    if temp is not None:
        config.generation.temperature = temp
    if context_length is not None:
        config.generation.context_length = context_length
    if max_tokens is not None:
        config.generation.max_tokens = max_tokens

    # Pipe mode: stdin → LLM → stdout (no TUI)
    if not sys.stdin.isatty():
        stdin_text = sys.stdin.read().strip()
        if not stdin_text:
            click.echo("No input received from stdin.", err=True)
            sys.exit(1)
        asyncio.run(
            _run_pipe(config, model, stdin_text, system_prompt=system)
        )
        return

    _run_app(config, mode="chat", model_id=model, system_prompt=system)


@main.command()
@click.argument("models", nargs=-1, required=True)
@click.pass_context
def compare(ctx: click.Context, models: tuple[str, ...]):
    """Compare models side-by-side with the same prompt.

    \b
    Example: llmforge compare llama3.2:3b qwen2.5:7b phi4:14b
    """
    if len(models) < 2:
        click.echo("Need at least 2 models to compare.", err=True)
        sys.exit(1)
    if len(models) > 4:
        click.echo("Maximum 4 models for comparison.", err=True)
        sys.exit(1)

    _run_app(ctx.obj["config"], mode="compare", model_ids=list(models))


@main.command()
@click.pass_context
def models(ctx: click.Context):
    """Browse the model library."""
    _run_app(ctx.obj["config"], mode="models")


@main.command(name="exp")
@click.pass_context
def experiments(ctx: click.Context):
    """View experiment history and run details."""
    _run_app(ctx.obj["config"], mode="experiments")


@main.command()
@click.pass_context
def sessions(ctx: click.Context):
    """Browse and resume previous chat sessions."""
    _run_app(ctx.obj["config"], mode="sessions")


@main.command()
@click.argument("model")
@click.pass_context
def sweep(ctx: click.Context, model: str):
    """Run a parameter sweep across temperature values.

    \b
    Example: llmforge sweep llama3.2:3b
    """
    _run_app(ctx.obj["config"], mode="sweep", model_id=model)


@main.command()
@click.argument("model")
@click.argument("prompt")
@click.option("--reference", "-r", help="Reference text for BLEU/ROUGE scoring")
@click.pass_context
def score(ctx: click.Context, model: str, prompt: str, reference: str | None):
    """Score a model response for quality.

    \b
    Example: llmforge score llama3.2:3b "Explain quantum computing"
    """
    asyncio.run(_run_score(ctx.obj["config"], model, prompt, reference))


@main.command()
@click.argument("model")
@click.argument("prompt")
@click.option("--max-tokens", type=int, default=2048, help="Max tokens")
@click.pass_context
def run(ctx: click.Context, model: str, prompt: str, max_tokens: int):
    """Run a single inference (non-interactive, stdout output).

    \b
    Example: llmforge run llama3.2:3b "Write a haiku about coding"
    """
    config = ctx.obj["config"]
    config.generation.max_tokens = max_tokens
    asyncio.run(_run_pipe(config, model, prompt))


# ── Non-TUI async runners ────────────────────────────────────────────────


@main.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option("--list", "list_docs", is_flag=True, help="List ingested documents")
@click.option("--delete", type=int, help="Delete a document by ID")
@click.pass_context
def ingest(ctx: click.Context, paths: tuple[str, ...], list_docs: bool, delete: int | None):
    """Ingest documents for RAG (Retrieval-Augmented Generation).

    \b
    Examples:
      llmforge ingest README.md docs/
      llmforge ingest --list
      llmforge ingest --delete 3
    """
    asyncio.run(_run_ingest(ctx.obj["config"], paths, list_docs, delete))


async def _run_ingest(
    config: Config, paths: tuple[str, ...], list_docs: bool, delete: int | None
):
    from pathlib import Path

    import aiosqlite

    from llmforge.config import db_path
    from llmforge.rag.store import RAGStore

    db = await aiosqlite.connect(str(db_path()))
    store = RAGStore(db)
    await store.ensure_schema()

    try:
        if list_docs:
            docs = await store.list_documents()
            if not docs:
                click.echo("No documents ingested yet.")
                return
            click.echo(f"{'ID':>4}  {'Chunks':>6}  {'Name'}")
            click.echo("─" * 50)
            for d in docs:
                click.echo(f"{d['id']:>4}  {d['chunk_count']:>6}  {d['name']}")
            total = await store.chunk_count()
            click.echo(f"\nTotal: {len(docs)} documents, {total} chunks")
            return

        if delete is not None:
            await store.delete_document(delete)
            click.echo(f"Deleted document {delete}.")
            return

        if not paths:
            click.echo("Provide file/directory paths to ingest, or use --list.", err=True)
            return

        ollama_url = config.ollama.base_url
        emb_model = config.rag.embedding_model

        for p_str in paths:
            p = Path(p_str)
            if p.is_dir():
                files = sorted(p.rglob("*"))
                files = [f for f in files if f.is_file() and not f.name.startswith(".")]
            else:
                files = [p]

            for f in files:
                try:
                    click.echo(f"Ingesting {f.name}...", nl=False)
                    doc_id = await store.add_document(
                        f,
                        embedding_model=emb_model,
                        chunk_size=config.rag.chunk_size,
                        overlap=config.rag.overlap,
                        ollama_url=ollama_url,
                    )
                    click.echo(f" done (id={doc_id})")
                except Exception as e:
                    click.echo(f" error: {e}")
    finally:
        await db.close()


@main.command(name="config")
@click.option("--show", is_flag=True, help="Show current config")
@click.option("--set", "set_vals", nargs=2, multiple=True,
              help="Set a config value: --set key value")
@click.pass_context
def config_cmd(ctx: click.Context, show: bool, set_vals: tuple[tuple[str, str], ...]):
    """View or edit configuration.

    \b
    Examples:
      llmforge config --show
      llmforge config --set backend anthropic
      llmforge config --set anthropic.api_key sk-ant-...
      llmforge config --set rag.enabled true
    """
    config = ctx.obj["config"]

    if show or not set_vals:
        import tomli_w
        click.echo(tomli_w.dumps(config.model_dump()))
        return

    for key, value in set_vals:
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                click.echo(f"Unknown config key: {key}", err=True)
                return

        field_name = parts[-1]
        if not hasattr(obj, field_name):
            click.echo(f"Unknown config key: {key}", err=True)
            return

        # Type coercion
        current = getattr(obj, field_name)
        if isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)

        setattr(obj, field_name, value)
        click.echo(f"Set {key} = {value}")

    config.save()
    click.echo("Config saved.")


@main.command()
@click.option("--port", type=int, default=8000, help="Port to listen on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.pass_context
def serve(ctx: click.Context, port: int, host: str):
    """Start an OpenAI-compatible API server.

    Other apps can use LLM Forge as a backend at http://host:port/v1

    \b
    Example:
      llmforge serve --port 8000
      llmforge -b llamacpp serve
    """
    config = ctx.obj["config"]
    backend = _create_backend(config)

    from llmforge.server.app import create_app
    app = create_app(backend, config)

    try:
        import uvicorn
    except ImportError:
        click.echo(
            "Server dependencies not installed. "
            "Run: pip install 'llmforge[server]'",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Starting LLM Forge API server on {host}:{port}")
    click.echo(f"Backend: {config.backend}")
    click.echo("Endpoints:")
    click.echo(f"  GET  http://{host}:{port}/v1/models")
    click.echo(f"  POST http://{host}:{port}/v1/chat/completions")
    click.echo(f"  GET  http://{host}:{port}/health")
    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command()
@click.argument("query", required=False)
@click.option("--list", "list_local", is_flag=True, help="List local GGUF models")
@click.option("--popular", is_flag=True, help="Show popular downloadable models")
@click.pass_context
def download(ctx: click.Context, query: str | None, list_local: bool, popular: bool):
    """Download GGUF models from HuggingFace.

    \b
    Examples:
      llmforge download --popular
      llmforge download --list
      llmforge download "llama 3.2"
    """
    asyncio.run(_run_download(ctx.obj["config"], query, list_local, popular))


async def _run_download(config: Config, query: str | None, list_local: bool, popular: bool):
    from llmforge.models.downloader import (
        MODELS_DIR,
        POPULAR_REPOS,
        download_gguf,
        list_local_gguf,
        search_huggingface_gguf,
    )

    if list_local:
        files = list_local_gguf(config.llamacpp.model_dirs)
        if not files:
            click.echo("No GGUF models found. Download with: llmforge download --popular")
            return
        click.echo(f"Local GGUF models ({MODELS_DIR}):\n")
        for f in files:
            size_gb = f.stat().st_size / (1024 ** 3)
            click.echo(f"  {f.name:40s}  {size_gb:.2f} GB")
        return

    if popular:
        click.echo("Popular GGUF models (Q4_K_M quantization):\n")
        for i, repo in enumerate(POPULAR_REPOS, 1):
            click.echo(f"  [{i}] {repo['name']:35s}  {repo['params']:>5s}  {repo['repo']}")
        click.echo("\nDownload: llmforge download --popular then enter number")
        choice = click.prompt("Download model number (0 to skip)", type=int, default=0)
        if 1 <= choice <= len(POPULAR_REPOS):
            repo = POPULAR_REPOS[choice - 1]
            for filename in repo["files"]:
                click.echo(f"Downloading {filename} from {repo['repo']}...")
                async for progress in download_gguf(repo["repo"], filename):
                    if progress.status == "complete":
                        click.echo(f"  Done! Saved to {MODELS_DIR / filename}")
                    elif progress.status.startswith("error"):
                        click.echo(f"  Error: {progress.status}")
                    else:
                        click.echo(f"  {progress.status}...")
        return

    if query:
        click.echo(f"Searching HuggingFace for '{query}'...")
        results = await search_huggingface_gguf(query)
        if not results:
            click.echo("No results found.")
            return
        for i, r in enumerate(results[:10], 1):
            click.echo(
                f"  [{i}] {r['repo_id']:50s}  "
                f"DL:{r['downloads']:>8,}"
            )
        return

    click.echo("Usage: llmforge download --popular | --list | <search query>")


def _create_backend(config: Config):
    """Create the appropriate backend from config."""
    from llmforge.ui.app import LLMForgeApp
    return LLMForgeApp._create_backend(config)


async def _run_pipe(
    config: Config,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
):
    """Pipe mode: generate response and print to stdout."""
    from llmforge.domain.models import ChatMessage, GenerationParams, InferenceRequest

    backend = _create_backend(config)
    try:
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        params = GenerationParams(
            temperature=config.generation.temperature,
            top_p=config.generation.top_p,
            top_k=config.generation.top_k,
            max_tokens=config.generation.max_tokens,
            context_length=config.generation.context_length,
            repeat_penalty=config.generation.repeat_penalty,
            seed=config.generation.seed,
        )

        request = InferenceRequest(
            model_id=model,
            messages=messages,
            params=params,
        )

        first_chunk = True
        got_tokens = False
        async for chunk in backend.generate(request):
            if chunk.is_final and first_chunk and chunk.text:
                # Single-chunk final = error from backend (connection refused, etc.)
                click.echo(chunk.text, err=True)
                sys.exit(1)
            if chunk.text:
                got_tokens = True
                print(chunk.text, end="", flush=True)
            first_chunk = False
            if chunk.is_final:
                break
        if got_tokens:
            print()  # Final newline
    finally:
        await backend.close()


async def _run_score(
    config: Config, model: str, prompt: str, reference: str | None
):
    from llmforge.domain.models import ChatMessage, InferenceRequest
    from llmforge.scoring import score_response

    backend = _create_backend(config)
    try:
        click.echo(f"Generating response from {model}...")

        request = InferenceRequest(
            model_id=model,
            messages=[ChatMessage(role="user", content=prompt)],
        )

        response_text = ""
        async for chunk in backend.generate(request):
            if chunk.text:
                response_text += chunk.text
            if chunk.is_final:
                break

        click.echo(f"\nResponse ({len(response_text)} chars):")
        click.echo(response_text[:500])
        click.echo("\nScoring...")

        scores = await score_response(
            prompt=prompt,
            response=response_text,
            reference=reference,
            judge_model=config.scoring.judge_model,
            ollama_url=config.ollama.base_url,
        )

        click.echo(f"\n{'─' * 40}")
        click.echo(f"  LLM Judge:  {scores.llm_judge:.1f}/10")
        if reference:
            click.echo(f"  BLEU:       {scores.bleu:.4f}")
            click.echo(f"  ROUGE-L:    {scores.rouge_l:.4f}")
        click.echo(f"{'─' * 40}")
    finally:
        await backend.close()


# ── App launcher ──────────────────────────────────────────────────────────


def _run_app(
    config: Config,
    mode: str = "chat",
    model_id: str | None = None,
    model_ids: list[str] | None = None,
    system_prompt: str | None = None,
):
    from llmforge.ui.app import LLMForgeApp

    app = LLMForgeApp(
        config=config,
        mode=mode,
        model_id=model_id,
        model_ids=model_ids,
        system_prompt=system_prompt,
    )
    app.run()


if __name__ == "__main__":
    main()
