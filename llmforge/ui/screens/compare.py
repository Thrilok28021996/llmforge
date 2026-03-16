"""A/B model comparison screen — side-by-side streaming from multiple models."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Markdown, Static

from llmforge.domain.models import ChatMessage, GenerationParams, InferenceRequest
from llmforge.domain.profiler import InferenceProfiler

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llmforge.backends.ollama import OllamaBackend
    from llmforge.storage.db import Database


@dataclass
class CompareColumn:
    model_id: str
    response_chunks: list[str] = field(default_factory=list)
    response: str = ""
    profiler: InferenceProfiler = field(default_factory=InferenceProfiler)
    done: bool = False
    error: str | None = None
    md_stream: object | None = field(default=None, repr=False)  # MarkdownStream


class CompareColumnWidget(Static):
    """One column in the comparison view."""

    DEFAULT_CSS = """
    CompareColumnWidget {
        width: 1fr;
        height: 100%;
        border-right: thick $primary-background-darken-2;
        padding: 0 1;
    }
    CompareColumnWidget:last-child {
        border-right: none;
    }
    CompareColumnWidget > .col-header {
        text-style: bold;
        color: $accent;
        text-align: center;
        height: 1;
        border-bottom: solid $primary-background-darken-2;
        margin-bottom: 1;
    }
    CompareColumnWidget > .col-response {
        height: 1fr;
        overflow-y: auto;
    }
    CompareColumnWidget > Markdown {
        height: 1fr;
        overflow-y: auto;
    }
    CompareColumnWidget > .col-metrics {
        dock: bottom;
        height: 4;
        border-top: solid $primary-background-darken-2;
        background: $surface-darken-1;
        padding: 0 1;
    }
    """

    def __init__(self, col: CompareColumn):
        super().__init__()
        self.col = col

    def compose(self) -> ComposeResult:
        status = (
            "[green]✓[/]" if self.col.done and not self.col.error else
            "[red]✗[/]" if self.col.error else
            "[yellow]●[/]"
        )
        yield Static(f"{status} {self.col.model_id}", classes="col-header")
        yield Markdown("", id=f"md-{id(self)}")

        m = self.col.profiler.metrics
        tps_str = f"{m.tokens_per_second:.1f}" if m.tokens_per_second else "—"
        ttft_str = f"{m.ttft_ms:.0f}ms" if m.ttft_ms else "—"
        tokens_str = str(m.token_count) if m.token_count else "—"

        yield Static(
            f"  [dim]TTFT:[/] [cyan]{ttft_str}[/]\n"
            f"  [dim]t/s:[/]  [bold]{tps_str}[/]\n"
            f"  [dim]Tokens:[/] {tokens_str}",
            classes="col-metrics",
        )


class CompareScreen(Screen):
    """Side-by-side model comparison with concurrent streaming."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "back", "Back"),
    ]

    DEFAULT_CSS = """
    CompareScreen {
        layout: vertical;
    }
    #compare-columns {
        layout: horizontal;
        height: 1fr;
    }
    #compare-input-area {
        dock: bottom;
        height: 3;
        border-top: thick $primary-background-darken-2;
        padding: 0 1;
    }
    #compare-status {
        dock: bottom;
        height: 1;
        background: $primary-background-darken-3;
        padding: 0 1;
    }
    """

    def __init__(self, model_ids: list[str], backend: OllamaBackend, db: Database):
        super().__init__()
        self.model_ids = model_ids
        self._ollama_config = backend._config
        self.db = db
        self.columns = [CompareColumn(model_id=mid) for mid in model_ids]
        self._streaming = False
        self._active_backends: list = []
        # Cached widget references (populated on mount)
        self._col_widgets: list[CompareColumnWidget] = []
        self._md_widgets: list[Markdown] = []
        self._header_widgets: list[Static] = []
        self._metrics_widgets: list[Static] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal(id="compare-columns"):
            for col in self.columns:
                yield CompareColumnWidget(col)

        models_str = " vs ".join(self.model_ids)
        yield Static(
            f"  [dim]Comparing:[/] [bold]{models_str}[/]    [dim]○ idle[/]",
            id="compare-status",
        )
        with Vertical(id="compare-input-area"):
            yield Input(
                placeholder="Type a prompt (sent to all models)...",
                id="compare-input",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#compare-input", Input).focus()
        # Cache widget references to avoid CSS queries on every token
        self._col_widgets = list(self.query(CompareColumnWidget))
        for cw in self._col_widgets:
            md_list = list(cw.query(Markdown))
            self._md_widgets.append(md_list[0] if md_list else None)
            hdr = list(cw.query(".col-header"))
            self._header_widgets.append(hdr[0] if hdr else None)
            met = list(cw.query(".col-metrics"))
            self._metrics_widgets.append(met[0] if met else None)

    @on(Input.Submitted, "#compare-input")
    async def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._streaming:
            return

        event.input.value = ""
        self._streaming = True

        # Reset columns and create markdown streams
        for i, col in enumerate(self.columns):
            col.response = ""
            col.response_chunks = []
            col.done = False
            col.error = None
            col.profiler = InferenceProfiler()
            # Reset markdown widget and create stream
            md = self._md_widgets[i] if i < len(self._md_widgets) else None
            if md:
                md.update("")
                col.md_stream = Markdown.get_stream(md)

        self._update_status(streaming=True)
        self._run_all(text)

    @work(thread=False)
    async def _run_all(self, prompt: str):
        """Run inference on all models concurrently."""
        self._active_backends = []
        tasks = []
        for i, col in enumerate(self.columns):
            tasks.append(self._run_one(col, prompt, i))
        await asyncio.gather(*tasks)

        self._streaming = False
        self._active_backends = []
        self._update_status(streaming=False)

    async def _run_one(self, col: CompareColumn, prompt: str, index: int):
        """Run inference for a single model with its own backend."""
        from llmforge.backends.ollama import OllamaBackend

        backend = OllamaBackend(self._ollama_config)
        self._active_backends.append(backend)
        request = InferenceRequest(
            model_id=col.model_id,
            messages=[ChatMessage(role="user", content=prompt)],
        )
        col.profiler.start()

        try:
            async for chunk in backend.generate(request):
                if chunk.is_final:
                    col.response = "".join(col.response_chunks)
                    col.done = True
                    col.profiler.finish()
                    self._refresh_column_metrics(index)
                    break

                if chunk.text:
                    col.response_chunks.append(chunk.text)
                    col.profiler.on_token(chunk.text)
                    # Stream markdown incrementally
                    if col.md_stream:
                        await col.md_stream.write(chunk.text)
                    self._refresh_column_metrics(index)

        except Exception as e:
            col.error = str(e)
            col.done = True
            if col.md_stream:
                await col.md_stream.write(f"\n\n**Error:** {e}")
            self._refresh_column_metrics(index)
        finally:
            # Stop stream and close backend
            if col.md_stream:
                try:
                    await col.md_stream.stop()
                except Exception:
                    pass
                col.md_stream = None
            col.response = "".join(col.response_chunks)
            await backend.close()

        # Record to DB
        try:
            await self.db.record_run(
                model_id=col.model_id,
                prompt=prompt,
                response=col.response,
                params=GenerationParams().to_dict(),
                ttft_ms=col.profiler.metrics.ttft_ms,
                tokens_per_second=col.profiler.metrics.tokens_per_second,
                total_latency_ms=col.profiler.metrics.total_latency_ms,
            )
        except Exception:
            logger.warning("Failed to record compare run for %s", col.model_id, exc_info=True)

    def _refresh_column_metrics(self, index: int):
        """Update a single column's header and metrics using cached widget refs."""
        if index >= len(self._col_widgets):
            return
        col = self.columns[index]

        hdr = self._header_widgets[index]
        if hdr:
            status = (
                "[green]✓[/]" if col.done and not col.error else
                "[red]✗[/]" if col.error else
                "[yellow]●[/]"
            )
            hdr.update(f"{status} {col.model_id}")

        mw = self._metrics_widgets[index]
        if mw:
            m = col.profiler.metrics
            tps_str = f"{m.tokens_per_second:.1f}" if m.tokens_per_second else "—"
            ttft_str = f"{m.ttft_ms:.0f}ms" if m.ttft_ms else "—"
            tokens_str = str(m.token_count) if m.token_count else "—"
            mw.update(
                f"  [dim]TTFT:[/] [cyan]{ttft_str}[/]\n"
                f"  [dim]t/s:[/]  [bold]{tps_str}[/]\n"
                f"  [dim]Tokens:[/] {tokens_str}"
            )

    def _update_status(self, streaming: bool):
        indicator = "[green bold]● streaming[/]" if streaming else "[dim]○ idle[/]"
        models_str = " vs ".join(self.model_ids)
        self.query_one("#compare-status", Static).update(
            f"  [dim]Comparing:[/] [bold]{models_str}[/]    {indicator}"
        )

    async def _cancel_all(self):
        """Cancel all active backends."""
        for backend in self._active_backends:
            try:
                await backend.cancel()
            except Exception:
                pass

    async def action_back(self) -> None:
        if self._streaming:
            await self._cancel_all()
        self.dismiss()

    async def action_quit(self) -> None:
        if self._streaming:
            await self._cancel_all()
        self.app.exit()
