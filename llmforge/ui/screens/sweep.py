"""Parameter sweep screen — explore hyperparameter space."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from llmforge.domain.models import ChatMessage, GenerationParams, InferenceRequest
from llmforge.domain.profiler import InferenceProfiler

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llmforge.backends.ollama import OllamaBackend
    from llmforge.storage.db import Database


@dataclass
class SweepResult:
    params: GenerationParams
    response: str
    ttft_ms: float | None
    tokens_per_second: float | None
    token_count: int
    total_latency_ms: float | None


class ParameterSweepScreen(Screen):
    """Run a prompt across different parameter combinations."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "back", "Back"),
    ]

    DEFAULT_CSS = """
    ParameterSweepScreen {
        layout: vertical;
    }
    #sweep-config {
        height: auto;
        max-height: 8;
        padding: 1;
        border-bottom: thick $primary-background-darken-2;
    }
    #sweep-table {
        height: 1fr;
    }
    #sweep-detail {
        height: 12;
        border-top: thick $primary-background-darken-2;
        padding: 1;
        overflow-y: auto;
    }
    #sweep-status {
        dock: bottom;
        height: 1;
        background: $primary-background-darken-3;
        padding: 0 1;
    }
    #sweep-prompt-input {
        width: 100%;
    }
    """

    def __init__(self, model_id: str, backend: OllamaBackend, db: Database):
        super().__init__()
        self.model_id = model_id
        self.backend = backend
        self.db = db
        self.results: list[SweepResult] = []
        self._running = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Vertical(id="sweep-config"):
            yield Static(
                f"[bold cyan]Parameter Sweep[/]  "
                f"[dim]Model:[/] [bold]{self.model_id}[/]"
            )
            yield Static(
                "[dim]Sweeps temperature: 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5[/]"
            )
            yield Input(
                placeholder="Enter prompt to sweep...",
                id="sweep-prompt-input",
            )

        yield DataTable(id="sweep-table", cursor_type="row", zebra_stripes=True)

        yield Static("[dim]Select a result to see the response.[/]", id="sweep-detail")

        yield Static(
            f"  [dim]Model: {self.model_id}[/]  "
            f"[cyan]Enter[/] [dim]run sweep[/]  "
            f"[cyan]Esc[/] [dim]back[/]",
            id="sweep-status",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#sweep-table", DataTable)
        table.add_columns(
            "Temp", "top_p", "t/s", "TTFT", "Tokens", "Latency", "Preview"
        )
        self.query_one("#sweep-prompt-input", Input).focus()

    @on(Input.Submitted, "#sweep-prompt-input")
    async def on_submit(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt or self._running:
            return
        self._running = True
        self.results.clear()
        self.query_one("#sweep-table", DataTable).clear()
        self.query_one("#sweep-status", Static).update(
            "  [green bold]● running sweep...[/]"
        )
        self._run_sweep(prompt)

    @work(thread=False)
    async def _run_sweep(self, prompt: str):
        """Run the prompt at different temperature values."""
        temps = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5]
        top_ps = [0.9]  # Fixed for now

        for temp in temps:
            for top_p in top_ps:
                params = GenerationParams(
                    temperature=temp,
                    top_p=top_p,
                    max_tokens=256,  # Keep short for sweep
                )
                request = InferenceRequest(
                    model_id=self.model_id,
                    messages=[ChatMessage(role="user", content=prompt)],
                    params=params,
                )

                profiler = InferenceProfiler()
                profiler.start()
                response = ""

                try:
                    async for chunk in self.backend.generate(request):
                        if chunk.is_final:
                            profiler.finish()
                            break
                        if chunk.text:
                            response += chunk.text
                            profiler.on_token(chunk.text)
                except Exception as e:
                    response = f"Error: {e}"
                    profiler.finish()

                result = SweepResult(
                    params=params,
                    response=response,
                    ttft_ms=profiler.metrics.ttft_ms,
                    tokens_per_second=profiler.metrics.tokens_per_second,
                    token_count=profiler.metrics.token_count,
                    total_latency_ms=profiler.metrics.total_latency_ms,
                )
                self.results.append(result)
                self._add_result_row(result)

                # Record to DB
                try:
                    await self.db.record_run(
                        model_id=self.model_id,
                        prompt=prompt,
                        response=response,
                        params=params.to_dict(),
                        ttft_ms=result.ttft_ms,
                        tokens_per_second=result.tokens_per_second,
                        total_latency_ms=result.total_latency_ms,
                    )
                except Exception:
                    logger.warning("Failed to record sweep run", exc_info=True)

        self._running = False
        self.query_one("#sweep-status", Static).update(
            f"  [dim]{len(self.results)} results[/]  "
            f"[cyan]↑↓[/] [dim]browse[/]  "
            f"[cyan]Esc[/] [dim]back[/]"
        )
        self.notify(f"Sweep complete: {len(self.results)} configurations tested")

    def _add_result_row(self, result: SweepResult):
        table = self.query_one("#sweep-table", DataTable)
        m = result
        tps = f"{m.tokens_per_second:.1f}" if m.tokens_per_second else "—"
        ttft = f"{m.ttft_ms:.0f}ms" if m.ttft_ms else "—"
        latency = f"{m.total_latency_ms:.0f}ms" if m.total_latency_ms else "—"
        preview = m.response[:60].replace("\n", " ")

        table.add_row(
            f"{m.params.temperature:.1f}",
            f"{m.params.top_p:.1f}",
            tps,
            ttft,
            str(m.token_count),
            latency,
            preview,
        )

    @on(DataTable.RowHighlighted)
    def on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        idx = event.cursor_row
        if idx is not None and idx < len(self.results):
            r = self.results[idx]
            detail = self.query_one("#sweep-detail", Static)
            detail.update(
                f"[bold]temp={r.params.temperature} "
                f"top_p={r.params.top_p}[/]\n\n"
                f"{r.response[:500]}"
            )

    def action_back(self) -> None:
        self.dismiss()

    def action_quit(self) -> None:
        self.app.exit()
