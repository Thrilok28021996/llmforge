"""Experiment tracking screen — browse and inspect past runs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

if TYPE_CHECKING:
    from llmforge.storage.db import Database


class RunDetailWidget(Static):
    """Detailed view of a single run."""

    DEFAULT_CSS = """
    RunDetailWidget {
        width: 100%;
        height: 100%;
        padding: 1 2;
        overflow-y: auto;
    }
    """

    def show_run(self, run: dict):
        params = {}
        try:
            params = json.loads(run.get("parameters", "{}"))
        except (json.JSONDecodeError, TypeError):
            pass

        tps = run.get("tokens_per_second")
        ttft = run.get("ttft_ms")
        latency = run.get("total_latency_ms")
        bleu = run.get("score_bleu")
        rouge = run.get("score_rouge_l")
        judge = run.get("score_llm_judge")

        lines = [
            f"[bold cyan]Run {run.get('id', '?')}[/]",
            "",
            f"[dim]Model:[/]    [bold]{run.get('model_id', '?')}[/]",
            f"[dim]Date:[/]     {run.get('created_at', '?')}",
            f"[dim]Device:[/]   {run.get('hw_device', '—')}",
            "",
            "[bold yellow]── Performance ──[/]",
            f"  [dim]TTFT:[/]     [cyan]{ttft:.0f}ms[/]" if ttft else "  [dim]TTFT:[/]     —",
            f"  [dim]t/s:[/]      [bold]{tps:.1f}[/]" if tps else "  [dim]t/s:[/]      —",
            f"  [dim]Latency:[/]  {latency:.0f}ms" if latency else "  [dim]Latency:[/]  —",
            f"  [dim]Tokens:[/]   {run.get('prompt_tokens', '?')}"
            f"→{run.get('completion_tokens', '?')}",
            "",
            "[bold yellow]── Hardware ──[/]",
            f"  [dim]CPU:[/]  {(run.get('hw_cpu_util_avg') or 0):.1f}%",
            f"  [dim]GPU:[/]  {(run.get('hw_gpu_util_avg') or 0):.1f}%",
            f"  [dim]RAM:[/]  {(run.get('hw_ram_used_gb') or 0):.1f}GB",
            "",
            "[bold yellow]── Parameters ──[/]",
        ]
        for k, v in params.items():
            lines.append(f"  [dim]{k}:[/] {v}")

        if bleu or rouge or judge:
            lines.append("")
            lines.append("[bold yellow]── Scores ──[/]")
            if bleu:
                lines.append(f"  [dim]BLEU:[/]   {bleu:.4f}")
            if rouge:
                lines.append(f"  [dim]ROUGE-L:[/] {rouge:.4f}")
            if judge:
                lines.append(f"  [dim]Judge:[/]  {judge:.1f}/10")

        lines.append("")
        lines.append("[bold yellow]── Prompt ──[/]")
        lines.append(run.get("prompt", "")[:500])
        lines.append("")
        lines.append("[bold yellow]── Response ──[/]")
        lines.append(run.get("response", "")[:1000])

        self.update("\n".join(lines))


class ExperimentsScreen(Screen):
    """Browse experiment runs with detail drill-down."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "back", "Back"),
    ]

    DEFAULT_CSS = """
    ExperimentsScreen {
        layout: horizontal;
    }
    #runs-list {
        width: 1fr;
        min-width: 50;
    }
    #run-detail-panel {
        width: 1fr;
        border-left: thick $primary-background-darken-2;
    }
    #runs-table {
        height: 1fr;
    }
    #runs-status {
        dock: bottom;
        height: 1;
        background: $primary-background-darken-3;
        padding: 0 1;
    }
    """

    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self._runs: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            with Vertical(id="runs-list"):
                yield DataTable(
                    id="runs-table", cursor_type="row", zebra_stripes=True
                )
                yield Static("  [dim]Loading runs...[/]", id="runs-status")

            with Vertical(id="run-detail-panel"):
                yield RunDetailWidget(
                    "[dim]Select a run to view details.[/]",
                    id="run-detail",
                )

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.add_columns("ID", "Model", "t/s", "TTFT", "Tokens", "Date")
        self._load_runs()

    @work(thread=False)
    async def _load_runs(self):
        self._runs = await self.db.list_runs(limit=200)

        table = self.query_one("#runs-table", DataTable)
        table.clear()

        for run in self._runs:
            tps = run.get("tokens_per_second")
            ttft = run.get("ttft_ms")
            comp_tokens = run.get("completion_tokens")
            created = run.get("created_at", "")

            table.add_row(
                run["id"],
                run["model_id"],
                f"{tps:.1f}" if tps else "—",
                f"{ttft:.0f}ms" if ttft else "—",
                str(comp_tokens) if comp_tokens else "—",
                created[:16] if created else "—",
                key=run["id"],
            )

        self.query_one("#runs-status", Static).update(
            f"  [dim]{len(self._runs)} runs[/]  "
            f"[cyan]Enter[/] [dim]detail[/]  "
            f"[cyan]Esc[/] [dim]back[/]"
        )

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        run_id = str(event.row_key.value) if event.row_key else None
        if run_id:
            for run in self._runs:
                if run["id"] == run_id:
                    self.query_one("#run-detail", RunDetailWidget).show_run(run)
                    break

    def action_back(self) -> None:
        self.dismiss()

    def action_quit(self) -> None:
        self.app.exit()
