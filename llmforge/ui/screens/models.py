"""Model library browser screen with download support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static

from llmforge.domain.models import ModelDescriptor, estimate_memory_bytes

if TYPE_CHECKING:
    from llmforge.backends.ollama import OllamaBackend
    from llmforge.domain.hardware import HardwareMonitor


class ModelLibraryScreen(Screen):
    """Browse and select from available models."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("slash", "filter", "Filter"),
        Binding("p", "pull_model", "Pull Model"),
        Binding("delete", "delete_model", "Delete"),
        Binding("escape", "back", "Back"),
    ]

    DEFAULT_CSS = """
    ModelLibraryScreen {
        layout: vertical;
    }
    #model-filter {
        dock: bottom;
        height: 3;
        padding: 0 1;
        border-top: thick $primary-background-darken-2;
    }
    #model-status {
        dock: bottom;
        height: 1;
        background: $primary-background-darken-3;
        padding: 0 1;
    }
    #model-table {
        height: 1fr;
    }
    """

    def __init__(self, backend: OllamaBackend, hw_monitor: HardwareMonitor):
        super().__init__()
        self.backend = backend
        self.hw_monitor = hw_monitor
        self.models: list[ModelDescriptor] = []
        self._filter_visible = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield DataTable(id="model-table", cursor_type="row", zebra_stripes=True)
        yield Static("  [dim]Loading models...[/]", id="model-status")
        yield Input(
            placeholder="Filter models...",
            id="model-filter",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#model-filter", Input).display = False

        table = self.query_one("#model-table", DataTable)
        table.add_columns(
            "Name", "Backend", "Size", "Params", "Quant", "Families", "Fit"
        )
        self._load_models()

    @work(thread=False)
    async def _load_models(self):
        """Fetch models from backend."""
        self.models = await self.backend.list_models()
        self._populate_table(self.models)
        self.query_one("#model-status", Static).update(
            f"  [dim]{len(self.models)} models[/]  "
            f"[cyan]/[/] [dim]filter[/]  "
            f"[cyan]Enter[/] [dim]select[/]  "
            f"[cyan]Esc[/] [dim]back[/]"
        )

    def _populate_table(self, models: list[ModelDescriptor]):
        table = self.query_one("#model-table", DataTable)
        table.clear()

        snap = self.hw_monitor.latest
        # Use cached snapshot to avoid blocking; fallback to 16GB if no data yet
        free_bytes = int(snap.ram_free_gb * 1024 ** 3) if snap.ram_total_gb > 0 else 16 * 1024 ** 3

        for m in models:
            estimated = estimate_memory_bytes(m)
            ratio = estimated / max(free_bytes, 1)

            size_str = f"{m.size_gb:.2f}GB" if m.size_gb else "—"
            param_str = f"{m.param_billions:.1f}B" if m.param_billions else "—"
            quant_str = m.quantization or "—"
            families_str = ", ".join(m.families[:2]) if m.families else "—"

            if ratio < 0.7:
                fit = "[green]●[/]"
            elif ratio < 0.9:
                fit = "[yellow]◐[/]"
            else:
                fit = "[red]○[/]"

            table.add_row(
                m.name, m.backend, size_str, param_str, quant_str, families_str, fit,
                key=m.id,
            )

    @on(Input.Changed, "#model-filter")
    def on_filter_changed(self, event: Input.Changed) -> None:
        query = event.value.lower()
        filtered = [m for m in self.models if query in m.name.lower()]
        self._populate_table(filtered)
        self.query_one("#model-status", Static).update(
            f"  [dim]{len(filtered)}/{len(self.models)} models[/]  "
            f"[dim]Filter: \"{event.value}\"[/]"
        )

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key:
            self.dismiss(str(event.row_key.value))

    def action_filter(self) -> None:
        filt = self.query_one("#model-filter", Input)
        filt.display = not filt.display
        if filt.display:
            filt.focus()
        else:
            filt.value = ""
            self._populate_table(self.models)

    async def action_pull_model(self) -> None:
        """Pull a model by name (Ollama only)."""
        if not hasattr(self.backend, "pull_model"):
            self.notify("Pull only available with Ollama backend", severity="warning")
            return

        filt = self.query_one("#model-filter", Input)
        filt.display = True
        filt.placeholder = "Enter model name to pull (e.g. llama3.2:3b)..."
        filt.focus()
        filt.value = ""
        self._pull_mode = True

    async def action_delete_model(self) -> None:
        """Delete the selected model (Ollama only)."""
        if not hasattr(self.backend, "delete_model"):
            self.notify("Delete only available with Ollama backend", severity="warning")
            return

        table = self.query_one("#model-table", DataTable)
        if table.cursor_row < 0 or table.cursor_row >= len(self.models):
            return

        model = self.models[table.cursor_row]
        if not hasattr(self, "_delete_pending") or not self._delete_pending:
            self._delete_pending = True
            self.notify(f"Press Delete again to remove {model.name}", severity="warning")
            return

        self._delete_pending = False
        try:
            await self.backend.delete_model(model.name)
            self.notify(f"Deleted {model.name}")
            self._load_models()
        except Exception as e:
            self.notify(f"Delete failed: {e}", severity="error")

    @on(Input.Submitted, "#model-filter")
    def on_filter_submitted(self, event: Input.Submitted) -> None:
        if getattr(self, "_pull_mode", False):
            self._pull_mode = False
            model_name = event.value.strip()
            if model_name:
                self._do_pull(model_name)
            event.input.placeholder = "Filter models..."
            event.input.value = ""
            event.input.display = False

    @work(thread=False)
    async def _do_pull(self, model_name: str):
        self.notify(f"Pulling {model_name}... (this may take a while)")
        status = self.query_one("#model-status", Static)
        try:
            async for data in self.backend.pull_model(model_name):
                completed = data.get("completed", 0)
                total = data.get("total", 1)
                pct = completed / max(total, 1) * 100 if total else 0
                status_text = data.get("status", "downloading")
                status.update(f"  [cyan]Pulling:[/] {status_text} [{pct:.0f}%]")
            self.notify(f"Successfully pulled {model_name}")
            self._load_models()
        except Exception as e:
            self.notify(f"Pull failed: {e}", severity="error")
        status.update(f"  [dim]{len(self.models)} models[/]")

    def action_back(self) -> None:
        self.dismiss(None)

    def action_quit(self) -> None:
        self.app.exit()
