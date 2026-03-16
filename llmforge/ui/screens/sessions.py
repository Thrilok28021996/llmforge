"""Session management screen — browse, resume, and manage chat sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import DataTable, Footer, Header, Static

if TYPE_CHECKING:
    from llmforge.storage.db import Database


class SessionListScreen(Screen):
    """Browse and resume previous chat sessions."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("d", "delete", "Delete"),
    ]

    DEFAULT_CSS = """
    SessionListScreen {
        layout: horizontal;
    }
    #session-list {
        width: 1fr;
        min-width: 50;
    }
    #session-preview {
        width: 1fr;
        border-left: thick $primary-background-darken-2;
        padding: 1;
        overflow-y: auto;
    }
    #sessions-table {
        height: 1fr;
    }
    #sessions-status {
        dock: bottom;
        height: 1;
        background: $primary-background-darken-3;
        padding: 0 1;
    }
    """

    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self._sessions: list[dict] = []
        self._preview_timer: Timer | None = None
        self._pending_preview_id: str | None = None
        self._delete_pending: str | None = None  # Session awaiting confirm

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            with Vertical(id="session-list"):
                yield DataTable(
                    id="sessions-table",
                    cursor_type="row",
                    zebra_stripes=True,
                )
                yield Static(
                    "  [dim]Loading sessions...[/]", id="sessions-status"
                )

            yield Static(
                "[dim]Select a session to preview.[/]",
                id="session-preview",
            )

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        table.add_columns("ID", "Name", "Model", "Messages", "Last Active")
        self._load_sessions()

    @work(thread=False)
    async def _load_sessions(self):
        self._sessions = await self.db.list_sessions()

        table = self.query_one("#sessions-table", DataTable)
        table.clear()

        for s in self._sessions:
            table.add_row(
                s["id"],
                s.get("name", "Untitled"),
                s.get("model_id", "—"),
                str(s.get("message_count", 0)),
                (s.get("updated_at") or "")[:16],
                key=s["id"],
            )

        count = len(self._sessions)
        self.query_one("#sessions-status", Static).update(
            f"  [dim]{count} sessions[/]  "
            f"[cyan]Enter[/] [dim]resume[/]  "
            f"[cyan]d[/] [dim]delete[/]  "
            f"[cyan]Esc[/] [dim]back[/]"
        )

    @on(DataTable.RowSelected)
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        session_id = str(event.row_key.value) if event.row_key else None
        if session_id:
            self.dismiss(session_id)

    @on(DataTable.RowHighlighted)
    def on_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        if not event.row_key:
            return
        self._pending_preview_id = str(event.row_key.value)
        # Debounce: wait 200ms before loading preview (avoids DB query per arrow key)
        if self._preview_timer:
            self._preview_timer.stop()
        self._preview_timer = self.set_timer(0.2, self._load_preview)

    @work(thread=False)
    async def _load_preview(self) -> None:
        session_id = self._pending_preview_id
        if not session_id:
            return
        for s in self._sessions:
            if s["id"] == session_id:
                preview = self.query_one("#session-preview", Static)
                messages = await self.db.get_session_messages(session_id, 10)
                lines = [
                    f"[bold cyan]{s.get('name', 'Untitled')}[/]",
                    f"[dim]Model: {s.get('model_id', '?')}[/]",
                    f"[dim]{s.get('message_count', 0)} messages[/]",
                    "",
                ]
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"][:200]
                    if role == "user":
                        lines.append(f"[cyan bold]YOU:[/] {content}")
                    else:
                        lines.append(f"[green bold]AI:[/] {content}")
                    lines.append("")
                preview.update("\n".join(lines))
                break

    async def action_delete(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        if table.cursor_row is None:
            return
        try:
            row_key, _ = table.coordinate_to_cell_key((table.cursor_row, 0))
        except Exception:
            return
        session_id = str(row_key.value)

        # Two-press confirmation: first press shows warning, second deletes
        if self._delete_pending == session_id:
            session_name = session_id
            for s in self._sessions:
                if s["id"] == session_id:
                    session_name = s.get("name", session_id)
                    break
            await self.db.delete_session(session_id)
            self._delete_pending = None
            self._load_sessions()
            self.notify(f"Deleted session: {session_name}")
        else:
            self._delete_pending = session_id
            session_name = session_id
            for s in self._sessions:
                if s["id"] == session_id:
                    session_name = s.get("name", session_id)
                    break
            self.notify(
                f"Press [bold]d[/] again to delete '{session_name}'",
                severity="warning",
                timeout=3,
            )

    def action_back(self) -> None:
        self.dismiss(None)

    def action_quit(self) -> None:
        self.app.exit()
