"""Prompt Template Library — create, edit, and use prompt templates."""

from __future__ import annotations

import json
import re

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Input, Static, TextArea

from llmforge.storage.db import Database


class TemplateEditorScreen(Screen):
    """Edit or create a prompt template."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    TemplateEditorScreen {
        background: $surface;
    }
    TemplateEditorScreen > Vertical > .editor-label {
        height: 1;
        margin: 1 1 0 1;
        color: $text-muted;
    }
    TemplateEditorScreen > Vertical > Input {
        margin: 0 1;
    }
    TemplateEditorScreen > Vertical > TextArea {
        margin: 0 1;
        height: 1fr;
    }
    TemplateEditorScreen > Vertical > .var-display {
        height: 3;
        margin: 0 1;
        padding: 1;
        color: $accent;
    }
    """

    def __init__(
        self,
        db: Database,
        template_id: str | None = None,
        name: str = "",
        content: str = "",
        variables: list[str] | None = None,
    ):
        super().__init__()
        self._db = db
        self._template_id = template_id
        self._name = name
        self._content = content
        self._variables = variables or []

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("Template Name:", classes="editor-label")
            yield Input(value=self._name, id="tpl-name", placeholder="e.g. Code Review")
            yield Static(
                "Template Content (use {{variable}} for placeholders):",
                classes="editor-label",
            )
            yield TextArea(self._content, id="tpl-content", language="markdown")
            yield Static(
                f"  Detected variables: {', '.join(self._variables) or 'none'}",
                id="var-display",
                classes="var-display",
            )
        yield Footer()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        content = event.text_area.text
        variables = re.findall(r"\{\{(\w+)\}\}", content)
        unique_vars = list(dict.fromkeys(variables))
        self._variables = unique_vars
        try:
            self.query_one("#var-display", Static).update(
                f"  Detected variables: {', '.join(unique_vars) or 'none'}"
            )
        except Exception:
            pass

    async def action_save(self) -> None:
        name = self.query_one("#tpl-name", Input).value.strip()
        content = self.query_one("#tpl-content", TextArea).text.strip()
        if not name or not content:
            self.notify("Name and content are required", severity="warning")
            return

        variables = re.findall(r"\{\{(\w+)\}\}", content)
        unique_vars = list(dict.fromkeys(variables))

        if self._template_id:
            await self._db.update_template(self._template_id, name, content, unique_vars)
            self.notify(f"Updated template: {name}")
        else:
            await self._db.save_template(name, content, unique_vars)
            self.notify(f"Saved template: {name}")

        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(None)


class PromptTemplateScreen(Screen):
    """Browse and manage prompt templates."""

    BINDINGS = [
        Binding("n", "new_template", "New"),
        Binding("e", "edit_template", "Edit"),
        Binding("d", "delete_template", "Delete"),
        Binding("enter", "use_template", "Use"),
        Binding("escape", "back", "Back"),
    ]

    DEFAULT_CSS = """
    PromptTemplateScreen {
        background: $surface;
    }
    PromptTemplateScreen > Vertical > .tpl-header {
        height: 1;
        text-align: center;
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    PromptTemplateScreen > Vertical > .tpl-preview {
        height: 8;
        margin: 1;
        padding: 1;
        border: tall $primary;
    }
    """

    def __init__(self, db: Database):
        super().__init__()
        self._db = db
        self._templates: list[dict] = []
        self._delete_pending = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static(" Prompt Templates ", classes="tpl-header")
            yield DataTable(id="tpl-table")
            yield Static("", id="tpl-preview", classes="tpl-preview")
        yield Footer()

    async def on_mount(self) -> None:
        table = self.query_one("#tpl-table", DataTable)
        table.add_columns("Name", "Variables", "Version")
        table.cursor_type = "row"
        await self._refresh()

    async def _refresh(self) -> None:
        table = self.query_one("#tpl-table", DataTable)
        table.clear()
        self._templates = await self._db.list_templates()
        for t in self._templates:
            variables = json.loads(t.get("variables", "[]")) if t.get("variables") else []
            table.add_row(
                t["name"],
                ", ".join(variables) if variables else "—",
                str(t.get("version", 1)),
                key=t["id"],
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None:
            return
        idx = event.cursor_row
        if 0 <= idx < len(self._templates):
            t = self._templates[idx]
            content = t.get("content", "")
            preview = content[:300] + ("..." if len(content) > 300 else "")
            try:
                self.query_one("#tpl-preview", Static).update(preview)
            except Exception:
                pass

    async def action_new_template(self) -> None:
        def on_done(result):
            if result:
                self.call_later(self._refresh)

        await self.app.push_screen(TemplateEditorScreen(self._db), on_done)

    async def action_edit_template(self) -> None:
        table = self.query_one("#tpl-table", DataTable)
        idx = table.cursor_row
        if idx < 0 or idx >= len(self._templates):
            return
        t = self._templates[idx]
        variables = json.loads(t.get("variables", "[]")) if t.get("variables") else []

        def on_done(result):
            if result:
                self.call_later(self._refresh)

        await self.app.push_screen(
            TemplateEditorScreen(
                self._db,
                template_id=t["id"],
                name=t["name"],
                content=t["content"],
                variables=variables,
            ),
            on_done,
        )

    async def action_delete_template(self) -> None:
        table = self.query_one("#tpl-table", DataTable)
        idx = table.cursor_row
        if idx < 0 or idx >= len(self._templates):
            return

        if not self._delete_pending:
            self._delete_pending = True
            self.notify("Press d again to confirm delete", severity="warning")
            return

        self._delete_pending = False
        t = self._templates[idx]
        await self._db.delete_template(t["id"])
        self.notify(f"Deleted: {t['name']}")
        await self._refresh()

    def action_use_template(self) -> None:
        table = self.query_one("#tpl-table", DataTable)
        idx = table.cursor_row
        if 0 <= idx < len(self._templates):
            self.dismiss(self._templates[idx])
        else:
            self.dismiss(None)

    def action_back(self) -> None:
        self.dismiss(None)
