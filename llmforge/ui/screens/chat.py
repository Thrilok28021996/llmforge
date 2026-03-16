"""Chat screen — the primary interface for conversing with a model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Markdown, Static

from llmforge.domain.models import (
    ChatMessage,
    GenerationParams,
    InferenceRequest,
)
from llmforge.domain.profiler import ContextWindowTracker, InferenceProfiler
from llmforge.ui.widgets.params import ParameterPanel
from llmforge.ui.widgets.profiler import ProfilerWidget

logger = logging.getLogger(__name__)

# Maximum response length (chars) to prevent runaway generation
MAX_RESPONSE_CHARS = 200_000

if TYPE_CHECKING:
    from llmforge.backends.ollama import OllamaBackend
    from llmforge.domain.hardware import HardwareMonitor
    from llmforge.storage.db import Database


class MessageWidget(Static):
    """A single chat message with role badge and streaming Markdown content."""

    DEFAULT_CSS = """
    MessageWidget {
        width: 100%;
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
    }
    MessageWidget.user-msg {
        border-left: thick $accent;
        background: $primary-background-darken-3;
    }
    MessageWidget.assistant-msg {
        border-left: thick $success;
        background: $surface-darken-1;
    }
    MessageWidget.system-msg {
        border-left: thick $warning;
        background: $surface-darken-2;
    }
    MessageWidget > .role-badge {
        text-style: bold;
        height: 1;
    }
    MessageWidget > Markdown {
        padding: 0 0 0 1;
        margin: 0;
    }
    MessageWidget > .msg-text {
        padding: 0 0 0 1;
    }
    """

    def __init__(self, role: str, content: str, model_name: str | None = None):
        super().__init__()
        self.role = role
        self.content = content
        self.model_name = model_name
        self.add_class(f"{role}-msg")

    def compose(self) -> ComposeResult:
        if self.role == "user":
            badge = "[black on cyan] YOU [/]"
        elif self.role == "assistant":
            name = self.model_name or "ASSISTANT"
            badge = f"[black on green] {name.upper()} [/]"
        else:
            badge = "[black on yellow] SYSTEM [/]"

        yield Static(badge, classes="role-badge")
        # Use Markdown widget for assistant, plain Static for user
        if self.role == "assistant":
            yield Markdown(self.content, id=f"md-{id(self)}")
        else:
            yield Static(self.content, classes="msg-text")


class ChatScreen(Screen):
    """Main chat interface with live profiler panel."""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear", "Clear chat", show=True),
        Binding("ctrl+u", "clear_input", "Clear input", show=False),
        Binding("ctrl+e", "export", "Export", show=True),
        Binding("ctrl+m", "switch_model", "Switch model", show=True),
        Binding("ctrl+t", "toggle_params", "Params", show=True),
        Binding("ctrl+r", "use_template", "Template", show=True),
        Binding("ctrl+f", "fork_chat", "Fork", show=True),
        Binding("ctrl+a", "toggle_agent", "Agent", show=True),
        Binding("ctrl+w", "toggle_web_search", "WebSearch", show=False),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("1", "preset_creative", "Creative", show=False),
        Binding("2", "preset_balanced", "Balanced", show=False),
        Binding("3", "preset_precise", "Precise", show=False),
        Binding("4", "preset_code", "Code", show=False),
    ]

    DEFAULT_CSS = """
    ChatScreen {
        layout: horizontal;
    }
    #messages-col {
        width: 3fr;
        min-width: 40;
    }
    #profiler-col {
        width: 1fr;
        min-width: 26;
        max-width: 42;
        border-left: thick $primary-background-darken-2;
    }
    #params-col {
        width: 1fr;
        min-width: 28;
        max-width: 38;
        border-left: thick $primary-background-darken-2;
        display: none;
    }
    #params-col.visible {
        display: block;
    }
    #message-scroll {
        height: 1fr;
    }
    #input-box {
        dock: bottom;
        height: 3;
        border-top: thick $primary-background-darken-2;
        padding: 0 1;
    }
    #chat-input {
        width: 100%;
    }
    #status-line {
        dock: bottom;
        height: 1;
        background: $primary-background-darken-3;
        padding: 0 1;
    }
    #welcome-container {
        align: center middle;
        width: 100%;
        height: 100%;
    }
    #welcome-panel {
        width: 64;
        height: auto;
        border: round $primary;
        padding: 1 2;
        background: $surface-darken-1;
    }
    #welcome-title {
        text-align: center;
        text-style: bold;
        color: $accent;
    }
    """

    def __init__(
        self,
        model_id: str,
        backend: OllamaBackend,
        hw_monitor: HardwareMonitor,
        db: Database,
        params: GenerationParams | None = None,
        session_id: str | None = None,
        initial_messages: list[ChatMessage] | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.backend = backend
        self.hw_monitor = hw_monitor
        self.db = db
        self.params = params or GenerationParams()
        self.system_prompt = system_prompt
        self.messages: list[ChatMessage] = initial_messages or []
        # Prepend system prompt as the first message if provided
        if system_prompt and not any(m.role == "system" for m in self.messages):
            self.messages.insert(0, ChatMessage(role="system", content=system_prompt))
        self.profiler = InferenceProfiler()
        self.profiler_widget = ProfilerWidget()
        self.param_panel = ParameterPanel(self.params)
        self.ctx_tracker = ContextWindowTracker(self.params.context_length)
        self._mcp_clients: list = []
        self._agent_mode = False
        self._web_search_enabled = False
        self._streaming = False
        self._current_response = ""
        self._current_msg_widget: MessageWidget | None = None
        self._md_stream = None  # MarkdownStream for incremental rendering
        self._session_id = session_id
        self._inference_worker = None
        self._generation_id: int = 0  # Incremented per inference, prevents stale workers
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._response_chunks: list[str] = []  # accumulate, join once

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with Horizontal():
            with Vertical(id="messages-col"):
                with VerticalScroll(id="message-scroll"):
                    with Vertical(id="welcome-container"):
                        with Vertical(id="welcome-panel"):
                            yield Static(
                                "[bold cyan]LLM Forge[/]",
                                id="welcome-title",
                            )
                            yield Static(
                                f"[dim]Model:[/] [bold]{self.model_id}[/]",
                            )
                            yield Static("")
                            yield Static(
                                "[dim]Type a message to start chatting.[/]"
                            )
                            yield Static(
                                f"[dim]temp={self.params.temperature}  "
                                f"ctx={self.params.context_length}  "
                                f"top_p={self.params.top_p}[/]"
                            )
                            yield Static("")
                            yield Static(
                                "[cyan]Enter[/] [dim]Send[/]  "
                                "[cyan]Ctrl+E[/] [dim]Export[/]  "
                                "[cyan]Ctrl+M[/] [dim]Switch model[/]"
                            )
                            yield Static(
                                "[cyan]Ctrl+L[/] [dim]Clear[/]  "
                                "[cyan]Esc[/] [dim]Cancel[/]  "
                                "[cyan]Ctrl+C[/] [dim]Quit[/]"
                            )
                yield Static(
                    self._format_status(streaming=False),
                    id="status-line",
                )
                with Vertical(id="input-box"):
                    yield Input(
                        placeholder="Type your message... (Enter to send)",
                        id="chat-input",
                    )

            with Vertical(id="profiler-col"):
                yield self.profiler_widget

            with Vertical(id="params-col"):
                yield self.param_panel

        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#chat-input", Input).focus()
        self._start_hw_polling()
        self._init_mcp()
        # If resuming a session, restore messages
        if self.messages:
            self._restore_messages()

    @work(thread=False)
    async def _init_mcp(self):
        """Connect to configured MCP servers."""
        from llmforge.mcp.client import MCPClient

        try:
            config = self.app.config  # type: ignore
            for srv in config.mcp.servers:
                client = MCPClient(srv.name, srv.command, srv.env)
                if await client.connect():
                    self._mcp_clients.append(client)
                    tools = [t.name for t in client.tools]
                    logger.info("MCP %s: %d tools (%s)", srv.name, len(tools), ", ".join(tools))
                else:
                    logger.warning("Failed to connect to MCP server: %s", srv.name)
        except Exception:
            logger.warning("MCP initialization failed", exc_info=True)

    @work(thread=False)
    async def _start_hw_polling(self):
        async def on_snap(snap):
            self.profiler_widget.update_hardware(snap)
        await self.hw_monitor.run(on_snap)

    def _restore_messages(self):
        """Restore previous messages when resuming a session."""
        welcome = self.query("#welcome-container")
        for w in welcome:
            w.display = False
        scroll = self.query_one("#message-scroll", VerticalScroll)
        for msg in self.messages:
            widget = MessageWidget(
                msg.role, msg.content, model_name=msg.model_id
            )
            scroll.mount(widget)
        scroll.scroll_end(animate=False)

    @staticmethod
    def _expand_file_refs(text: str) -> str:
        """Expand @file references: @path/to/file → file contents inline."""
        import re

        def replace_ref(match: re.Match) -> str:
            path = Path(match.group(1)).expanduser()
            if path.is_file():
                try:
                    content = path.read_text(errors="replace")
                    if len(content) > 50_000:
                        content = content[:50_000] + "\n... (truncated at 50k chars)"
                    return f"\n```{path.suffix.lstrip('.')}\n# {path.name}\n{content}\n```\n"
                except Exception:
                    return match.group(0)  # Keep original if unreadable
            return match.group(0)

        return re.sub(r"@([/\w.~\-]+\.\w+)", replace_ref, text)

    @on(Input.Submitted, "#chat-input")
    async def on_submit(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self._streaming:
            return

        event.input.value = ""

        # Expand @file references before sending
        text = self._expand_file_refs(text)

        # Hide welcome on first message
        welcome = self.query("#welcome-container")
        if welcome:
            for w in welcome:
                w.display = False

        # Create session on first message
        if self._session_id is None:
            try:
                self._session_id = await self.db.create_session(
                    model_id=self.model_id,
                    name=text[:50],
                    params=self.params.to_dict(),
                )
            except Exception:
                logger.warning("Failed to create session", exc_info=True)

        # Add user message and track context
        user_msg = ChatMessage(role="user", content=text)
        self.messages.append(user_msg)
        self.ctx_tracker.add_message(text)
        self.profiler_widget.update_context(
            self.ctx_tracker.tokens_used, self.ctx_tracker.context_length
        )
        scroll = self.query_one("#message-scroll", VerticalScroll)
        user_widget = MessageWidget("user", text)
        await scroll.mount(user_widget)

        # Persist to session
        if self._session_id:
            try:
                await self.db.add_session_message(
                    self._session_id, "user", text
                )
            except Exception:
                logger.warning("Failed to save user message", exc_info=True)

        # Add assistant placeholder
        self._current_response = ""
        self._response_chunks = []
        self._current_msg_widget = MessageWidget(
            "assistant", "", model_name=self.model_id
        )
        await scroll.mount(self._current_msg_widget)
        scroll.anchor()  # Auto-scroll to bottom as content grows

        # Set up MarkdownStream for incremental rendering
        md_widget = self._current_msg_widget.query_one(Markdown)
        self._md_stream = Markdown.get_stream(md_widget)

        # Start streaming
        self._streaming = True
        self._generation_id += 1
        self._update_status(streaming=True)
        self.profiler.start()
        self._inference_worker = self._run_inference(text, self._generation_id)

    @work(thread=False)
    async def _run_inference(self, prompt: str, gen_id: int):
        # Build context from RAG and web search
        context_parts: list[str] = []

        rag_context = await self._get_rag_context(prompt)
        if rag_context:
            context_parts.append(rag_context)

        web_context = await self._get_web_search_context(prompt)
        if web_context:
            context_parts.append(web_context)

        system_ctx = "\n\n".join(context_parts) if context_parts else None

        request = InferenceRequest(
            model_id=self.model_id,
            messages=self.messages,
            params=self.params,
            system_prompt=system_ctx,
        )

        prompt_tokens = 0
        completion_tokens = 0
        total_chars = 0

        # Agent mode: plan/execute with built-in tools + MCP
        if self._agent_mode:
            from llmforge.tools.agent import run_agent_loop

            config = self.app.config  # type: ignore
            ws_cfg = config.web_search
            stream = run_agent_loop(
                self.backend,
                self.model_id,
                self.messages,
                self.params,
                system_prompt=system_ctx,
                web_search_config={
                    "provider": ws_cfg.provider,
                    "max_results": ws_cfg.max_results,
                    "searxng_url": ws_cfg.searxng_url,
                    "tavily_api_key": ws_cfg.tavily_api_key,
                },
                mcp_clients=self._mcp_clients,
            )
        elif self._mcp_clients:
            # Use MCP tool loop if tools are available
            from llmforge.mcp.tool_loop import run_with_tools
            stream = run_with_tools(self.backend, request, self._mcp_clients)
        else:
            stream = self.backend.generate(request)

        try:
            async for chunk in stream:
                if chunk.is_final:
                    self.profiler.finish()
                    self.profiler_widget.update_metrics(self.profiler.metrics)
                    if chunk.prompt_tokens:
                        prompt_tokens = chunk.prompt_tokens
                    if chunk.completion_tokens:
                        completion_tokens = chunk.completion_tokens
                    break

                if chunk.text:
                    total_chars += len(chunk.text)
                    # Guard against runaway generation
                    if total_chars > MAX_RESPONSE_CHARS:
                        self._response_chunks.append(
                            "\n\n*[Response truncated — exceeded "
                            f"{MAX_RESPONSE_CHARS // 1000}k chars]*"
                        )
                        if self._md_stream:
                            await self._md_stream.write(
                                "\n\n*[Response truncated]*"
                            )
                        await self.backend.cancel()
                        self.profiler.finish()
                        break

                    self._response_chunks.append(chunk.text)
                    self.profiler.on_token(chunk.text)
                    self.profiler_widget.update_metrics(self.profiler.metrics)
                    # Incremental append via MarkdownStream (batches internally)
                    if self._md_stream:
                        await self._md_stream.write(chunk.text)

        except Exception as e:
            error_msg = f"\n\n**Error:** {e}"
            self._response_chunks.append(error_msg)
            if self._md_stream:
                await self._md_stream.write(error_msg)

        # Stop the markdown stream (flushes remaining content)
        if self._md_stream:
            try:
                await self._md_stream.stop()
            except Exception:
                pass
            self._md_stream = None

        # Join final response text
        self._current_response = "".join(self._response_chunks)

        # Guard: if a newer generation started, don't modify shared state
        if gen_id != self._generation_id:
            return

        # Update token counters
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens

        # Finalize
        self._streaming = False
        self._update_status(streaming=False)

        if self._current_response:
            self.messages.append(
                ChatMessage(
                    role="assistant",
                    content=self._current_response,
                    model_id=self.model_id,
                )
            )
            self.ctx_tracker.add_message(self._current_response)
            self.profiler_widget.update_context(
                self.ctx_tracker.tokens_used, self.ctx_tracker.context_length
            )
            if self.ctx_tracker.needs_compaction:
                self.notify(
                    "Context window >75% full — responses may degrade",
                    severity="warning",
                )
            # Persist to session
            if self._session_id:
                try:
                    await self.db.add_session_message(
                        self._session_id,
                        "assistant",
                        self._current_response,
                        model_id=self.model_id,
                    )
                except Exception:
                    logger.warning("Failed to save assistant message", exc_info=True)

        # Record run
        try:
            await self.db.record_run(
                model_id=self.model_id,
                prompt=(
                    self.messages[-2].content
                    if len(self.messages) >= 2
                    else ""
                ),
                response=self._current_response,
                params=self.params.to_dict(),
                ttft_ms=self.profiler.metrics.ttft_ms,
                tokens_per_second=self.profiler.metrics.tokens_per_second,
                total_latency_ms=self.profiler.metrics.total_latency_ms,
                prompt_tokens=prompt_tokens or None,
                completion_tokens=completion_tokens or None,
                hw_device=self.hw_monitor.latest.device_name,
                hw_cpu_util_avg=self.hw_monitor.latest.cpu_percent,
                hw_ram_used_gb=self.hw_monitor.latest.ram_used_gb,
            )
        except Exception:
            logger.warning("Failed to record run", exc_info=True)

        self._current_msg_widget = None

    def _format_status(self, streaming: bool) -> str:
        indicator = (
            "[green bold]● streaming[/]" if streaming else "[dim]○ idle[/]"
        )
        tokens_str = ""
        total = self._total_prompt_tokens + self._total_completion_tokens
        if total > 0:
            tokens_str = (
                f"  [dim]tokens:[/] {self._total_prompt_tokens}"
                f"+{self._total_completion_tokens}"
            )
        mode_badges = ""
        if self._agent_mode:
            mode_badges += "  [bold magenta]AGENT[/]"
        if self._web_search_enabled:
            mode_badges += "  [bold blue]WEB[/]"
        return (
            f"  [dim]Model:[/] [bold]{self.model_id}[/]  "
            f"[dim]temp:{self.params.temperature}  "
            f"ctx:{self.params.context_length}[/]"
            f"{tokens_str}{mode_badges}    {indicator}"
        )

    def _update_status(self, streaming: bool):
        status = self.query_one("#status-line", Static)
        status.update(self._format_status(streaming))

    def action_clear(self) -> None:
        self.messages.clear()
        self._session_id = None
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self.ctx_tracker.reset()
        # Remove only message widgets, not the welcome container
        for widget in list(self.query(MessageWidget)):
            widget.remove()
        welcome = self.query("#welcome-container")
        for w in welcome:
            w.display = True
        self._update_status(streaming=False)

    def action_clear_input(self) -> None:
        self.query_one("#chat-input", Input).value = ""

    async def action_cancel(self) -> None:
        if self._streaming:
            await self.backend.cancel()
            if self._inference_worker and self._inference_worker.is_running:
                self._inference_worker.cancel()
            if self._md_stream:
                try:
                    await self._md_stream.stop()
                except Exception:
                    pass
                self._md_stream = None
            self._current_response = "".join(self._response_chunks)
            self._streaming = False
            self._inference_worker = None
            self._update_status(streaming=False)

    def action_export(self) -> None:
        """Export chat as Markdown file."""
        if not self.messages:
            self.notify("No messages to export.", severity="warning")
            return

        lines = [f"# LLM Forge Chat — {self.model_id}\n"]
        lines.append(
            f"Parameters: temp={self.params.temperature}, "
            f"ctx={self.params.context_length}, "
            f"top_p={self.params.top_p}\n"
        )
        lines.append("---\n")

        for msg in self.messages:
            if msg.role == "user":
                lines.append(f"## You\n\n{msg.content}\n")
            elif msg.role == "assistant":
                name = msg.model_id or self.model_id
                lines.append(f"## {name}\n\n{msg.content}\n")
            else:
                lines.append(f"## System\n\n{msg.content}\n")

        export_dir = Path.home() / ".llmforge" / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        import re
        safe_model = re.sub(r"[^\w\-.]", "_", self.model_id)
        filename = f"chat_{safe_model}_{ts}.md"
        path = export_dir / filename
        try:
            path.write_text("\n".join(lines))
            self.notify(f"Exported to {path}")
        except OSError as e:
            self.notify(f"Export failed: {e}", severity="error")

    async def action_switch_model(self) -> None:
        """Open model picker overlay to switch models mid-chat."""
        from llmforge.ui.screens.models import ModelLibraryScreen

        def on_selected(model_id: str | None):
            if model_id:
                from llmforge.domain.models import strip_backend_prefix
                actual_id = strip_backend_prefix(model_id)
                self.model_id = actual_id
                self._update_status(streaming=False)
                self.notify(f"Switched to {actual_id}")

        await self.app.push_screen(
            ModelLibraryScreen(
                backend=self.backend,
                hw_monitor=self.hw_monitor,
            ),
            on_selected,
        )

    def action_toggle_params(self) -> None:
        """Toggle the parameter panel visibility."""
        params_col = self.query_one("#params-col")
        params_col.toggle_class("visible")

    def on_parameter_panel_params_changed(self, event: ParameterPanel.ParamsChanged) -> None:
        """Update params when sliders change."""
        self.params = event.params
        self.ctx_tracker = ContextWindowTracker(self.params.context_length)
        self._update_status(streaming=False)

    def action_preset_creative(self) -> None:
        self.param_panel.apply_preset("creative")

    def action_preset_balanced(self) -> None:
        self.param_panel.apply_preset("balanced")

    def action_preset_precise(self) -> None:
        self.param_panel.apply_preset("precise")

    def action_preset_code(self) -> None:
        self.param_panel.apply_preset("code")

    async def _get_rag_context(self, query: str) -> str | None:
        """Build RAG context if enabled and documents exist."""
        try:
            config = self.app.config  # type: ignore
            if not config.rag.enabled:
                return None

            from llmforge.rag.context import build_rag_context
            from llmforge.rag.store import RAGStore

            store = RAGStore(self.db.db)
            await store.ensure_schema()

            return await build_rag_context(
                query,
                store,
                top_k=config.rag.top_k,
                embedding_model=config.rag.embedding_model,
                ollama_url=config.ollama.base_url,
            )
        except Exception:
            logger.warning("RAG context retrieval failed", exc_info=True)
            return None

    async def action_use_template(self) -> None:
        """Open template browser and insert selected template into input."""
        from llmforge.ui.screens.templates import PromptTemplateScreen

        def on_template(result):
            if result and isinstance(result, dict):
                content = result.get("content", "")
                inp = self.query_one("#chat-input", Input)
                inp.value = content
                inp.focus()

        await self.app.push_screen(PromptTemplateScreen(self.db), on_template)

    async def action_fork_chat(self) -> None:
        """Fork the current conversation from the current point."""
        if not self._session_id or not self.messages:
            self.notify("No conversation to fork", severity="warning")
            return
        try:
            msg_count = len([m for m in self.messages if m.role != "system"])
            new_id = await self.db.fork_session(
                self._session_id, len(self.messages) - 1
            )
            self.notify(f"Forked session ({msg_count} messages) → {new_id[:8]}")
        except Exception as e:
            self.notify(f"Fork failed: {e}", severity="error")

    def action_toggle_agent(self) -> None:
        """Toggle agent mode (tool use + planning)."""
        self._agent_mode = not self._agent_mode
        mode = "ON" if self._agent_mode else "OFF"
        self.notify(f"Agent mode: {mode}")
        self._update_status(streaming=False)

    def action_toggle_web_search(self) -> None:
        """Toggle web search augmentation."""
        self._web_search_enabled = not self._web_search_enabled
        mode = "ON" if self._web_search_enabled else "OFF"
        self.notify(f"Web search: {mode}")

    async def _get_web_search_context(self, query: str) -> str | None:
        """Run a web search and format results as context."""
        if not self._web_search_enabled:
            return None
        try:
            config = self.app.config  # type: ignore
            ws_cfg = config.web_search
            from llmforge.rag.web_search import format_search_context, web_search

            results = await web_search(
                query,
                provider=ws_cfg.provider,
                max_results=ws_cfg.max_results,
                searxng_url=ws_cfg.searxng_url,
                tavily_api_key=ws_cfg.tavily_api_key,
            )
            return format_search_context(results)
        except Exception:
            logger.warning("Web search failed", exc_info=True)
            return None

    async def action_quit(self) -> None:
        # Clean up MCP clients
        for client in self._mcp_clients:
            await client.close()
        self.hw_monitor.stop()
        self.app.exit()
