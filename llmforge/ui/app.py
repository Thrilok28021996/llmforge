"""Main Textual application with command palette and screen routing."""

from __future__ import annotations

from pathlib import Path

from textual.app import App
from textual.binding import Binding
from textual.command import Hit, Hits, Provider

from llmforge.config import Config
from llmforge.domain.hardware import HardwareMonitor
from llmforge.domain.models import GenerationParams, strip_backend_prefix
from llmforge.storage.db import Database
from llmforge.ui.screens.chat import ChatScreen
from llmforge.ui.screens.compare import CompareScreen
from llmforge.ui.screens.experiments import ExperimentsScreen
from llmforge.ui.screens.models import ModelLibraryScreen
from llmforge.ui.screens.sessions import SessionListScreen
from llmforge.ui.screens.sweep import ParameterSweepScreen
from llmforge.ui.screens.templates import PromptTemplateScreen

CSS_PATH = Path(__file__).parent / "css" / "app.tcss"


class ForgeCommands(Provider):
    """Command palette provider for LLM Forge actions."""

    async def search(self, query: str) -> Hits:
        app = self.app
        if not isinstance(app, LLMForgeApp):
            return

        commands = [
            ("Open Model Library", "Browse available models", "models"),
            ("Open Experiments", "View run history", "experiments"),
            ("Open Sessions", "Browse chat sessions", "sessions"),
            ("New Chat", "Start a new chat", "new_chat"),
            ("Export Chat", "Export current chat as Markdown", "export"),
            ("Clear Chat", "Clear current conversation", "clear"),
            ("Prompt Templates", "Browse and use prompt templates", "templates"),
        ]

        for name, help_text, action in commands:
            if query.lower() in name.lower() or query.lower() in help_text.lower():
                yield Hit(
                    score=1,
                    match_display=name,
                    command=lambda a=action: app.run_command(a),
                    help=help_text,
                )

    async def discover(self) -> Hits:
        commands = [
            ("Open Model Library", "Browse available models", "models"),
            ("Open Experiments", "View run history", "experiments"),
            ("Open Sessions", "Browse chat sessions", "sessions"),
            ("New Chat", "Start a new chat", "new_chat"),
            ("Export Chat", "Export current chat as Markdown", "export"),
            ("Prompt Templates", "Browse and use prompt templates", "templates"),
        ]
        for name, help_text, action in commands:
            app = self.app
            yield Hit(
                score=1,
                match_display=name,
                command=lambda a=action: app.run_command(a),  # type: ignore
                help=help_text,
            )


class LLMForgeApp(App):
    """LLM Forge — Terminal-based local LLM developer toolkit."""

    TITLE = "LLM Forge"
    SUB_TITLE = "Local LLM Developer Toolkit"
    CSS_PATH = CSS_PATH
    COMMANDS = {ForgeCommands}

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+p", "command_palette", "Commands", show=True),
    ]

    def __init__(
        self,
        config: Config,
        mode: str = "chat",
        model_id: str | None = None,
        model_ids: list[str] | None = None,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self.config = config
        self.mode = mode
        self.model_id = model_id
        self.model_ids = model_ids or []
        self.system_prompt = system_prompt
        self.backend = self._create_backend(config)
        self.hw_monitor = HardwareMonitor(config.profiler.poll_interval_ms)
        self.db = Database()

    @staticmethod
    def _create_backend(config: Config):
        if config.backend == "openai-compat":
            from llmforge.backends.openai_compat import OpenAICompatBackend
            return OpenAICompatBackend(
                base_url=config.openai_compat.base_url,
                api_key=config.openai_compat.api_key,
                timeout=config.openai_compat.timeout_secs,
            )
        if config.backend == "anthropic":
            from llmforge.backends.anthropic import AnthropicBackend
            return AnthropicBackend(config.anthropic)
        if config.backend == "google":
            from llmforge.backends.google import GoogleBackend
            return GoogleBackend(config.google)
        if config.backend == "openrouter":
            from llmforge.backends.openrouter import OpenRouterBackend
            return OpenRouterBackend(config.openrouter)
        if config.backend == "llamacpp":
            from llmforge.backends.llamacpp import LlamaCppBackend
            return LlamaCppBackend(config.llamacpp)
        from llmforge.backends.ollama import OllamaBackend
        return OllamaBackend(config.ollama)

    def _make_params(self) -> GenerationParams:
        return GenerationParams(
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            top_k=self.config.generation.top_k,
            max_tokens=self.config.generation.max_tokens,
            context_length=self.config.generation.context_length,
            repeat_penalty=self.config.generation.repeat_penalty,
            seed=self.config.generation.seed,
        )

    async def on_mount(self) -> None:
        try:
            await self.db.connect()
        except Exception as e:
            self.notify(
                f"Database error: {e} — running without persistence",
                severity="error",
                timeout=5,
            )
        params = self._make_params()

        if self.mode == "chat" and self.model_id:
            await self.push_screen(
                ChatScreen(
                    model_id=self.model_id,
                    backend=self.backend,
                    hw_monitor=self.hw_monitor,
                    db=self.db,
                    params=params,
                    system_prompt=self.system_prompt,
                )
            )
        elif self.mode == "models":
            await self.push_screen(
                ModelLibraryScreen(
                    backend=self.backend, hw_monitor=self.hw_monitor
                )
            )
        elif self.mode == "compare" and self.model_ids:
            await self.push_screen(
                CompareScreen(
                    model_ids=self.model_ids,
                    backend=self.backend,
                    db=self.db,
                )
            )
        elif self.mode == "experiments":
            await self.push_screen(ExperimentsScreen(db=self.db))
        elif self.mode == "sessions":
            await self._open_sessions()
        elif self.mode == "sweep" and self.model_id:
            await self.push_screen(
                ParameterSweepScreen(
                    model_id=self.model_id,
                    backend=self.backend,
                    db=self.db,
                )
            )
        else:
            # Default: model picker → chat
            await self._open_model_picker(params)

    async def _open_model_picker(self, params: GenerationParams | None = None):
        params = params or self._make_params()
        screen = ModelLibraryScreen(
            backend=self.backend, hw_monitor=self.hw_monitor
        )

        def on_model_selected(model_id: str | None):
            if model_id:
                actual_id = strip_backend_prefix(model_id)
                self.push_screen(
                    ChatScreen(
                        model_id=actual_id,
                        backend=self.backend,
                        hw_monitor=self.hw_monitor,
                        db=self.db,
                        params=params,
                    )
                )
            else:
                self.exit()

        await self.push_screen(screen, on_model_selected)

    async def _open_sessions(self):
        screen = SessionListScreen(db=self.db)

        def on_session_selected(session_id: str | None):
            if session_id:
                self.call_later(
                    lambda: self._resume_session(session_id)
                )
            else:
                self.exit()

        await self.push_screen(screen, on_session_selected)

    async def _resume_session(self, session_id: str):
        """Load session messages and open chat."""
        from llmforge.domain.models import ChatMessage

        messages_data = await self.db.get_session_messages(session_id)
        messages = [
            ChatMessage(
                role=m["role"],
                content=m["content"],
                model_id=m.get("model_id"),
            )
            for m in messages_data
        ]

        # Get session metadata
        session = await self.db.get_session(session_id)
        model_id = session["model_id"] if session else "unknown"

        await self.push_screen(
            ChatScreen(
                model_id=model_id,
                backend=self.backend,
                hw_monitor=self.hw_monitor,
                db=self.db,
                params=self._make_params(),
                session_id=session_id,
                initial_messages=messages,
            )
        )

    async def run_command(self, action: str) -> None:
        """Execute a command palette action."""
        if action == "models":
            await self._open_model_picker()
        elif action == "experiments":
            await self.push_screen(ExperimentsScreen(db=self.db))
        elif action == "sessions":
            await self._open_sessions()
        elif action == "new_chat":
            await self._open_model_picker()
        elif action == "export":
            # Delegate to current screen if it's a ChatScreen
            screen = self.screen
            if isinstance(screen, ChatScreen):
                screen.action_export()
        elif action == "clear":
            screen = self.screen
            if isinstance(screen, ChatScreen):
                screen.action_clear()
        elif action == "templates":
            await self.push_screen(PromptTemplateScreen(db=self.db))

    async def on_unmount(self) -> None:
        self.hw_monitor.stop()
        await self.backend.close()
        await self.db.close()
