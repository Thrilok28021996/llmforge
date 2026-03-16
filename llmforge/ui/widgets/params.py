"""Parameter configuration panel — LM Studio-style tuning for generation params."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, ProgressBar, Static

from llmforge.domain.models import GenerationParams


class ParamInput(Widget):
    """A single parameter: label + editable value + visual bar."""

    DEFAULT_CSS = """
    ParamInput {
        height: 3;
        padding: 0 1;
    }
    ParamInput > .param-header {
        height: 1;
    }
    ParamInput > ProgressBar {
        height: 1;
        margin: 0;
    }
    ParamInput > Horizontal {
        height: 1;
    }
    ParamInput > Horizontal > .param-minus, ParamInput > Horizontal > .param-plus {
        width: 3;
        text-align: center;
        color: $accent;
    }
    ParamInput > Horizontal > Input {
        width: 1fr;
        height: 1;
    }
    """

    class Changed(Message):
        def __init__(self, param_id: str, value: float):
            super().__init__()
            self.param_id = param_id
            self.value = value

    def __init__(
        self,
        label: str,
        param_id: str,
        min_val: float,
        max_val: float,
        step: float,
        default: float,
    ):
        super().__init__()
        self.label = label
        self.param_id = param_id
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self._value = default

    def compose(self) -> ComposeResult:
        yield Static(
            f"  [dim]{self.label}:[/] [bold]{self._format(self._value)}[/]",
            classes="param-header",
            id=f"lbl-{self.param_id}",
        )
        yield ProgressBar(total=100, show_eta=False, show_percentage=False,
                          id=f"bar-{self.param_id}")
        with Horizontal():
            yield Static("[cyan][-][/]", classes="param-minus",
                         id=f"minus-{self.param_id}")
            yield Input(
                value=self._format(self._value),
                id=f"input-{self.param_id}",
            )
            yield Static("[cyan][+][/]", classes="param-plus",
                         id=f"plus-{self.param_id}")

    def on_mount(self) -> None:
        try:
            self.query_one(f"#bar-{self.param_id}", ProgressBar).update(
                progress=self._value_pct()
            )
        except Exception:
            pass

    def on_click(self, event) -> None:
        """Handle +/- clicks."""
        target_id = ""
        if hasattr(event, "widget") and event.widget:
            target_id = event.widget.id or ""

        if f"minus-{self.param_id}" in target_id:
            self._adjust(-self.step)
        elif f"plus-{self.param_id}" in target_id:
            self._adjust(self.step)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Parse manual input."""
        try:
            val = float(event.value)
            val = max(self.min_val, min(self.max_val, val))
            self._value = val
            self._sync_display()
            self.post_message(self.Changed(self.param_id, self._value))
        except ValueError:
            # Reset to current value
            event.input.value = self._format(self._value)

    def _adjust(self, delta: float) -> None:
        new_val = self._value + delta
        new_val = max(self.min_val, min(self.max_val, new_val))
        if new_val != self._value:
            self._value = new_val
            self._sync_display()
            self.post_message(self.Changed(self.param_id, self._value))

    def _sync_display(self) -> None:
        try:
            self.query_one(f"#lbl-{self.param_id}", Static).update(
                f"  [dim]{self.label}:[/] [bold]{self._format(self._value)}[/]"
            )
            self.query_one(f"#bar-{self.param_id}", ProgressBar).update(
                progress=self._value_pct()
            )
            self.query_one(f"#input-{self.param_id}", Input).value = self._format(
                self._value
            )
        except Exception:
            pass

    @property
    def value(self) -> float:
        return self._value

    def set_value(self, val: float) -> None:
        self._value = max(self.min_val, min(self.max_val, val))
        self._sync_display()

    def _value_pct(self) -> float:
        rng = self.max_val - self.min_val
        if rng == 0:
            return 0
        return ((self._value - self.min_val) / rng) * 100

    def _format(self, val: float) -> str:
        if self.step >= 1:
            return str(int(val))
        if self.step >= 0.1:
            return f"{val:.1f}"
        return f"{val:.2f}"


class ParameterPanel(Widget):
    """Panel with all generation parameter controls."""

    class ParamsChanged(Message):
        """Emitted when any parameter changes."""

        def __init__(self, params: GenerationParams):
            super().__init__()
            self.params = params

    DEFAULT_CSS = """
    ParameterPanel {
        width: 100%;
        height: 100%;
        padding: 0;
        overflow-y: auto;
    }
    ParameterPanel > .panel-title {
        text-align: center;
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    ParameterPanel > .seed-row {
        height: 3;
        padding: 0 1;
    }
    ParameterPanel > .preset-row {
        height: 2;
        padding: 0 1;
        margin-top: 1;
    }
    """

    def __init__(self, params: GenerationParams | None = None):
        super().__init__()
        self._params = params or GenerationParams()

    def compose(self) -> ComposeResult:
        p = self._params
        yield Static(" Parameters ", classes="panel-title")

        yield ParamInput("Temperature", "temp", 0.0, 2.0, 0.05, p.temperature)
        yield ParamInput("Top P", "top_p", 0.0, 1.0, 0.05, p.top_p)
        yield ParamInput("Top K", "top_k", 1, 100, 1, p.top_k)
        yield ParamInput("Max Tokens", "max_tokens", 64, 32768, 64, p.max_tokens)
        yield ParamInput("Context Length", "ctx_len", 512, 131072, 512, p.context_length)
        yield ParamInput("Repeat Penalty", "repeat", 1.0, 2.0, 0.05, p.repeat_penalty)

        with Vertical(classes="seed-row"):
            yield Static("  [dim]Seed (blank = random):[/]")
            yield Input(
                value=str(p.seed) if p.seed is not None else "",
                placeholder="random",
                id="seed-input",
            )

        # Preset keys
        yield Static(
            "  [cyan]Presets:[/] "
            "[dim][[/][bold]1[/][dim]] Creative[/]  "
            "[dim][[/][bold]2[/][dim]] Balanced[/]  "
            "[dim][[/][bold]3[/][dim]] Precise[/]  "
            "[dim][[/][bold]4[/][dim]] Code[/]",
            classes="preset-row",
        )

    def on_param_input_changed(self, event: ParamInput.Changed) -> None:
        """Propagate any parameter change."""
        self.post_message(self.ParamsChanged(self.get_params()))

    def get_params(self) -> GenerationParams:
        """Read current values into a GenerationParams."""
        inputs = {p.param_id: p for p in self.query(ParamInput)}
        seed_val = None
        try:
            seed_text = self.query_one("#seed-input", Input).value.strip()
            if seed_text:
                seed_val = int(seed_text)
        except (ValueError, Exception):
            pass

        return GenerationParams(
            temperature=inputs["temp"].value if "temp" in inputs else 0.7,
            top_p=inputs["top_p"].value if "top_p" in inputs else 0.9,
            top_k=int(inputs["top_k"].value) if "top_k" in inputs else 40,
            max_tokens=int(inputs["max_tokens"].value) if "max_tokens" in inputs else 2048,
            context_length=int(inputs["ctx_len"].value) if "ctx_len" in inputs else 4096,
            repeat_penalty=inputs["repeat"].value if "repeat" in inputs else 1.1,
            seed=seed_val,
        )

    def set_params(self, params: GenerationParams) -> None:
        """Set all inputs from a GenerationParams."""
        self._params = params
        mapping = {
            "temp": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "max_tokens": params.max_tokens,
            "ctx_len": params.context_length,
            "repeat": params.repeat_penalty,
        }
        for param_input in self.query(ParamInput):
            if param_input.param_id in mapping:
                param_input.set_value(mapping[param_input.param_id])
        try:
            self.query_one("#seed-input", Input).value = (
                str(params.seed) if params.seed is not None else ""
            )
        except Exception:
            pass

    def apply_preset(self, preset: str) -> None:
        """Apply a named preset."""
        presets = {
            "creative": GenerationParams(
                temperature=1.2, top_p=0.95, top_k=80,
                max_tokens=4096, repeat_penalty=1.05,
            ),
            "balanced": GenerationParams(
                temperature=0.7, top_p=0.9, top_k=40,
                max_tokens=2048, repeat_penalty=1.1,
            ),
            "precise": GenerationParams(
                temperature=0.2, top_p=0.8, top_k=20,
                max_tokens=2048, repeat_penalty=1.15,
            ),
            "code": GenerationParams(
                temperature=0.1, top_p=0.95, top_k=40,
                max_tokens=4096, repeat_penalty=1.0,
            ),
        }
        if preset in presets:
            p = presets[preset]
            p.context_length = self._params.context_length
            self.set_params(p)
            self.post_message(self.ParamsChanged(p))
