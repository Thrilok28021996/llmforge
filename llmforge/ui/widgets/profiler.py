"""Live profiler widget — shows metrics, gauges, sparklines during inference."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import ProgressBar, Sparkline, Static

from llmforge.domain.hardware import HardwareSnapshot
from llmforge.domain.profiler import ProfileMetrics, SparklineBuffer


class ProfilerWidget(Widget):
    """Right-side profiler panel — updates individual elements, no recompose."""

    DEFAULT_CSS = """
    ProfilerWidget {
        width: 100%;
        height: 100%;
        padding: 0 1;
    }
    ProfilerWidget > .section-title {
        color: $warning;
        text-style: bold;
        text-align: center;
        margin-bottom: 1;
    }
    ProfilerWidget > .metric-row {
        height: 1;
        margin: 0;
    }
    ProfilerWidget > .gauge-label {
        height: 1;
        margin-top: 1;
    }
    ProfilerWidget > ProgressBar {
        margin: 0;
    }
    ProfilerWidget > Sparkline {
        height: 3;
        margin-top: 1;
    }
    ProfilerWidget > .device-label {
        color: $text-muted;
        text-align: center;
        dock: bottom;
        height: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.tps_history = SparklineBuffer(60)
        self.gpu_history = SparklineBuffer(60)
        self._metrics = ProfileMetrics()
        self._hw = HardwareSnapshot()

    def update_metrics(self, metrics: ProfileMetrics):
        """Update metrics display without recomposing the widget tree."""
        self._metrics = metrics
        self.tps_history.push(metrics.tokens_per_second)

        tps = metrics.tokens_per_second
        tps_color = (
            "green"
            if tps > 40
            else "#b4dc64"
            if tps > 15
            else "yellow"
            if tps > 5
            else "red"
        )

        ttft_part = (
            f"[cyan]{metrics.ttft_ms:.0f}ms[/]"
            if metrics.ttft_ms
            else "[dim]—[/]"
        )

        try:
            self.query_one("#prof-tps", Static).update(
                f"  [{tps_color} bold]{tps:.1f}[/] [dim]t/s[/]"
                f"    [dim]TTFT[/] {ttft_part}"
            )
            self.query_one("#prof-tokens", Static).update(
                f"  [dim]{metrics.token_count} tokens[/]"
                f"    [dim]peak[/] [{tps_color}]{metrics.peak_tps:.1f}[/]"
            )
            # Update sparkline data
            spark = self.query_one("#spark-tps", Sparkline)
            spark.data = self.tps_history.as_list()
            self.query_one("#spark-tps-label", Static).update(
                f"  [dim italic]t/s (peak {self.tps_history.max:.0f})[/]"
            )
        except Exception:
            pass  # Widget not yet mounted

    def update_context(self, tokens_used: int, context_length: int):
        """Update context window usage display."""
        pct = min(100, (tokens_used / max(1, context_length)) * 100)
        color = "red" if pct > 85 else "yellow" if pct > 70 else "green"
        try:
            self.query_one("#prof-ctx-label", Static).update(
                f"  [{color}]Context {tokens_used}/{context_length}[/]"
            )
            self.query_one("#prof-ctx-bar", ProgressBar).update(progress=pct)
        except Exception:
            pass

    def update_hardware(self, snap: HardwareSnapshot):
        """Update hardware display without recomposing."""
        self._hw = snap
        self.gpu_history.push(snap.gpu_util)

        gpu_pct = max(0, min(100, snap.gpu_util))
        gpu_color = (
            "red" if gpu_pct > 95 else "yellow" if gpu_pct > 85 else "green"
        )
        ram_frac = snap.ram_used_fraction * 100
        ram_color = (
            "red" if ram_frac > 92 else "yellow" if ram_frac > 80 else "green"
        )

        try:
            self.query_one("#prof-gpu-label", Static).update(
                f"  [{gpu_color}]GPU {gpu_pct:>5.1f}%[/]"
            )
            bars = list(self.query(ProgressBar))
            if len(bars) >= 1:
                bars[0].update(progress=gpu_pct)
            if len(bars) >= 2:
                bars[1].update(progress=ram_frac)

            self.query_one("#prof-ram-label", Static).update(
                f"  [{ram_color}]RAM "
                f"{snap.ram_used_gb:.1f}/{snap.ram_total_gb:.1f}GB[/]"
            )

            # GPU sparkline
            spark = self.query_one("#spark-gpu", Sparkline)
            spark.data = self.gpu_history.as_list()
            self.query_one("#spark-gpu-label", Static).update(
                f"  [dim italic]GPU% (peak {self.gpu_history.max:.0f})[/]"
            )

            if snap.device_name:
                self.query_one("#prof-device", Static).update(
                    f" {snap.device_name} "
                )
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        yield Static(" Profiler ", classes="section-title")

        # Metrics rows (updated in-place)
        yield Static(
            "  [dim]0.0[/] [dim]t/s[/]    [dim]TTFT[/] [dim]—[/]",
            id="prof-tps",
            classes="metric-row",
        )
        yield Static(
            "  [dim]0 tokens[/]    [dim]peak[/] [dim]0.0[/]",
            id="prof-tokens",
            classes="metric-row",
        )

        # GPU gauge
        yield Static("  [green]GPU   0.0%[/]", id="prof-gpu-label",
                      classes="gauge-label")
        yield ProgressBar(total=100, show_eta=False, show_percentage=False)

        # RAM gauge
        yield Static("  [green]RAM 0.0/0.0GB[/]", id="prof-ram-label",
                      classes="gauge-label")
        yield ProgressBar(total=100, show_eta=False, show_percentage=False)

        # Context window gauge
        yield Static("  [dim]Context[/] [dim]—[/]", id="prof-ctx-label",
                      classes="gauge-label")
        yield ProgressBar(total=100, show_eta=False, show_percentage=False,
                          id="prof-ctx-bar")

        # Sparklines
        yield Static("  [dim italic]t/s[/]", id="spark-tps-label")
        yield Sparkline([], summary_function=max, id="spark-tps")

        yield Static("  [dim italic]GPU%[/]", id="spark-gpu-label")
        yield Sparkline([], summary_function=max, id="spark-gpu")

        # Device label
        yield Static("", id="prof-device", classes="device-label")
