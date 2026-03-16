"""Live inference profiling — TTFT, tokens/sec, token count."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import time


@dataclass
class ProfileMetrics:
    tokens_per_second: float = 0.0
    ttft_ms: float | None = None
    token_count: int = 0
    total_latency_ms: float = 0.0
    peak_tps: float = 0.0


class SparklineBuffer:
    """Fixed-size ring buffer for sparkline data."""

    def __init__(self, capacity: int = 60):
        self._data: deque[float] = deque(maxlen=capacity)
        self._max: float = 0.0

    def push(self, value: float):
        at_capacity = len(self._data) == self._data.maxlen
        self._data.append(value)
        # Recalculate max when buffer is full (old max may have rolled out)
        if at_capacity:
            self._max = max(self._data) if self._data else 0.0
        elif value > self._max:
            self._max = value

    def as_list(self) -> list[float]:
        return list(self._data)

    @property
    def max(self) -> float:
        return self._max

    @property
    def latest(self) -> float:
        return self._data[-1] if self._data else 0.0

    def clear(self):
        self._data.clear()
        self._max = 0.0


class InferenceProfiler:
    """Tracks metrics for a single inference run."""

    def __init__(self):
        self.metrics = ProfileMetrics()
        self._start_time: float | None = None
        self._first_token_time: float | None = None
        # Use deque to avoid unbounded growth — keep last 500 timestamps
        self._token_times: deque[float] = deque(maxlen=500)
        self._window_size = 5.0  # seconds for rolling t/s

    def start(self):
        """Call when inference request is sent."""
        self._start_time = time()
        self._first_token_time = None
        self._token_times.clear()
        self.metrics = ProfileMetrics()

    def on_token(self, token_text: str):
        """Call for each token received."""
        now = time()

        if self._start_time is None:
            self._start_time = now

        if self._first_token_time is None:
            self._first_token_time = now
            self.metrics.ttft_ms = (now - self._start_time) * 1000

        self.metrics.token_count += 1

        # Prune old entries BEFORE appending so the new token is never pruned
        cutoff = now - self._window_size
        while self._token_times and self._token_times[0] <= cutoff:
            self._token_times.popleft()

        self._token_times.append(now)

        if len(self._token_times) >= 2:
            elapsed = self._token_times[-1] - self._token_times[0]
            if elapsed > 0:
                tps = (len(self._token_times) - 1) / elapsed
                self.metrics.tokens_per_second = tps
                if tps > self.metrics.peak_tps:
                    self.metrics.peak_tps = tps

    def finish(self):
        """Call when inference completes."""
        if self._start_time:
            self.metrics.total_latency_ms = (
                (time() - self._start_time) * 1000
            )


class ContextWindowTracker:
    """Estimates token usage against the context window limit."""

    def __init__(self, context_length: int = 4096):
        self.context_length = context_length
        self._estimated_tokens = 0

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return max(1, len(text) // 4)

    def add_message(self, content: str):
        self._estimated_tokens += self.estimate_tokens(content)

    def reset(self):
        self._estimated_tokens = 0

    @property
    def usage_fraction(self) -> float:
        return min(1.0, self._estimated_tokens / max(1, self.context_length))

    @property
    def tokens_used(self) -> int:
        return self._estimated_tokens

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.context_length - self._estimated_tokens)

    @property
    def needs_compaction(self) -> bool:
        """True when usage exceeds 75% — time to auto-compact."""
        return self.usage_fraction > 0.75
