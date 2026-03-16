"""Hardware monitoring — CPU, RAM, GPU metrics via psutil + macOS APIs."""

from __future__ import annotations

import asyncio
import platform
import subprocess
from dataclasses import dataclass, field
from time import time

import psutil


@dataclass
class HardwareSnapshot:
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_util: float = 0.0
    gpu_memory_gb: float = 0.0
    device_name: str = ""
    timestamp: float = field(default_factory=time)

    @property
    def ram_used_fraction(self) -> float:
        if self.ram_total_gb <= 0:
            return 0.0
        return self.ram_used_gb / self.ram_total_gb

    @property
    def ram_free_gb(self) -> float:
        return max(0.0, self.ram_total_gb - self.ram_used_gb)


class HardwareMonitor:
    """Polls system metrics at a configurable interval.

    All blocking calls run in a thread executor to avoid freezing the UI.
    """

    def __init__(self, poll_interval_ms: int = 200):
        self._interval = poll_interval_ms / 1000.0
        self._running = False
        self._latest = HardwareSnapshot()
        self._device_name: str | None = None
        self._gpu_failed = False

    def _detect_device(self) -> str:
        """Detect hardware name (runs once, synchronous)."""
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for line in result.stdout.splitlines():
                    if "Chip" in line and ":" in line:
                        return line.split(":", 1)[1].strip()
            except Exception:
                pass
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.stdout.strip()
            except Exception:
                pass
        return platform.processor() or "Unknown"

    @property
    def latest(self) -> HardwareSnapshot:
        return self._latest

    def poll_once(self) -> HardwareSnapshot:
        """Synchronous single poll — call from executor, not event loop."""
        if self._device_name is None:
            self._device_name = self._detect_device()

        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)

        gpu_util = 0.0
        if platform.system() == "Darwin" and not self._gpu_failed:
            gpu_util = self._poll_macos_gpu()

        self._latest = HardwareSnapshot(
            cpu_percent=cpu,
            ram_used_gb=mem.used / (1024**3),
            ram_total_gb=mem.total / (1024**3),
            gpu_util=gpu_util,
            device_name=self._device_name,
        )
        return self._latest

    def _poll_macos_gpu(self) -> float:
        """Get Apple Silicon GPU utilization via ioreg."""
        try:
            result = subprocess.run(
                [
                    "ioreg",
                    "-r",
                    "-d",
                    "1",
                    "-c",
                    "IOAccelerator",
                ],
                capture_output=True,
                text=True,
                timeout=1,
            )
            # Parse GPU utilization from IOAccelerator
            for line in result.stdout.splitlines():
                stripped = line.strip()
                if '"Device Utilization %"' in stripped:
                    # Format: "Device Utilization %" = 42
                    parts = stripped.split("=")
                    if len(parts) == 2:
                        try:
                            return float(parts[1].strip())
                        except ValueError:
                            pass
                if '"gpu-util"' in stripped.lower():
                    parts = stripped.split("=")
                    if len(parts) == 2:
                        try:
                            return float(parts[1].strip())
                        except ValueError:
                            pass
            return 0.0
        except Exception:
            self._gpu_failed = True
            return 0.0

    async def run(self, callback) -> None:
        """Continuously poll (in thread executor) and invoke callback."""
        self._running = True
        loop = asyncio.get_running_loop()

        # Initial CPU reading (psutil needs two calls for accuracy)
        await loop.run_in_executor(
            None, lambda: psutil.cpu_percent(interval=None)
        )
        await asyncio.sleep(0.1)

        while self._running:
            snap = await loop.run_in_executor(None, self.poll_once)
            await callback(snap)
            await asyncio.sleep(self._interval)

    def stop(self):
        self._running = False
