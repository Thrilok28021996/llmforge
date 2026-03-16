"""Multi-language sandboxed code execution for the code interpreter."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 10_000
DEFAULT_TIMEOUT = 30

# Language configs: extension, command builder, env overrides
LANGUAGES: dict[str, dict] = {
    "python": {
        "ext": ".py",
        "cmd": lambda path: ["python3", "-u", str(path)],
        "env_extra": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        },
    },
    "javascript": {
        "ext": ".js",
        "cmd": lambda path: ["node", str(path)],
        "env_extra": {},
    },
    "typescript": {
        "ext": ".ts",
        "cmd": lambda path: ["npx", "tsx", str(path)],
        "env_extra": {},
    },
    "bash": {
        "ext": ".sh",
        "cmd": lambda path: ["bash", str(path)],
        "env_extra": {},
    },
    "ruby": {
        "ext": ".rb",
        "cmd": lambda path: ["ruby", str(path)],
        "env_extra": {},
    },
    "go": {
        "ext": ".go",
        "cmd": lambda path: ["go", "run", str(path)],
        "env_extra": {},
    },
    "rust": {
        "ext": ".rs",
        "cmd": lambda path: _rust_cmd(path),
        "env_extra": {},
    },
    "c": {
        "ext": ".c",
        "cmd": lambda path: _c_cmd(path),
        "env_extra": {},
    },
    "cpp": {
        "ext": ".cpp",
        "cmd": lambda path: _cpp_cmd(path),
        "env_extra": {},
    },
}


def _rust_cmd(path: Path) -> list[str]:
    """Compile and run a Rust file."""
    out = path.with_suffix("")
    return ["sh", "-c", f"rustc {path} -o {out} && {out}"]


def _c_cmd(path: Path) -> list[str]:
    """Compile and run a C file."""
    out = path.with_suffix("")
    return ["sh", "-c", f"cc {path} -o {out} && {out}"]


def _cpp_cmd(path: Path) -> list[str]:
    """Compile and run a C++ file."""
    out = path.with_suffix("")
    return ["sh", "-c", f"c++ {path} -o {out} && {out}"]


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    return_code: int
    language: str = "python"
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.return_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"STDERR:\n{self.stderr.strip()}")
        if self.timed_out:
            parts.append(f"[Timed out after {DEFAULT_TIMEOUT}s]")
        return "\n".join(parts) if parts else "(no output)"


def detect_language(code: str) -> str:
    """Auto-detect language from code content."""
    code_stripped = code.strip()

    # Explicit markers
    if code_stripped.startswith("#!/usr/bin/env python"):
        return "python"
    if code_stripped.startswith("#!/usr/bin/env node"):
        return "javascript"
    if code_stripped.startswith("#!/bin/bash") or code_stripped.startswith("#!/bin/sh"):
        return "bash"

    # Language-specific patterns (order matters — check specific before generic)
    if "package main" in code or ("func main()" in code and "fmt." in code):
        return "go"
    if "#include" in code:
        if "iostream" in code or "std::" in code or "cout" in code:
            return "cpp"
        return "c"
    if "fn main()" in code or "println!" in code or "let mut " in code:
        return "rust"
    if "console.log" in code or "const " in code or "require(" in code:
        return "javascript"
    if "puts " in code and "end\n" in code:
        return "ruby"
    if "def " in code or "import " in code or "print(" in code:
        return "python"

    return "python"  # Default


def available_languages() -> list[str]:
    """Return languages whose runtimes are available on this system."""
    available = []
    runtime_checks = {
        "python": "python3",
        "javascript": "node",
        "typescript": "npx",
        "bash": "bash",
        "ruby": "ruby",
        "go": "go",
        "rust": "rustc",
        "c": "cc",
        "cpp": "c++",
    }
    for lang, cmd in runtime_checks.items():
        if shutil.which(cmd):
            available.append(lang)
    return available


async def execute_code(
    code: str,
    language: str = "auto",
    timeout: int = DEFAULT_TIMEOUT,
) -> ExecResult:
    """Execute code in the specified language.

    If language is "auto", attempts to detect from code content.
    """
    if language == "auto":
        language = detect_language(code)

    lang_config = LANGUAGES.get(language)
    if not lang_config:
        return ExecResult(
            stdout="",
            stderr=f"Unsupported language: {language}",
            return_code=-1,
            language=language,
        )

    with tempfile.TemporaryDirectory(prefix="llmforge_exec_") as tmpdir:
        script_path = Path(tmpdir) / f"script{lang_config['ext']}"
        script_path.write_text(code, encoding="utf-8")

        cmd = lang_config["cmd"](script_path)

        env = {
            "PATH": "/usr/bin:/usr/local/bin:/opt/homebrew/bin",
            "HOME": tmpdir,
            "TMPDIR": tmpdir,
        }
        env.update(lang_config.get("env_extra", {}))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
                env=env,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                stdout = stdout_bytes.decode(
                    "utf-8", errors="replace"
                )[:MAX_OUTPUT_CHARS]
                stderr = stderr_bytes.decode(
                    "utf-8", errors="replace"
                )[:MAX_OUTPUT_CHARS]
                return ExecResult(
                    stdout=stdout,
                    stderr=stderr,
                    return_code=proc.returncode or 0,
                    language=language,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return ExecResult(
                    stdout="",
                    stderr="",
                    return_code=-1,
                    language=language,
                    timed_out=True,
                )

        except FileNotFoundError:
            return ExecResult(
                stdout="",
                stderr=(
                    f"Runtime not found for {language}. "
                    f"Install the {language} runtime to use this."
                ),
                return_code=-1,
                language=language,
            )
        except Exception as e:
            return ExecResult(
                stdout="",
                stderr=str(e),
                return_code=-1,
                language=language,
            )


# Backwards compatibility alias
async def execute_python(
    code: str, timeout: int = DEFAULT_TIMEOUT
) -> ExecResult:
    """Execute Python code (convenience wrapper)."""
    return await execute_code(code, language="python", timeout=timeout)
