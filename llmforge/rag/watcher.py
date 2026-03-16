"""Folder watcher for automatic RAG document ingestion.

Watches configured directories for file changes and auto-ingests
new/modified documents. Like GPT4All's LocalDocs but for the terminal.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# File extensions to watch
WATCHED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".html", ".css", ".json", ".yaml", ".yml", ".toml", ".xml",
    ".csv", ".log", ".sh", ".bash", ".zsh", ".sql", ".r",
    ".tex", ".rst", ".org", ".pdf",
}


class RAGFolderWatcher:
    """Watches directories and auto-ingests documents into RAG store."""

    def __init__(
        self,
        watch_dirs: list[str],
        store,  # RAGStore
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        chunk_size: int = 512,
        overlap: int = 64,
    ):
        self._watch_dirs = [
            Path(d).expanduser() for d in watch_dirs
        ]
        self._store = store
        self._embedding_model = embedding_model
        self._ollama_url = ollama_url
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._task: asyncio.Task | None = None
        self._known_files: dict[str, float] = {}  # path → mtime

    async def start(self) -> None:
        """Start watching in the background."""
        if self._task is not None:
            return

        # Initial scan
        await self._initial_scan()

        self._task = asyncio.create_task(self._watch_loop())
        logger.info(
            "RAG folder watcher started for: %s",
            [str(d) for d in self._watch_dirs],
        )

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _initial_scan(self) -> None:
        """Scan all directories and note existing files."""
        existing_docs = await self._store.list_documents()
        existing_paths = {d.get("path", "") for d in existing_docs}

        for d in self._watch_dirs:
            if not d.is_dir():
                continue
            for f in d.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in WATCHED_EXTENSIONS:
                    continue
                if f.name.startswith("."):
                    continue

                path_str = str(f)
                mtime = f.stat().st_mtime
                self._known_files[path_str] = mtime

                # Ingest if not already in store
                if path_str not in existing_paths:
                    await self._ingest_file(f)

    async def _watch_loop(self) -> None:
        """Watch for file changes using watchfiles."""
        try:
            from watchfiles import Change, awatch
        except ImportError:
            logger.warning(
                "watchfiles not installed — folder watching disabled. "
                "Install with: pip install watchfiles"
            )
            return

        watch_paths = [
            str(d) for d in self._watch_dirs if d.is_dir()
        ]
        if not watch_paths:
            return

        try:
            async for changes in awatch(*watch_paths):
                for change_type, path_str in changes:
                    path = Path(path_str)
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in WATCHED_EXTENSIONS:
                        continue
                    if path.name.startswith("."):
                        continue

                    if change_type in (Change.added, Change.modified):
                        new_mtime = path.stat().st_mtime
                        old_mtime = self._known_files.get(
                            path_str, 0
                        )
                        if new_mtime != old_mtime:
                            self._known_files[path_str] = new_mtime
                            await self._ingest_file(path)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Folder watcher error", exc_info=True
            )

    async def _ingest_file(self, path: Path) -> None:
        """Ingest a single file into the RAG store."""
        try:
            await self._store.add_document(
                path,
                embedding_model=self._embedding_model,
                chunk_size=self._chunk_size,
                overlap=self._overlap,
                ollama_url=self._ollama_url,
            )
            logger.info("Auto-ingested: %s", path.name)
        except Exception as e:
            logger.debug(
                "Failed to auto-ingest %s: %s", path.name, e
            )
