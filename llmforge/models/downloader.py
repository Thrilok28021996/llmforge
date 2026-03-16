"""HuggingFace GGUF model downloader — no Ollama registry dependency.

Downloads GGUF files directly from HuggingFace Hub repos.
Supports progress callbacks for TUI integration.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from llmforge.config import data_dir

logger = logging.getLogger(__name__)

MODELS_DIR = data_dir() / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Curated popular GGUF repos on HuggingFace
POPULAR_REPOS: list[dict] = [
    {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "name": "Llama 3.2 3B Instruct",
        "params": "3B",
        "files": ["Llama-3.2-3B-Instruct-Q4_K_M.gguf"],
    },
    {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "name": "Llama 3.2 1B Instruct",
        "params": "1B",
        "files": ["Llama-3.2-1B-Instruct-Q4_K_M.gguf"],
    },
    {
        "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "name": "Qwen 2.5 7B Instruct",
        "params": "7B",
        "files": ["Qwen2.5-7B-Instruct-Q4_K_M.gguf"],
    },
    {
        "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
        "name": "Qwen 2.5 3B Instruct",
        "params": "3B",
        "files": ["Qwen2.5-3B-Instruct-Q4_K_M.gguf"],
    },
    {
        "repo": "bartowski/Phi-4-mini-instruct-GGUF",
        "name": "Phi 4 Mini Instruct",
        "params": "3.8B",
        "files": ["Phi-4-mini-instruct-Q4_K_M.gguf"],
    },
    {
        "repo": "bartowski/gemma-2-9b-it-GGUF",
        "name": "Gemma 2 9B IT",
        "params": "9B",
        "files": ["gemma-2-9b-it-Q4_K_M.gguf"],
    },
    {
        "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "name": "Mistral 7B Instruct v0.3",
        "params": "7B",
        "files": ["Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"],
    },
    {
        "repo": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "name": "DeepSeek R1 Distill 7B",
        "params": "7B",
        "files": [
            "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
        ],
    },
]


@dataclass
class DownloadProgress:
    filename: str
    downloaded_bytes: int
    total_bytes: int
    status: str  # "downloading" | "complete" | "error"

    @property
    def percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100


async def download_gguf(
    repo_id: str,
    filename: str,
    dest_dir: Path | None = None,
) -> AsyncIterator[DownloadProgress]:
    """Download a GGUF file from HuggingFace Hub with progress.

    Uses huggingface_hub for authenticated + cached downloads.
    """
    import asyncio

    dest = dest_dir or MODELS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    dest_file = dest / filename

    if dest_file.exists():
        size = dest_file.stat().st_size
        yield DownloadProgress(
            filename=filename,
            downloaded_bytes=size,
            total_bytes=size,
            status="complete",
        )
        return

    def _download():
        """Blocking download via huggingface_hub."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )

    yield DownloadProgress(
        filename=filename,
        downloaded_bytes=0,
        total_bytes=0,
        status="downloading",
    )

    loop = asyncio.get_event_loop()
    try:
        result_path = await loop.run_in_executor(None, _download)
        final_path = Path(result_path)
        size = final_path.stat().st_size
        yield DownloadProgress(
            filename=filename,
            downloaded_bytes=size,
            total_bytes=size,
            status="complete",
        )
    except Exception as e:
        logger.error("Download failed for %s/%s: %s", repo_id, filename, e)
        yield DownloadProgress(
            filename=filename,
            downloaded_bytes=0,
            total_bytes=0,
            status=f"error: {e}",
        )


def list_local_gguf(search_dirs: list[str] | None = None) -> list[Path]:
    """Find all .gguf files in default + configured directories."""
    dirs = [MODELS_DIR]
    if search_dirs:
        dirs.extend(Path(d).expanduser() for d in search_dirs)

    found = []
    seen: set[str] = set()
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*.gguf")):
            if f.name not in seen:
                seen.add(f.name)
                found.append(f)
    return found


async def search_huggingface_gguf(
    query: str, limit: int = 20
) -> list[dict]:
    """Search HuggingFace for GGUF model repos."""
    import asyncio

    def _search():
        from huggingface_hub import HfApi
        api = HfApi()
        # Search for repos with GGUF in the name
        results = api.list_models(
            search=f"{query} GGUF",
            sort="downloads",
            direction=-1,
            limit=limit,
        )
        return [
            {
                "repo_id": m.id,
                "name": m.id.split("/")[-1],
                "downloads": m.downloads,
                "likes": m.likes,
            }
            for m in results
        ]

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _search)
    except Exception as e:
        logger.error("HuggingFace search failed: %s", e)
        return []


async def list_repo_gguf_files(repo_id: str) -> list[dict]:
    """List .gguf files in a HuggingFace repo with sizes."""
    import asyncio

    def _list():
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id)
        result = []
        for f in files:
            if f.endswith(".gguf"):
                try:
                    info = api.model_info(repo_id)
                    siblings = info.siblings or []
                    size = None
                    for s in siblings:
                        if s.rfilename == f:
                            size = s.size
                            break
                    result.append({
                        "filename": f,
                        "size_bytes": size,
                    })
                except Exception:
                    result.append({"filename": f, "size_bytes": None})
        return result

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _list)
    except Exception as e:
        logger.error("Failed to list repo files for %s: %s", repo_id, e)
        return []
