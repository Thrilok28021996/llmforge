"""Document chunking for RAG."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by paragraph boundaries.

    chunk_size and overlap are in estimated tokens (~4 chars/token).
    """
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    # Split by double newlines (paragraphs) first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > char_size and current:
            # Flush current chunk
            chunk_text_str = "\n\n".join(current)
            chunks.append(chunk_text_str)
            # Keep overlap from end of current
            overlap_text = chunk_text_str[-char_overlap:] if char_overlap else ""
            current = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text[:char_size]] if text.strip() else []


def ingest_file(path: Path) -> str:
    """Read a file and return its text content."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in (".md", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml",
                   ".toml", ".ini", ".cfg", ".log", ".py", ".js", ".ts",
                   ".rs", ".go", ".java", ".c", ".cpp", ".h", ".html",
                   ".css", ".xml", ".sh", ".bash", ".zsh"):
        return path.read_text(errors="replace")

    # Try as text
    try:
        return path.read_text(errors="replace")
    except Exception:
        logger.warning("Cannot read %s as text", path)
        return ""


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)
    except ImportError:
        logger.warning(
            "PyMuPDF not installed. Install with: pip install pymupdf"
        )
        return ""
    except Exception as e:
        logger.warning("Failed to read PDF %s: %s", path, e)
        return ""
