"""Vector storage in SQLite for RAG."""

from __future__ import annotations

import logging
import math
import struct
from pathlib import Path

import aiosqlite

from llmforge.rag.chunker import chunk_text, ingest_file
from llmforge.rag.embeddings import embed_texts

logger = logging.getLogger(__name__)

RAG_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    path TEXT,
    chunk_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rag_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,
    FOREIGN KEY (document_id) REFERENCES rag_documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(document_id);
"""


def _pack_embedding(vec: list[float]) -> bytes:
    """Pack a float vector into a compact binary blob."""
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a binary blob into a float vector."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RAGStore:
    """SQLite-backed vector store for RAG documents."""

    def __init__(self, db: aiosqlite.Connection):
        self._db = db

    async def ensure_schema(self) -> None:
        """Create RAG tables if they don't exist."""
        await self._db.executescript(RAG_SCHEMA)
        await self._db.commit()

    async def add_document(
        self,
        path: Path,
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 512,
        overlap: int = 64,
        ollama_url: str = "http://localhost:11434",
    ) -> int:
        """Ingest a document: read, chunk, embed, store. Returns document ID."""
        text = ingest_file(path)
        if not text.strip():
            raise ValueError(f"No text content extracted from {path}")

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        logger.info("Chunked %s into %d chunks", path.name, len(chunks))

        # Embed all chunks
        embeddings = await embed_texts(
            chunks, model=embedding_model, ollama_url=ollama_url
        )
        if not embeddings:
            raise RuntimeError(
                f"Embedding failed. Ensure '{embedding_model}' is pulled in Ollama."
            )

        # Store document
        cursor = await self._db.execute(
            "INSERT INTO rag_documents (name, path, chunk_count) VALUES (?, ?, ?)",
            (path.name, str(path), len(chunks)),
        )
        doc_id = cursor.lastrowid

        # Store chunks with embeddings
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            await self._db.execute(
                "INSERT INTO rag_chunks (document_id, chunk_index, text, embedding) "
                "VALUES (?, ?, ?, ?)",
                (doc_id, i, chunk, _pack_embedding(emb)),
            )

        await self._db.commit()
        logger.info("Stored %d chunks for document %s (id=%d)", len(chunks), path.name, doc_id)
        return doc_id

    async def search(
        self,
        query: str,
        top_k: int = 3,
        embedding_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
    ) -> list[tuple[str, float]]:
        """Search for the most relevant chunks. Returns (text, score) pairs."""
        # Embed the query
        query_embeddings = await embed_texts(
            [query], model=embedding_model, ollama_url=ollama_url
        )
        if not query_embeddings:
            return []
        query_vec = query_embeddings[0]

        # Brute-force cosine similarity against all chunks
        cursor = await self._db.execute(
            "SELECT text, embedding FROM rag_chunks WHERE embedding IS NOT NULL"
        )
        rows = await cursor.fetchall()

        scored: list[tuple[str, float]] = []
        for text, emb_blob in rows:
            chunk_vec = _unpack_embedding(emb_blob)
            score = _cosine_similarity(query_vec, chunk_vec)
            scored.append((text, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def list_documents(self) -> list[dict]:
        """List all ingested documents."""
        cursor = await self._db.execute(
            "SELECT id, name, path, chunk_count, created_at FROM rag_documents "
            "ORDER BY created_at DESC"
        )
        cols = [d[0] for d in cursor.description]
        rows = await cursor.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    async def delete_document(self, doc_id: int) -> None:
        """Delete a document and its chunks."""
        await self._db.execute("DELETE FROM rag_chunks WHERE document_id=?", (doc_id,))
        await self._db.execute("DELETE FROM rag_documents WHERE id=?", (doc_id,))
        await self._db.commit()

    async def chunk_count(self) -> int:
        """Total number of chunks in the store."""
        cursor = await self._db.execute("SELECT COUNT(*) FROM rag_chunks")
        row = await cursor.fetchone()
        return row[0] if row else 0
