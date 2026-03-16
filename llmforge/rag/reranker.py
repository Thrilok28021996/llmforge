"""Re-ranker for RAG results — improves retrieval precision.

Uses a lightweight cross-encoder scoring approach:
1. LLM-based re-ranking (sends query+chunk to the model for relevance scoring)
2. Keyword overlap scoring (zero-dependency fallback)
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

import httpx

logger = logging.getLogger(__name__)


async def rerank(
    query: str,
    chunks: list[tuple[str, float]],
    method: str = "auto",
    top_k: int = 3,
    ollama_url: str = "http://localhost:11434",
    rerank_model: str = "llama3.2:3b",
) -> list[tuple[str, float]]:
    """Re-rank retrieved chunks for better relevance.

    Args:
        query: The user's question
        chunks: List of (text, initial_score) tuples from vector search
        method: "auto" | "llm" | "keyword"
        top_k: Number of chunks to return after re-ranking
        ollama_url: Ollama endpoint for LLM re-ranking
        rerank_model: Model to use for LLM-based re-ranking
    """
    if not chunks:
        return []

    if method == "keyword":
        return _rerank_keyword(query, chunks, top_k)

    if method == "llm":
        return await _rerank_llm(
            query, chunks, top_k, ollama_url, rerank_model
        )

    # Auto: try LLM, fall back to keyword
    try:
        result = await _rerank_llm(
            query, chunks, top_k, ollama_url, rerank_model
        )
        if result:
            return result
    except Exception:
        pass

    return _rerank_keyword(query, chunks, top_k)


def _rerank_keyword(
    query: str,
    chunks: list[tuple[str, float]],
    top_k: int,
) -> list[tuple[str, float]]:
    """Re-rank using BM25-style keyword overlap scoring.

    Combines:
    - Original vector similarity score (weighted 0.4)
    - BM25-style term overlap (weighted 0.6)
    """
    query_terms = _tokenize(query)
    if not query_terms:
        return chunks[:top_k]

    query_counts = Counter(query_terms)

    # Compute IDF across all chunks
    doc_count = len(chunks)
    doc_freq: Counter[str] = Counter()
    for text, _ in chunks:
        terms = set(_tokenize(text))
        for t in terms:
            doc_freq[t] += 1

    scored = []
    for text, vec_score in chunks:
        chunk_terms = _tokenize(text)
        chunk_counts = Counter(chunk_terms)
        chunk_len = len(chunk_terms)
        avg_len = sum(
            len(_tokenize(t)) for t, _ in chunks
        ) / max(len(chunks), 1)

        # BM25 scoring
        k1, b = 1.5, 0.75
        bm25 = 0.0
        for term, qf in query_counts.items():
            if term not in chunk_counts:
                continue
            tf = chunk_counts[term]
            df = doc_freq.get(term, 0)
            idf = math.log(
                (doc_count - df + 0.5) / (df + 0.5) + 1
            )
            tf_norm = (tf * (k1 + 1)) / (
                tf + k1 * (1 - b + b * chunk_len / max(avg_len, 1))
            )
            bm25 += idf * tf_norm * qf

        # Normalize BM25 to 0-1 range (approximate)
        bm25_norm = min(bm25 / 10.0, 1.0)

        # Combined score
        combined = 0.4 * vec_score + 0.6 * bm25_norm
        scored.append((text, combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


async def _rerank_llm(
    query: str,
    chunks: list[tuple[str, float]],
    top_k: int,
    ollama_url: str,
    model: str,
) -> list[tuple[str, float]]:
    """Re-rank using an LLM to score relevance (0-10)."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0)
    ) as client:
        scored = []
        for text, vec_score in chunks:
            prompt = (
                f"Rate how relevant this text passage is to the "
                f"query on a scale of 0-10. "
                f"Reply with ONLY a number.\n\n"
                f"Query: {query}\n\n"
                f"Passage: {text[:500]}\n\n"
                f"Relevance score (0-10):"
            )

            try:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 5,
                        },
                    },
                )
                if resp.status_code != 200:
                    return []

                data = resp.json()
                response_text = data.get("response", "").strip()
                # Extract first number from response
                match = re.search(r"(\d+(?:\.\d+)?)", response_text)
                if match:
                    llm_score = float(match.group(1)) / 10.0
                    llm_score = min(max(llm_score, 0.0), 1.0)
                else:
                    llm_score = vec_score

                # Blend LLM score with vector score
                combined = 0.3 * vec_score + 0.7 * llm_score
                scored.append((text, combined))

            except Exception:
                # Fall back to original score
                scored.append((text, vec_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())
