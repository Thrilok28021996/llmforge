"""Embedding with multiple backends — Ollama, llama.cpp native, or TF-IDF fallback.

Priority:
1. Ollama (if running)
2. llama-cpp-python (if installed + embedding model available)
3. TF-IDF bag-of-words (always available, no dependencies)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from collections import Counter

import httpx

logger = logging.getLogger(__name__)


async def embed_texts(
    texts: list[str],
    model: str = "nomic-embed-text",
    ollama_url: str = "http://localhost:11434",
    method: str = "auto",
) -> list[list[float]]:
    """Embed texts using the best available method.

    method: "auto" | "ollama" | "llamacpp" | "tfidf"
    """
    if method == "ollama":
        return await _embed_ollama(texts, model, ollama_url)
    if method == "llamacpp":
        return await _embed_llamacpp(texts, model)
    if method == "tfidf":
        return _embed_tfidf(texts)

    # Auto: try Ollama first, then llama.cpp, then TF-IDF
    result = await _embed_ollama(texts, model, ollama_url)
    if result:
        return result

    result = await _embed_llamacpp(texts, model)
    if result:
        return result

    logger.info(
        "No embedding backend available, using TF-IDF fallback "
        "(install llama-cpp-python or run Ollama for better results)"
    )
    return _embed_tfidf(texts)


async def _embed_ollama(
    texts: list[str], model: str, ollama_url: str
) -> list[list[float]]:
    """Embed via Ollama's /api/embed endpoint."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120.0, connect=10.0)
    ) as client:
        try:
            resp = await client.post(
                f"{ollama_url}/api/embed",
                json={"model": model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("embeddings", [])
        except httpx.ConnectError:
            return []
        except Exception as e:
            logger.debug("Ollama embedding failed: %s", e)
            return []


async def _embed_llamacpp(
    texts: list[str], model: str
) -> list[list[float]]:
    """Embed via llama-cpp-python's built-in embedding support."""
    try:
        from llama_cpp import Llama
    except ImportError:
        return []

    # Look for an embedding model in the models directory
    from llmforge.models.downloader import MODELS_DIR

    model_path = None
    for f in MODELS_DIR.rglob("*.gguf"):
        name_lower = f.name.lower()
        if "embed" in name_lower or "nomic" in name_lower:
            model_path = str(f)
            break

    if not model_path:
        return []

    def _do_embed():
        llm = Llama(
            model_path=model_path,
            embedding=True,
            verbose=False,
            n_ctx=512,
        )
        results = []
        for text in texts:
            emb = llm.embed(text)
            results.append(emb)
        del llm
        return results

    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _do_embed)
    except Exception as e:
        logger.debug("llama.cpp embedding failed: %s", e)
        return []


# ── TF-IDF Fallback (zero dependencies) ─────────────────────────────

# Vocabulary for consistent dimensionality
_TFIDF_DIM = 384  # Match common embedding dimensions


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r"\w+", text.lower())


def _text_to_hash_vector(text: str, dim: int = _TFIDF_DIM) -> list[float]:
    """Convert text to a fixed-dimension vector via feature hashing.

    Uses the hashing trick for a zero-vocabulary embedding.
    Good enough for cosine similarity ranking in RAG.
    """
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * dim

    # Bigrams for some context sensitivity
    features = list(tokens)
    for i in range(len(tokens) - 1):
        features.append(f"{tokens[i]}_{tokens[i + 1]}")

    counts = Counter(features)
    vec = [0.0] * dim

    for feature, count in counts.items():
        # Hash to a bucket
        h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
        bucket = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        # TF component with sublinear scaling
        vec[bucket] += sign * (1.0 + math.log(count))

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]

    return vec


def _embed_tfidf(texts: list[str]) -> list[list[float]]:
    """Fallback: hash-based TF-IDF embedding. Always works, no deps."""
    return [_text_to_hash_vector(t) for t in texts]
