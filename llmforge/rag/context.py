"""RAG context builder — retrieves, re-ranks, and formats context."""

from __future__ import annotations

from llmforge.rag.store import RAGStore


async def build_rag_context(
    query: str,
    store: RAGStore,
    top_k: int = 3,
    embedding_model: str = "nomic-embed-text",
    ollama_url: str = "http://localhost:11434",
    rerank: bool = True,
    rerank_model: str = "llama3.2:3b",
) -> str | None:
    """Search the RAG store, re-rank, and build a context string.

    Returns None if no relevant chunks are found.
    """
    total = await store.chunk_count()
    if total == 0:
        return None

    # Retrieve more candidates than needed for re-ranking
    fetch_k = top_k * 3 if rerank else top_k

    results = await store.search(
        query,
        top_k=fetch_k,
        embedding_model=embedding_model,
        ollama_url=ollama_url,
    )

    if not results:
        return None

    # Filter out very low-relevance chunks
    relevant = [(text, score) for text, score in results if score > 0.2]
    if not relevant:
        return None

    # Re-rank for better precision
    if rerank and len(relevant) > top_k:
        from llmforge.rag.reranker import rerank as do_rerank

        relevant = await do_rerank(
            query,
            relevant,
            top_k=top_k,
            ollama_url=ollama_url,
            rerank_model=rerank_model,
        )
    else:
        relevant = relevant[:top_k]

    parts = ["Use the following context to help answer the user's question:\n"]
    for i, (text, score) in enumerate(relevant, 1):
        parts.append(f"--- Context {i} (relevance: {score:.2f}) ---")
        parts.append(text)
        parts.append("")

    parts.append("--- End of context ---\n")
    return "\n".join(parts)
