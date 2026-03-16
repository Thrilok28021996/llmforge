"""Web search providers for live RAG — DuckDuckGo, SearXNG, Tavily."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


async def search_duckduckgo(
    query: str, max_results: int = 5
) -> list[SearchResult]:
    """Search via DuckDuckGo Lite (no API key needed)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://lite.duckduckgo.com/lite",
            params={"q": query},
            headers={"User-Agent": "LLMForge/1.0"},
        )
        resp.raise_for_status()

    # Parse the HTML table results from DDG Lite
    html = resp.text
    results: list[SearchResult] = []

    # DDG Lite returns results in <a class="result-link"> and <td class="result-snippet">
    import re

    links = re.findall(
        r'class="result-link"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', html
    )
    snippets = re.findall(
        r'class="result-snippet">([^<]+)</td>', html
    )

    for i, (url, title) in enumerate(links[:max_results]):
        snippet = snippets[i].strip() if i < len(snippets) else ""
        results.append(SearchResult(title=title.strip(), url=url, snippet=snippet))

    return results


async def search_searxng(
    query: str, base_url: str = "http://localhost:8080", max_results: int = 5
) -> list[SearchResult]:
    """Search via a SearXNG instance (self-hosted)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{base_url.rstrip('/')}/search",
            params={"q": query, "format": "json", "categories": "general"},
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
        )
    return results


async def search_tavily(
    query: str, api_key: str, max_results: int = 5
) -> list[SearchResult]:
    """Search via Tavily API (optimized for AI/RAG)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
        )
    return results


async def web_search(
    query: str,
    provider: str = "duckduckgo",
    max_results: int = 5,
    searxng_url: str = "http://localhost:8080",
    tavily_api_key: str = "",
) -> list[SearchResult]:
    """Unified web search dispatcher."""
    try:
        if provider == "tavily" and tavily_api_key:
            return await search_tavily(query, tavily_api_key, max_results)
        if provider == "searxng":
            return await search_searxng(query, searxng_url, max_results)
        return await search_duckduckgo(query, max_results)
    except Exception as e:
        logger.warning("Web search failed (%s): %s", provider, e)
        return []


def format_search_context(results: list[SearchResult]) -> str | None:
    """Format search results into a context string for the LLM."""
    if not results:
        return None

    parts = ["Web search results:\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r.title}")
        parts.append(f"    URL: {r.url}")
        if r.snippet:
            parts.append(f"    {r.snippet}")
        parts.append("")

    return "\n".join(parts)
