"""Agent framework — plan/execute loop with built-in tools."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from llmforge.domain.models import ChatMessage, InferenceRequest, TokenChunk
from llmforge.tools.code_exec import execute_code

logger = logging.getLogger(__name__)

MAX_AGENT_ROUNDS = 10

# Built-in tool definitions for agent mode
BUILT_IN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": (
                "Execute code and return the output. Supports "
                "python, javascript, bash, go, rust, c, cpp, ruby."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code to execute",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language (auto-detected if omitted)",
                        "enum": [
                            "python", "javascript", "bash",
                            "go", "rust", "c", "cpp", "ruby", "auto",
                        ],
                        "default": "auto",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


@dataclass
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool = False


@dataclass
class AgentStep:
    thought: str = ""
    tool_calls: list[ToolExecution] = field(default_factory=list)
    response: str = ""


async def _execute_builtin_tool(
    name: str,
    arguments: dict[str, Any],
    web_search_config: dict | None = None,
) -> str:
    """Execute a built-in agent tool."""
    if name in ("run_code", "run_python"):
        code = arguments.get("code", "")
        language = arguments.get("language", "auto")
        result = await execute_code(code, language=language)
        return result.output

    if name == "web_search":
        query = arguments.get("query", "")
        try:
            from llmforge.rag.web_search import format_search_context, web_search

            cfg = web_search_config or {}
            results = await web_search(
                query,
                provider=cfg.get("provider", "duckduckgo"),
                max_results=cfg.get("max_results", 5),
                searxng_url=cfg.get("searxng_url", "http://localhost:8080"),
                tavily_api_key=cfg.get("tavily_api_key", ""),
            )
            return format_search_context(results) or "No results found."
        except Exception as e:
            return f"Search error: {e}"

    return f"Unknown tool: {name}"


def _parse_tool_calls_from_text(text: str) -> list[dict]:
    """Parse tool calls from model output text (for models without native tool support)."""
    calls = []
    pattern = r"<tool_call>\s*\{(.*?)\}\s*</tool_call>"
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            data = json.loads("{" + match.group(1) + "}")
            if "name" in data:
                calls.append({
                    "name": data["name"],
                    "arguments": data.get("arguments", {}),
                })
        except json.JSONDecodeError:
            continue
    return calls


async def run_agent_loop(
    backend,
    model_id: str,
    messages: list[ChatMessage],
    params,
    system_prompt: str | None = None,
    web_search_config: dict | None = None,
    mcp_clients: list | None = None,
) -> AsyncIterator[TokenChunk]:
    """Run an agent loop: generate → detect tool calls → execute → re-generate.

    Yields TokenChunk for each step so the UI can stream progress.
    """
    working_messages = list(messages)

    # Gather all available tools
    tools = list(BUILT_IN_TOOLS)
    mcp_tool_map: dict[str, Any] = {}

    if mcp_clients:
        for client in mcp_clients:
            for tool in client.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                })
                mcp_tool_map[tool.name] = client

    agent_system = (system_prompt or "") + (
        "\n\nYou are an agent with access to tools. When you need to use a tool, "
        "format your call as:\n"
        "<tool_call>{\"name\": \"tool_name\", "
        "\"arguments\": {...}}</tool_call>\n"
        "Wait for the result before proceeding. If you don't need a tool, respond normally."
    )

    for round_num in range(MAX_AGENT_ROUNDS):
        request = InferenceRequest(
            model_id=model_id,
            messages=working_messages,
            params=params,
            system_prompt=agent_system,
            tools=tools,
        )

        # Collect response
        full_text = ""
        async for chunk in backend.generate(request):
            if chunk.text:
                full_text += chunk.text
                yield chunk
            if chunk.is_final:
                break

        # Check for tool calls in the response
        tool_calls = _parse_tool_calls_from_text(full_text)

        if not tool_calls:
            # No tools needed — agent is done
            yield TokenChunk.final()
            return

        # Execute tool calls
        working_messages.append(ChatMessage(role="assistant", content=full_text))

        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]

            yield TokenChunk(text=f"\n\n🔧 Running `{name}`...\n")

            if name in mcp_tool_map:
                # MCP tool
                client = mcp_tool_map[name]
                result = await client.call_tool(name, args)
                result_text = result.content
            else:
                # Built-in tool
                result_text = await _execute_builtin_tool(
                    name, args, web_search_config
                )

            yield TokenChunk(text=f"```\n{result_text[:2000]}\n```\n\n")

            working_messages.append(
                ChatMessage(
                    role="tool",
                    content=f"Tool '{name}' result:\n{result_text}",
                )
            )

    yield TokenChunk(text="\n[Agent reached maximum rounds]")
    yield TokenChunk.final()
