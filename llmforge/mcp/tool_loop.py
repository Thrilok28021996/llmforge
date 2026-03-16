"""Tool-calling loop — runs inference with MCP tool support."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from llmforge.domain.models import ChatMessage, InferenceRequest, TokenChunk
from llmforge.mcp.client import MCPClient
from llmforge.mcp.types import ToolCall, ToolDefinition

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10


async def run_with_tools(
    backend,
    request: InferenceRequest,
    mcp_clients: list[MCPClient],
    on_tool_call=None,
    on_tool_result=None,
) -> AsyncIterator[TokenChunk]:
    """Run inference with tool-calling loop.

    Yields TokenChunks as they stream. When the model makes tool calls,
    executes them via MCP and re-submits with results.

    on_tool_call(tool_call: ToolCall) — optional callback for UI display
    on_tool_result(name: str, result: str) — optional callback for UI display
    """
    # Collect all tools from all MCP clients
    all_tools: list[ToolDefinition] = []
    tool_to_client: dict[str, MCPClient] = {}
    for client in mcp_clients:
        for tool in client.tools:
            all_tools.append(tool)
            tool_to_client[tool.name] = client

    if not all_tools:
        # No tools — just pass through to normal inference
        async for chunk in backend.generate(request):
            yield chunk
        return

    # Add tools to request (OpenAI format, compatible with most backends)
    tools_json = [t.to_openai_format() for t in all_tools]

    messages = list(request.messages)
    rounds = 0

    while rounds < MAX_TOOL_ROUNDS:
        rounds += 1

        # Build request with current messages and tools
        current_request = InferenceRequest(
            model_id=request.model_id,
            messages=messages,
            params=request.params,
            system_prompt=request.system_prompt,
            tools=tools_json,
        )

        # Accumulate the full response to detect tool calls
        response_text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        has_tool_calls = False

        async for chunk in backend.generate(current_request):
            if chunk.tool_calls:
                # Backend returned structured tool calls
                has_tool_calls = True
                for tc in chunk.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=json.loads(
                            tc.get("function", {}).get("arguments", "{}")
                        ),
                    ))
            elif chunk.text:
                response_text_parts.append(chunk.text)
                yield chunk

            if chunk.is_final:
                if not has_tool_calls:
                    # Check if response text contains JSON tool calls
                    # (for models that emit tool calls as text)
                    full_text = "".join(response_text_parts)
                    parsed = _parse_text_tool_calls(full_text)
                    if parsed:
                        has_tool_calls = True
                        tool_calls = parsed
                break

        if not has_tool_calls or not tool_calls:
            # No tool calls — we're done
            return

        # Execute tool calls
        full_response = "".join(response_text_parts)
        messages.append(ChatMessage(role="assistant", content=full_response))

        for tc in tool_calls:
            if on_tool_call:
                on_tool_call(tc)

            client = tool_to_client.get(tc.name)
            if not client:
                result_text = f"Error: Unknown tool '{tc.name}'"
                is_error = True
            else:
                result = await client.call_tool(tc.name, tc.arguments)
                result_text = result.content
                is_error = result.is_error

            if on_tool_result:
                on_tool_result(tc.name, result_text)

            # Add tool result as a message
            messages.append(ChatMessage(
                role="tool",
                content=json.dumps({
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": result_text,
                    "is_error": is_error,
                }),
            ))

            # Yield a visible indicator
            yield TokenChunk(
                text=f"\n\n> **Tool: {tc.name}** → {result_text[:200]}\n\n"
            )

    # Max rounds exceeded
    yield TokenChunk(
        text="\n\n*[Tool calling limit reached]*\n",
    )


def _parse_text_tool_calls(text: str) -> list[ToolCall]:
    """Try to parse tool calls from model text output.

    Looks for patterns like:
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    or ```json\n{"name": "...", "arguments": {...}}\n```
    """
    import re

    calls = []

    # Pattern 1: <tool_call>...</tool_call>
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            data = json.loads(match.group(1).strip())
            calls.append(ToolCall(
                id=data.get("id", ""),
                name=data["name"],
                arguments=data.get("arguments", {}),
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    return calls
