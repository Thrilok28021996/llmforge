"""MCP client — communicates with MCP servers via stdio JSON-RPC."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from llmforge.mcp.types import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages a single MCP server subprocess via stdio JSON-RPC 2.0."""

    def __init__(self, name: str, command: list[str], env: dict[str, str] | None = None):
        self.name = name
        self._command = command
        self._env = env
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._tools: list[ToolDefinition] = []

    async def connect(self) -> bool:
        """Start the MCP server subprocess and initialize."""
        try:
            import os
            merged_env = {**os.environ, **(self._env or {})}
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=merged_env,
            )
        except FileNotFoundError:
            logger.error("MCP server command not found: %s", self._command)
            return False
        except Exception as e:
            logger.error("Failed to start MCP server %s: %s", self.name, e)
            return False

        # Send initialize request
        result = await self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "llmforge", "version": "0.1.0"},
        })
        if result is None:
            logger.error("MCP server %s failed to initialize", self.name)
            return False

        # Send initialized notification
        await self._notify("notifications/initialized", {})

        # Fetch tools
        await self.refresh_tools()
        return True

    async def refresh_tools(self) -> list[ToolDefinition]:
        """Fetch available tools from the server."""
        result = await self._rpc("tools/list", {})
        if result is None:
            return []

        self._tools = []
        for tool_data in result.get("tools", []):
            self._tools.append(ToolDefinition(
                name=tool_data["name"],
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
            ))
        return self._tools

    @property
    def tools(self) -> list[ToolDefinition]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Call a tool on the MCP server."""
        result = await self._rpc("tools/call", {
            "name": name,
            "arguments": arguments,
        })

        if result is None:
            return ToolResult(
                tool_call_id="",
                content=f"MCP server {self.name} returned no result",
                is_error=True,
            )

        content_parts = result.get("content", [])
        text_parts = []
        for part in content_parts:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))

        return ToolResult(
            tool_call_id="",
            content="\n".join(text_parts) if text_parts else str(result),
            is_error=result.get("isError", False),
        )

    async def close(self) -> None:
        """Shut down the MCP server."""
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except TimeoutError:
                self._process.kill()
            except Exception:
                pass

    async def _rpc(self, method: str, params: dict) -> dict | None:
        """Send a JSON-RPC 2.0 request and wait for the response."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            msg = json.dumps(request) + "\n"
            self._process.stdin.write(msg.encode())
            await self._process.stdin.drain()

            line = await asyncio.wait_for(
                self._process.stdout.readline(), timeout=30.0
            )
            if not line:
                return None

            response = json.loads(line.decode())
            if "error" in response:
                logger.warning(
                    "MCP error from %s: %s", self.name, response["error"]
                )
                return None
            return response.get("result")

        except TimeoutError:
            logger.warning("MCP request to %s timed out", self.name)
            return None
        except Exception as e:
            logger.warning("MCP RPC error with %s: %s", self.name, e)
            return None

    async def _notify(self, method: str, params: dict) -> None:
        """Send a JSON-RPC 2.0 notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        try:
            msg = json.dumps(notification) + "\n"
            self._process.stdin.write(msg.encode())
            await self._process.stdin.drain()
        except Exception:
            pass
