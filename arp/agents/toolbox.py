"""MCP client side: one stdio server process per principal, tool calls traced as spans."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any, Protocol

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ..obs.trace import Tracer

MAX_RESULT_CHARS = 3500  # keep tool output small: local models have 8k-token slots


class Toolbox(Protocol):
    tools: list[dict]

    async def call(self, agent: str, name: str, args: dict[str, Any]) -> tuple[str, bool]: ...


class MCPToolbox:
    """Spawns ``python -m arp.mcp_server.server`` bound to one principal and talks MCP over stdio."""

    def __init__(
        self,
        principal: str,
        tracer: Tracer | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 60.0,
    ):
        self.principal = principal
        self.tracer = tracer
        self.env = {**os.environ, **(env or {}), "ARP_PRINCIPAL": principal}
        self.timeout = timeout
        self.tools: list[dict] = []
        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> MCPToolbox:
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "arp.mcp_server.server"], env=self.env, cwd=os.getcwd()
        )
        read, write = await self._stack.enter_async_context(
            stdio_client(params, errlog=open(os.devnull, "w"))
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        listed = await self._session.list_tools()
        self.tools = [
            {
                "name": t.name,
                "description": (t.description or "").strip(),
                "input_schema": _clean_schema(t.input_schema),
            }
            for t in listed.tools
        ]
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()

    def subset(self, names: list[str]) -> list[dict]:
        return [t for t in self.tools if t["name"] in names]

    async def call(self, agent: str, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        assert self._session is not None
        if self.tracer is None:
            return await self._call(name, args)
        with self.tracer.span("tool", name, agent=agent, principal=self.principal, args=args) as sp:
            text, is_error = await self._call(name, args)
            sp.set(result=text, is_error=is_error)
            return text, is_error

    async def _call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        try:
            res = await asyncio.wait_for(self._session.call_tool(name, args), self.timeout)
        except TimeoutError:
            return f"tool {name} timed out after {self.timeout}s", True
        text = "\n".join(getattr(c, "text", "") for c in res.content)
        text = text.replace(f"Error executing tool {name}: ", "")
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + f"\n...[truncated {len(text) - MAX_RESULT_CHARS} chars]"
        return text, bool(res.is_error)


def _clean_schema(schema: dict) -> dict:
    """Drop pydantic titles; small models do better with plain schemas."""

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(json.loads(json.dumps(schema)))
