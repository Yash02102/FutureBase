from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import anyio
import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client


@dataclass
class MCPClient:
    url: str
    timeout: float = 30.0

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        return anyio.run(self._call_tool, name, arguments or {})

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        timeout = httpx.Timeout(self.timeout)
        async_client = create_mcp_http_client(timeout=timeout)
        async with streamable_http_client(self.url, http_client=async_client) as streams:
            read_stream, write_stream, _get_session_id = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
        return _format_result(result)


def _format_result(result) -> str:
    structured = getattr(result, "structuredContent", None)
    if structured:
        return json.dumps(structured, ensure_ascii=True)
    contents = getattr(result, "content", [])
    if not contents:
        return ""
    parts = []
    for item in contents:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
        else:
            parts.append(str(item))
    return "\n".join(parts)
