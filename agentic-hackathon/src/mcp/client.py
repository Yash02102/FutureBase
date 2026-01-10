from typing import Dict, List, Optional, Protocol

from .schemas import MCPServerConfig, MCPToolSpec


class MCPTransport(Protocol):
    def connect(self, server: MCPServerConfig) -> None:
        ...

    def list_tools(self, server: MCPServerConfig) -> List[MCPToolSpec]:
        ...

    def call_tool(
        self,
        server: MCPServerConfig,
        tool_name: str,
        arguments: Dict[str, str],
    ) -> str:
        ...


class InMemoryTransport:
    def __init__(self, tools: Optional[List[MCPToolSpec]] = None) -> None:
        self.tools = tools or []

    def connect(self, server: MCPServerConfig) -> None:
        return None

    def list_tools(self, server: MCPServerConfig) -> List[MCPToolSpec]:
        return self.tools

    def call_tool(
        self,
        server: MCPServerConfig,
        tool_name: str,
        arguments: Dict[str, str],
    ) -> str:
        return (
            f"MCP transport stub: '{tool_name}' on {server.name} called with {arguments}."
        )


class MCPClient:
    def __init__(self, server: MCPServerConfig, transport: MCPTransport) -> None:
        self.server = server
        self.transport = transport

    def connect(self) -> None:
        self.transport.connect(self.server)

    def list_tools(self) -> List[MCPToolSpec]:
        return self.transport.list_tools(self.server)

    def call_tool(self, tool_name: str, arguments: Dict[str, str]) -> str:
        return self.transport.call_tool(self.server, tool_name, arguments)
