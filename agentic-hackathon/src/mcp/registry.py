import os
from typing import Dict, List, Optional

from .client import InMemoryTransport, MCPClient
from .schemas import MCPServerConfig, MCPToolSpec


class MCPRegistry:
    def __init__(self) -> None:
        self._servers: Dict[str, MCPServerConfig] = {}
        self._transports: Dict[str, InMemoryTransport] = {}

    def register_server(self, server: MCPServerConfig) -> None:
        self._servers[server.name] = server
        if server.name not in self._transports:
            self._transports[server.name] = InMemoryTransport()

    def add_tools(self, server_name: str, tools: List[MCPToolSpec]) -> None:
        if server_name not in self._transports:
            self._transports[server_name] = InMemoryTransport()
        self._transports[server_name].tools.extend(tools)

    def get_client(self, server_name: str) -> Optional[MCPClient]:
        server = self._servers.get(server_name)
        if not server:
            return None
        transport = self._transports.get(server_name, InMemoryTransport())
        return MCPClient(server=server, transport=transport)

    @classmethod
    def from_env(cls) -> "MCPRegistry":
        registry = cls()
        servers = os.getenv("MCP_SERVERS", "")
        for entry in filter(None, [chunk.strip() for chunk in servers.split(",")]):
            if "=" not in entry:
                continue
            name, url = entry.split("=", 1)
            registry.register_server(MCPServerConfig(name=name, url=url))
        return registry
