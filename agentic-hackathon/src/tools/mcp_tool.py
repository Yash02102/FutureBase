from typing import Dict

from ..mcp import MCPRegistry, MCPToolSpec


_registry = MCPRegistry.from_env()


def register_mcp_tool(server_name: str, tool: MCPToolSpec) -> None:
    _registry.add_tools(server_name, [tool])


def call_mcp_tool(server_name: str, tool_name: str, arguments: Dict[str, str]) -> str:
    client = _registry.get_client(server_name)
    if not client:
        return f"MCP server '{server_name}' not registered. Set MCP_SERVERS env var."
    client.connect()
    return client.call_tool(tool_name, arguments)
