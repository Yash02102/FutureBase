from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.tools import BaseTool

from .mcp import tools as mcp_tools
from .tools import create_http_get_tool, create_rag_tool


@dataclass(frozen=True)
class ToolSets:
    all_tools: List[BaseTool]
    generic_tools: List[BaseTool]


def build_tool_sets(rag_pipeline) -> ToolSets:
    rag_tool = create_rag_tool(rag_pipeline)
    http_tool = create_http_get_tool()
    all_tools = [
        rag_tool,
        http_tool,
        mcp_tools.catalog_search,
        mcp_tools.inventory_check,
        mcp_tools.pricing_lookup,
        mcp_tools.promo_check,
        mcp_tools.cart_add,
        mcp_tools.cart_remove,
        mcp_tools.cart_view,
        mcp_tools.checkout,
        mcp_tools.order_status,
        mcp_tools.track_shipment,
        mcp_tools.return_request,
        mcp_tools.refund_status,
        mcp_tools.reorder,
        mcp_tools.support_ticket,
    ]
    tool_map = {tool.name: tool for tool in all_tools}
    generic_names = [
        "rag_search",
        "http_get",
        "catalog_search",
        "inventory_check",
        "pricing_lookup",
        "promo_check",
        "cart_view",
        "order_status",
        "track_shipment",
        "refund_status",
    ]
    generic_tools = [tool_map[name] for name in generic_names if name in tool_map]
    return ToolSets(all_tools=all_tools, generic_tools=generic_tools)
