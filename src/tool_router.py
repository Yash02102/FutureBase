from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from langchain_core.tools import BaseTool

from .mcp import tools as mcp_tools
from .tools import create_http_get_tool, create_rag_tool


@dataclass(frozen=True)
class ToolSpec:
    name: str
    tool: BaseTool
    tags: Sequence[str]
    cost: int = 1
    always: bool = False


class ToolRouter:
    def __init__(self, specs: Iterable[ToolSpec], max_tools: int, max_cost: int, min_score: float) -> None:
        self.specs = list(specs)
        self.max_tools = max_tools
        self.max_cost = max_cost
        self.min_score = min_score

    def select(self, task: str, context: str) -> List[BaseTool]:
        text = f"{task} {context}".lower()
        scored: List[tuple[float, ToolSpec]] = []
        for spec in self.specs:
            score = sum(1 for tag in spec.tags if tag in text)
            if spec.always:
                score += 1.0
            if score >= self.min_score or spec.always:
                scored.append((score, spec))
        scored.sort(key=lambda item: (item[0], -item[1].cost), reverse=True)
        selected: List[BaseTool] = []
        total_cost = 0
        for _score, spec in scored:
            if len(selected) >= self.max_tools:
                continue
            if total_cost + spec.cost > self.max_cost:
                continue
            selected.append(spec.tool)
            total_cost += spec.cost
        return selected


def build_tool_specs(rag_pipeline) -> List[ToolSpec]:
    rag_tool = create_rag_tool(rag_pipeline)
    http_tool = create_http_get_tool()
    return [
        ToolSpec(
            name=rag_tool.name,
            tool=rag_tool,
            tags=["rag", "docs", "context", "policy", "spec"],
            cost=3,
        ),
        ToolSpec(
            name=http_tool.name,
            tool=http_tool,
            tags=["api", "http", "endpoint", "fetch"],
            cost=4,
        ),
        ToolSpec(
            name=mcp_tools.catalog_search.name,
            tool=mcp_tools.catalog_search,
            tags=["buy", "search", "catalog", "find", "product"],
            cost=1,
            always=True,
        ),
        ToolSpec(
            name=mcp_tools.inventory_check.name,
            tool=mcp_tools.inventory_check,
            tags=["inventory", "stock", "availability"],
            cost=1,
        ),
        ToolSpec(
            name=mcp_tools.pricing_lookup.name,
            tool=mcp_tools.pricing_lookup,
            tags=["price", "cost", "discount"],
            cost=1,
        ),
        ToolSpec(
            name=mcp_tools.promo_check.name,
            tool=mcp_tools.promo_check,
            tags=["promo", "coupon", "offer"],
            cost=1,
        ),
        ToolSpec(
            name=mcp_tools.cart_add.name,
            tool=mcp_tools.cart_add,
            tags=["cart", "add", "buy"],
            cost=2,
        ),
        ToolSpec(
            name=mcp_tools.cart_remove.name,
            tool=mcp_tools.cart_remove,
            tags=["remove", "cart"],
            cost=2,
        ),
        ToolSpec(
            name=mcp_tools.cart_view.name,
            tool=mcp_tools.cart_view,
            tags=["cart", "view"],
            cost=2,
        ),
        ToolSpec(
            name=mcp_tools.checkout.name,
            tool=mcp_tools.checkout,
            tags=["checkout", "pay", "order"],
            cost=3,
        ),
        ToolSpec(
            name=mcp_tools.order_status.name,
            tool=mcp_tools.order_status,
            tags=["track", "status", "order"],
            cost=2,
        ),
        ToolSpec(
            name=mcp_tools.track_shipment.name,
            tool=mcp_tools.track_shipment,
            tags=["track", "shipment", "delivery"],
            cost=2,
        ),
        ToolSpec(
            name=mcp_tools.return_request.name,
            tool=mcp_tools.return_request,
            tags=["return", "refund"],
            cost=2,
        ),
        ToolSpec(
            name=mcp_tools.refund_status.name,
            tool=mcp_tools.refund_status,
            tags=["refund", "status"],
            cost=2,
        ),
        ToolSpec(
            name=mcp_tools.reorder.name,
            tool=mcp_tools.reorder,
            tags=["reorder", "again", "repeat"],
            cost=2,
        ),
        ToolSpec(
            name=mcp_tools.support_ticket.name,
            tool=mcp_tools.support_ticket,
            tags=["support", "issue", "ticket"],
            cost=2,
        ),
    ]
