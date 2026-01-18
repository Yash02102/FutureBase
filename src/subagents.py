from __future__ import annotations

from typing import List, Sequence

from deepagents.middleware.subagents import SubAgent
from langchain_core.tools import BaseTool


SPECIALISTS = [
    {
        "name": "CatalogAgent",
        "description": "Research products, inventory, and pricing for comparisons or recommendations.",
        "system_prompt": (
            "You are a catalog specialist. Use tools to search the catalog, check inventory, "
            "and compare prices. Return a concise set of options with key tradeoffs."
        ),
        "tool_names": {"catalog_search", "inventory_check", "pricing_lookup", "rag_search"},
    },
    {
        "name": "CartAgent",
        "description": "Manage cart contents when the user asks to add, remove, or review items.",
        "system_prompt": (
            "You manage cart updates. Add or remove items only when the user intent is explicit. "
            "Summarize the updated cart and any relevant promo guidance."
        ),
        "tool_names": {"cart_add", "cart_remove", "cart_view", "promo_check"},
    },
    {
        "name": "CheckoutAgent",
        "description": "Finalize checkout when required details are confirmed.",
        "system_prompt": (
            "You complete checkout only after required details are confirmed (user_id, address, "
            "payment_method). Confirm cart totals before submitting."
        ),
        "tool_names": {"checkout", "cart_view"},
    },
    {
        "name": "OrderSupportAgent",
        "description": "Handle order status and shipment tracking requests.",
        "system_prompt": "You provide order status and shipment tracking updates.",
        "tool_names": {"order_status", "track_shipment"},
    },
    {
        "name": "ReturnsAgent",
        "description": "Handle return requests and refund status checks.",
        "system_prompt": "You create return requests and report refund status when asked.",
        "tool_names": {"return_request", "refund_status"},
    },
    {
        "name": "ReorderAgent",
        "description": "Place reorders for previous purchases once the user confirms.",
        "system_prompt": "You reorder items only after the user confirms the order_id and user_id.",
        "tool_names": {"reorder"},
    },
    {
        "name": "SupportAgent",
        "description": "Open support tickets for issues that need escalation.",
        "system_prompt": "You open support tickets with a clear subject and concise issue summary.",
        "tool_names": {"support_ticket"},
    },
]


def build_subagents(llm, tools: Sequence[BaseTool]) -> List[SubAgent]:
    tool_map = {tool.name: tool for tool in tools}
    subagents: List[SubAgent] = []
    for spec in SPECIALISTS:
        selected = [tool_map[name] for name in spec["tool_names"] if name in tool_map]
        subagents.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "system_prompt": spec["system_prompt"],
                "tools": selected,
                "model": llm,
            }
        )
    return subagents