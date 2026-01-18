from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from langchain_core.tools import tool

from .client import MCPClient


def _mcp_url() -> str:
    return os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")


@lru_cache(maxsize=1)
def _client() -> MCPClient:
    return MCPClient(url=_mcp_url())


@tool
def catalog_search(query: str, limit: int = 5, max_price: Optional[float] = None, category: Optional[str] = None) -> str:
    """Search the catalog for products."""
    return _client().call_tool(
        "catalog_search",
        {"query": query, "limit": limit, "max_price": max_price, "category": category},
    )


@tool
def inventory_check(sku: str) -> str:
    """Check inventory for a SKU."""
    return _client().call_tool("inventory_check", {"sku": sku})


@tool
def pricing_lookup(sku: str, user_id: Optional[str] = None) -> str:
    """Fetch pricing for a SKU."""
    return _client().call_tool("pricing_lookup", {"sku": sku, "user_id": user_id})


@tool
def promo_check(user_id: str, cart_total: float, code: Optional[str] = None) -> str:
    """Check promotion eligibility."""
    return _client().call_tool(
        "promo_check", {"user_id": user_id, "cart_total": cart_total, "code": code}
    )


@tool
def cart_add(user_id: str, sku: str, quantity: int = 1) -> str:
    """Add an item to the cart."""
    return _client().call_tool(
        "cart_add", {"user_id": user_id, "sku": sku, "quantity": quantity}
    )


@tool
def cart_remove(user_id: str, sku: str) -> str:
    """Remove an item from the cart."""
    return _client().call_tool("cart_remove", {"user_id": user_id, "sku": sku})


@tool
def cart_view(user_id: str) -> str:
    """View the cart."""
    return _client().call_tool("cart_view", {"user_id": user_id})


@tool
def checkout(user_id: str, payment_method: str, address: str) -> str:
    """Checkout the current cart."""
    return _client().call_tool(
        "checkout", {"user_id": user_id, "payment_method": payment_method, "address": address}
    )


@tool
def order_status(order_id: str) -> str:
    """Get order status."""
    return _client().call_tool("order_status", {"order_id": order_id})


@tool
def track_shipment(tracking_id: str) -> str:
    """Track shipment status."""
    return _client().call_tool("track_shipment", {"tracking_id": tracking_id})


@tool
def return_request(order_id: str, reason: str) -> str:
    """Create a return request."""
    return _client().call_tool("return_request", {"order_id": order_id, "reason": reason})


@tool
def refund_status(order_id: str) -> str:
    """Fetch refund status."""
    return _client().call_tool("refund_status", {"order_id": order_id})


@tool
def reorder(user_id: str, order_id: str) -> str:
    """Reorder a previous order."""
    return _client().call_tool("reorder", {"user_id": user_id, "order_id": order_id})


@tool
def support_ticket(user_id: str, subject: str, description: str) -> str:
    """Create a support ticket."""
    return _client().call_tool(
        "support_ticket", {"user_id": user_id, "subject": subject, "description": description}
    )
