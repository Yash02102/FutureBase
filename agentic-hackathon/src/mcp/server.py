from __future__ import annotations

import os
import time
import uuid
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP


class CommerceStore:
    def __init__(self) -> None:
        self.catalog = [
            {"sku": "WH-1000XM5", "name": "Sony WH-1000XM5", "price": 29999, "category": "headphones"},
            {"sku": "QC45", "name": "Bose QuietComfort 45", "price": 24999, "category": "headphones"},
            {"sku": "BUDS2", "name": "Galaxy Buds2", "price": 8999, "category": "earbuds"},
            {"sku": "JBL760", "name": "JBL Tune 760NC", "price": 6999, "category": "headphones"},
            {"sku": "BOAT120", "name": "boAt Rockerz 120", "price": 1299, "category": "earphones"},
        ]
        self.inventory = {item["sku"]: 12 for item in self.catalog}
        self.carts: Dict[str, Dict[str, int]] = {}
        self.orders: Dict[str, Dict[str, str]] = {}
        self.tickets: Dict[str, Dict[str, str]] = {}

    def search(self, query: str, limit: int, max_price: Optional[float], category: Optional[str]) -> List[Dict]:
        results = []
        query_lower = query.lower()
        for item in self.catalog:
            if query_lower and query_lower not in item["name"].lower():
                continue
            if category and category.lower() not in item["category"].lower():
                continue
            if max_price is not None and item["price"] > max_price:
                continue
            results.append(item)
            if len(results) >= limit:
                break
        return results


_STORE = CommerceStore()

mcp = FastMCP(name="commerce-mcp", instructions="Commerce tools for catalog, orders, and support.")


@mcp.tool()
def catalog_search(query: str, limit: int = 5, max_price: Optional[float] = None, category: Optional[str] = None) -> Dict:
    return {"results": _STORE.search(query, limit, max_price, category)}


@mcp.tool()
def inventory_check(sku: str) -> Dict:
    qty = _STORE.inventory.get(sku, 0)
    return {"sku": sku, "available": qty > 0, "quantity": qty}


@mcp.tool()
def pricing_lookup(sku: str, user_id: Optional[str] = None) -> Dict:
    item = next((item for item in _STORE.catalog if item["sku"] == sku), None)
    if not item:
        return {"error": "unknown_sku"}
    base_price = item["price"]
    discount = 0.05 if user_id else 0.0
    return {
        "sku": sku,
        "price": base_price,
        "discount": discount,
        "final_price": int(base_price * (1 - discount)),
        "currency": "INR",
    }


@mcp.tool()
def promo_check(user_id: str, cart_total: float, code: Optional[str] = None) -> Dict:
    eligible = cart_total > 5000
    if code and code.lower() == "save10":
        return {"eligible": True, "discount": 0.1}
    return {"eligible": eligible, "discount": 0.05 if eligible else 0.0}


@mcp.tool()
def cart_add(user_id: str, sku: str, quantity: int = 1) -> Dict:
    cart = _STORE.carts.setdefault(user_id, {})
    cart[sku] = cart.get(sku, 0) + quantity
    return {"user_id": user_id, "cart": cart}


@mcp.tool()
def cart_remove(user_id: str, sku: str) -> Dict:
    cart = _STORE.carts.setdefault(user_id, {})
    if sku in cart:
        del cart[sku]
    return {"user_id": user_id, "cart": cart}


@mcp.tool()
def cart_view(user_id: str) -> Dict:
    return {"user_id": user_id, "cart": _STORE.carts.get(user_id, {})}


@mcp.tool()
def checkout(user_id: str, payment_method: str, address: str) -> Dict:
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    _STORE.orders[order_id] = {
        "order_id": order_id,
        "status": "processing",
        "user_id": user_id,
        "payment_method": payment_method,
        "address": address,
    }
    return {"order_id": order_id, "status": "processing"}


@mcp.tool()
def order_status(order_id: str) -> Dict:
    order = _STORE.orders.get(order_id)
    if not order:
        return {"error": "order_not_found"}
    return {"order_id": order_id, "status": order["status"], "updated_at": int(time.time())}


@mcp.tool()
def track_shipment(tracking_id: str) -> Dict:
    return {"tracking_id": tracking_id, "status": "in_transit", "eta_days": 2}


@mcp.tool()
def return_request(order_id: str, reason: str) -> Dict:
    return {"order_id": order_id, "status": "return_requested", "reason": reason}


@mcp.tool()
def refund_status(order_id: str) -> Dict:
    return {"order_id": order_id, "status": "pending"}


@mcp.tool()
def reorder(user_id: str, order_id: str) -> Dict:
    return {"user_id": user_id, "order_id": order_id, "status": "reordered"}


@mcp.tool()
def support_ticket(user_id: str, subject: str, description: str) -> Dict:
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    _STORE.tickets[ticket_id] = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "subject": subject,
        "description": description,
    }
    return {"ticket_id": ticket_id, "status": "open"}


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
