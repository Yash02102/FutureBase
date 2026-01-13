import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from .mcp_tool import call_mcp_tool

_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "sku_1001",
        "name": "EchoLite Bluetooth Speaker",
        "price": 3499,
        "rating": 4.4,
        "stock": 12,
        "category": "audio",
    },
    {
        "id": "sku_1002",
        "name": "Nimbus Wireless Headset",
        "price": 4899,
        "rating": 4.6,
        "stock": 5,
        "category": "audio",
    },
    {
        "id": "sku_1003",
        "name": "VoltAir Noise-Canceling Earbuds",
        "price": 2999,
        "rating": 4.2,
        "stock": 0,
        "category": "audio",
    },
    {
        "id": "sku_2001",
        "name": "Drift Car Phone Mount",
        "price": 899,
        "rating": 4.1,
        "stock": 27,
        "category": "auto",
    },
    {
        "id": "sku_3001",
        "name": "Orion Smartwatch Active",
        "price": 5499,
        "rating": 4.5,
        "stock": 9,
        "category": "wearables",
    },
]

_CARTS: Dict[str, List[Dict[str, Any]]] = {}
_ORDERS: Dict[str, Dict[str, Any]] = {}
_SHIPMENTS: Dict[str, Dict[str, Any]] = {}
_TICKETS: Dict[str, Dict[str, Any]] = {}


def _api_mode() -> str:
    return os.getenv("COMMERCE_API_MODE", "mock").lower()


def _api_base_url() -> str:
    return os.getenv("COMMERCE_API_BASE_URL", "http://localhost:8002").rstrip("/")


def _request(method: str, path: str, payload: Optional[dict] = None, params: Optional[dict] = None) -> Any:
    url = f"{_api_base_url()}/{path.lstrip('/')}"
    response = requests.request(method, url, json=payload, params=params, timeout=10)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return response.text


def _mcp_enabled(tool_env: str) -> bool:
    return bool(os.getenv("COMMERCE_MCP_SERVER") and os.getenv(tool_env))


def _mcp_strict() -> bool:
    return os.getenv("COMMERCE_MCP_STRICT", "false").lower() == "true"


def _mcp_call(tool_env: str, args: Dict[str, Any]) -> Optional[str]:
    if not _mcp_enabled(tool_env):
        return None
    server = os.getenv("COMMERCE_MCP_SERVER", "")
    tool_name = os.getenv(tool_env, "")
    result = call_mcp_tool(server, tool_name, _stringify_args(args))
    if _mcp_strict():
        return result
    if _is_json(result):
        return result
    return None


def _stringify_args(args: Dict[str, Any]) -> Dict[str, str]:
    return {key: json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list)) else str(value) for key, value in args.items() if value is not None}


def _is_json(value: str) -> bool:
    try:
        json.loads(value)
        return True
    except json.JSONDecodeError:
        return False


def _as_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True)


def _find_product(sku: str) -> Optional[Dict[str, Any]]:
    for product in _CATALOG:
        if product["id"] == sku:
            return product
    return None


def catalog_search(
    query: str,
    limit: int = 5,
    max_price: Optional[float] = None,
    category: Optional[str] = None,
) -> str:
    mcp_result = _mcp_call(
        "COMMERCE_MCP_CATALOG_TOOL",
        {"query": query, "limit": limit, "max_price": max_price, "category": category},
    )
    if mcp_result is not None:
        return mcp_result
    if _api_mode() == "remote":
        data = _request(
            "GET",
            "/catalog/search",
            params={
                "q": query,
                "limit": limit,
                "max_price": max_price,
                "category": category,
            },
        )
        return _as_json(data)

    query_lower = query.lower().strip()
    results = []
    for product in _CATALOG:
        if query_lower and query_lower not in product["name"].lower():
            if not product["category"].lower().startswith(query_lower):
                continue
        if category and product["category"] != category:
            continue
        if max_price is not None and product["price"] > max_price:
            continue
        results.append(product)

    results = sorted(results, key=lambda item: item["rating"], reverse=True)[:limit]
    return _as_json({"count": len(results), "items": results})


def inventory_check(sku: str) -> str:
    mcp_result = _mcp_call("COMMERCE_MCP_INVENTORY_TOOL", {"sku": sku})
    if mcp_result is not None:
        return mcp_result
    if _api_mode() == "remote":
        return _as_json(_request("GET", f"/inventory/{sku}"))

    product = _find_product(sku)
    if not product:
        return _as_json({"sku": sku, "available": False, "stock": 0})
    return _as_json({"sku": sku, "available": product["stock"] > 0, "stock": product["stock"]})


def pricing_lookup(sku: str, user_id: Optional[str] = None) -> str:
    if _api_mode() == "remote":
        return _as_json(_request("GET", f"/pricing/{sku}", params={"user_id": user_id}))

    product = _find_product(sku)
    if not product:
        return _as_json(
            {
                "sku": sku,
                "base_price": None,
                "final_price": None,
                "currency": "INR",
                "discount_pct": 0,
            }
        )
    price = float(product["price"])
    vip_users = {entry.strip() for entry in os.getenv("COMMERCE_VIP_USERS", "").split(",") if entry.strip()}
    discount = 0.1 if user_id and user_id in vip_users else 0.0
    final_price = int(price * (1 - discount))
    return _as_json(
        {
            "sku": sku,
            "base_price": product["price"],
            "final_price": final_price,
            "currency": "INR",
            "discount_pct": int(discount * 100),
        }
    )


def promo_check(user_id: str, cart_total: float, code: Optional[str] = None) -> str:
    if _api_mode() == "remote":
        return _as_json(
            _request(
                "POST",
                "/promotions/check",
                payload={"user_id": user_id, "cart_total": cart_total, "code": code},
            )
        )

    discount = 0
    reason = "No promotion applied."
    if code and code.upper() == "SAVE10" and cart_total >= 1000:
        discount = int(cart_total * 0.1)
        reason = "SAVE10 applied."
    return _as_json({"discount": discount, "reason": reason})


def cart_add(user_id: str, sku: str, quantity: int = 1) -> str:
    if _api_mode() == "remote":
        return _as_json(
            _request(
                "POST",
                f"/cart/{user_id}/items",
                payload={"sku": sku, "quantity": quantity},
            )
        )

    cart = _CARTS.setdefault(user_id, [])
    for item in cart:
        if item["sku"] == sku:
            item["quantity"] += quantity
            break
    else:
        cart.append({"sku": sku, "quantity": quantity})
    return _as_json({"user_id": user_id, "items": cart})


def cart_remove(user_id: str, sku: str) -> str:
    if _api_mode() == "remote":
        return _as_json(_request("DELETE", f"/cart/{user_id}/items/{sku}"))

    cart = _CARTS.setdefault(user_id, [])
    cart = [item for item in cart if item["sku"] != sku]
    _CARTS[user_id] = cart
    return _as_json({"user_id": user_id, "items": cart})


def cart_view(user_id: str) -> str:
    if _api_mode() == "remote":
        return _as_json(_request("GET", f"/cart/{user_id}"))
    return _as_json({"user_id": user_id, "items": _CARTS.get(user_id, [])})


def fraud_check(user_id: str, order_total: float, order_id: Optional[str] = None) -> str:
    if _api_mode() == "remote":
        return _as_json(
            _request(
                "POST",
                "/fraud/check",
                payload={"user_id": user_id, "order_total": order_total, "order_id": order_id},
            )
        )
    risk = "low" if order_total < 20000 else "review"
    return _as_json({"order_id": order_id, "risk": risk})


def checkout(user_id: str, payment_method: str, address: str) -> str:
    if _api_mode() == "remote":
        return _as_json(
            _request(
                "POST",
                "/checkout",
                payload={"user_id": user_id, "payment_method": payment_method, "address": address},
            )
        )

    cart = _CARTS.get(user_id, [])
    if not cart:
        return _as_json({"error": "Cart is empty."})
    total = 0
    for item in cart:
        product = _find_product(item["sku"])
        if not product:
            continue
        total += product["price"] * item["quantity"]
    order_id = f"ord_{int(time.time())}"
    tracking_id = f"trk_{int(time.time())}"
    _ORDERS[order_id] = {
        "order_id": order_id,
        "user_id": user_id,
        "items": cart,
        "total": total,
        "status": "confirmed",
        "payment_method": payment_method,
    }
    _SHIPMENTS[tracking_id] = {
        "tracking_id": tracking_id,
        "order_id": order_id,
        "status": "label_created",
    }
    _CARTS[user_id] = []
    return _as_json({"order_id": order_id, "total": total, "tracking_id": tracking_id})


def order_status(order_id: str) -> str:
    if _api_mode() == "remote":
        return _as_json(_request("GET", f"/orders/{order_id}"))

    order = _ORDERS.get(order_id)
    if not order:
        return _as_json({"order_id": order_id, "status": "not_found"})
    return _as_json(order)


def track_shipment(tracking_id: str) -> str:
    if _api_mode() == "remote":
        return _as_json(_request("GET", f"/logistics/track/{tracking_id}"))

    shipment = _SHIPMENTS.get(tracking_id)
    if not shipment:
        return _as_json({"tracking_id": tracking_id, "status": "not_found"})
    return _as_json(shipment)


def return_request(order_id: str, reason: str) -> str:
    if _api_mode() == "remote":
        return _as_json(
            _request(
                "POST",
                "/returns",
                payload={"order_id": order_id, "reason": reason},
            )
        )

    order = _ORDERS.get(order_id)
    if not order:
        return _as_json({"order_id": order_id, "status": "not_found"})
    order["status"] = "return_requested"
    order["return_reason"] = reason
    return _as_json({"order_id": order_id, "status": "return_requested"})


def refund_status(order_id: str) -> str:
    if _api_mode() == "remote":
        return _as_json(_request("GET", f"/refunds/{order_id}"))

    order = _ORDERS.get(order_id)
    if not order:
        return _as_json({"order_id": order_id, "refund_status": "not_found"})
    refund_status = "pending" if order.get("status") == "return_requested" else "n/a"
    return _as_json({"order_id": order_id, "refund_status": refund_status})


def reorder(user_id: str, order_id: str) -> str:
    if _api_mode() == "remote":
        return _as_json(
            _request(
                "POST",
                "/orders/reorder",
                payload={"user_id": user_id, "order_id": order_id},
            )
        )

    order = _ORDERS.get(order_id)
    if not order:
        return _as_json({"order_id": order_id, "status": "not_found"})
    _CARTS[user_id] = list(order["items"])
    return _as_json({"user_id": user_id, "items": _CARTS[user_id]})


def support_ticket(user_id: str, subject: str, description: str) -> str:
    if _api_mode() == "remote":
        return _as_json(
            _request(
                "POST",
                "/support/tickets",
                payload={"user_id": user_id, "subject": subject, "description": description},
            )
        )

    ticket_id = f"tic_{int(time.time())}"
    _TICKETS[ticket_id] = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "subject": subject,
        "description": description,
        "status": "open",
    }
    return _as_json(_TICKETS[ticket_id])
