from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class ToolSchema:
    args_model: Optional[Type[BaseModel]]
    output_model: Optional[Type[BaseModel]]


class Product(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    price: int
    rating: float
    stock: int
    category: str


class CatalogSearchArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    limit: int = Field(default=5, ge=1, le=10)
    max_price: Optional[float] = Field(default=None, ge=0)
    category: Optional[str] = None


class CatalogSearchOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    count: int
    items: List[Product]


class InventoryArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sku: str


class InventoryOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sku: str
    available: bool
    stock: int


class PricingArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sku: str
    user_id: Optional[str] = None


class PricingOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sku: str
    base_price: Optional[int] = None
    final_price: Optional[int] = None
    currency: str
    discount_pct: int


class PromoArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    cart_total: float = Field(ge=0)
    code: Optional[str] = None


class PromoOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    discount: int
    reason: str


class CartItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sku: str
    quantity: int = Field(ge=1)


class CartAddArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    sku: str
    quantity: int = Field(default=1, ge=1)


class CartRemoveArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    sku: str


class CartViewArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str


class CartOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    items: List[CartItem]


class FraudArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    order_total: float = Field(ge=0)
    order_id: Optional[str] = None


class FraudOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: Optional[str] = None
    risk: str


class CheckoutArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    payment_method: str
    address: str


class CheckoutOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: Optional[str] = None
    total: Optional[int] = None
    tracking_id: Optional[str] = None
    error: Optional[str] = None


class OrderStatusArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str


class OrderStatusOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str
    status: str


class TrackingArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tracking_id: str


class TrackingOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tracking_id: str
    status: str


class ReturnArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str
    reason: str


class ReturnOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str
    status: str


class RefundArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str


class RefundOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    order_id: str
    refund_status: str


class ReorderArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    order_id: str


class ReorderOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    items: List[CartItem]


class SupportArgs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: str
    subject: str
    description: str


class SupportOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticket_id: str
    user_id: str
    subject: str
    description: str
    status: str


TOOL_SCHEMAS: Dict[str, ToolSchema] = {
    "catalog_search_tool": ToolSchema(CatalogSearchArgs, CatalogSearchOutput),
    "inventory_check_tool": ToolSchema(InventoryArgs, InventoryOutput),
    "pricing_tool": ToolSchema(PricingArgs, PricingOutput),
    "promo_tool": ToolSchema(PromoArgs, PromoOutput),
    "cart_add_tool": ToolSchema(CartAddArgs, CartOutput),
    "cart_remove_tool": ToolSchema(CartRemoveArgs, CartOutput),
    "cart_view_tool": ToolSchema(CartViewArgs, CartOutput),
    "fraud_check_tool": ToolSchema(FraudArgs, FraudOutput),
    "checkout_tool": ToolSchema(CheckoutArgs, CheckoutOutput),
    "order_status_tool": ToolSchema(OrderStatusArgs, OrderStatusOutput),
    "logistics_tool": ToolSchema(TrackingArgs, TrackingOutput),
    "return_tool": ToolSchema(ReturnArgs, ReturnOutput),
    "refund_tool": ToolSchema(RefundArgs, RefundOutput),
    "reorder_tool": ToolSchema(ReorderArgs, ReorderOutput),
    "support_tool": ToolSchema(SupportArgs, SupportOutput),
}
