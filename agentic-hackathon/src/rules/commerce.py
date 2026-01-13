import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .engine import Intent, Rule, RuleContext, RuleDecision


COMMERCE_KEYWORDS = {
    "purchase": ["buy", "purchase", "checkout", "place order", "add to cart"],
    "compare_products": ["compare", "comparison", "vs", "which is better"],
    "track_order": ["track", "tracking", "order status", "where is my order"],
    "return_request": ["return", "exchange", "send back"],
    "refund_request": ["refund", "chargeback", "money back"],
    "reorder": ["reorder", "buy again", "order again"],
    "support_ticket": ["support", "help", "ticket", "issue", "problem"],
    "product_search": ["find", "search", "recommend", "looking for", "need", "get"],
    "address_change": ["change address", "update address", "shipping address"],
}

_INTENT_PRIORITY = [
    "refund_request",
    "return_request",
    "track_order",
    "reorder",
    "purchase",
    "compare_products",
    "support_ticket",
    "product_search",
    "address_change",
]


class CommerceIntentClassifier:
    def __init__(self, keyword_map: Dict[str, List[str]]) -> None:
        self.keyword_map = keyword_map

    def classify(self, task: str) -> Intent:
        text = _normalize(task)
        entities = _extract_entities(text)
        scores = {intent: 0 for intent in self.keyword_map}
        for intent, keywords in self.keyword_map.items():
            for keyword in keywords:
                if keyword in text:
                    scores[intent] += 1
        _apply_entity_hints(scores, entities, text)
        best_intent, best_score = _pick_intent(scores)
        confidence = min(1.0, best_score / 3) if best_score else 0.0
        tags = list(entities.keys())
        if not best_score:
            best_intent = "general"
        return Intent(name=best_intent, confidence=confidence, tags=tags, entities=entities)


def commerce_intent_classifier() -> CommerceIntentClassifier:
    return CommerceIntentClassifier(COMMERCE_KEYWORDS)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _apply_entity_hints(scores: Dict[str, int], entities: Dict[str, str], text: str) -> None:
    if entities.get("tracking_id"):
        scores["track_order"] += 2
    if entities.get("order_id") and "refund" in text:
        scores["refund_request"] += 2
    if entities.get("order_id") and "return" in text:
        scores["return_request"] += 2
    if entities.get("order_id") and "status" in text:
        scores["track_order"] += 2
    if entities.get("order_id") and "reorder" in text:
        scores["reorder"] += 2
    if entities.get("product_query"):
        scores["product_search"] += 1
    if "compare" in text or " vs " in text:
        scores["compare_products"] += 2


def _pick_intent(scores: Dict[str, int]) -> tuple[str, int]:
    best_intent = ""
    best_score = 0
    for intent, score in scores.items():
        if score > best_score:
            best_intent = intent
            best_score = score
    if not best_intent:
        return "general", 0
    tied = [intent for intent, score in scores.items() if score == best_score]
    if len(tied) > 1:
        for intent in _INTENT_PRIORITY:
            if intent in tied:
                return intent, best_score
    return best_intent, best_score


def _extract_entities(text: str) -> Dict[str, str]:
    entities: Dict[str, str] = {}
    order_id = _match_any(
        text,
        [
            r"\b(?:order id|order #|order)\s*[:#]?\s*([a-z0-9_-]{4,})",
            r"\bord[_-]?\d+\b",
        ],
    )
    if order_id:
        entities["order_id"] = order_id
    tracking_id = _match_any(
        text,
        [
            r"\b(?:tracking id|tracking #|tracking)\s*[:#]?\s*([a-z0-9_-]{4,})",
            r"\btrk[_-]?\d+\b",
        ],
    )
    if tracking_id:
        entities["tracking_id"] = tracking_id
    sku = _match_any(text, [r"\bsku[_-]?\d+\b"])
    if sku:
        entities["sku"] = sku
    quantity = _match_any(text, [r"\b(?:qty|quantity|x)\s*([0-9]+)\b"])
    if quantity:
        entities["quantity"] = quantity
    max_price = _extract_max_price(text)
    if max_price is not None:
        entities["max_price"] = str(max_price)
    product_query = _extract_product_query(text)
    if product_query:
        entities["product_query"] = product_query
    return entities


def _match_any(text: str, patterns: List[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return ""


def _extract_max_price(text: str) -> Optional[int]:
    match = re.search(
        r"(?:under|below|less than|max|<=)\s*([0-9][0-9,\.]*\s*[kK]?)",
        text,
    )
    if not match:
        return None
    return _parse_amount(match.group(1))


def _parse_amount(value: str) -> Optional[int]:
    cleaned = value.lower().replace(",", "").strip()
    multiplier = 1
    if cleaned.endswith("k"):
        multiplier = 1000
        cleaned = cleaned[:-1]
    try:
        amount = float(cleaned)
    except ValueError:
        return None
    return int(amount * multiplier)


def _extract_product_query(text: str) -> str:
    match = re.search(
        r"(?:buy|purchase|find|search for|looking for|need|recommend|get)\s+(.*)",
        text,
    )
    if not match:
        return ""
    query = match.group(1)
    query = re.sub(r"(?:under|below|less than|max|<=)\s*.*", "", query).strip()
    query = re.sub(r"\bfor\s+my\b", "for my", query).strip()
    return query.strip(" .")


@dataclass(frozen=True)
class IntentMatchRule(Rule):
    intent: str
    decision: RuleDecision

    def apply(self, context: RuleContext) -> Optional[RuleDecision]:
        if context.intent.name == self.intent:
            return self.decision
        return None


def commerce_rules() -> List[Rule]:
    return [
        IntentMatchRule(
            intent="refund_request",
            decision=RuleDecision(
                require_approval=True,
                notes=["Refunds require human approval before responding."],
                system_instructions=[
                    "Do not promise refunds or reversals without approval.",
                    "Collect order ID and reason only.",
                ],
                allowed_tools=["order_status_tool", "refund_tool", "support_tool"],
            ),
        ),
        IntentMatchRule(
            intent="return_request",
            decision=RuleDecision(
                notes=["Returns require eligibility checks before confirming."],
                system_instructions=[
                    "Collect order ID, item details, and return reason.",
                    "Confirm return window and condition requirements.",
                ],
                allowed_tools=["order_status_tool", "return_tool", "refund_tool"],
            ),
        ),
        IntentMatchRule(
            intent="address_change",
            decision=RuleDecision(
                require_approval=True,
                notes=["Address changes require verification and approval."],
                system_instructions=[
                    "Verify identity before confirming address updates.",
                ],
                allowed_tools=["support_tool"],
            ),
        ),
        IntentMatchRule(
            intent="track_order",
            decision=RuleDecision(
                notes=["Prefer internal order systems for status."],
                system_instructions=[
                    "Use internal order status tools when available.",
                ],
                allowed_tools=["order_status_tool", "logistics_tool"],
            ),
        ),
        IntentMatchRule(
            intent="reorder",
            decision=RuleDecision(
                notes=["Reorders should confirm items before checkout."],
                system_instructions=[
                    "Confirm items and quantities before reordering.",
                ],
                allowed_tools=["order_status_tool", "reorder_tool", "cart_view_tool"],
            ),
        ),
        IntentMatchRule(
            intent="support_ticket",
            decision=RuleDecision(
                notes=["Open a support ticket for unresolved issues."],
                system_instructions=[
                    "Collect clear subject and description for the ticket.",
                ],
                allowed_tools=["support_tool"],
            ),
        ),
        IntentMatchRule(
            intent="product_search",
            decision=RuleDecision(
                notes=["Recommend products with clear disclaimers on availability."],
                system_instructions=[
                    "Avoid guaranteeing inventory or delivery dates.",
                ],
                allowed_tools=["catalog_search_tool", "inventory_check_tool", "pricing_tool"],
            ),
        ),
        IntentMatchRule(
            intent="compare_products",
            decision=RuleDecision(
                notes=["Provide side-by-side comparisons using catalog data."],
                system_instructions=[
                    "Focus on price, rating, and category fit.",
                ],
                allowed_tools=["catalog_search_tool", "pricing_tool"],
            ),
        ),
        IntentMatchRule(
            intent="purchase",
            decision=RuleDecision(
                notes=["Confirm item, price, and payment method before checkout."],
                system_instructions=[
                    "Ask for confirmation before placing an order.",
                    "Verify shipping address before payment.",
                ],
                allowed_tools=[
                    "catalog_search_tool",
                    "inventory_check_tool",
                    "pricing_tool",
                    "promo_tool",
                    "cart_add_tool",
                    "cart_view_tool",
                    "fraud_check_tool",
                    "checkout_tool",
                ],
            ),
        ),
    ]
