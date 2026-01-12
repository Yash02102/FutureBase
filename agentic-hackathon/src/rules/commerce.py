from dataclasses import dataclass
from typing import List, Optional

from .engine import Intent, KeywordIntentClassifier, Rule, RuleContext, RuleDecision


COMMERCE_KEYWORDS = {
    "order_status": ["track", "tracking", "order status", "where is my order"],
    "refund_request": ["refund", "chargeback", "return", "money back"],
    "product_search": ["find", "search", "recommend", "looking for"],
    "address_change": ["change address", "update address", "shipping address"],
}


def commerce_intent_classifier() -> KeywordIntentClassifier:
    return KeywordIntentClassifier(COMMERCE_KEYWORDS)


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
            ),
        ),
        IntentMatchRule(
            intent="order_status",
            decision=RuleDecision(
                notes=["Prefer internal order systems for status."],
                system_instructions=[
                    "Use internal order status tools when available.",
                ],
            ),
        ),
        IntentMatchRule(
            intent="product_search",
            decision=RuleDecision(
                notes=["Recommend products with clear disclaimers on availability."],
                system_instructions=[
                    "Avoid guaranteeing inventory or delivery dates.",
                ],
            ),
        ),
    ]
