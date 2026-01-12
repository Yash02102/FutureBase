from src.rules.engine import Intent, KeywordIntentClassifier, RuleContext, RulesEngine
from src.rules.commerce import commerce_rules


def test_keyword_intent_classifier():
    classifier = KeywordIntentClassifier({"refund_request": ["refund"]})
    intent = classifier.classify("I need a refund.")
    assert intent.name == "refund_request"


def test_rules_engine_requires_approval_for_refunds():
    engine = RulesEngine(commerce_rules())
    context = RuleContext(task="refund this order", intent=Intent(name="refund_request"))
    decision = engine.evaluate(context)
    assert decision.require_approval is True
