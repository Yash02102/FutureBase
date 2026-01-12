import os
from typing import Tuple

from .commerce import commerce_intent_classifier, commerce_rules
from .engine import IntentClassifier, KeywordIntentClassifier, Rule


def default_intent_classifier() -> IntentClassifier:
    return KeywordIntentClassifier({})


def default_rules() -> list[Rule]:
    return []


def rules_from_env() -> Tuple[IntentClassifier, list[Rule], str]:
    ruleset = os.getenv("RULESET", "default").lower()
    if ruleset == "commerce":
        return commerce_intent_classifier(), commerce_rules(), ruleset
    return default_intent_classifier(), default_rules(), ruleset
