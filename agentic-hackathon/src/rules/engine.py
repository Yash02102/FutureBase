from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol


@dataclass(frozen=True)
class Intent:
    name: str
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)
    entities: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RuleContext:
    task: str
    intent: Intent
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RuleDecision:
    allow: bool = True
    require_approval: bool = False
    notes: List[str] = field(default_factory=list)
    system_instructions: List[str] = field(default_factory=list)
    allowed_tools: Optional[List[str]] = None


class Rule(Protocol):
    def apply(self, context: RuleContext) -> Optional[RuleDecision]:
        ...


class IntentClassifier(Protocol):
    def classify(self, task: str) -> Intent:
        ...


class KeywordIntentClassifier:
    def __init__(self, keyword_map: Dict[str, List[str]]) -> None:
        self.keyword_map = keyword_map

    def classify(self, task: str) -> Intent:
        text = task.lower()
        best_intent = "general"
        best_score = 0
        for intent, keywords in self.keyword_map.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_intent = intent
        confidence = min(1.0, best_score / 3) if best_score else 0.0
        return Intent(name=best_intent, confidence=confidence)


class RulesEngine:
    def __init__(self, rules: Iterable[Rule]) -> None:
        self.rules = list(rules)

    def evaluate(self, context: RuleContext) -> RuleDecision:
        decision = RuleDecision()
        for rule in self.rules:
            outcome = rule.apply(context)
            if not outcome:
                continue
            decision = _merge_decisions(decision, outcome)
        return decision


def _merge_decisions(base: RuleDecision, update: RuleDecision) -> RuleDecision:
    base.allow = base.allow and update.allow
    base.require_approval = base.require_approval or update.require_approval
    base.notes.extend(update.notes)
    base.system_instructions.extend(update.system_instructions)
    if update.allowed_tools is not None:
        if base.allowed_tools is None:
            base.allowed_tools = list(update.allowed_tools)
        else:
            base.allowed_tools = [
                tool for tool in base.allowed_tools if tool in update.allowed_tools
            ]
    return base
