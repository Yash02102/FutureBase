import os
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol


@dataclass(frozen=True)
class GuardrailVerdict:
    passed: bool
    notes: List[str] = field(default_factory=list)
    sanitized_output: Optional[str] = None


class Guardrail(Protocol):
    def check_input(self, task: str, context: str) -> GuardrailVerdict:
        ...

    def check_output(self, task: str, result: str) -> GuardrailVerdict:
        ...


class MaxLengthGuardrail:
    def __init__(self, max_input_chars: int, max_output_chars: int) -> None:
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars

    def check_input(self, task: str, context: str) -> GuardrailVerdict:
        length = len(task) + len(context)
        if length > self.max_input_chars:
            return GuardrailVerdict(
                passed=False,
                notes=[f"Input exceeds {self.max_input_chars} characters."],
            )
        return GuardrailVerdict(passed=True)

    def check_output(self, task: str, result: str) -> GuardrailVerdict:
        if len(result) > self.max_output_chars:
            return GuardrailVerdict(
                passed=False,
                notes=[f"Output exceeds {self.max_output_chars} characters."],
            )
        return GuardrailVerdict(passed=True)


class RegexBlocklistGuardrail:
    def __init__(self, patterns: Iterable[str]) -> None:
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _check_text(self, text: str) -> GuardrailVerdict:
        matches = [pattern.pattern for pattern in self.patterns if pattern.search(text)]
        if matches:
            return GuardrailVerdict(
                passed=False,
                notes=[f"Blocked by guardrail patterns: {', '.join(matches)}"],
            )
        return GuardrailVerdict(passed=True)

    def check_input(self, task: str, context: str) -> GuardrailVerdict:
        return self._check_text(f"{task}\n{context}")

    def check_output(self, task: str, result: str) -> GuardrailVerdict:
        return self._check_text(result)


class Guardrails:
    def __init__(self, guardrails: Iterable[Guardrail]) -> None:
        self.guardrails = list(guardrails)

    def check_input(self, task: str, context: str) -> GuardrailVerdict:
        return self._run(lambda guardrail: guardrail.check_input(task, context))

    def check_output(self, task: str, result: str) -> GuardrailVerdict:
        return self._run(lambda guardrail: guardrail.check_output(task, result))

    def _run(self, check_fn) -> GuardrailVerdict:
        passed = True
        notes: List[str] = []
        sanitized: Optional[str] = None
        for guardrail in self.guardrails:
            verdict = check_fn(guardrail)
            if not verdict.passed:
                passed = False
                notes.extend(verdict.notes)
            if verdict.sanitized_output:
                sanitized = verdict.sanitized_output
        return GuardrailVerdict(passed=passed, notes=notes, sanitized_output=sanitized)


def guardrails_from_env() -> Guardrails:
    blocklist_env = os.getenv("GUARDRAIL_BLOCKLIST", "")
    patterns = [entry.strip() for entry in blocklist_env.split(",") if entry.strip()]
    max_input_chars = int(os.getenv("GUARDRAIL_MAX_INPUT_CHARS", "4000"))
    max_output_chars = int(os.getenv("GUARDRAIL_MAX_OUTPUT_CHARS", "8000"))
    guardrails: List[Guardrail] = [MaxLengthGuardrail(max_input_chars, max_output_chars)]
    if patterns:
        guardrails.append(RegexBlocklistGuardrail(patterns))
    return Guardrails(guardrails)
