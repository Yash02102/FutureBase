import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol


@dataclass(frozen=True)
class InputRequest:
    task: str
    step: str
    fields: List[str]
    notes: str = ""


class HumanInputProvider(Protocol):
    def request_inputs(self, request: InputRequest) -> Dict[str, str]:
        ...


class AutoInputProvider:
    def request_inputs(self, request: InputRequest) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for field in request.fields:
            default = _default_for(field)
            if default:
                values[field] = default
        return values


class ConsoleInputProvider:
    def request_inputs(self, request: InputRequest) -> Dict[str, str]:
        print("\nInput required")
        print("Step:", request.step)
        print("Task:", request.task)
        if request.notes:
            print("Notes:", request.notes)
        values: Dict[str, str] = {}
        for field in request.fields:
            value = input(f"Provide {field}: ").strip()
            if value:
                values[field] = value
        return values


def human_input_provider_from_env() -> HumanInputProvider:
    mode = os.getenv("HUMAN_INPUT_MODE", "auto").lower()
    if mode == "manual":
        return ConsoleInputProvider()
    return AutoInputProvider()


def _default_for(field: str) -> Optional[str]:
    env_key = f"HUMAN_INPUT_DEFAULT_{field.upper()}"
    return os.getenv(env_key)
