import os
from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class ApprovalRequest:
    task: str
    intent: str
    notes: List[str]
    result_preview: Optional[str] = None


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    notes: str = ""


class ApprovalProvider(Protocol):
    def request(self, request: ApprovalRequest) -> ApprovalDecision:
        ...


class AutoApproveProvider:
    def request(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(approved=True, notes="Auto-approved.")


class ConsoleApprovalProvider:
    def request(self, request: ApprovalRequest) -> ApprovalDecision:
        print("\nApproval required")
        print("Intent:", request.intent)
        print("Task:", request.task)
        if request.notes:
            print("Notes:", " | ".join(request.notes))
        if request.result_preview:
            print("Preview:", request.result_preview[:500])
        decision = input("Approve? (y/N): ").strip().lower()
        approved = decision == "y"
        return ApprovalDecision(
            approved=approved,
            notes="Approved by operator." if approved else "Rejected by operator.",
        )


def approval_provider_from_env() -> ApprovalProvider:
    mode = os.getenv("APPROVAL_MODE", "auto").lower()
    if mode == "manual":
        return ConsoleApprovalProvider()
    return AutoApproveProvider()
