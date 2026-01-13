import os
import sys
import uuid
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv

from .approvals import ApprovalRequest, approval_provider_from_env
from .commerce_workflow import CommerceWorkflowEngine, CommerceWorkflowRunner
from .executor import Executor
from .guardrails import guardrails_from_env
from .llm import LLMSettings, get_chat_model
from .memory import MemoryStore, memory_backend_from_env
from .planner import Planner
from .rules import Intent, RuleContext, RulesEngine, rules_from_env
from .utils.logging import setup_logging
from .utils.trace_recorder import TraceEvent, trace_recorder_from_env
from .verifier import Verifier


class CommerceAgentState(TypedDict):
    task: str
    intent: str
    plan_steps: List[str]
    workflow_steps: List[str]
    result: str
    verified: bool
    notes: str
    policy_notes: List[str]
    approval_status: str
    approval_notes: str
    blocked: bool
    session_id: str


def _ensure_commerce_ruleset() -> None:
    if not os.getenv("RULESET"):
        os.environ["RULESET"] = "commerce"


def run(
    task: str,
    session_id: Optional[str] = None,
    initial_entities: Optional[Dict[str, str]] = None,
) -> CommerceAgentState:
    _ensure_commerce_ruleset()
    settings = LLMSettings.from_env()
    llm = get_chat_model(settings)
    planner = Planner(llm=llm)
    executor = Executor(llm=llm, max_retries=2)
    verifier = Verifier(llm=llm)
    guardrails = guardrails_from_env()
    approval_provider = approval_provider_from_env()
    intent_classifier, rules, _ruleset = rules_from_env()
    rules_engine = RulesEngine(rules)
    workflow_engine = CommerceWorkflowEngine()
    trace_recorder = trace_recorder_from_env()
    session_id = session_id or str(uuid.uuid4())
    memory = MemoryStore(
        llm=llm, session_id=session_id, backend=memory_backend_from_env()
    )

    memory.add_turn("user", task)

    intent: Intent = intent_classifier.classify(task)
    intent_entities = {**intent.entities, **(initial_entities or {})}
    policy = rules_engine.evaluate(
        RuleContext(task=task, intent=intent, metadata=intent_entities)
    )
    blocked = not policy.allow
    approval_status = "not_required"
    approval_notes = ""

    if policy.require_approval and not blocked:
        request = ApprovalRequest(
            task=task,
            intent=intent.name,
            notes=policy.notes,
        )
        decision = approval_provider.request(request)
        if not decision.approved:
            blocked = True
            approval_status = "rejected"
            approval_notes = decision.notes
        else:
            approval_status = "approved"
            approval_notes = decision.notes

    guardrail_in = guardrails.check_input(task, "")
    if not guardrail_in.passed:
        blocked = True

    if blocked:
        result = "Request blocked by policy or guardrails."
        return CommerceAgentState(
            task=task,
            intent=intent.name,
            plan_steps=[],
            workflow_steps=[],
            result=result,
            verified=False,
            notes="Blocked.",
            policy_notes=policy.notes,
            approval_status=approval_status,
            approval_notes=approval_notes,
            blocked=True,
            session_id=session_id,
        )

    plan = planner.generate_plan(
        task, trace_recorder=trace_recorder, session_id=session_id
    )
    memory.set_plan(plan)
    workflow = workflow_engine.build(intent.name, plan)
    workflow_steps = [step.name for step in workflow]

    trace_recorder.record(
        TraceEvent(
            session_id=session_id,
            step="plan",
            event="plan_generated",
            status="success",
            data={"plan": [step.step for step in plan.steps], "intent": intent.name},
        )
    )

    runner = CommerceWorkflowRunner(
        executor=executor,
        verifier=verifier,
        memory=memory,
        trace_recorder=trace_recorder,
        approval_provider=approval_provider,
    )
    workflow_result = runner.run(
        task=task,
        intent=intent.name,
        steps=workflow,
        policy=policy,
        session_id=session_id,
        entities=intent_entities,
    )

    final_entities = workflow_result.entities or intent_entities
    final_context = memory.compile_context(
        task, intent.name, current_step="RESPONSE", entities=final_entities
    )
    final_task = (
        "Provide a concise customer-ready response. Summarize results, ask for any missing "
        "details, and confirm next actions."
    )
    final_execution = executor.execute_detailed(
        final_task,
        final_context,
        policy=policy,
        allowed_tools=[],
        trace_recorder=trace_recorder,
        session_id=session_id,
        step_name="final_response",
    )
    result = final_execution.content
    memory.add_turn("assistant", result)

    guardrail_out = guardrails.check_output(task, result)
    if not guardrail_out.passed:
        result = guardrail_out.sanitized_output or "Response blocked by guardrails."

    verification = verifier.verify(task, result)

    return CommerceAgentState(
        task=task,
        intent=intent.name,
        plan_steps=[step.step for step in plan.steps],
        workflow_steps=workflow_steps,
        result=result,
        verified=verification.is_valid,
        notes=verification.notes,
        policy_notes=policy.notes,
        approval_status=approval_status,
        approval_notes=approval_notes,
        blocked=workflow_result.status != "success",
        session_id=session_id,
    )


def main(args: List[str]) -> None:
    setup_logging()
    load_dotenv()

    if not args:
        print("Usage: python -m src.commerce_agent \"your task here\"")
        sys.exit(1)

    task = " ".join(args)
    state = run(task)
    print("Intent:", state.get("intent"))
    print("Plan:", ", ".join(state.get("plan_steps", [])))
    print("Workflow:", ", ".join(state.get("workflow_steps", [])))
    if state.get("policy_notes"):
        print("Policy notes:", " | ".join(state.get("policy_notes")))
    print("Approval:", state.get("approval_status"), state.get("approval_notes"))
    print("\nResult:\n", state.get("result", ""))
    print("\nVerified:", state.get("verified"))
    print("Notes:", state.get("notes"))


if __name__ == "__main__":
    main(sys.argv[1:])
