import sys
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .tools.rag_tool import rag_lookup
from .approvals import ApprovalRequest, approval_provider_from_env
from .evals import EvalRecord, eval_recorder_from_env
from .guardrails import guardrails_from_env
from .llm import LLMSettings, get_chat_model
from .rules import Intent, RuleContext, RuleDecision, RulesEngine, rules_from_env
from .utils.logging import setup_logging
from .utils.schemas import Plan


class AgentState(TypedDict):
    task: str
    plan: Plan
    context: str
    result: str
    verified: bool
    notes: str
    intent: str
    ruleset: str
    policy_notes: List[str]
    policy: Optional[RuleDecision]
    approval_status: str
    approval_notes: str
    guardrail_input_passed: bool
    guardrail_output_passed: bool
    guardrail_notes: List[str]
    blocked: bool


def build_graph():
    settings = LLMSettings.from_env()
    llm = get_chat_model(settings)
    planner = Planner(llm=llm)
    executor = Executor(llm=llm)
    verifier = Verifier(llm=llm)
    guardrails = guardrails_from_env()
    approval_provider = approval_provider_from_env()
    intent_classifier, rules, ruleset = rules_from_env()
    rules_engine = RulesEngine(rules)

    def plan_step(state: AgentState) -> AgentState:
        plan = planner.generate_plan(state["task"])
        return {**state, "plan": plan}

    def classify_step(state: AgentState) -> AgentState:
        intent = intent_classifier.classify(state["task"])
        return {**state, "intent": intent.name, "ruleset": ruleset}

    def retrieve_step(state: AgentState) -> AgentState:
        context = rag_lookup(state["task"])
        return {**state, "context": context}

    def rules_step(state: AgentState) -> AgentState:
        intent_name = state.get("intent", "general")
        context = RuleContext(task=state["task"], intent=Intent(name=intent_name))
        decision = rules_engine.evaluate(context)
        blocked = state.get("blocked", False) or not decision.allow
        approval_status = "required" if decision.require_approval and not blocked else "not_required"
        return {
            **state,
            "policy_notes": decision.notes,
            "policy": decision,
            "approval_status": approval_status,
            "blocked": blocked,
            "result": "Blocked by rules." if blocked else state.get("result", ""),
        }

    def guardrail_input_step(state: AgentState) -> AgentState:
        verdict = guardrails.check_input(state["task"], state.get("context", ""))
        blocked = state.get("blocked", False) or not verdict.passed
        result = state.get("result", "")
        if not verdict.passed and not result:
            result = "Blocked by guardrails."
        return {
            **state,
            "guardrail_input_passed": verdict.passed,
            "guardrail_notes": verdict.notes,
            "blocked": blocked,
            "result": result,
        }

    def approval_step(state: AgentState) -> AgentState:
        if state.get("blocked"):
            return state
        policy = state.get("policy")
        if not policy or not policy.require_approval:
            return {**state, "approval_status": "not_required"}
        request = ApprovalRequest(
            task=state["task"],
            intent=state.get("intent", "general"),
            notes=state.get("policy_notes", []),
        )
        decision = approval_provider.request(request)
        if not decision.approved:
            return {
                **state,
                "approval_status": "rejected",
                "approval_notes": decision.notes,
                "blocked": True,
                "result": "Approval rejected.",
            }
        return {
            **state,
            "approval_status": "approved",
            "approval_notes": decision.notes,
        }

    def execute_step(state: AgentState) -> AgentState:
        if state.get("blocked"):
            return state
        policy = state.get("policy")
        result = executor.execute(state["task"], state.get("context", ""), policy=policy)
        return {**state, "result": result}

    def guardrail_output_step(state: AgentState) -> AgentState:
        if state.get("blocked"):
            return state
        verdict = guardrails.check_output(state["task"], state.get("result", ""))
        if not verdict.passed:
            notes = list(state.get("guardrail_notes", [])) + verdict.notes
            return {
                **state,
                "guardrail_output_passed": False,
                "guardrail_notes": notes,
                "blocked": True,
                "result": verdict.sanitized_output or "Output blocked by guardrails.",
            }
        return {**state, "guardrail_output_passed": True}

    def verify_step(state: AgentState) -> AgentState:
        if state.get("blocked"):
            return {**state, "verified": False, "notes": "Verification skipped."}
        verdict = verifier.verify(state["task"], state.get("result", ""))
        return {**state, "verified": verdict.is_valid, "notes": verdict.notes}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_step)
    graph.add_node("classify", classify_step)
    graph.add_node("retrieve", retrieve_step)
    graph.add_node("rules", rules_step)
    graph.add_node("guardrail_input", guardrail_input_step)
    graph.add_node("approval", approval_step)
    graph.add_node("execute", execute_step)
    graph.add_node("guardrail_output", guardrail_output_step)
    graph.add_node("verify", verify_step)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "rules")
    graph.add_edge("rules", "guardrail_input")
    graph.add_edge("guardrail_input", "approval")
    graph.add_edge("approval", "execute")
    graph.add_edge("execute", "guardrail_output")
    graph.add_edge("guardrail_output", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


def run(task: str) -> AgentState:
    app = build_graph()
    initial_state: AgentState = {
        "task": task,
        "plan": Plan(goal=task, steps=[]),
        "context": "",
        "result": "",
        "verified": False,
        "notes": "",
        "intent": "general",
        "ruleset": "default",
        "policy_notes": [],
        "policy": None,
        "approval_status": "not_required",
        "approval_notes": "",
        "guardrail_input_passed": True,
        "guardrail_output_passed": True,
        "guardrail_notes": [],
        "blocked": False,
    }
    state = app.invoke(initial_state)

    eval_recorder = eval_recorder_from_env()
    plan_dump = state["plan"].model_dump() if state.get("plan") else None
    record = EvalRecord(
        task=state["task"],
        intent=state.get("intent", "general"),
        plan=plan_dump,
        result=state.get("result", ""),
        verified=state.get("verified", False),
        notes=state.get("notes", ""),
        metadata={
            "ruleset": state.get("ruleset"),
            "approval_status": state.get("approval_status"),
            "approval_notes": state.get("approval_notes"),
            "guardrail_input_passed": state.get("guardrail_input_passed"),
            "guardrail_output_passed": state.get("guardrail_output_passed"),
            "guardrail_notes": state.get("guardrail_notes"),
            "policy_notes": state.get("policy_notes"),
            "blocked": state.get("blocked"),
        },
    )
    eval_recorder.record(record)
    return state


def main(args: List[str]) -> None:
    setup_logging()
    load_dotenv()

    if not args:
        print("Usage: python -m src.agent_core \"your task here\"")
        sys.exit(1)

    task = " ".join(args)
    state = run(task)
    print("Plan:")
    for step in state["plan"].steps:
        print(f"- {step.step}")
    print("\nIntent:", state.get("intent"))
    if state.get("policy_notes"):
        print("Policy notes:", " | ".join(state.get("policy_notes")))
    print("Approval:", state.get("approval_status"), state.get("approval_notes"))
    if not state.get("guardrail_input_passed", True) or not state.get("guardrail_output_passed", True):
        print("Guardrails:", " | ".join(state.get("guardrail_notes", [])))
    print("\nResult:\n", state["result"])
    print("\nVerified:", state["verified"])
    print("Notes:", state["notes"])


if __name__ == "__main__":
    main(sys.argv[1:])
