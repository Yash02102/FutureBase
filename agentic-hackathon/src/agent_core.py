import os
import sys
from typing import List, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .tools.rag_tool import rag_lookup
from .utils.logging import setup_logging
from .utils.schemas import Plan


class AgentState(TypedDict):
    task: str
    plan: Plan
    context: str
    result: str
    verified: bool
    notes: str


def build_graph(model: str):
    planner = Planner(model=model)
    executor = Executor(model=model)
    verifier = Verifier(model=model)

    def plan_step(state: AgentState) -> AgentState:
        plan = planner.generate_plan(state["task"])
        return {**state, "plan": plan}

    def retrieve_step(state: AgentState) -> AgentState:
        context = rag_lookup(state["task"])
        return {**state, "context": context}

    def execute_step(state: AgentState) -> AgentState:
        result = executor.execute(state["task"], state.get("context", ""))
        return {**state, "result": result}

    def verify_step(state: AgentState) -> AgentState:
        verdict = verifier.verify(state["task"], state.get("result", ""))
        return {**state, "verified": verdict.is_valid, "notes": verdict.notes}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_step)
    graph.add_node("retrieve", retrieve_step)
    graph.add_node("execute", execute_step)
    graph.add_node("verify", verify_step)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "execute")
    graph.add_edge("execute", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


def run(task: str) -> AgentState:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    app = build_graph(model=model)
    initial_state: AgentState = {
        "task": task,
        "plan": Plan(goal=task, steps=[]),
        "context": "",
        "result": "",
        "verified": False,
        "notes": "",
    }
    return app.invoke(initial_state)


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
    print("\nResult:\n", state["result"])
    print("\nVerified:", state["verified"])
    print("Notes:", state["notes"])


if __name__ == "__main__":
    main(sys.argv[1:])