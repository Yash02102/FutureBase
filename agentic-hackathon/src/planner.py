from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from .utils.schemas import Plan
from .utils.trace_recorder import TraceEvent, TraceRecorder


class Planner:
    def __init__(self, llm, max_retries: int = 1):
        self.llm = llm
        self.max_retries = max_retries

    def generate_plan(
        self,
        goal: str,
        trace_recorder: Optional[TraceRecorder] = None,
        session_id: Optional[str] = None,
    ) -> Plan:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise planner. Return a concise plan with 3-7 steps.",
                ),
                ("human", "Goal: {goal}"),
            ]
        )
        chain = prompt | self.llm.with_structured_output(Plan)
        for attempt in range(self.max_retries + 1):
            messages = prompt.format_messages(goal=goal)
            self._record_trace(
                trace_recorder,
                session_id,
                "plan_request",
                {
                    "goal": goal,
                    "attempt": attempt + 1,
                    "messages": self._serialize_messages(messages),
                },
            )
            plan = chain.invoke({"goal": goal})
            error = self._validate_plan(plan)
            self._record_trace(
                trace_recorder,
                session_id,
                "plan_response",
                {
                    "plan": [step.step for step in plan.steps],
                    "error": error or "",
                    "attempt": attempt + 1,
                },
            )
            if not error:
                return plan
            if attempt >= self.max_retries:
                return plan
            prompt = self._repair_prompt(goal, plan, error)
            chain = prompt | self.llm.with_structured_output(Plan)
        return plan

    @staticmethod
    def _validate_plan(plan: Plan) -> str:
        if not plan.steps:
            return "Plan has no steps."
        if len(plan.steps) < 3 or len(plan.steps) > 7:
            return "Plan must contain 3-7 steps."
        for step in plan.steps:
            if not step.step.strip():
                return "Plan contains an empty step."
        return ""

    @staticmethod
    def _repair_prompt(goal: str, plan: Plan, error: str) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a precise planner. Fix the plan to satisfy constraints.",
                ),
                (
                    "human",
                    "Goal: {goal}\n\n"
                    "Previous plan:\n{plan}\n\n"
                    "Issue: {error}\n\n"
                    "Return a corrected plan with 3-7 concrete steps.",
                ),
            ]
        ).partial(goal=goal, plan="\n".join(step.step for step in plan.steps), error=error)

    @staticmethod
    def _record_trace(
        trace_recorder: Optional[TraceRecorder],
        session_id: Optional[str],
        event: str,
        data: dict,
    ) -> None:
        if not trace_recorder or not session_id:
            return
        trace_recorder.record(
            TraceEvent(
                session_id=session_id,
                step="planner",
                event=event,
                status="success",
                data=data,
            )
        )

    @staticmethod
    def _serialize_messages(messages: list) -> list[dict]:
        payload: list[dict] = []
        for message in messages:
            role = getattr(message, "type", message.__class__.__name__).lower()
            payload.append({"role": role, "content": message.content})
        return payload
