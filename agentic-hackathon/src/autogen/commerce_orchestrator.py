import json
from typing import Dict, List, Optional, Protocol

from .multi_agent import AgentConfig, AgentMessage, MultiAgentOrchestrator
from ..executor import Executor
from ..utils.trace_recorder import TraceEvent, TraceRecorder


class StepLike(Protocol):
    name: str
    description: str


def run_autogen_insights(
    executor: Executor,
    task: str,
    step: StepLike,
    entities: Dict[str, str],
    rounds: int = 1,
    trace_recorder: Optional[TraceRecorder] = None,
    session_id: Optional[str] = None,
) -> List[str]:
    agents = [
        AgentConfig(
            name="Planner",
            role="planner",
            system_prompt="Break the step into clear sub-tasks.",
        ),
        AgentConfig(
            name="Ops",
            role="executor",
            system_prompt="Identify required data and tool usage for the step.",
        ),
        AgentConfig(
            name="Verifier",
            role="verifier",
            system_prompt="State what success looks like for this step.",
        ),
    ]

    def responder(config: AgentConfig, history: List[AgentMessage], task_input: str) -> AgentMessage:
        recent = " | ".join(message.content for message in history[-2:])
        prompt = (
            f"{config.system_prompt}\n\n"
            f"Step: {step.name} - {step.description}\n"
            f"Task: {task_input}\n"
            f"Entities: {json.dumps(entities, ensure_ascii=True)}\n"
            f"Recent: {recent}\n"
            "Provide 2-3 concise bullets."
        )
        result = executor.execute_detailed(
            prompt,
            "",
            allowed_tools=[],
            trace_recorder=trace_recorder,
            session_id=session_id,
            step_name=f"{step.name}:{config.name}",
        )
        _record_trace(
            trace_recorder,
            session_id,
            step.name,
            config.name,
            result.content,
        )
        return AgentMessage(sender=config.name, content=result.content)

    orchestrator = MultiAgentOrchestrator(agents, responder)
    messages = orchestrator.run(task, rounds=rounds)
    return [f"{message.sender}: {message.content}" for message in messages]


def _record_trace(
    trace_recorder: Optional[TraceRecorder],
    session_id: Optional[str],
    step_name: str,
    agent_name: str,
    output: str,
) -> None:
    if not trace_recorder or not session_id:
        return
    trace_recorder.record(
        TraceEvent(
            session_id=session_id,
            step=step_name,
            event="autogen",
            status="success",
            data={"agent": agent_name, "output": output},
        )
    )
