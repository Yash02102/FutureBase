from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional


@dataclass
class AgentMessage:
    sender: str
    content: str


@dataclass
class AgentConfig:
    name: str
    role: str
    system_prompt: str
    metadata: Dict[str, str] = field(default_factory=dict)


AgentResponder = Callable[[AgentConfig, List[AgentMessage], str], AgentMessage]


def default_responder(config: AgentConfig, history: List[AgentMessage], task: str) -> AgentMessage:
    summary = " | ".join([message.content for message in history[-2:]])
    content = (
        f"[{config.name}::{config.role}] stub response for '{task}'. "
        f"Recent context: {summary}"
    )
    return AgentMessage(sender=config.name, content=content)


class MultiAgentOrchestrator:
    def __init__(
        self,
        agents: Iterable[AgentConfig],
        responder: AgentResponder = default_responder,
    ) -> None:
        self.agents = list(agents)
        self.responder = responder

    def run(self, task: str, rounds: int = 2) -> List[AgentMessage]:
        history: List[AgentMessage] = []
        for _ in range(rounds):
            for agent in self.agents:
                message = self.responder(agent, history, task)
                history.append(message)
        return history

    def run_once(self, task: str) -> Optional[AgentMessage]:
        if not self.agents:
            return None
        return self.responder(self.agents[0], [], task)
