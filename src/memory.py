from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.prompts import ChatPromptTemplate


@dataclass(frozen=True)
class MemorySettings:
    max_turns: int
    summarize: bool


@dataclass
class MemoryState:
    summary: str = ""
    messages: List[BaseMessage] = field(default_factory=list)


class MemoryBackend(Protocol):
    def load(self, session_id: str) -> MemoryState:
        ...

    def save(self, session_id: str, state: MemoryState) -> None:
        ...


class EphemeralMemoryBackend:
    def load(self, session_id: str) -> MemoryState:
        return MemoryState()

    def save(self, session_id: str, state: MemoryState) -> None:
        return None


class FileMemoryBackend:
    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)

    def load(self, session_id: str) -> MemoryState:
        path = self._path(session_id)
        if not os.path.exists(path):
            return MemoryState()
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        summary = payload.get("summary", "")
        messages = messages_from_dict(payload.get("messages", []))
        return MemoryState(summary=summary, messages=messages)

    def save(self, session_id: str, state: MemoryState) -> None:
        path = self._path(session_id)
        payload = {
            "summary": state.summary,
            "messages": messages_to_dict(state.messages),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    def _path(self, session_id: str) -> str:
        safe = session_id.replace("/", "_")
        return os.path.join(self.root, f"{safe}.json")


class ConversationMemory:
    def __init__(
        self,
        llm,
        settings: MemorySettings,
        backend: MemoryBackend,
        session_id: str,
        llm_config: dict | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings
        self.backend = backend
        self.session_id = session_id
        self.llm_config = llm_config
        self.state = backend.load(session_id)

    def add_user(self, content: str) -> None:
        self.state.messages.append(HumanMessage(content=content))
        self._persist()

    def add_assistant(self, content: str) -> None:
        if content:
            self.state.messages.append(AIMessage(content=content))
            self._persist()

    def add_system(self, content: str) -> None:
        if content:
            self.state.messages.append(SystemMessage(content=content))
            self._persist()

    def maybe_summarize(self) -> None:
        if not self.settings.summarize:
            return
        if len(self.state.messages) <= self.settings.max_turns:
            return
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Summarize the conversation so far into concise bullets. Preserve user preferences and decisions.",
                ),
                (
                    "human",
                    "Existing summary:\n{summary}\n\nNew messages:\n{messages}",
                ),
            ]
        )
        serialized = "\n".join(
            f"{msg.type}: {getattr(msg, 'content', '')}" for msg in self.state.messages
        )
        if self.llm_config:
            response = (summary_prompt | self.llm).invoke(
                {"summary": self.state.summary, "messages": serialized},
                config=self.llm_config,
            )
        else:
            response = (summary_prompt | self.llm).invoke(
                {"summary": self.state.summary, "messages": serialized}
            )
        self.state.summary = getattr(response, "content", str(response))
        self.state.messages = self.state.messages[-self.settings.max_turns :]
        self._persist()

    def build_context(self, rag_context: str) -> str:
        sections: List[str] = []
        if self.state.summary:
            sections.append(f"Summary:\n{self.state.summary}")
        if self.state.messages:
            transcript = "\n".join(
                f"{msg.type}: {getattr(msg, 'content', '')}" for msg in self.state.messages[-6:]
            )
            sections.append(f"Recent turns:\n{transcript}")
        if rag_context:
            sections.append(f"RAG context:\n{rag_context}")
        return "\n\n".join(sections).strip()

    def _persist(self) -> None:
        self.backend.save(self.session_id, self.state)
