import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from langchain_core.prompts import ChatPromptTemplate

from .utils.schemas import Plan, ToolResult


@dataclass(frozen=True)
class MemorySettings:
    max_history_turns: int
    max_tool_results: int
    summarize: bool

    @classmethod
    def from_env(cls) -> "MemorySettings":
        max_history = int(os.getenv("MEMORY_MAX_TURNS", "8"))
        max_tools = int(os.getenv("MEMORY_MAX_TOOL_RESULTS", "4"))
        summarize = os.getenv("MEMORY_SUMMARIZE", "false").lower() == "true"
        return cls(max_history_turns=max_history, max_tool_results=max_tools, summarize=summarize)


@dataclass
class MemoryState:
    user_profile: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    active_cart: Dict[str, Any] = field(default_factory=dict)
    order_history: List[Dict[str, Any]] = field(default_factory=list)
    history_summary: str = ""
    episodic_notes: List[str] = field(default_factory=list)
    recent_turns: List[Dict[str, str]] = field(default_factory=list)
    last_tool_outputs: List[ToolResult] = field(default_factory=list)
    current_plan: Optional[Plan] = None


class MemoryBackend(Protocol):
    def load(self, session_id: str) -> MemoryState:
        ...

    def save(self, session_id: str, state: MemoryState) -> None:
        ...


class NoopMemoryBackend:
    def load(self, session_id: str) -> MemoryState:
        return MemoryState()

    def save(self, session_id: str, state: MemoryState) -> None:
        return None


class SqliteMemoryBackend:
    def __init__(self, path: str) -> None:
        self.path = path
        self._ensure_schema()

    def load(self, session_id: str) -> MemoryState:
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute(
                "SELECT short_term, working, long_term, episodic, history_summary "
                "FROM memory_state WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
        if not row:
            return MemoryState()
        short_term = _safe_json(row[0]) or {}
        working = _safe_json(row[1]) or {}
        long_term = _safe_json(row[2]) or {}
        episodic = _safe_json(row[3]) or {}
        history_summary = row[4] or ""
        return MemoryState(
            user_profile=long_term.get("user_profile", {}),
            preferences=long_term.get("preferences", {}),
            active_cart=long_term.get("active_cart", {}),
            order_history=long_term.get("order_history", []),
            history_summary=history_summary,
            episodic_notes=episodic.get("episodic_notes", []),
            recent_turns=short_term.get("recent_turns", []),
            last_tool_outputs=_hydrate_tool_results(working.get("last_tool_outputs", [])),
            current_plan=_hydrate_plan(working.get("current_plan")),
        )

    def save(self, session_id: str, state: MemoryState) -> None:
        short_term = {"recent_turns": state.recent_turns}
        working = {
            "last_tool_outputs": [tool.model_dump() for tool in state.last_tool_outputs],
            "current_plan": state.current_plan.model_dump()
            if state.current_plan
            else None,
        }
        long_term = {
            "user_profile": state.user_profile,
            "preferences": state.preferences,
            "active_cart": state.active_cart,
            "order_history": state.order_history,
        }
        episodic = {"episodic_notes": state.episodic_notes}
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO memory_state "
                "(session_id, short_term, working, long_term, episodic, history_summary, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "short_term=excluded.short_term, "
                "working=excluded.working, "
                "long_term=excluded.long_term, "
                "episodic=excluded.episodic, "
                "history_summary=excluded.history_summary, "
                "updated_at=excluded.updated_at",
                (
                    session_id,
                    json.dumps(short_term, ensure_ascii=True),
                    json.dumps(working, ensure_ascii=True),
                    json.dumps(long_term, ensure_ascii=True),
                    json.dumps(episodic, ensure_ascii=True),
                    state.history_summary,
                    time.time(),
                ),
            )

    def _ensure_schema(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_state ("
                "session_id TEXT PRIMARY KEY,"
                "short_term TEXT,"
                "working TEXT,"
                "long_term TEXT,"
                "episodic TEXT,"
                "history_summary TEXT,"
                "updated_at REAL"
                ")"
            )


def memory_backend_from_env() -> MemoryBackend:
    backend = os.getenv("MEMORY_BACKEND", "sqlite").lower()
    if backend == "sqlite":
        path = os.getenv("MEMORY_DB_PATH", "./data/commerce_memory.db")
        return SqliteMemoryBackend(path)
    return NoopMemoryBackend()


class MemoryStore:
    def __init__(
        self,
        llm,
        session_id: str,
        settings: Optional[MemorySettings] = None,
        backend: Optional[MemoryBackend] = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or MemorySettings.from_env()
        self.session_id = session_id
        self.backend = backend or memory_backend_from_env()
        self.state = self.backend.load(session_id)

    def add_turn(self, role: str, content: str) -> None:
        self.state.recent_turns.append({"role": role, "content": content})
        if len(self.state.recent_turns) > self.settings.max_history_turns:
            if self.settings.summarize:
                self.state.history_summary = self._summarize_turns(self.state.recent_turns)
            self.state.recent_turns = self.state.recent_turns[-self.settings.max_history_turns :]
        self._persist()

    def set_plan(self, plan: Plan) -> None:
        self.state.current_plan = plan
        self._persist()

    def add_tool_result(self, result: ToolResult) -> None:
        self.state.last_tool_outputs.append(result)
        if len(self.state.last_tool_outputs) > self.settings.max_tool_results:
            self.state.last_tool_outputs = self.state.last_tool_outputs[
                -self.settings.max_tool_results :
            ]
        self._update_state_from_tool(result)
        self._persist()

    def add_episodic_note(self, note: str) -> None:
        self.state.episodic_notes.append(note)
        self._persist()

    def get_cached_value(self, key: str) -> Optional[str]:
        if key in self.state.user_profile:
            return str(self.state.user_profile[key])
        if key in self.state.preferences:
            return str(self.state.preferences[key])
        if key in {"order_id", "tracking_id"}:
            for entry in reversed(self.state.order_history):
                if key in entry:
                    return str(entry[key])
        return None

    def set_user_detail(self, key: str, value: str) -> None:
        self.state.user_profile[key] = value
        self._persist()

    def set_preference(self, key: str, value: str) -> None:
        self.state.preferences[key] = value
        self._persist()

    def compile_context(
        self,
        task: str,
        intent: str,
        current_step: Optional[str] = None,
        entities: Optional[Dict[str, str]] = None,
    ) -> str:
        sections: List[str] = []
        if self.state.user_profile:
            sections.append(f"User profile: {self._as_inline(self.state.user_profile)}")
        if self.state.preferences:
            sections.append(f"Preferences: {self._as_inline(self.state.preferences)}")
        if self.state.active_cart:
            sections.append(f"Active cart: {self._as_inline(self.state.active_cart)}")
        if self.state.order_history:
            sections.append(
                f"Order history: {self._as_inline(self.state.order_history[-3:])}"
            )
        if self.state.history_summary:
            sections.append(f"History summary: {self.state.history_summary}")
        if self.state.recent_turns:
            turns = "; ".join(
                f"{turn['role']}: {self._truncate(turn['content'], 120)}"
                for turn in self.state.recent_turns[-4:]
            )
            sections.append(f"Recent turns: {turns}")
        if self.state.current_plan:
            plan_steps = ", ".join(step.step for step in self.state.current_plan.steps)
            sections.append(f"Plan: {plan_steps}")
        if current_step:
            sections.append(f"Current step: {current_step}")
        if self.state.last_tool_outputs:
            outputs = "; ".join(
                f"{item.tool}: {self._truncate(item.output, 140)}"
                for item in self.state.last_tool_outputs
            )
            sections.append(f"Tool outputs: {outputs}")
        sections.append(f"Intent: {intent}")
        if entities:
            sections.append(f"Entities: {self._as_inline(entities)}")
        sections.append(f"Task: {task}")
        return "\n".join(sections)

    def _update_state_from_tool(self, result: ToolResult) -> None:
        payload = self._safe_json(result.output)
        if not isinstance(payload, dict):
            return
        if result.tool in {"cart_add_tool", "cart_remove_tool", "cart_view_tool"}:
            self.state.active_cart = payload
        if result.tool == "checkout_tool" and "order_id" in payload:
            self.state.order_history.append(payload)
        if result.tool == "order_status_tool" and "order_id" in payload:
            self.state.order_history.append(payload)
        if result.tool == "support_tool" and "ticket_id" in payload:
            self.state.episodic_notes.append(f"Opened ticket {payload['ticket_id']}.")

    def _summarize_turns(self, turns: List[Dict[str, str]]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Summarize the recent conversation in 2-3 sentences."),
                ("human", "{turns}"),
            ]
        )
        serialized = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
        response = (prompt | self.llm).invoke({"turns": serialized})
        return getattr(response, "content", str(response))

    def _persist(self) -> None:
        if self.backend:
            self.backend.save(self.session_id, self.state)

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _as_inline(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True)

    @staticmethod
    def _safe_json(value: str) -> Any:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None


def _safe_json(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _hydrate_tool_results(raw: List[Dict[str, Any]]) -> List[ToolResult]:
    results: List[ToolResult] = []
    for entry in raw:
        try:
            results.append(ToolResult.model_validate(entry))
        except Exception:
            continue
    return results


def _hydrate_plan(raw: Optional[Dict[str, Any]]) -> Optional[Plan]:
    if not raw:
        return None
    try:
        return Plan.model_validate(raw)
    except Exception:
        return None
