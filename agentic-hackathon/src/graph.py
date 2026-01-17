from __future__ import annotations

from typing import Dict, List, TypedDict

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.errors import Command

from .cache import build_cache
from .config import AppConfig
from .llm import ModelSettings, get_chat_model, get_embeddings
from .memory import ConversationMemory, MemorySettings
from .rag import RAGPipeline
from .subagents import build_subagents, run_parallel_specialists
from .tool_router import ToolRouter, build_tool_specs


class AgentState(TypedDict):
    session_id: str
    task: str
    summary: str
    messages: List[BaseMessage]
    rag_context: str
    subagent_notes: List[str]
    tool_names: List[str]
    result: str
    todos: List[str]


def build_agent_graph(config: AppConfig):
    llm_settings = ModelSettings(
        provider=config.llm_provider,
        model=config.llm_model,
        temperature=config.llm_temperature,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    )
    llm = get_chat_model(llm_settings)

    embeddings = None
    if config.rag_enabled:
        embeddings = get_embeddings(
            provider=config.embedding_provider,
            model=config.embedding_model,
            api_key=config.embedding_api_key,
        )
    rag_pipeline = (
        RAGPipeline(
            config.rag_index_path,
            embeddings,
            llm=llm,
            vector_k=config.rag_vector_k,
            bm25_k=config.rag_bm25_k,
            rewrite_count=config.rag_rewrite_count,
            rerank_top_k=config.rag_rerank_top_k,
        )
        if embeddings
        else None
    )

    memory_backend = _build_memory_backend(config)
    memory_settings = MemorySettings(
        max_turns=config.memory_max_turns,
        summarize=config.memory_summarize,
    )

    tool_specs = build_tool_specs(rag_pipeline)
    tool_router = ToolRouter(
        tool_specs,
        max_tools=config.tool_router_max_tools,
        max_cost=config.tool_router_max_cost,
        min_score=config.tool_router_min_score,
    )

    cache = build_cache(config.cache_mode)

    def ingest_step(state: AgentState) -> AgentState:
        memory = ConversationMemory(
            llm=llm,
            settings=memory_settings,
            backend=memory_backend,
            session_id=state["session_id"],
        )
        memory.add_user(state["task"])
        memory.maybe_summarize()
        return {
            **state,
            "summary": memory.state.summary,
            "messages": list(memory.state.messages),
        }

    def rag_step(state: AgentState) -> AgentState:
        if not rag_pipeline:
            return {**state, "rag_context": ""}
        return {**state, "rag_context": rag_pipeline.lookup(state["task"]) }

    def subagent_step(state: AgentState) -> AgentState:
        if not config.subagents_enabled:
            return {**state, "subagent_notes": []}
        context = _build_context(state)
        notes = run_parallel_specialists(
            llm=llm,
            task=state["task"],
            context=context,
            tools=[spec.tool for spec in tool_specs],
            max_workers=config.subagent_parallelism,
        )
        return {**state, "subagent_notes": notes}

    def tool_router_step(state: AgentState) -> AgentState:
        context = _build_context(state)
        tools = tool_router.select(state["task"], context)
        return {**state, "tool_names": [tool.name for tool in tools]}

    def agent_step(state: AgentState) -> AgentState:
        context = _build_context(state)
        selected_tools = _select_tools_by_name(tool_specs, state.get("tool_names", []))
        subagents = build_subagents(llm, selected_tools)
        interrupt_on = _build_interrupts(config)
        backend = _build_filesystem_backend(config)
        agent = create_deep_agent(
            model=llm,
            tools=selected_tools,
            system_prompt=_system_prompt(),
            subagents=subagents,
            backend=backend,
            cache=cache,
            interrupt_on=interrupt_on or None,
        )
        messages = _assemble_messages(state["messages"], context)
        payload = agent.invoke({"messages": messages})
        reply = _extract_reply(payload)
        todos = _extract_todos(payload)
        if _needs_repair(payload, reply):
            reply = _repair_reply(llm, context, state["task"], reply)
        memory = ConversationMemory(
            llm=llm,
            settings=memory_settings,
            backend=memory_backend,
            session_id=state["session_id"],
        )
        memory.add_assistant(reply)
        memory.maybe_summarize()
        return {**state, "result": reply, "todos": todos}

    graph = StateGraph(AgentState)
    graph.add_node("ingest", ingest_step)
    graph.add_node("rag", rag_step)
    graph.add_node("subagents", subagent_step)
    graph.add_node("tools", tool_router_step)
    graph.add_node("agent", agent_step)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "rag")
    graph.add_edge("rag", "subagents")
    graph.add_edge("subagents", "tools")
    graph.add_edge("tools", "agent")
    graph.add_edge("agent", END)

    checkpointer = InMemorySaver() if config.hitl_mode == "manual" else None
    return graph.compile(checkpointer=checkpointer)


def run_agent(task: str, session_id: str, config: AppConfig) -> Dict[str, object]:
    app = build_agent_graph(config)
    initial_state: AgentState = {
        "session_id": session_id,
        "task": task,
        "summary": "",
        "messages": [],
        "rag_context": "",
        "subagent_notes": [],
        "tool_names": [],
        "result": "",
        "todos": [],
    }
    config_map = {"configurable": {"thread_id": session_id}}
    state = app.invoke(initial_state, config=config_map)
    if "__interrupt__" in state:
        state = _handle_interrupts(app, state, config_map)
    return state


def _handle_interrupts(app, state: Dict[str, object], config_map: Dict[str, object]) -> Dict[str, object]:
    interrupts = state.get("__interrupt__", [])
    resume_map: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for interrupt in interrupts:
        payload = getattr(interrupt, "value", {})
        actions = payload.get("action_requests", []) if isinstance(payload, dict) else []
        decisions: List[Dict[str, str]] = []
        for action in actions:
            name = action.get("name", "tool")
            args = action.get("args", {})
            print(f"Approval required for {name} with args {args}")
            choice = input("Approve? (y/n): ").strip().lower()
            if choice == "y":
                decisions.append({"type": "approve"})
            else:
                decisions.append({"type": "reject", "message": "rejected by operator"})
        resume_map[interrupt.id] = {"decisions": decisions}
    if not interrupts:
        return state
    return app.invoke(Command(resume=resume_map), config=config_map)


def _system_prompt() -> str:
    return (
        "You are a commerce agent. Use tools to search products, check inventory, price items, "
        "manage carts, and place orders. Ask for missing details like user_id, address, "
        "or payment_method before checkout. Be concise and action-oriented."
    )


def _assemble_messages(messages: List[BaseMessage], context: str) -> List[BaseMessage]:
    if not context:
        return messages
    return [SystemMessage(content=context)] + messages


def _build_context(state: AgentState) -> str:
    sections: List[str] = []
    if state.get("summary"):
        sections.append(f"Summary:\n{state['summary']}")
    if state.get("rag_context"):
        sections.append(f"RAG context:\n{state['rag_context']}")
    if state.get("subagent_notes"):
        sections.append("Subagent notes:\n" + "\n".join(state["subagent_notes"]))
    return "\n\n".join(sections)


def _select_tools_by_name(tool_specs, names: List[str]):
    name_set = set(names)
    return [spec.tool for spec in tool_specs if spec.tool.name in name_set]


def _extract_reply(payload) -> str:
    if not isinstance(payload, dict):
        return ""
    messages = payload.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content if part)
        if content:
            return str(content)
    return ""


def _extract_todos(payload) -> List[str]:
    if not isinstance(payload, dict):
        return []
    todos = payload.get("todos", [])
    items: List[str] = []
    for todo in todos:
        if isinstance(todo, dict):
            content = todo.get("content", "")
        else:
            content = getattr(todo, "content", "")
        if content:
            items.append(str(content))
    return items


def _needs_repair(payload, reply: str) -> bool:
    if reply.strip():
        return False
    if not isinstance(payload, dict):
        return True
    messages = payload.get("messages", [])
    if not messages:
        return True
    last = messages[-1]
    tool_calls = None
    if isinstance(last, dict):
        tool_calls = last.get("tool_calls")
    else:
        tool_calls = getattr(last, "tool_calls", None)
    return bool(tool_calls)


def _repair_reply(llm, context: str, task: str, reply: str) -> str:
    prompt = [
        SystemMessage(
            content=(
                "You are repairing a tool run. Provide a final customer response. "
                "Do not mention tool errors. Keep it concise."
            )
        ),
        HumanMessage(content=f"Task: {task}\nContext: {context}\nPrevious: {reply}"),
    ]
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def _build_memory_backend(config: AppConfig):
    if config.memory_backend == "filesystem":
        from .memory import FileMemoryBackend

        return FileMemoryBackend(config.memory_path)
    from .memory import EphemeralMemoryBackend

    return EphemeralMemoryBackend()


def _build_filesystem_backend(config: AppConfig):
    if not config.filesystem_root:
        return None
    from deepagents.backends.filesystem import FilesystemBackend

    return FilesystemBackend(root_dir=config.filesystem_root)


def _build_interrupts(config: AppConfig) -> Dict[str, bool] | None:
    if config.hitl_mode != "manual":
        return None
    return {name: True for name in config.hitl_tools}
