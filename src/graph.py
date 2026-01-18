from __future__ import annotations

from typing import Dict, List, TypedDict

from deepagents import create_deep_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.errors import Command

from .cache import build_cache
from .config import AppConfig
from .llm import ModelSettings, get_chat_model, get_embeddings
from .memory import ConversationMemory, MemorySettings
from .observability import LangfuseContext, langfuse_context
from .rag import RAGPipeline
from .subagents import build_subagents
from .tooling import build_tool_sets


class AgentState(TypedDict):
    session_id: str
    task: str
    summary: str
    messages: List[BaseMessage]
    rag_context: str
    result: str
    todos: List[str]
    trace_id: str


def build_agent_graph(config: AppConfig, observability: LangfuseContext | None):
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

    def _llm_config() -> Dict[str, object] | None:
        if not observability:
            return None
        return observability.build_config()

    tool_sets = build_tool_sets(rag_pipeline, config_factory=_llm_config)
    generic_tools = tool_sets.generic_tools

    cache = build_cache(config.cache_mode)

    def ingest_step(state: AgentState) -> AgentState:
        memory = ConversationMemory(
            llm=llm,
            settings=memory_settings,
            backend=memory_backend,
            session_id=state["session_id"],
            llm_config=_llm_config(),
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
        return {**state, "rag_context": rag_pipeline.lookup(state["task"], config=_llm_config())}

    def agent_step(state: AgentState) -> AgentState:
        context = _build_context(state)
        interrupt_on = _build_interrupts(config)
        tools = generic_tools if config.subagents_enabled else tool_sets.all_tools
        subagents = (
            build_subagents(llm, tool_sets.all_tools, interrupt_on=interrupt_on)
            if config.subagents_enabled
            else None
        )
        backend = _build_filesystem_backend(config)
        agent = create_deep_agent(
            model=llm,
            tools=tools,
            system_prompt=_system_prompt(),
            subagents=subagents,
            backend=backend,
            cache=cache,
            interrupt_on=interrupt_on or None,
        )
        messages = _assemble_messages(state["messages"], context)
        llm_config = _llm_config()
        if llm_config:
            payload = agent.invoke({"messages": messages}, config=llm_config)
        else:
            payload = agent.invoke({"messages": messages})
        reply = _extract_reply(payload)
        todos = _extract_todos(payload)
        if _needs_repair(payload, reply):
            reply = _repair_reply(llm, context, state["task"], reply, llm_config)
        memory = ConversationMemory(
            llm=llm,
            settings=memory_settings,
            backend=memory_backend,
            session_id=state["session_id"],
            llm_config=_llm_config(),
        )
        memory.add_assistant(reply)
        memory.maybe_summarize()
        return {**state, "result": reply, "todos": todos}

    graph = StateGraph(AgentState)
    graph.add_node("ingest", ingest_step)
    graph.add_node("rag", rag_step)
    graph.add_node("agent", agent_step)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "rag")
    graph.add_edge("rag", "agent")
    graph.add_edge("agent", END)

    checkpointer = InMemorySaver() if config.hitl_mode == "manual" else None
    return graph.compile(checkpointer=checkpointer)


def run_agent(task: str, session_id: str, config: AppConfig) -> Dict[str, object]:
    with langfuse_context(config, session_id, task) as observability:
        app = build_agent_graph(config, observability)
        initial_state: AgentState = {
            "session_id": session_id,
            "task": task,
            "summary": "",
            "messages": [],
            "rag_context": "",
            "result": "",
            "todos": [],
            "trace_id": "",
        }
        config_map = {"configurable": {"thread_id": session_id}}
        if observability:
            config_map.update(observability.build_config())
        state = app.invoke(initial_state, config=config_map)
        if "__interrupt__" in state:
            state = _handle_interrupts(app, state, config_map)
        if observability:
            observability.update_output(
                {
                    "result": state.get("result", ""),
                    "todos": state.get("todos", []),
                }
            )
            trace_id = observability.trace_id()
            if trace_id:
                state = {**state, "trace_id": trace_id}
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
    return "\n\n".join(sections)


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


def _repair_reply(
    llm,
    context: str,
    task: str,
    reply: str,
    llm_config: Dict[str, object] | None,
) -> str:
    prompt = [
        SystemMessage(
            content=(
                "You are repairing a tool run. Provide a final customer response. "
                "Do not mention tool errors. Keep it concise."
            )
        ),
        HumanMessage(content=f"Task: {task}\nContext: {context}\nPrevious: {reply}"),
    ]
    if llm_config:
        response = llm.invoke(prompt, config=llm_config)
    else:
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
