from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence

from deepagents import create_deep_agent
from deepagents.middleware.subagents import SubAgent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool


SPECIALISTS = [
    {
        "name": "CatalogResearch",
        "description": "Find relevant products and summarize options.",
        "system_prompt": "You are a catalog researcher. Use tools to list products and summarize tradeoffs.",
        "tool_names": {"catalog_search", "inventory_check", "pricing_lookup", "rag_search"},
    },
    {
        "name": "PolicyCheck",
        "description": "Flag missing info or risky actions.",
        "system_prompt": "Identify missing details, risks, or policy concerns. Ask for clarifications.",
        "tool_names": {"rag_search"},
    },
    {
        "name": "OrderSupport",
        "description": "Handle order status or returns when relevant.",
        "system_prompt": "Check order/return workflows and suggest next steps.",
        "tool_names": {"order_status", "track_shipment", "return_request", "refund_status"},
    },
]


def build_subagents(llm, tools: Sequence[BaseTool]) -> List[SubAgent]:
    tool_map = {tool.name: tool for tool in tools}
    subagents: List[SubAgent] = []
    for spec in SPECIALISTS:
        selected = [tool_map[name] for name in spec["tool_names"] if name in tool_map]
        subagents.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "system_prompt": spec["system_prompt"],
                "tools": selected,
                "model": llm,
            }
        )
    return subagents


def run_parallel_specialists(
    llm,
    task: str,
    context: str,
    tools: Sequence[BaseTool],
    max_workers: int,
) -> List[str]:
    tool_map = {tool.name: tool for tool in tools}
    prompts = []
    for spec in SPECIALISTS:
        selected = [tool_map[name] for name in spec["tool_names"] if name in tool_map]
        prompts.append(
            (
                spec["name"],
                spec["system_prompt"],
                selected,
            )
        )

    def _run_agent(name: str, system_prompt: str, agent_tools: Sequence[BaseTool]) -> str:
        agent = create_deep_agent(model=llm, tools=list(agent_tools), system_prompt=system_prompt)
        messages = []
        if context:
            messages.append(SystemMessage(content=f"Context:\n{context}"))
        messages.append(HumanMessage(content=task))
        payload = agent.invoke({"messages": messages})
        return _extract_reply(payload, name)

    results: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_agent, name, prompt, tlist) for name, prompt, tlist in prompts]
        for future in futures:
            results.append(future.result())
    return [note for note in results if note]


def _extract_reply(payload, name: str) -> str:
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    for message in reversed(messages):
        content = getattr(message, "content", None) if not isinstance(message, dict) else message.get("content")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content if part)
        if content:
            return f"{name}: {content}"
    return ""
