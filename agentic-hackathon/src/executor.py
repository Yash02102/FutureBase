from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from .tools.rag_tool import rag_lookup
from .tools.search_tool import web_search
from .tools.browser_tool import open_browser
from .tools.db_tool import query_internal_api
from .tools.custom_tool import run_custom_tool
from .tools.mcp_tool import call_mcp_tool
from .rules.engine import RuleDecision


@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant context from the local FAISS index."""
    return rag_lookup(query)


@tool
def search_tool(query: str) -> str:
    """Optional web search for quick background info."""
    return web_search(query)


@tool
def browser_tool(url: str) -> str:
    """Browser automation stub; replace with Playwright or similar."""
    return open_browser(url)


@tool
def internal_api_tool(endpoint: str) -> str:
    """Call an internal API endpoint and return raw response."""
    return query_internal_api(endpoint)


@tool
def custom_tool(payload: str) -> str:
    """Custom domain tool placeholder."""
    return run_custom_tool(payload)


@tool
def mcp_tool(server_name: str, tool_name: str, arguments: Dict[str, str]) -> str:
    """Call a remote MCP server tool by name."""
    return call_mcp_tool(server_name, tool_name, arguments)


class Executor:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [
            rag_tool,
            search_tool,
            browser_tool,
            internal_api_tool,
            custom_tool,
            mcp_tool,
        ]

    def execute(self, task: str, context: str, policy: Optional[RuleDecision] = None) -> str:
        system_parts = ["Use tools when helpful. Keep responses concise."]
        if policy and policy.system_instructions:
            system_parts.extend(policy.system_instructions)
        allowed_tools = self.tools
        if policy and policy.allowed_tools is not None:
            allowed_tools = [tool for tool in self.tools if tool.name in policy.allowed_tools]
        messages = [
            SystemMessage(content="\n".join(system_parts)),
            HumanMessage(content=f"Task: {task}\nContext: {context}"),
        ]
        response = self.llm.bind_tools(allowed_tools).invoke(messages)

        if response.tool_calls:
            outputs: List[str] = []
            for call in response.tool_calls:
                tool_fn = {t.name: t for t in allowed_tools}.get(call["name"])
                if not tool_fn:
                    outputs.append(f"{call['name']}: blocked by tool policy")
                    continue
                tool_out = tool_fn.invoke(call["args"])
                outputs.append(f"{call['name']}: {tool_out}")
            return "\n".join(outputs)

        return response.content
