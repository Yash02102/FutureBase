from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from .tools.rag_tool import rag_lookup
from .tools.search_tool import web_search
from .tools.browser_tool import open_browser
from .tools.db_tool import query_internal_api
from .tools.custom_tool import run_custom_tool
from .tools.mcp_tool import call_mcp_tool


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
    def __init__(self, model: str):
        self.llm = ChatOpenAI(model=model)
        self.tools = [
            rag_tool,
            search_tool,
            browser_tool,
            internal_api_tool,
            custom_tool,
            mcp_tool,
        ]

    def execute(self, task: str, context: str) -> str:
        messages = [
            SystemMessage(content="Use tools when helpful. Keep responses concise."),
            HumanMessage(content=f"Task: {task}\nContext: {context}"),
        ]
        response = self.llm.bind_tools(self.tools).invoke(messages)

        if response.tool_calls:
            outputs: List[str] = []
            for call in response.tool_calls:
                tool_fn = {t.name: t for t in self.tools}[call["name"]]
                tool_out = tool_fn.invoke(call["args"])
                outputs.append(f"{call['name']}: {tool_out}")
            return "\n".join(outputs)

        return response.content
