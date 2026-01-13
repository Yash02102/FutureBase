import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from .tools.rag_tool import rag_lookup
from .tools.search_tool import web_search
from .tools.browser_tool import open_browser
from .tools.db_tool import query_internal_api
from .tools.custom_tool import run_custom_tool
from .tools.mcp_tool import call_mcp_tool
from .tools.commerce_tools import (
    cart_add,
    cart_remove,
    cart_view,
    catalog_search,
    checkout,
    fraud_check,
    inventory_check,
    order_status,
    pricing_lookup,
    promo_check,
    refund_status,
    reorder,
    return_request,
    support_ticket,
    track_shipment,
)
from .rules.engine import RuleDecision
from .utils.schemas import ToolResult
from .utils.trace_recorder import TraceEvent, TraceRecorder
from .tools.schema_registry import TOOL_SCHEMAS, ToolSchema


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


@tool
def catalog_search_tool(
    query: str,
    limit: int = 5,
    max_price: Optional[float] = None,
    category: Optional[str] = None,
) -> str:
    """Search the commerce catalog."""
    return catalog_search(query=query, limit=limit, max_price=max_price, category=category)


@tool
def inventory_check_tool(sku: str) -> str:
    """Check inventory for a specific SKU."""
    return inventory_check(sku)


@tool
def pricing_tool(sku: str, user_id: Optional[str] = None) -> str:
    """Fetch pricing for a specific SKU."""
    return pricing_lookup(sku, user_id=user_id)


@tool
def promo_tool(user_id: str, cart_total: float, code: Optional[str] = None) -> str:
    """Check promotion eligibility or discount value."""
    return promo_check(user_id=user_id, cart_total=cart_total, code=code)


@tool
def cart_add_tool(user_id: str, sku: str, quantity: int = 1) -> str:
    """Add an item to the user's cart."""
    return cart_add(user_id=user_id, sku=sku, quantity=quantity)


@tool
def cart_remove_tool(user_id: str, sku: str) -> str:
    """Remove an item from the user's cart."""
    return cart_remove(user_id=user_id, sku=sku)


@tool
def cart_view_tool(user_id: str) -> str:
    """View the user's cart."""
    return cart_view(user_id=user_id)


@tool
def fraud_check_tool(user_id: str, order_total: float, order_id: Optional[str] = None) -> str:
    """Perform a basic fraud check."""
    return fraud_check(user_id=user_id, order_total=order_total, order_id=order_id)


@tool
def checkout_tool(user_id: str, payment_method: str, address: str) -> str:
    """Place an order for the current cart."""
    return checkout(user_id=user_id, payment_method=payment_method, address=address)


@tool
def order_status_tool(order_id: str) -> str:
    """Fetch order status."""
    return order_status(order_id)


@tool
def logistics_tool(tracking_id: str) -> str:
    """Track shipment for an order."""
    return track_shipment(tracking_id)


@tool
def return_tool(order_id: str, reason: str) -> str:
    """Create a return request for an order."""
    return return_request(order_id=order_id, reason=reason)


@tool
def refund_tool(order_id: str) -> str:
    """Check refund status."""
    return refund_status(order_id)


@tool
def reorder_tool(user_id: str, order_id: str) -> str:
    """Reorder a previous order."""
    return reorder(user_id=user_id, order_id=order_id)


@tool
def support_tool(user_id: str, subject: str, description: str) -> str:
    """Create a support ticket."""
    return support_ticket(user_id=user_id, subject=subject, description=description)


class ToolExecutionError(RuntimeError):
    pass


class ToolPolicyError(RuntimeError):
    pass


class EmptyResponseError(RuntimeError):
    pass


@dataclass
class ExecutionResult:
    content: str
    tool_outputs: List[ToolResult] = field(default_factory=list)
    error: Optional[str] = None
    tool_calls: List[Dict[str, str]] = field(default_factory=list)


class Executor:
    def __init__(self, llm, max_retries: int = 1):
        self.llm = llm
        self.max_retries = max_retries
        self.tools = [
            rag_tool,
            search_tool,
            browser_tool,
            internal_api_tool,
            custom_tool,
            mcp_tool,
            catalog_search_tool,
            inventory_check_tool,
            pricing_tool,
            promo_tool,
            cart_add_tool,
            cart_remove_tool,
            cart_view_tool,
            fraud_check_tool,
            checkout_tool,
            order_status_tool,
            logistics_tool,
            return_tool,
            refund_tool,
            reorder_tool,
            support_tool,
        ]

    def execute(self, task: str, context: str, policy: Optional[RuleDecision] = None) -> str:
        result = self.execute_detailed(task, context, policy=policy)
        return result.content

    def execute_detailed(
        self,
        task: str,
        context: str,
        policy: Optional[RuleDecision] = None,
        allowed_tools: Optional[List[str]] = None,
        trace_recorder: Optional[TraceRecorder] = None,
        session_id: Optional[str] = None,
        step_name: str = "execute",
    ) -> ExecutionResult:
        system_parts = ["Use tools when helpful. Keep responses concise."]
        if policy and policy.system_instructions:
            system_parts.extend(policy.system_instructions)
        allowed = self._resolve_allowed_tools(policy, allowed_tools)
        messages = [
            SystemMessage(content="\n".join(system_parts)),
            HumanMessage(content=f"Task: {task}\nContext: {context}"),
        ]

        for attempt in range(self.max_retries + 1):
            self._record_llm_event(
                trace_recorder,
                session_id,
                step_name,
                "llm_request",
                {"messages": self._serialize_messages(messages)},
            )
            response = self.llm.bind_tools(allowed).invoke(messages)
            self._record_llm_event(
                trace_recorder,
                session_id,
                step_name,
                "llm_response",
                {
                    "content": getattr(response, "content", ""),
                    "tool_calls": getattr(response, "tool_calls", []),
                },
            )
            try:
                return self._handle_response(
                    response,
                    allowed,
                    trace_recorder,
                    session_id,
                    step_name,
                )
            except (ToolExecutionError, ToolPolicyError, EmptyResponseError) as exc:
                if attempt >= self.max_retries:
                    return ExecutionResult(content=str(exc), error=str(exc))
                messages = self._repair_messages(task, context, response, str(exc), allowed)

        return ExecutionResult(content="Execution failed after retries.", error="retry_exhausted")

    def _handle_response(
        self,
        response,
        allowed_tools: List,
        trace_recorder: Optional[TraceRecorder],
        session_id: Optional[str],
        step_name: str,
    ) -> ExecutionResult:
        if response.tool_calls:
            outputs: List[str] = []
            tool_results: List[ToolResult] = []
            tool_map = {t.name: t for t in allowed_tools}
            for call in response.tool_calls:
                tool_fn = tool_map.get(call["name"])
                if not tool_fn:
                    raise ToolPolicyError(f"{call['name']}: blocked by tool policy")
                schema = TOOL_SCHEMAS.get(call["name"])
                started = time.time()
                try:
                    self._validate_tool_args(schema, call)
                except ToolExecutionError as exc:
                    self._record_trace(
                        trace_recorder,
                        session_id,
                        step_name,
                        call["name"],
                        "error",
                        {"args": call.get("args", {}), "error": str(exc)},
                        started,
                    )
                    raise
                args = call.get("args", {})
                try:
                    tool_out = tool_fn.invoke(args)
                except Exception as exc:
                    self._record_trace(
                        trace_recorder,
                        session_id,
                        step_name,
                        call["name"],
                        "error",
                        {"args": call.get("args", {}), "error": str(exc)},
                        started,
                    )
                    raise ToolExecutionError(f"{call['name']}: {exc}") from exc
                try:
                    self._validate_tool_output(schema, call["name"], tool_out)
                except ToolExecutionError as exc:
                    self._record_trace(
                        trace_recorder,
                        session_id,
                        step_name,
                        call["name"],
                        "error",
                        {"args": args, "output": tool_out, "error": str(exc)},
                        started,
                    )
                    raise
                self._record_trace(
                    trace_recorder,
                    session_id,
                    step_name,
                    call["name"],
                    "success",
                    {"args": args, "output": tool_out},
                    started,
                )
                outputs.append(f"{call['name']}: {tool_out}")
                tool_results.append(ToolResult(tool=call["name"], output=str(tool_out)))
            return ExecutionResult(
                content="\n".join(outputs),
                tool_outputs=tool_results,
                tool_calls=response.tool_calls,
            )
        if not response.content:
            raise EmptyResponseError("LLM returned an empty response.")
        return ExecutionResult(content=response.content, tool_calls=[])

    def _resolve_allowed_tools(
        self, policy: Optional[RuleDecision], allowed_tools: Optional[List[str]]
    ) -> List:
        tools = self.tools
        if policy and policy.allowed_tools is not None:
            tools = [tool for tool in tools if tool.name in policy.allowed_tools]
        if allowed_tools is not None:
            tools = [tool for tool in tools if tool.name in allowed_tools]
        return tools

    def _repair_messages(
        self,
        task: str,
        context: str,
        response,
        error: str,
        allowed_tools: List,
    ) -> List:
        tool_names = ", ".join(tool.name for tool in allowed_tools) or "none"
        last_output = getattr(response, "content", "") or str(getattr(response, "tool_calls", ""))
        system = (
            "You are a self-healing executor. Diagnose and fix the previous failure."
        )
        human = (
            "Original task:\n"
            f"{task}\n\nContext:\n{context}\n\nFailure:\n{error}\n\n"
            f"Previous response:\n{last_output}\n\nAvailable tools: {tool_names}\n\n"
            "Return a corrected tool call or a concise response."
        )
        return [SystemMessage(content=system), HumanMessage(content=human)]

    def _validate_tool_args(self, schema: Optional[ToolSchema], call: Dict[str, str]) -> None:
        if not schema or not schema.args_model:
            return
        args = call.get("args")
        if not isinstance(args, dict):
            raise ToolExecutionError(f"{call.get('name')}: tool args must be a JSON object.")
        try:
            schema.args_model.model_validate(args)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"{call.get('name')}: tool args failed schema validation: {exc}"
            ) from exc

    def _validate_tool_output(
        self, schema: Optional[ToolSchema], tool_name: str, output: str
    ) -> None:
        if not schema or not schema.output_model:
            return
        data = self._parse_json(output)
        if data is None:
            raise ToolExecutionError(f"{tool_name}: tool output must be valid JSON.")
        try:
            schema.output_model.model_validate(data)
        except ValidationError as exc:
            raise ToolExecutionError(
                f"{tool_name}: tool output failed schema validation: {exc}"
            ) from exc

    @staticmethod
    def _parse_json(value: str) -> Optional[object]:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _serialize_messages(messages: List) -> List[Dict[str, str]]:
        payload: List[Dict[str, str]] = []
        for message in messages:
            role = getattr(message, "type", message.__class__.__name__).lower()
            payload.append({"role": role, "content": message.content})
        return payload

    @staticmethod
    def _record_llm_event(
        trace_recorder: Optional[TraceRecorder],
        session_id: Optional[str],
        step_name: str,
        event: str,
        data: Dict[str, object],
    ) -> None:
        if not trace_recorder or not session_id:
            return
        trace_recorder.record(
            TraceEvent(
                session_id=session_id,
                step=step_name,
                event=event,
                status="success",
                data=data,
            )
        )

    @staticmethod
    def _record_trace(
        trace_recorder: Optional[TraceRecorder],
        session_id: Optional[str],
        step_name: str,
        tool_name: str,
        status: str,
        data: Dict[str, str],
        start_time: float,
    ) -> None:
        if not trace_recorder or not session_id:
            return
        latency_ms = int((time.time() - start_time) * 1000)
        trace_recorder.record(
            TraceEvent(
                session_id=session_id,
                step=step_name,
                event=tool_name,
                status=status,
                data=data,
                latency_ms=latency_ms,
            )
        )
