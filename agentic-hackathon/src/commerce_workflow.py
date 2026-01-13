import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .approvals import ApprovalRequest, ApprovalProvider, approval_provider_from_env
from .executor import Executor
from .deepagents_harness import DeepAgentsHarness, StepResult
from .human_input import HumanInputProvider, InputRequest, human_input_provider_from_env
from .memory import MemoryStore
from .rules import RuleDecision
from .tools.rag_tool import rag_lookup
from .utils.schemas import ToolResult
from .utils.trace_recorder import TraceEvent, TraceRecorder
from .verifier import Verification, Verifier
from .autogen.commerce_orchestrator import run_autogen_insights
from .utils.schemas import Plan


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    description: str
    required_tools: List[str] = field(default_factory=list)
    requires_confirmation: bool = False
    input_fields: List[str] = field(default_factory=list)


_WORKFLOWS: Dict[str, List[WorkflowStep]] = {
    "purchase": [
        WorkflowStep(
            name="PRODUCT_SEARCH",
            description="Search catalog for matching products.",
            required_tools=["catalog_search_tool"],
        ),
        WorkflowStep(
            name="INVENTORY_CHECK",
            description="Validate inventory for shortlisted SKUs.",
            required_tools=["inventory_check_tool"],
        ),
        WorkflowStep(
            name="PRICING",
            description="Fetch pricing and promotions if applicable.",
            required_tools=["pricing_tool", "promo_tool"],
        ),
        WorkflowStep(
            name="RECOMMEND",
            description="Rank and recommend best-fit options.",
        ),
        WorkflowStep(
            name="CART",
            description="Add selected item to cart and confirm.",
            required_tools=["cart_add_tool", "cart_view_tool"],
            requires_confirmation=True,
            input_fields=["user_id"],
        ),
        WorkflowStep(
            name="PAYMENT",
            description="Run fraud checks and request payment method confirmation.",
            required_tools=["fraud_check_tool"],
            requires_confirmation=True,
            input_fields=["user_id"],
        ),
        WorkflowStep(
            name="CONFIRM",
            description="Place the order and provide confirmation.",
            required_tools=["checkout_tool"],
            requires_confirmation=True,
            input_fields=["user_id", "payment_method", "address"],
        ),
    ],
    "product_search": [
        WorkflowStep(
            name="PRODUCT_SEARCH",
            description="Search catalog for matching products.",
            required_tools=["catalog_search_tool"],
        ),
        WorkflowStep(
            name="RECOMMEND",
            description="Recommend top matches or alternatives.",
        ),
    ],
    "compare_products": [
        WorkflowStep(
            name="PRODUCT_SEARCH",
            description="Search catalog for compared products.",
            required_tools=["catalog_search_tool"],
        ),
        WorkflowStep(
            name="RECOMMEND",
            description="Provide comparison and highlight differences.",
        ),
    ],
    "track_order": [
        WorkflowStep(
            name="ORDER_STATUS",
            description="Fetch order status.",
            required_tools=["order_status_tool"],
            input_fields=["order_id"],
        ),
        WorkflowStep(
            name="LOGISTICS",
            description="Check shipment tracking if available.",
            required_tools=["logistics_tool"],
            input_fields=["tracking_id"],
        ),
    ],
    "return_request": [
        WorkflowStep(
            name="ORDER_STATUS",
            description="Validate order eligibility for return.",
            required_tools=["order_status_tool"],
            input_fields=["order_id"],
        ),
        WorkflowStep(
            name="RETURN_CREATE",
            description="Create a return request with reason.",
            required_tools=["return_tool"],
            requires_confirmation=True,
            input_fields=["order_id", "reason"],
        ),
        WorkflowStep(
            name="REFUND_STATUS",
            description="Share refund status or next steps.",
            required_tools=["refund_tool"],
        ),
    ],
    "refund_request": [
        WorkflowStep(
            name="ORDER_STATUS",
            description="Validate order eligibility for refund.",
            required_tools=["order_status_tool"],
            input_fields=["order_id"],
        ),
        WorkflowStep(
            name="REFUND_STATUS",
            description="Provide refund status and next steps.",
            required_tools=["refund_tool"],
        ),
    ],
    "reorder": [
        WorkflowStep(
            name="ORDER_STATUS",
            description="Retrieve past order details.",
            required_tools=["order_status_tool"],
            input_fields=["order_id"],
        ),
        WorkflowStep(
            name="REORDER",
            description="Recreate the order in cart.",
            required_tools=["reorder_tool", "cart_view_tool"],
            requires_confirmation=True,
            input_fields=["user_id"],
        ),
    ],
    "support_ticket": [
        WorkflowStep(
            name="SUPPORT",
            description="Create or update a support ticket.",
            required_tools=["support_tool"],
            requires_confirmation=True,
            input_fields=["user_id", "subject", "description"],
        ),
    ],
}


_STEP_HINTS: Dict[str, str] = {
    "search": "PRODUCT_SEARCH",
    "find": "PRODUCT_SEARCH",
    "inventory": "INVENTORY_CHECK",
    "stock": "INVENTORY_CHECK",
    "price": "PRICING",
    "offer": "PRICING",
    "discount": "PRICING",
    "recommend": "RECOMMEND",
    "cart": "CART",
    "checkout": "CONFIRM",
    "payment": "PAYMENT",
    "order status": "ORDER_STATUS",
    "track": "LOGISTICS",
    "return": "RETURN_CREATE",
    "refund": "REFUND_STATUS",
    "support": "SUPPORT",
    "ticket": "SUPPORT",
}


class CommerceWorkflowEngine:
    def build(self, intent: str, plan: Plan) -> List[WorkflowStep]:
        if intent == "order_status":
            intent = "track_order"
        workflow = list(_WORKFLOWS.get(intent, []))
        mapped = self._map_plan_steps(plan, intent)
        return mapped or workflow or self._default_workflow()

    def _map_plan_steps(self, plan: Plan, intent: str) -> List[WorkflowStep]:
        workflow = _WORKFLOWS.get(intent, [])
        if not plan.steps:
            return []
        mapped_steps: List[WorkflowStep] = []
        for step in plan.steps:
            name = self._match_step_name(step.step)
            if not name:
                continue
            workflow_step = next((entry for entry in workflow if entry.name == name), None)
            if workflow_step and workflow_step not in mapped_steps:
                mapped_steps.append(workflow_step)
        return mapped_steps

    @staticmethod
    def _match_step_name(step_text: str) -> str:
        lowered = step_text.lower()
        for hint, name in _STEP_HINTS.items():
            if hint in lowered:
                return name
        return ""

    @staticmethod
    def _default_workflow() -> List[WorkflowStep]:
        return [
            WorkflowStep(name="PRODUCT_SEARCH", description="Search catalog.", required_tools=["catalog_search_tool"]),
            WorkflowStep(name="RECOMMEND", description="Recommend the best options."),
        ]


@dataclass
class WorkflowStepState:
    step: WorkflowStep
    status: str = "pending"
    attempts: int = 0
    last_output: str = ""
    verified: bool = False
    verification_notes: str = ""
    error: Optional[str] = None


@dataclass
class WorkflowRunResult:
    status: str
    step_states: List[WorkflowStepState]
    entities: Dict[str, str] = field(default_factory=dict)


class CommerceWorkflowRunner:
    def __init__(
        self,
        executor: Executor,
        verifier: Verifier,
        memory: MemoryStore,
        trace_recorder: Optional[TraceRecorder] = None,
        approval_provider: Optional[ApprovalProvider] = None,
        input_provider: Optional[HumanInputProvider] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        self.executor = executor
        self.verifier = verifier
        self.memory = memory
        self.trace_recorder = trace_recorder
        self.approval_provider = approval_provider or approval_provider_from_env()
        self.input_provider = input_provider or human_input_provider_from_env()
        self.force_rag = os.getenv("FORCE_RAG", "false").lower() == "true"
        self.deepagents_enabled = os.getenv("DEEPAGENTS_ENABLED", "false").lower() == "true"
        self.autogen_enabled = os.getenv("AUTOGEN_ENABLED", "false").lower() == "true"
        self.deepagents_rounds = int(os.getenv("DEEPAGENTS_ROUNDS", "1"))
        self.autogen_rounds = int(os.getenv("AUTOGEN_ROUNDS", "1"))
        env_retries = os.getenv("WORKFLOW_MAX_RETRIES")
        self.max_retries = max_retries if max_retries is not None else int(env_retries or "1")

    def run(
        self,
        task: str,
        intent: str,
        steps: List[WorkflowStep],
        policy: RuleDecision,
        session_id: str,
        entities: Optional[Dict[str, str]] = None,
    ) -> WorkflowRunResult:
        step_states: List[WorkflowStepState] = []
        overall_status = "success"
        runtime_entities = dict(entities or {})
        for step in steps:
            state = WorkflowStepState(step=step, status="running")
            step_states.append(state)
            missing = self._collect_inputs(task, step, runtime_entities, session_id)
            if missing:
                state.status = "failed"
                state.error = f"Missing required inputs: {', '.join(missing)}"
                state.verification_notes = state.error
                overall_status = "blocked"
                self._record_step_event(session_id, step.name, state)
                break
            if step.requires_confirmation:
                if not self._request_confirmation(task, step, session_id):
                    state.status = "failed"
                    state.error = "Approval rejected."
                    state.verification_notes = state.error
                    overall_status = "blocked"
                    self._record_step_event(session_id, step.name, state)
                    break
            rag_context = self._maybe_retrieve_rag(task, step, runtime_entities, session_id)
            insights = self._collect_insights(task, step, runtime_entities, session_id)
            verified = False
            for attempt in range(self.max_retries + 1):
                state.attempts = attempt + 1
                retry_note = ""
                if state.error:
                    retry_note = f"Previous attempt failed: {state.error}"
                step_context = self._build_step_context(
                    task,
                    intent,
                    step.name,
                    retry_note,
                    runtime_entities,
                    rag_context,
                    insights,
                )
                step_task = f"{step.name}: {step.description}\nUser goal: {task}"
                allowed_tools = self._step_allowed_tools(step, policy)
                execution = self.executor.execute_detailed(
                    step_task,
                    step_context,
                    policy=policy,
                    allowed_tools=allowed_tools,
                    trace_recorder=self.trace_recorder,
                    session_id=session_id,
                    step_name=step.name,
                )
                for tool_output in execution.tool_outputs:
                    self.memory.add_tool_result(tool_output)
                self.memory.add_turn("assistant", execution.content)
                state.last_output = execution.content
                state.error = execution.error
                state.status = "running"
                verification = self._verify_step(step, step_task, execution)
                state.verified = verification.is_valid
                state.verification_notes = verification.notes
                if not verification.is_valid:
                    state.error = verification.notes
                self._record_step_event(session_id, step.name, state)
                if verification.is_valid:
                    verified = True
                    break
                state.status = "retrying"
            if not verified:
                state.status = "failed"
                overall_status = "failed"
                break
            state.status = "verified"
        return WorkflowRunResult(
            status=overall_status,
            step_states=step_states,
            entities=runtime_entities,
        )

    def _verify_step(self, step: WorkflowStep, step_task: str, execution) -> Verification:
        if execution.error:
            return Verification(is_valid=False, notes=f"Execution error: {execution.error}")
        if step.required_tools and not execution.tool_calls:
            return Verification(is_valid=False, notes="Required tool call missing.")
        return self.verifier.verify(step_task, execution.content)

    def _build_step_context(
        self,
        task: str,
        intent: str,
        step_name: str,
        retry_note: str,
        entities: Optional[Dict[str, str]],
        rag_context: str,
        insights: List[str],
    ) -> str:
        context = self.memory.compile_context(
            task, intent, current_step=step_name, entities=entities
        )
        if retry_note:
            context = f"{context}\nRetry note: {retry_note}"
        if rag_context:
            context = f"{context}\n\nRAG context:\n{rag_context}"
        if insights:
            context = f"{context}\n\nSubagent insights:\n" + "\n".join(insights)
        return context

    def _collect_inputs(
        self,
        task: str,
        step: WorkflowStep,
        entities: Dict[str, str],
        session_id: str,
    ) -> List[str]:
        missing: List[str] = []
        for field in step.input_fields:
            if field in entities and entities[field]:
                continue
            cached = self.memory.get_cached_value(field)
            if cached:
                entities[field] = cached
                continue
            missing.append(field)
        if not missing:
            return []
        request = InputRequest(task=task, step=step.name, fields=missing)
        provided = self.input_provider.request_inputs(request)
        self._record_input_event(session_id, step.name, missing, provided)
        for field, value in provided.items():
            if not value:
                continue
            entities[field] = value
            self._store_input(field, value)
        return [field for field in missing if field not in entities]

    def _request_confirmation(self, task: str, step: WorkflowStep, session_id: str) -> bool:
        request = ApprovalRequest(
            task=task,
            intent=step.name,
            notes=[step.description],
        )
        decision = self.approval_provider.request(request)
        self._record_approval_event(session_id, step.name, decision.approved, decision.notes)
        return decision.approved

    def _maybe_retrieve_rag(
        self,
        task: str,
        step: WorkflowStep,
        entities: Dict[str, str],
        session_id: str,
    ) -> str:
        if not self.force_rag:
            return ""
        query = f"{task}\nStep: {step.name} - {step.description}\nEntities: {entities}"
        output = rag_lookup(query)
        self.memory.add_tool_result(ToolResult(tool="rag_tool", output=output))
        if self.trace_recorder:
            self.trace_recorder.record(
                TraceEvent(
                    session_id=session_id,
                    step=step.name,
                    event="rag_lookup",
                    status="success",
                    data={"query": query, "output": output},
                )
            )
        return output

    def _collect_insights(
        self,
        task: str,
        step: WorkflowStep,
        entities: Dict[str, str],
        session_id: str,
    ) -> List[str]:
        insights: List[str] = []
        if self.deepagents_enabled:
            insights.extend(self._run_deepagents(task, step, entities, session_id))
        if self.autogen_enabled:
            insights.extend(
                run_autogen_insights(
                    executor=self.executor,
                    task=task,
                    step=step,
                    entities=entities,
                    rounds=self.autogen_rounds,
                    trace_recorder=self.trace_recorder,
                    session_id=session_id,
                )
            )
        return insights

    def _run_deepagents(
        self,
        task: str,
        step: WorkflowStep,
        entities: Dict[str, str],
        session_id: str,
    ) -> List[str]:
        def _subagent(name: str, prompt: str) -> StepResult:
            result = self.executor.execute_detailed(
                prompt,
                "",
                allowed_tools=[],
                trace_recorder=self.trace_recorder,
                session_id=session_id,
                step_name=f"{step.name}:{name}",
            )
            return StepResult(name=name, output=result.content)

        task_context = f"Step: {step.name} - {step.description}\nEntities: {entities}"
        results: List[str] = []
        for round_idx in range(self.deepagents_rounds):
            harness = DeepAgentsHarness(
                [
                    lambda t: _subagent(
                        "Planner",
                        f"Provide a short execution plan.\nTask: {t}\n{task_context}",
                    ),
                    lambda t: _subagent(
                        "RiskCheck",
                        f"List missing info or risks.\nTask: {t}\n{task_context}",
                    ),
                ]
            )
            round_results = harness.run(task)
            for result in round_results:
                prefix = f"Round {round_idx + 1} {result.name}"
                results.append(f"{prefix}: {result.output}")
        return results

    def _step_allowed_tools(self, step: WorkflowStep, policy: RuleDecision) -> Optional[List[str]]:
        if not step.required_tools:
            return None
        if policy.allowed_tools is None:
            return self._with_rag(step.required_tools)
        intersection = [tool for tool in step.required_tools if tool in policy.allowed_tools]
        return self._with_rag(intersection)

    @staticmethod
    def _with_rag(tool_names: List[str]) -> List[str]:
        if "rag_tool" in tool_names:
            return tool_names
        return tool_names + ["rag_tool"]

    def _record_step_event(self, session_id: str, step_name: str, state: WorkflowStepState) -> None:
        if not self.trace_recorder:
            return
        self.trace_recorder.record(
            TraceEvent(
                session_id=session_id,
                step=step_name,
                event="verification",
                status="success" if state.verified else "error",
                data={
                    "attempts": state.attempts,
                    "output": state.last_output,
                    "verification_notes": state.verification_notes,
                },
            )
        )

    def _record_input_event(
        self,
        session_id: str,
        step_name: str,
        required: List[str],
        provided: Dict[str, str],
    ) -> None:
        if not self.trace_recorder:
            return
        self.trace_recorder.record(
            TraceEvent(
                session_id=session_id,
                step=step_name,
                event="input_request",
                status="success",
                data={"required": required, "provided": provided},
            )
        )

    def _record_approval_event(
        self,
        session_id: str,
        step_name: str,
        approved: bool,
        notes: str,
    ) -> None:
        if not self.trace_recorder:
            return
        self.trace_recorder.record(
            TraceEvent(
                session_id=session_id,
                step=step_name,
                event="approval",
                status="success" if approved else "rejected",
                data={"notes": notes},
            )
        )

    def _store_input(self, field: str, value: str) -> None:
        if field in {"user_id", "address"}:
            self.memory.set_user_detail(field, value)
        elif field in {"payment_method"}:
            self.memory.set_preference(field, value)
        elif field in {"reason", "subject", "description"}:
            self.memory.add_episodic_note(f"{field}: {value}")
        else:
            self.memory.set_preference(field, value)
