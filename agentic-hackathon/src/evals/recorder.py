import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Protocol


logger = logging.getLogger(__name__)


@dataclass
class EvalRecord:
    task: str
    intent: str
    plan: Optional[Dict[str, Any]]
    result: str
    verified: bool
    notes: str
    metadata: Dict[str, Any]


class EvalRecorder(Protocol):
    def record(self, record: EvalRecord) -> None:
        ...


class NoopEvalRecorder:
    def record(self, record: EvalRecord) -> None:
        return None


class JsonlEvalRecorder:
    def __init__(self, path: str) -> None:
        self.path = path

    def record(self, record: EvalRecord) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record)) + "\n")


class LangSmithEvalRecorder:
    def __init__(self, project: str) -> None:
        try:
            from langsmith import Client
        except ImportError as exc:
            raise RuntimeError(
                "langsmith is not installed. Add it to requirements.txt to use it."
            ) from exc
        self.client = Client()
        self.project = project

    def record(self, record: EvalRecord) -> None:
        self.client.create_run(
            project_name=self.project,
            name="agent_run",
            inputs={"task": record.task, "intent": record.intent},
            outputs={"result": record.result},
            extra=record.metadata,
        )


def eval_recorder_from_env() -> EvalRecorder:
    mode = os.getenv("EVAL_RECORDER", "noop").lower()
    if mode == "jsonl":
        path = os.getenv("EVAL_OUTPUT_PATH", "./evals/records.jsonl")
        return JsonlEvalRecorder(path)
    if mode == "langsmith":
        project = os.getenv("LANGSMITH_PROJECT", "agentic-hackathon")
        try:
            return LangSmithEvalRecorder(project)
        except RuntimeError as exc:
            logger.warning("LangSmith unavailable: %s", exc)
            path = os.getenv("EVAL_OUTPUT_PATH", "./evals/records.jsonl")
            return JsonlEvalRecorder(path)
    return NoopEvalRecorder()
