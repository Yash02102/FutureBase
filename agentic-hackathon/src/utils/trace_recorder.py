import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class TraceEvent:
    session_id: str
    step: str
    event: str
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


class TraceRecorder(Protocol):
    def record(self, event: TraceEvent) -> None:
        ...


class NoopTraceRecorder:
    def record(self, event: TraceEvent) -> None:
        return None


class JsonlTraceRecorder:
    def __init__(self, path: str) -> None:
        self.path = path

    def record(self, event: TraceEvent) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), ensure_ascii=True) + "\n")


def trace_recorder_from_env() -> TraceRecorder:
    mode = os.getenv("TRACE_RECORDER", "noop").lower()
    if mode == "jsonl":
        path = os.getenv("TRACE_OUTPUT_PATH", "./traces/commerce.jsonl")
        return JsonlTraceRecorder(path)
    return NoopTraceRecorder()
