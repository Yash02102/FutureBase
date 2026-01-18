from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List

from .config import AppConfig


def _parse_tags(raw: List[str]) -> List[str]:
    return [tag for tag in raw if tag]


@dataclass(frozen=True)
class LangfuseContext:
    handler: Any
    metadata: Dict[str, object]
    root_span: Any | None
    handler_factory: Callable[[], Any]

    def build_config(self, fresh: bool = False) -> Dict[str, object]:
        handler = self.handler_factory() if fresh else self.handler
        config: Dict[str, object] = {"callbacks": [handler]}
        if self.metadata:
            config["metadata"] = dict(self.metadata)
        return config

    def trace_id(self) -> str | None:
        return getattr(self.handler, "last_trace_id", None)

    def update_output(self, output: Dict[str, object]) -> None:
        if not self.root_span:
            return
        updater = getattr(self.root_span, "update_trace", None)
        if callable(updater):
            updater(output=output)


@contextmanager
def langfuse_context(
    config: AppConfig, session_id: str, task: str
) -> Iterator[LangfuseContext | None]:
    if not config.langfuse_enabled:
        yield None
        return
    if not config.langfuse_public_key or not config.langfuse_secret_key:
        raise RuntimeError(
            "Langfuse is enabled but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are missing."
        )
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Langfuse is enabled but the langfuse package is not installed.") from exc

    langfuse = get_client()

    def _new_handler() -> Any:
        return CallbackHandler()

    tags = _parse_tags(config.langfuse_tags)
    metadata: Dict[str, object] = {}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    if config.langfuse_user_id:
        metadata["langfuse_user_id"] = config.langfuse_user_id
    if tags:
        metadata["langfuse_tags"] = tags

    trace_attrs: Dict[str, object] = {}
    if session_id:
        trace_attrs["session_id"] = session_id
    if config.langfuse_user_id:
        trace_attrs["user_id"] = config.langfuse_user_id
    if tags:
        trace_attrs["tags"] = tags

    with langfuse.start_as_current_observation(
        as_type="span",
        name=config.langfuse_trace_name,
    ) as root_span:
        handler = _new_handler()
        if trace_attrs or task:
            payload = dict(trace_attrs)
            if task:
                payload["input"] = {"task": task}
            root_span.update_trace(**payload)
        try:
            yield LangfuseContext(
                handler=handler,
                metadata=metadata,
                root_span=root_span,
                handler_factory=_new_handler,
            )
        finally:
            flush = getattr(langfuse, "flush", None)
            if callable(flush):
                flush()
