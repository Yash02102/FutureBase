import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, Protocol

from langchain_openai import ChatOpenAI


class ChatModel(Protocol):
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def bind_tools(self, tools: list) -> "ChatModel":
        ...

    def with_structured_output(self, schema: Any) -> "ChatModel":
        ...


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float

    @classmethod
    def from_env(cls) -> "LLMSettings":
        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        return cls(provider=provider, model=model, temperature=temperature)


class LLMProvider(Protocol):
    def get_chat_model(self, settings: LLMSettings) -> ChatModel:
        ...


class OpenAIProvider:
    def get_chat_model(self, settings: LLMSettings) -> ChatModel:
        api_key = os.getenv("LLM_API_KEY")
        kwargs: Dict[str, Any] = {
            "model": settings.model,
            "temperature": settings.temperature,
        }
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)


class OpenAICompatibleProvider:
    def get_chat_model(self, settings: LLMSettings) -> ChatModel:
        base_url = os.getenv("LLM_BASE_URL")
        api_key = os.getenv("LLM_API_KEY")
        kwargs: Dict[str, Any] = {
            "model": settings.model,
            "temperature": settings.temperature,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        return ChatOpenAI(**kwargs)


class AnthropicProvider:
    def get_chat_model(self, settings: LLMSettings) -> ChatModel:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "langchain-anthropic is not installed. Add it to requirements.txt to use it."
            ) from exc
        return ChatAnthropic(model=settings.model, temperature=settings.temperature)


class MockChatResponse:
    def __init__(self, content: str, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class MockChatModel:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        structured_schema: Any | None = None,
        tools: list | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._structured_schema = structured_schema
        self._tools = tools or []

    def bind_tools(self, tools: list) -> "MockChatModel":
        clone = copy.copy(self)
        clone._tools = tools
        return clone

    def with_structured_output(self, schema: Any) -> "MockChatModel":
        clone = copy.copy(self)
        clone._structured_schema = schema
        return clone

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        if self._structured_schema is not None:
            return _default_structured(self._structured_schema)
        return MockChatResponse(content="mock-response", tool_calls=[])


class MockProvider:
    def get_chat_model(self, settings: LLMSettings) -> ChatModel:
        return MockChatModel(model=settings.model, temperature=settings.temperature)


_PROVIDERS: Dict[str, LLMProvider] = {
    "openai": OpenAIProvider(),
    "openai_compatible": OpenAICompatibleProvider(),
    "anthropic": AnthropicProvider(),
    "mock": MockProvider(),
}


def get_chat_model(settings: LLMSettings) -> ChatModel:
    provider = _PROVIDERS.get(settings.provider)
    if not provider:
        raise ValueError(f"Unknown LLM provider '{settings.provider}'.")
    return provider.get_chat_model(settings)


def _default_structured(schema: Any) -> Any:
    if hasattr(schema, "model_construct"):
        fields = getattr(schema, "model_fields", {})
        values: Dict[str, Any] = {}
        if "goal" in fields and "steps" in fields:
            values = {"goal": "mock-goal", "steps": []}
        elif "is_valid" in fields and "notes" in fields:
            values = {"is_valid": True, "notes": "mock-verification"}
        return schema.model_construct(**values)
    return schema()
