from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any, Dict, Protocol

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class ChatModel(Protocol):
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


class ModelProvider(Protocol):
    def get_chat_model(self, settings: ModelSettings) -> ChatModel:
        ...


class OpenAIProvider:
    def get_chat_model(self, settings: ModelSettings) -> ChatModel:
        api_key = settings.api_key or os.getenv("OPENAI_API_KEY")
        kwargs: Dict[str, Any] = {
            "model": settings.model,
            "temperature": settings.temperature,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        return ChatOpenAI(**kwargs)


class OpenAICompatibleProvider(OpenAIProvider):
    def get_chat_model(self, settings: ModelSettings) -> ChatModel:
        return super().get_chat_model(settings)


class AnthropicProvider:
    def get_chat_model(self, settings: ModelSettings) -> ChatModel:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "langchain-anthropic is not installed. Add it to requirements.txt to use it."
            ) from exc
        kwargs: Dict[str, Any] = {
            "model": settings.model,
            "temperature": settings.temperature,
        }
        api_key = settings.api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            key_param = _resolve_api_key_param(ChatAnthropic)
            if key_param:
                kwargs[key_param] = api_key
        return ChatAnthropic(**kwargs)


_PROVIDERS: Dict[str, ModelProvider] = {
    "openai": OpenAIProvider(),
    "openai_compatible": OpenAICompatibleProvider(),
    "anthropic": AnthropicProvider(),
}


def get_chat_model(settings: ModelSettings) -> ChatModel:
    provider = _PROVIDERS.get(settings.provider)
    if not provider:
        raise ValueError(f"Unknown LLM provider '{settings.provider}'.")
    return provider.get_chat_model(settings)


def get_embeddings(provider: str, model: str, api_key: str | None = None):
    provider = provider.lower()
    if provider != "openai":
        raise ValueError(f"Unsupported embedding provider '{provider}'.")
    kwargs: Dict[str, Any] = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)


def _resolve_api_key_param(cls: type) -> str | None:
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return None
    if "anthropic_api_key" in params:
        return "anthropic_api_key"
    if "api_key" in params:
        return "api_key"
    return None
