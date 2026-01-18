from __future__ import annotations

from langgraph.cache.memory import InMemoryCache


def build_cache(mode: str):
    if mode == "memory":
        return InMemoryCache()
    return None
