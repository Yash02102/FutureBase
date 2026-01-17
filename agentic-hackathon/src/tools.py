from __future__ import annotations

from typing import Optional

import requests
from langchain_core.tools import tool


def create_rag_tool(rag_pipeline):
    @tool
    def rag_search(query: str, k: int = 4) -> str:
        """Retrieve relevant context from the RAG index."""
        if rag_pipeline is None:
            return ""
        return rag_pipeline.lookup(query, k=k)

    return rag_search


def create_http_get_tool(timeout: float = 10.0, max_chars: int = 2000):
    @tool
    def http_get(url: str) -> str:
        """Fetch a URL over HTTP for quick API checks."""
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        text = response.text
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    return http_get
