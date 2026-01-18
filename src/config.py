from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


def _get_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() == "true"


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str
    llm_model: str
    llm_temperature: float
    llm_api_key: Optional[str]
    llm_base_url: Optional[str]
    embedding_provider: str
    embedding_model: str
    embedding_api_key: Optional[str]
    rag_enabled: bool
    rag_index_path: str
    rag_vector_k: int
    rag_bm25_k: int
    rag_rewrite_count: int
    rag_rerank_top_k: int
    mcp_server_url: str
    memory_max_turns: int
    memory_summarize: bool
    memory_backend: str
    memory_path: str
    hitl_mode: str
    hitl_tools: List[str]
    cache_mode: str
    filesystem_root: Optional[str]
    subagents_enabled: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        llm_model = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        llm_api_key = os.getenv("LLM_API_KEY")
        llm_base_url = os.getenv("LLM_BASE_URL")

        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        embedding_api_key = os.getenv("EMBEDDING_API_KEY")

        rag_enabled = _get_bool("RAG_ENABLED", "true")
        rag_index_path = os.getenv("RAG_INDEX_PATH", "./data/faiss_index")
        rag_vector_k = int(os.getenv("RAG_VECTOR_K", "6"))
        rag_bm25_k = int(os.getenv("RAG_BM25_K", "6"))
        rag_rewrite_count = int(os.getenv("RAG_REWRITE_COUNT", "2"))
        rag_rerank_top_k = int(os.getenv("RAG_RERANK_TOP_K", "6"))

        mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

        memory_max_turns = int(os.getenv("MEMORY_MAX_TURNS", "12"))
        memory_summarize = _get_bool("MEMORY_SUMMARIZE", "true")
        memory_backend = os.getenv("MEMORY_BACKEND", "ephemeral").lower()
        memory_path = os.getenv("MEMORY_PATH", "./data/memory")

        hitl_mode = os.getenv("HITL_MODE", "auto").lower()
        hitl_tools = [
            item.strip()
            for item in os.getenv("HITL_TOOLS", "checkout,cart_add").split(",")
            if item.strip()
        ]

        cache_mode = os.getenv("CACHE_MODE", "memory").lower()
        filesystem_root = os.getenv("FILESYSTEM_ROOT")

        subagents_enabled = _get_bool("SUBAGENTS_ENABLED", "true")

        return cls(
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_api_key=embedding_api_key,
            rag_enabled=rag_enabled,
            rag_index_path=rag_index_path,
            rag_vector_k=rag_vector_k,
            rag_bm25_k=rag_bm25_k,
            rag_rewrite_count=rag_rewrite_count,
            rag_rerank_top_k=rag_rerank_top_k,
            mcp_server_url=mcp_server_url,
            memory_max_turns=memory_max_turns,
            memory_summarize=memory_summarize,
            memory_backend=memory_backend,
            memory_path=memory_path,
            hitl_mode=hitl_mode,
            hitl_tools=hitl_tools,
            cache_mode=cache_mode,
            filesystem_root=filesystem_root,
            subagents_enabled=subagents_enabled,
        )
