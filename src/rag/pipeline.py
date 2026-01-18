from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_index(input_dir: str, index_path: str, embeddings) -> None:
    docs = _load_documents(input_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    store.save_local(index_path)


class RAGPipeline:
    def __init__(
        self,
        index_path: str,
        embeddings,
        llm=None,
        vector_k: int = 6,
        bm25_k: int = 6,
        rewrite_count: int = 2,
        rerank_top_k: int = 6,
    ) -> None:
        self.index_path = index_path
        self.embeddings = embeddings
        self.llm = llm
        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.rewrite_count = rewrite_count
        self.rerank_top_k = rerank_top_k
        self._store: FAISS | None = None
        self._bm25: BM25Retriever | None = None

    def lookup(self, query: str, k: int = 4) -> str:
        store = self._load_store()
        if not store:
            return ""
        queries = self._rewrite_queries(query)
        docs = self._hybrid_retrieve(store, queries)
        docs = self._rerank(query, docs)
        if k:
            docs = docs[:k]
        return "\n\n".join(doc.page_content for doc in docs)

    def _load_store(self) -> FAISS | None:
        if self._store is not None:
            return self._store
        if not os.path.exists(self.index_path):
            return None
        self._store = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return self._store

    def _load_bm25(self, store: FAISS) -> BM25Retriever | None:
        if self._bm25 is not None:
            return self._bm25
        docs = list(getattr(store.docstore, "_dict", {}).values())
        if not docs:
            return None
        try:
            self._bm25 = BM25Retriever.from_documents(docs)
        except ImportError:
            return None
        self._bm25.k = self.bm25_k
        return self._bm25

    def _rewrite_queries(self, query: str) -> List[str]:
        if not self.llm or self.rewrite_count <= 0:
            return [query]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Generate short alternate search queries for the user request.",
                ),
                ("human", "Request: {query}\nReturn {count} variants."),
            ]
        )
        response = (prompt | self.llm).invoke(
            {"query": query, "count": self.rewrite_count}
        )
        content = getattr(response, "content", str(response))
        variants = [line.strip("- ").strip() for line in content.splitlines() if line.strip()]
        variants = [variant for variant in variants if variant]
        return [query] + variants[: self.rewrite_count]

    def _hybrid_retrieve(self, store: FAISS, queries: Sequence[str]) -> List[Document]:
        results: List[Document] = []
        bm25 = self._load_bm25(store)
        for q in queries:
            results.extend(store.similarity_search(q, k=self.vector_k))
            if bm25:
                results.extend(bm25.invoke(q))
        return _dedupe_documents(results)

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not self.llm or not docs:
            return docs
        scored: List[tuple[int, Document]] = []
        for doc in docs[: max(self.rerank_top_k * 2, len(docs))]:
            snippet = doc.page_content[:600]
            score = _score_with_llm(self.llm, query, snippet)
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _score, doc in scored]


def _load_documents(input_dir: str) -> List[Document]:
    allowed = {".txt", ".md"}
    docs: List[Document] = []
    for path in _iter_files(input_dir, allowed):
        content = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=content, metadata={"source": str(path)}))
    return docs


def _dedupe_documents(documents: Iterable[Document]) -> List[Document]:
    seen = set()
    unique: List[Document] = []
    for doc in documents:
        key = (doc.page_content[:200], str(doc.metadata.get("source", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _score_with_llm(llm, query: str, snippet: str) -> int:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Score relevance from 0 to 3. Return only the number.",
            ),
            ("human", "Query: {query}\nSnippet: {snippet}"),
        ]
    )
    response = (prompt | llm).invoke({"query": query, "snippet": snippet})
    content = getattr(response, "content", "0").strip()
    try:
        return max(0, min(3, int(content.split()[0])))
    except (ValueError, IndexError):
        return 0


def _iter_files(root: str, allowed: Iterable[str]) -> Iterable[Path]:
    base = Path(root)
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path
