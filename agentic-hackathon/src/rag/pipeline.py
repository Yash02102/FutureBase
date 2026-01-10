import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


@dataclass(frozen=True)
class RAGConfig:
    index_path: str = "./data/faiss_index"
    k: int = 4
    embedding_model: str = "text-embedding-3-small"
    missing_index_message: str = "RAG index missing. Run the ingest script to build it."
    allow_dangerous_deserialization: bool = True


VectorstoreLoader = Callable[[str, str, bool], FAISS]


def default_vectorstore_loader(index_path: str, embedding_model: str, allow_dangerous: bool) -> FAISS:
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=allow_dangerous)


class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig,
        loader: VectorstoreLoader = default_vectorstore_loader,
    ) -> None:
        self.config = config
        self.loader = loader

    @classmethod
    def from_env(cls) -> "RAGPipeline":
        index_path = os.getenv("RAG_INDEX_PATH", "./data/faiss_index")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        k = int(os.getenv("RAG_TOP_K", "4"))
        config = RAGConfig(index_path=index_path, embedding_model=embedding_model, k=k)
        return cls(config=config)

    def retrieve(self, query: str) -> List[Document]:
        vectorstore = self.loader(
            self.config.index_path,
            self.config.embedding_model,
            self.config.allow_dangerous_deserialization,
        )
        return vectorstore.similarity_search(query, k=self.config.k)

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        chunks = []
        for doc in docs:
            meta = doc.metadata or {}
            source = meta.get("source", "unknown")
            chunks.append(f"[source: {source}]\n{doc.page_content}")
        return "\n\n".join(chunks)

    def lookup(self, query: str) -> str:
        if not os.path.exists(self.config.index_path):
            return self.config.missing_index_message
        docs = self.retrieve(query)
        return self.format_docs(docs)
