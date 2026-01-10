import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


def load_faiss_index(index_path: str) -> FAISS:
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def retrieve(query: str, index_path: str, k: int = 4) -> List[Document]:
    vectorstore = load_faiss_index(index_path)
    return vectorstore.similarity_search(query, k=k)


def format_docs(docs: List[Document]) -> str:
    chunks = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "unknown")
        chunks.append(f"[source: {source}]\n{doc.page_content}")
    return "\n\n".join(chunks)


def rag_lookup(query: str) -> str:
    index_path = os.getenv("RAG_INDEX_PATH", "./data/faiss_index")
    if not os.path.exists(index_path):
        return "RAG index missing. Run the ingest script to build it."
    docs = retrieve(query, index_path=index_path)
    return format_docs(docs)