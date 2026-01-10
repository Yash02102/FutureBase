from ..rag import RAGPipeline


def rag_lookup(query: str) -> str:
    pipeline = RAGPipeline.from_env()
    return pipeline.lookup(query)
