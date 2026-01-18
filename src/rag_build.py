import argparse

from dotenv import load_dotenv

from .config import AppConfig
from .llm import get_embeddings
from .rag import build_index


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build a FAISS RAG index.")
    parser.add_argument("--input", required=True, help="Directory of documents")
    parser.add_argument("--index", required=True, help="Output index path")
    args = parser.parse_args()

    config = AppConfig.from_env()
    embeddings = get_embeddings(
        provider=config.embedding_provider,
        model=config.embedding_model,
        api_key=config.embedding_api_key,
    )
    build_index(args.input, args.index, embeddings)
    print(f"Index built at {args.index}")


if __name__ == "__main__":
    main()
