import argparse
import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


def load_documents(input_dir: str) -> List:
    docs = []
    pdf_loader = DirectoryLoader(input_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(input_dir, glob="**/*.txt", loader_cls=TextLoader)
    md_loader = DirectoryLoader(input_dir, glob="**/*.md", loader_cls=TextLoader)
    for loader in (pdf_loader, txt_loader, md_loader):
        docs.extend(loader.load())
    return docs


def build_index(input_dir: str, index_path: str) -> None:
    documents = load_documents(input_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from docs")
    parser.add_argument("--input", required=True, help="Input directory with docs")
    parser.add_argument("--index", required=True, help="Output index path")
    args = parser.parse_args()

    build_index(args.input, args.index)
    print(f"FAISS index built at {args.index}")


if __name__ == "__main__":
    main()