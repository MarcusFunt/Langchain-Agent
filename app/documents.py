"""Document loading utilities for populating the vector store."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .config import DATA_PATH


def load_documents(data_path: str = DATA_PATH) -> list[Document]:
    """Load documents from disk to populate the vector store."""

    base_path = Path(data_path)
    if not base_path.exists():
        return []

    loaders = [
        DirectoryLoader(str(base_path), glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(str(base_path), glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(str(base_path), glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]

    documents: list[Document] = []
    for loader in loaders:
        documents.extend(loader.load())

    for doc in documents:
        source = doc.metadata.get("source")
        if source:
            path = Path(source)
            doc.metadata["basename"] = path.name
            doc.metadata["source_dir"] = str(path.parent)

    return documents
