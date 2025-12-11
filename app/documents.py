"""Document loading utilities for populating the vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .config import DATA_PATH


GENERAL_KNOWLEDGE_DATASET_ID = "MuskumPillerum/General-Knowledge"


def load_documents(data_path: str = DATA_PATH) -> list[Document]:
    """Load documents from disk to populate the vector store."""

    base_path = Path(data_path)
    if not base_path.exists():
        documents: list[Document] = []
    else:
        loaders = [
            DirectoryLoader(str(base_path), glob="**/*.md", loader_cls=TextLoader),
            DirectoryLoader(str(base_path), glob="**/*.txt", loader_cls=TextLoader),
            DirectoryLoader(str(base_path), glob="**/*.pdf", loader_cls=PyPDFLoader),
        ]

        documents: list[Document] = []
        for loader in loaders:
            documents.extend(loader.load())

    if not documents:
        documents = load_general_knowledge_documents()

    for doc in documents:
        source = doc.metadata.get("source")
        if source:
            path = Path(source)
            doc.metadata["basename"] = path.name
            doc.metadata["source_dir"] = str(path.parent)

    return documents


def load_general_knowledge_documents() -> list[Document]:
    """Load a public Q&A corpus to ensure Chroma always has data."""

    dataset = _load_hf_general_knowledge()
    documents: list[Document] = []

    for idx, item in enumerate(dataset):
        question = _clean_field(item.get("Question"))
        answer = _clean_field(item.get("Answer"))

        if not question and not answer:
            continue

        content_lines = []
        if question:
            content_lines.append(f"Question: {question}")
        if answer:
            content_lines.append(f"Answer: {answer}")

        documents.append(
            Document(
                page_content="\n".join(content_lines),
                metadata={
                    "source": GENERAL_KNOWLEDGE_DATASET_ID,
                    "basename": f"general-knowledge-{idx}",
                    "source_dir": GENERAL_KNOWLEDGE_DATASET_ID,
                },
            )
        )

    return documents


def _clean_field(value: str | None) -> str:
    """Strip whitespace and normalize a dataset field."""

    return value.strip() if value else ""


def _load_hf_general_knowledge() -> Iterable[dict]:
    """Load the General Knowledge dataset safely."""

    try:
        from datasets import load_dataset
    except Exception:
        return []

    try:
        dataset = load_dataset(GENERAL_KNOWLEDGE_DATASET_ID, split="train")
    except Exception:
        return []

    return dataset
