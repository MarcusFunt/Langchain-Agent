"""Chroma vector store configuration and construction."""

from __future__ import annotations

from urllib.parse import urlparse

from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from .config import CHROMA_DISTANCE_METRIC, CHROMA_PERSIST_DIR, CHROMA_SERVER_URL, DATA_PATH
from .documents import load_documents
from .embeddings import resolve_embeddings


def build_chroma_settings(
    persist_directory: str = CHROMA_PERSIST_DIR, server_url: str | None = CHROMA_SERVER_URL
) -> Settings:
    """Configure Chroma to persist locally or talk to a remote server."""

    if server_url:
        parsed = urlparse(server_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            raise ValueError("CHROMA_SERVER_URL must include a hostname")

        return Settings(
            anonymized_telemetry=False,
            chroma_api_impl="rest",
            chroma_server_host=host,
            chroma_server_http_port=port,
            chroma_server_ssl=parsed.scheme == "https",
        )

    return Settings(anonymized_telemetry=False, persist_directory=persist_directory)


def build_vector_store() -> Chroma:
    """Create a Chroma vector store populated with repository documents."""

    documents = load_documents()
    if not documents:
        documents = [
            Document(
                page_content=(
                    "No documents were loaded. Add Markdown, text, or PDF files under "
                    f"'{DATA_PATH}' to ground responses."
                ),
                metadata={"source": "bootstrap", "basename": "bootstrap"},
            )
        ]

    embeddings = resolve_embeddings()
    client_settings = build_chroma_settings()

    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        client_settings=client_settings,
        collection_name="langchain-agent-docs",
        collection_metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
    )

    if not CHROMA_SERVER_URL:
        vector_store.persist()

    return vector_store
