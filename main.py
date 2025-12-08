"""Example script showing LangChain with an in-memory Chroma vector store."""

import posthog
from chromadb.config import Settings
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Silence telemetry calls that rely on networked analytics backends.
posthog.capture = lambda *_, **__: None

CLIENT_SETTINGS = Settings(anonymized_telemetry=False)


def build_vector_store() -> Chroma:
    """Create a simple Chroma vector store populated with demo documents."""
    documents = [
        Document(page_content="LangChain streamlines building LLM-powered apps."),
        Document(page_content="Chroma provides a developer-friendly vector database."),
        Document(page_content="Containers make it easy to run code the same way everywhere."),
    ]

    embeddings = FakeEmbeddings(size=32)
    return Chroma.from_documents(documents, embeddings, client_settings=CLIENT_SETTINGS)


def main() -> None:
    vector_store = build_vector_store()

    query = "What is LangChain useful for?"
    results = vector_store.similarity_search(query, k=1)

    for idx, doc in enumerate(results, start=1):
        print(f"Result {idx}: {doc.page_content}")


if __name__ == "__main__":
    main()
