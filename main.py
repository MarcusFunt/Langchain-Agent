from __future__ import annotations

"""Example script showing LangChain with an in-memory Chroma vector store."""

import importlib
import os

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


def load_vllm_model(model_id: str | None = None) -> tuple[str, VLLM] | tuple[None, None]:
    """Instantiate a vLLM-backed model when a model ID is provided.

    The function looks for the provided ``model_id`` first and falls back to the
    ``VLLM_MODEL_ID`` environment variable. It returns ``(None, None)`` when no
    model should be loaded so the rest of the script can proceed without GPU
    requirements.
    """

    selected_model = model_id or os.getenv("VLLM_MODEL_ID")
    if not selected_model:
        return None, None

    if importlib.util.find_spec("langchain_community.llms.vllm") is None:
        raise ImportError(
            "vLLM support requires installing the optional `vllm` dependency."
        )

    from langchain_community.llms import VLLM

    llm = VLLM(model=selected_model, trust_remote_code=True)
    return selected_model, llm


def main() -> None:
    vector_store = build_vector_store()

    model_name, vllm_llm = load_vllm_model()
    if vllm_llm:
        print(f"Loaded model '{model_name}' with vLLM and ready for use.")

    query = "What is LangChain useful for?"
    results = vector_store.similarity_search(query, k=1)

    for idx, doc in enumerate(results, start=1):
        print(f"Result {idx}: {doc.page_content}")


if __name__ == "__main__":
    main()
