"""Embedding selection helpers."""

from __future__ import annotations

import os
from typing import Any

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings


def resolve_embeddings() -> Any:
    """Pick an embedding model based on environment configuration."""

    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformer").lower()
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

        return OpenAIEmbeddings(model=model_name, api_key=api_key)

    return SentenceTransformerEmbeddings(model_name=model_name)
