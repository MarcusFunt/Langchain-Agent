"""Configuration helpers and constants for the LangChain chatbot."""

from __future__ import annotations

import os

FRONTEND_DEV_MODE = os.getenv("FRONTEND_DEV_MODE", "").lower() in {"1", "true", "yes", "on"}

MODEL_CATALOG = {
    "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8": "Quantized 1B instruct (int4)",
    "meta-llama/Llama-3.1-8B": "Standard 8B instruct",
}

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8"
DEFAULT_SYSTEM_PROMPT = "You are a concise assistant who answers using the provided snippets."
DEFAULT_PERSONA_PROMPT = os.getenv("PERSONA_PROMPT", "")

DATA_PATH = os.getenv("DATA_PATH", "data")
CHROMA_DISTANCE_METRIC = os.getenv("CHROMA_DISTANCE_METRIC", "cosine")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
CHROMA_SERVER_URL = os.getenv("CHROMA_SERVER_URL")
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))
MEMORY_TOKEN_LIMIT = int(os.getenv("MEMORY_TOKEN_LIMIT", "2048"))
