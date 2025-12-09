"""Large language model utilities."""

from __future__ import annotations

import importlib
import os
from typing import Any

from .config import DEFAULT_MODEL_ID, MODEL_CATALOG


def ensure_docker_runtime() -> None:
    """Fail fast when the service is not running inside a container."""

    if not os.path.exists("/.dockerenv"):
        raise RuntimeError(
            "Docker is required. Please run the service inside the provided container image."
        )


def load_vllm_model(model_id: str | None = None) -> tuple[str, Any]:
    """Instantiate a vLLM-backed model using one of the approved Hugging Face IDs."""

    selected_model = model_id or os.getenv("VLLM_MODEL_ID") or DEFAULT_MODEL_ID

    if selected_model not in MODEL_CATALOG:
        raise ValueError(
            "Unsupported model id. Choose one of: " + ", ".join(MODEL_CATALOG)
        )

    if importlib.util.find_spec("langchain_community.llms.vllm") is None:
        raise ImportError(
            "vLLM is required. Install dependencies from requirements.txt before running."
        )

    from langchain_community.llms import VLLM

    llm = VLLM(model=selected_model, trust_remote_code=True)
    return selected_model, llm
