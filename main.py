from __future__ import annotations

"""Simple FastAPI chatbot backed by a Chroma vector store."""

import os

import posthog
from dotenv import load_dotenv
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from app.chat import build_response_generator
from app.config import (
    DEFAULT_PERSONA_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    FRONTEND_DEV_MODE,
    MEMORY_TOKEN_LIMIT,
    RETRIEVER_K,
)
from app.frontend import render_index_html
from app.llm import ensure_docker_runtime, load_vllm_model
from app.memory import build_memory_manager
from app.prompting import build_prompt
from app.vector_store import build_vector_store

# Silence telemetry calls that rely on networked analytics backends.
posthog.capture = lambda *_, **__: None

load_dotenv()
chat_prompt = build_prompt()
system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

if not FRONTEND_DEV_MODE:
    ensure_docker_runtime()
    vector_store = build_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": RETRIEVER_K}
    )
    _, vllm_llm = load_vllm_model()
else:
    vector_store = None
    retriever = None
    vllm_llm = None

get_session_memory = build_memory_manager(
    FRONTEND_DEV_MODE, vllm_llm, MEMORY_TOKEN_LIMIT
)
build_response = build_response_generator(
    FRONTEND_DEV_MODE,
    retriever,
    chat_prompt,
    system_prompt,
    DEFAULT_PERSONA_PROMPT,
    get_session_memory,
    vllm_llm,
)

app = FastAPI(title="LangChain Chatbot")


class ChatRequest(BaseModel):
    """Request payload for chat messages."""

    message: str
    session_id: str | None = None
    persona: str | None = None


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve a minimal chat window powered by the API."""

    return render_index_html()


@app.post("/api/chat")
async def chat(payload: ChatRequest = Body(...)) -> StreamingResponse:
    """Handle chat messages from the UI via Server-Sent Events."""

    session_id = payload.session_id or "default"

    def event_stream():
        for token in build_response(payload.message, session_id, payload.persona):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
