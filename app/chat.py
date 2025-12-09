"""Chat response orchestration."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .config import DEFAULT_PERSONA_PROMPT
from .prompting import build_history_block, format_docs


def mock_frontend_response(message: str) -> Iterable[str]:
    """Yield a light-weight, streaming-style mock response for UI work."""

    preview = (message[:80] + "…") if len(message) > 80 else message
    yield "Frontend-only mode active. "
    yield "Backend calls are skipped. "
    yield f"You asked: {preview}"


def build_response_generator(
    frontend_dev_mode: bool,
    retriever: Any | None,
    chat_prompt: Any,
    system_prompt: str,
    default_persona_prompt: str = DEFAULT_PERSONA_PROMPT,
    get_session_memory: Callable[[str], Any] | None = None,
    vllm_llm: Any | None = None,
) -> Callable[[str, str, str | None], Iterable[str]]:
    """Create a streaming response builder with injected dependencies."""

    def build_response(message: str, session_id: str, persona: str | None = None):
        if frontend_dev_mode:
            yield from mock_frontend_response(message)
            return

        if not vllm_llm:
            raise RuntimeError(
                "vLLM must be configured. Ensure requirements are installed and a model id is set."
            )

        session_memory = get_session_memory(session_id) if get_session_memory else None
        docs = retriever.get_relevant_documents(message) if retriever else []
        context = format_docs(docs) if docs else ""
        persona_block = f"Persona: {persona or default_persona_prompt}\n" if (persona or default_persona_prompt) else ""
        history_messages = session_memory.load_memory_variables({}).get("history", []) if session_memory else []
        history_block = build_history_block(history_messages)
        prompt = chat_prompt.format(
            system_prompt=system_prompt,
            persona_block=persona_block,
            context=context,
            history=history_block,
            question=message,
        )

        reply_tokens: list[str] = []
        for chunk in vllm_llm.stream(prompt):
            token = chunk if isinstance(chunk, str) else getattr(chunk, "text", str(chunk))
            reply_tokens.append(token)
            yield token

        full_reply = "".join(reply_tokens).strip()
        if session_memory:
            session_memory.save_context(
                {"question": message},
                {"response": full_reply},
            )

    return build_response
