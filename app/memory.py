"""Session memory management for chat conversations."""

from __future__ import annotations

from typing import Any, Callable

from langchain_classic.memory import ConversationTokenBufferMemory


class NullMemory:
    """Simple stand-in memory for frontend-only development."""

    def load_memory_variables(self, _: dict[str, Any]) -> dict[str, list[Any]]:
        return {"history": []}

    def save_context(self, *_: Any, **__: Any) -> None:
        return None


def build_memory_manager(
    frontend_dev_mode: bool, llm: Any | None, memory_token_limit: int
) -> Callable[[str], Any]:
    """Create a session-aware memory accessor based on runtime configuration."""

    _session_memories: dict[str, Any] = {}

    def get_session_memory(session_id: str) -> Any:
        if session_id not in _session_memories:
            if frontend_dev_mode:
                _session_memories[session_id] = NullMemory()
            else:
                _session_memories[session_id] = ConversationTokenBufferMemory(
                    llm=llm,
                    return_messages=True,
                    memory_key="history",
                    input_key="question",
                    output_key="response",
                    max_token_limit=memory_token_limit,
                )
        return _session_memories[session_id]

    return get_session_memory
