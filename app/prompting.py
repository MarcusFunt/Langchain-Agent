"""Prompt construction helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


def format_docs(docs: list[Document]) -> str:
    """Combine document contents for prompt injection."""

    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt() -> PromptTemplate:
    """Create a simple prompt that injects retrieved context."""

    return PromptTemplate.from_template(
        """{system_prompt}\n{persona_block}Context:\n{context}\n\nConversation so far:\n{history}\n\nQuestion: {question}\nAnswer:"""
    )


def build_history_block(messages: list[Any]) -> str:
    """Render structured prior turns for the prompt."""

    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            lines.append(f"System: {msg.content}")
        elif isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
        else:
            lines.append(str(msg))
    return "\n".join(lines)
