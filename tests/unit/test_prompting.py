from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from app.prompting import build_history_block, build_prompt, format_docs


def test_build_prompt():
    ""'Test that the prompt template is created correctly.""'
    prompt = build_prompt()
    assert isinstance(prompt, PromptTemplate)
    expected_template = """{system_prompt}\n{persona_block}Context:\n{context}\n\nConversation so far:\n{history}\n\nQuestion: {question}\nAnswer:"""
    assert prompt.template == expected_template


def test_format_docs():
    ""'Test that the document formatter works as expected.""'
    docs = [
        Document(page_content="This is the first document."),
        Document(page_content="This is the second document."),
    ]
    formatted_docs = format_docs(docs)
    expected_output = "This is the first document.\n\nThis is the second document."
    assert formatted_docs == expected_output


def test_build_history_block():
    ""'Test that the history block is created correctly.""'
    messages = [
        SystemMessage(content="This is a system message."),
        HumanMessage(content="This is a human message."),
        AIMessage(content="This is an AI message."),
    ]
    history_block = build_history_block(messages)
    expected_output = "System: This is a system message.\nUser: This is a human message.\nAssistant: This is an AI message."
    assert history_block == expected_output
