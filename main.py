from __future__ import annotations

"""Simple FastAPI chatbot backed by a demo Chroma vector store."""

import importlib
import os
from typing import Any

import posthog
from chromadb.config import Settings
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

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


def load_vllm_model(model_id: str | None = None) -> tuple[str, Any] | tuple[None, None]:
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


def format_docs(docs: list[Document]) -> str:
    """Combine document contents for prompt injection."""

    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt() -> PromptTemplate:
    """Create a simple prompt that injects retrieved context."""

    return PromptTemplate.from_template(
        """You are a concise assistant who answers using the provided snippets.\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"""
    )


vector_store = build_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
chat_prompt = build_prompt()
model_name, vllm_llm = load_vllm_model()

app = FastAPI(title="LangChain Chatbot")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Serve a minimal chat window powered by the API."""

    return """
    <!doctype html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
        <title>LangChain Chatbot</title>
        <style>
            :root { --bg: #0f172a; --panel: #111827; --accent: #22d3ee; --text: #e5e7eb; --muted: #94a3b8; }
            * { box-sizing: border-box; }
            body { margin: 0; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, sans-serif; }
            header { padding: 16px 20px; border-bottom: 1px solid #1f2937; display: flex; align-items: center; gap: 12px; }
            header h1 { margin: 0; font-size: 20px; letter-spacing: 0.4px; }
            main { display: flex; justify-content: center; padding: 24px; }
            .chat { width: min(820px, 100%); background: var(--panel); border: 1px solid #1f2937; border-radius: 14px; overflow: hidden; display: grid; grid-template-rows: 1fr auto; min-height: 72vh; }
            .messages { padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
            .bubble { padding: 12px 14px; border-radius: 12px; line-height: 1.5; max-width: 92%; white-space: pre-wrap; }
            .user { align-self: flex-end; background: #1f2937; border: 1px solid #22d3ee44; }
            .bot { align-self: flex-start; background: #0b2530; border: 1px solid #0ea5e9; }
            form { display: flex; gap: 10px; padding: 12px; border-top: 1px solid #1f2937; background: #0b1220; }
            textarea { flex: 1; resize: none; padding: 12px; border-radius: 10px; border: 1px solid #1f2937; background: #0f172a; color: var(--text); min-height: 56px; font-size: 15px; }
            button { background: linear-gradient(135deg, #06b6d4, #38bdf8); border: none; color: #0b1220; font-weight: 700; border-radius: 10px; padding: 0 18px; cursor: pointer; min-width: 88px; }
            button:disabled { opacity: 0.55; cursor: not-allowed; }
            .meta { color: var(--muted); font-size: 13px; }
        </style>
    </head>
    <body>
        <header>
            <svg width=\"28\" height=\"28\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"#22d3ee\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><circle cx=\"12\" cy=\"12\" r=\"10\"/><path d=\"M8 12h8M12 8v8\"/></svg>
            <div>
                <h1>LangChain Chatbot</h1>
                <div class=\"meta\">Ask about LangChain, Chroma, or containers.</div>
            </div>
        </header>
        <main>
            <section class=\"chat\">
                <div id=\"messages\" class=\"messages\"></div>
                <form id=\"chat-form\"> 
                    <textarea id=\"message\" placeholder=\"Ask me anything...\" required></textarea>
                    <button type=\"submit\">Send</button>
                </form>
            </section>
        </main>
        <script>
            const form = document.getElementById('chat-form');
            const textarea = document.getElementById('message');
            const messages = document.getElementById('messages');

            function addBubble(text, type = 'bot') {
                const bubble = document.createElement('div');
                bubble.className = `bubble ${type}`;
                bubble.textContent = text;
                messages.appendChild(bubble);
                messages.scrollTop = messages.scrollHeight;
            }

            async function sendMessage(evt) {
                evt.preventDefault();
                const content = textarea.value.trim();
                if (!content) return;
                addBubble(content, 'user');
                textarea.value = '';

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: content })
                    });
                    const data = await res.json();
                    addBubble(data.reply);
                } catch (err) {
                    addBubble('Sorry, something went wrong.');
                }
            }

            form.addEventListener('submit', sendMessage);
            addBubble('Hi! I am a tiny LangChain agent. Ask me something about the demo context.');
        </script>
    </body>
    </html>
    """


def build_response(message: str) -> str:
    """Generate an answer from retrieved context or a vLLM model."""

    docs = retriever.get_relevant_documents(message)
    context = format_docs(docs) if docs else ""

    if vllm_llm:
        prompt = chat_prompt.format(context=context, question=message)
        reply = vllm_llm.invoke(prompt)
        return reply.strip()

    if context:
        return (
            "Here's what I found:\n" + context + "\n\n" + "I hope that helps!"
        )

    return "I'm not sure yet, but I'll learn more soon."


@app.post("/api/chat")
async def chat(message: str = Body(embed=True)) -> dict[str, str]:
    """Handle chat messages from the UI."""

    reply = build_response(message)
    return {"reply": reply, "model": model_name}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
