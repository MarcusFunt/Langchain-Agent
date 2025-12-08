from __future__ import annotations

"""Simple FastAPI chatbot backed by a Chroma vector store."""

import importlib
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import posthog
from chromadb.config import Settings
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

# Silence telemetry calls that rely on networked analytics backends.
posthog.capture = lambda *_, **__: None

MODEL_CATALOG = {
    "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8": "Quantized 1B instruct (int4)",
    "meta-llama/Llama-3.1-8B": "Standard 8B instruct",
}
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8"


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


def load_documents(data_path: str) -> list[Document]:
    """Load documents from disk to populate the vector store."""

    base_path = Path(data_path)
    if not base_path.exists():
        return []

    loaders = [
        DirectoryLoader(str(base_path), glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(str(base_path), glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(str(base_path), glob="**/*.pdf", loader_cls=PyPDFLoader),
    ]

    documents: list[Document] = []
    for loader in loaders:
        documents.extend(loader.load())

    for doc in documents:
        source = doc.metadata.get("source")
        if source:
            path = Path(source)
            doc.metadata["basename"] = path.name
            doc.metadata["source_dir"] = str(path.parent)

    return documents


def build_chroma_settings(persist_directory: str, server_url: str | None) -> Settings:
    """Configure Chroma to persist locally or talk to a remote server."""

    if server_url:
        parsed = urlparse(server_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if not host:
            raise ValueError("CHROMA_SERVER_URL must include a hostname")

        return Settings(
            anonymized_telemetry=False,
            chroma_api_impl="rest",
            chroma_server_host=host,
            chroma_server_http_port=port,
            chroma_server_ssl=parsed.scheme == "https",
        )

    return Settings(anonymized_telemetry=False, persist_directory=persist_directory)


def build_vector_store() -> Chroma:
    """Create a Chroma vector store populated with repository documents."""

    data_path = os.getenv("DATA_PATH", "data")
    distance_metric = os.getenv("CHROMA_DISTANCE_METRIC", "cosine")
    persist_directory = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    server_url = os.getenv("CHROMA_SERVER_URL")

    documents = load_documents(data_path)
    if not documents:
        documents = [
            Document(
                page_content=(
                    "No documents were loaded. Add Markdown, text, or PDF files under "
                    f"'{data_path}' to ground responses."
                ),
                metadata={"source": "bootstrap", "basename": "bootstrap"},
            )
        ]

    embeddings = resolve_embeddings()
    client_settings = build_chroma_settings(persist_directory, server_url)

    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        client_settings=client_settings,
        collection_name="langchain-agent-docs",
        collection_metadata={"hnsw:space": distance_metric},
    )

    if not server_url:
        vector_store.persist()

    return vector_store


def ensure_docker_runtime() -> None:
    """Fail fast when the service is not running inside a container."""

    if not os.path.exists("/.dockerenv"):
        raise RuntimeError(
            "Docker is required. Please run the service inside the provided container image."
        )


def load_vllm_model(model_id: str | None = None) -> tuple[str, Any]:
    """Instantiate a vLLM-backed model using one of the approved Hugging Face IDs.

    The function looks for the provided ``model_id`` first and falls back to the
    ``VLLM_MODEL_ID`` environment variable or the default quantized model. It
    raises errors when vLLM is unavailable or when an unsupported identifier is
    provided to ensure GPU-backed inference is always configured.
    """

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


def format_docs(docs: list[Document]) -> str:
    """Combine document contents for prompt injection."""

    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt() -> PromptTemplate:
    """Create a simple prompt that injects retrieved context."""

    return PromptTemplate.from_template(
        """You are a concise assistant who answers using the provided snippets.\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"""
    )


ensure_docker_runtime()
vector_store = build_vector_store()
retriever_k = int(os.getenv("RETRIEVER_K", "4"))
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": retriever_k})
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
    """Generate an answer from retrieved context using the configured vLLM model."""

    if not vllm_llm:
        raise RuntimeError(
            "vLLM must be configured. Ensure requirements are installed and a model id is set."
        )

    docs = retriever.get_relevant_documents(message)
    context = format_docs(docs) if docs else ""
    prompt = chat_prompt.format(context=context, question=message)
    reply = vllm_llm.invoke(prompt)
    return reply.strip()


@app.post("/api/chat")
async def chat(message: str = Body(embed=True)) -> dict[str, str]:
    """Handle chat messages from the UI."""

    reply = build_response(message)
    return {"reply": reply, "model": model_name}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
