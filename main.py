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
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationTokenBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings

# Silence telemetry calls that rely on networked analytics backends.
posthog.capture = lambda *_, **__: None

FRONTEND_DEV_MODE = os.getenv("FRONTEND_DEV_MODE", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

MODEL_CATALOG = {
    "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8": "Quantized 1B instruct (int4)",
    "meta-llama/Llama-3.1-8B": "Standard 8B instruct",
}
DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8"
DEFAULT_SYSTEM_PROMPT = "You are a concise assistant who answers using the provided snippets."
DEFAULT_PERSONA_PROMPT = os.getenv("PERSONA_PROMPT", "")


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
        """{system_prompt}\n{persona_block}Context:\n{context}\n\nConversation so far:\n{history}\n\nQuestion: {question}\nAnswer:"""
    )


if not FRONTEND_DEV_MODE:
    ensure_docker_runtime()
    vector_store = build_vector_store()
    retriever_k = int(os.getenv("RETRIEVER_K", "4"))
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": retriever_k}
    )
    model_name, vllm_llm = load_vllm_model()
else:
    vector_store = None
    retriever = None
    model_name, vllm_llm = "frontend-dev", None

chat_prompt = build_prompt()
system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
memory_token_limit = int(os.getenv("MEMORY_TOKEN_LIMIT", "2048"))
_session_memories: dict[str, ConversationTokenBufferMemory] = {}


class NullMemory:
    """Simple stand-in memory for frontend-only development."""

    def load_memory_variables(self, _: dict[str, Any]) -> dict[str, list[Any]]:
        return {"history": []}

    def save_context(self, *_: Any, **__: Any) -> None:
        return None

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
            .chat { width: min(820px, 100%); background: var(--panel); border: 1px solid #1f2937; border-radius: 14px; overflow: hidden; display: grid; grid-template-rows: auto 1fr auto; min-height: 72vh; }
            .messages { padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
            .settings { padding: 12px 16px; border-bottom: 1px solid #1f2937; display: flex; align-items: center; gap: 10px; background: #0b1220; }
            .settings label { color: var(--muted); font-size: 14px; }
            .settings input { flex: 1; padding: 10px; border-radius: 10px; border: 1px solid #1f2937; background: #0f172a; color: var(--text); }
            .bubble { padding: 12px 14px; border-radius: 12px; line-height: 1.5; max-width: 92%; white-space: pre-wrap; }
            .user { align-self: flex-end; background: #1f2937; border: 1px solid #22d3ee44; }
            .bot { align-self: flex-start; background: #0b2530; border: 1px solid #0ea5e9; }
            form { display: flex; gap: 10px; padding: 12px; border-top: 1px solid #1f2937; background: #0b1220; }
            textarea { flex: 1; resize: none; padding: 12px; border-radius: 10px; border: 1px solid #1f2937; background: #0f172a; color: var(--text); min-height: 56px; font-size: 15px; }
            button { background: linear-gradient(135deg, #06b6d4, #38bdf8); border: none; color: #0b1220; font-weight: 700; border-radius: 10px; padding: 0 18px; cursor: pointer; min-width: 88px; }
            button:disabled { opacity: 0.55; cursor: not-allowed; }
            .meta { color: var(--muted); font-size: 13px; }
            .spinner { width: 14px; height: 14px; border-radius: 50%; border: 2px solid #38bdf8; border-top-color: transparent; display: inline-block; animation: spin 1s linear infinite; margin-right: 8px; vertical-align: middle; }
            .status { color: #fbbf24; font-size: 13px; }
            @keyframes spin { to { transform: rotate(360deg); } }
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
                <div class=\"settings\">
                    <label for=\"persona\">Persona</label>
                    <input id=\"persona\" name=\"persona\" placeholder=\"Optional tone or role...\" />
                </div>
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
            const personaInput = document.getElementById('persona');

            const SESSION_KEY = 'langchain-agent-session';
            const PERSONA_KEY = 'langchain-agent-persona';

            function getSessionId() {
                let sid = localStorage.getItem(SESSION_KEY);
                if (!sid) {
                    sid = crypto.randomUUID();
                    localStorage.setItem(SESSION_KEY, sid);
                }
                return sid;
            }

            function getPersona() {
                return localStorage.getItem(PERSONA_KEY) || '';
            }

            function persistPersona(value) {
                localStorage.setItem(PERSONA_KEY, value.trim());
            }

            function addBubble(text, type = 'bot') {
                const bubble = document.createElement('div');
                bubble.className = `bubble ${type}`;
                bubble.textContent = text;
                messages.appendChild(bubble);
                messages.scrollTop = messages.scrollHeight;
                return bubble;
            }

            function createStreamBubble() {
                const bubble = document.createElement('div');
                bubble.className = 'bubble bot';
                const spinner = document.createElement('span');
                spinner.className = 'spinner';
                const textSpan = document.createElement('span');
                textSpan.className = 'stream-text';
                bubble.appendChild(spinner);
                bubble.appendChild(textSpan);
                messages.appendChild(bubble);
                messages.scrollTop = messages.scrollHeight;
                return { bubble, spinner, textSpan };
            }

            function setStatus(message) {
                const status = document.createElement('div');
                status.className = 'status';
                status.textContent = message;
                messages.appendChild(status);
                messages.scrollTop = messages.scrollHeight;
            }

            async function sendMessage(evt) {
                evt.preventDefault();
                const content = textarea.value.trim();
                if (!content) return;
                addBubble(content, 'user');
                textarea.value = '';

                const { bubble, spinner, textSpan } = createStreamBubble();
                const controller = new AbortController();
                let accumulated = '';
                let buffer = '';
                const decoder = new TextDecoder();
                const timeout = setTimeout(() => controller.abort(), 90000);

                try {
                    const res = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Accept': 'text/event-stream'
                        },
                        body: JSON.stringify({
                            message: content,
                            session_id: getSessionId(),
                            persona: getPersona()
                        }),
                        signal: controller.signal
                    });

                    if (!res.ok || !res.body) {
                        throw new Error('Network response was not OK');
                    }

                    const reader = res.body.getReader();
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buffer += decoder.decode(value, { stream: true });
                        const events = buffer.split('\n\n');
                        buffer = events.pop();

                        for (const evt of events) {
                            const line = evt.trim();
                            if (!line.startsWith('data: ')) continue;
                            const payload = line.slice(6);
                            if (payload === '[DONE]') {
                                spinner.remove();
                                return;
                            }
                            accumulated += payload;
                            textSpan.textContent = accumulated;
                            messages.scrollTop = messages.scrollHeight;
                        }
                    }

                    if (buffer) {
                        const payload = buffer.replace(/^data: /, '').trim();
                        if (payload && payload !== '[DONE]') {
                            accumulated += payload;
                            textSpan.textContent = accumulated;
                        }
                    }
                } catch (err) {
                    if (spinner.isConnected) spinner.remove();
                    bubble.classList.add('bot');
                    textSpan.textContent = accumulated || 'Connection dropped. Please try again.';
                    setStatus('Streaming interrupted: ' + (err?.message || 'unknown error'));
                } finally {
                    clearTimeout(timeout);
                    if (spinner.isConnected) spinner.remove();
                    messages.scrollTop = messages.scrollHeight;
                }
            }

            form.addEventListener('submit', sendMessage);
            personaInput.value = getPersona();
            personaInput.addEventListener('input', (evt) => {
                persistPersona(evt.target.value);
            });
            addBubble('Hi! I am a tiny LangChain agent. Ask me something about the demo context.');
        </script>
    </body>
    </html>
    """


class ChatRequest(BaseModel):
    """Request payload for chat messages."""

    message: str
    session_id: str | None = None
    persona: str | None = None


def get_session_memory(session_id: str) -> ConversationTokenBufferMemory:
    """Return the session-scoped memory, creating one when necessary."""

    if session_id not in _session_memories:
        if FRONTEND_DEV_MODE:
            _session_memories[session_id] = NullMemory()
        else:
            _session_memories[session_id] = ConversationTokenBufferMemory(
                llm=vllm_llm,
                return_messages=True,
                memory_key="history",
                input_key="question",
                output_key="response",
                max_token_limit=memory_token_limit,
            )
    return _session_memories[session_id]


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


def mock_frontend_response(message: str):
    """Yield a light-weight, streaming-style mock response for UI work."""

    preview = (message[:80] + "…") if len(message) > 80 else message
    yield "Frontend-only mode active. "
    yield "Backend calls are skipped. "
    yield f"You asked: {preview}"


def build_response(message: str, session_id: str, persona: str | None = None):
    """Stream an answer using the vLLM streaming API, yielding tokens incrementally."""

    if FRONTEND_DEV_MODE:
        yield from mock_frontend_response(message)
        return

    if not vllm_llm:
        raise RuntimeError(
            "vLLM must be configured. Ensure requirements are installed and a model id is set."
        )

    session_memory = get_session_memory(session_id)
    docs = retriever.get_relevant_documents(message) if retriever else []
    context = format_docs(docs) if docs else ""
    persona_block = f"Persona: {persona}\n" if persona else ""
    history_messages = session_memory.load_memory_variables({}).get("history", [])
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
    session_memory.save_context(
        {"question": message},
        {"response": full_reply},
    )


@app.post("/api/chat")
async def chat(payload: ChatRequest = Body(...)) -> StreamingResponse:
    """Handle chat messages from the UI via Server-Sent Events."""

    session_id = payload.session_id or "default"
    persona = payload.persona or DEFAULT_PERSONA_PROMPT

    def event_stream():
        for token in build_response(payload.message, session_id, persona):
            yield f"data: {token}\n\n"
        yield f"data: [DONE]\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
