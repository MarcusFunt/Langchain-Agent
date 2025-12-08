# Langchain-Agent

This repository now exposes a tiny chatbot UI powered by LangChain, Chroma, and an optional vLLM-backed model. It runs a FastAPI app that serves a minimal chat window and responds with answers grounded in a persisted Chroma vector store populated from local Markdown, text, or PDF files.

## Requirements
- Docker (mandatory runtime environment)
- vLLM installed via `requirements.txt` (mandatory for inference)
- Python 3.11+ with the packages from `requirements.txt` (for local dev)

vLLM is required because the chatbot always serves answers from one of the
following Hugging Face models:

- `meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8` (default, int4 quantized)
- `meta-llama/Llama-3.1-8B`

## Run locally (GPU with vLLM)
Install dependencies and start the FastAPI server with your chosen model:

```bash
pip install -r requirements.txt
VLLM_MODEL_ID="meta-llama/Llama-3.1-8B" python main.py
```

If you omit `VLLM_MODEL_ID`, the app automatically loads
`meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8`. Then open
[http://localhost:8000](http://localhost:8000) to chat.

Key configuration options (all via environment variables):

- `DATA_PATH`: Directory containing Markdown (`*.md`), text (`*.txt`), or PDF (`*.pdf`) files to ingest into the vector store. Defaults to `./data`.
- `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`: Choose the embedding backend and model name. Defaults to `sentence_transformer` / `all-MiniLM-L6-v2`. Set `EMBEDDING_PROVIDER=openai` and `EMBEDDING_MODEL=text-embedding-3-small` to use OpenAI (requires `OPENAI_API_KEY`).
- `CHROMA_PERSIST_DIR`: Directory where Chroma stores vectors locally (default: `./chroma_db`).
- `CHROMA_SERVER_URL`: Point Chroma to an external server (e.g., `http://chroma:8000`); overrides local persistence.
- `CHROMA_DISTANCE_METRIC`: Similarity metric for the collection (e.g., `cosine`, `l2`, `ip`).
- `RETRIEVER_K`: Number of documents to inject into the prompt (default: `4`).
- `SYSTEM_PROMPT`: System-level instruction prepended to every conversation (default: concise assistant grounded in retrieved snippets).
- `PERSONA_PROMPT`: Optional persona block included ahead of user turns. Can also be provided per-request via the UI settings field.
- `MEMORY_TOKEN_LIMIT`: Token budget for the conversational memory buffer per session (default: `2048`).

## Personas and system prompt

The model receives a system prompt followed by an optional persona block before the user message. Configure them with environment variables when starting the service:

```bash
SYSTEM_PROMPT="You are a witty container expert." \
PERSONA_PROMPT="Respond as a friendly DevOps mentor." \
python main.py
```

The UI also exposes a **Persona** text field. The value is stored locally (via `localStorage`) and sent with each `/api/chat` request so you can experiment without restarting the server. Leave it empty to fall back to `PERSONA_PROMPT` or omit persona altogether.

## Session continuity

The frontend now creates a persistent session identifier (saved to `localStorage`) and sends it alongside each chat request. The backend keeps a token-limited `ConversationTokenBufferMemory` per session ID so conversations stay contextual while trimming older turns to respect the configured token budget. Clearing browser storage or supplying a new session ID starts a fresh history.

## Build and run with Docker (required path for deployment)

Build the container:

```bash
docker build -t langchain-agent .
```

Run it:

```bash
docker run --rm -p 8000:8000 \
  -e VLLM_MODEL_ID="meta-llama/Llama-3.1-8B" \
  langchain-agent
```

vLLM is no longer optional: the app will refuse to start unless the dependency
is available and one of the approved model identifiers is provided (or the
default is used).
