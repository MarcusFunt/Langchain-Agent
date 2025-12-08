# Langchain-Agent

Langchain-Agent is a work-in-progress AI copilot for writing, brainstorming, and rapid research. The stack runs entirely as a local web app backed by a FastAPI server and LangChain pipeline packaged in Docker so you can spin up the experience with a single container. Conversations are grounded in a persisted Chroma vector store built from your local Markdown, text, or PDF files, and responses stream through vLLM-hosted models to keep interactions fast and contextual.

## Project scope and roadmap

- **Core experience:** Deliver a reliable local-first chatbot UI reachable in the browser at `http://localhost:8000` when the Docker container is running. Users can upload or point the service at documents, then chat with an assistant that retrieves relevant passages and streams answers.
- **Model flexibility:** Prioritize strong on-device defaults while allowing easy switches between supported vLLM models via environment variables. Future iterations should expand the approved model list and add graceful fallbacks when hardware is limited.
- **Retrieval quality:** Maintain high-quality embeddings, configurable distance metrics, and tunable retriever parameters so the agent can balance precision and recall for different writing or ideation tasks.
- **Persona controls:** Keep both global (env-based) and per-session persona controls so users can experiment with tone and style without restarts.
- **Session management:** Persist session identifiers in the browser and keep token-budgeted conversation memory on the server so context survives across turns without unbounded growth.
- **Deployment path:** Optimize for Docker-first usage with sensible defaults, health checks, and logs that make it straightforward to run locally or behind reverse proxies. A production-hardened image with minimal size and faster cold starts is a near-term goal.
- **Observability and resilience:** Add structured logging, surface streaming failures clearly in the UI, and plan for metrics and tracing to make the agent reliable under long-running workloads.
- **Extensibility:** Keep the codebase modular so new tools (e.g., web search, code execution, citation formatting) can plug into the pipeline without destabilizing the core chat flow.

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

## Streaming responses

The `/api/chat` endpoint streams tokens via Server-Sent Events (SSE) using vLLM's streaming API. The browser client consumes the stream with the Fetch API and updates the assistant bubble as tokens arrive while showing a spinner. If the connection drops or times out (after ~90 seconds), the UI surfaces a warning and keeps partial output.

SSE is widely supported in evergreen browsers. Legacy Edge and very old mobile browsers may not fully support streaming Fetch; for those environments, prefer modern Chromium/Firefox/Safari or fall back to the non-streaming JSON flow by adapting the client.

## Session continuity

The frontend now creates a persistent session identifier (saved to `localStorage`) and sends it alongside each chat request. The backend keeps a token-limited `ConversationTokenBufferMemory` per session ID so conversations stay contextual while trimming older turns to respect the configured token budget. Clearing browser storage or supplying a new session ID starts a fresh history.

## Build and run with Docker (required path for deployment)

Generate your environment file and build the image:

```bash
cp .env.example .env
make build
```

Run it (Ctrl+C to stop):

```bash
make run
```

To rebuild and restart in one shot:

```bash
make dev
```

With the container running, open [http://localhost:8000](http://localhost:8000) to access the web UI backed by the local FastAPI server.

### Optional external Chroma server

The included `docker-compose.yml` defines a Chroma service you can opt into with Compose profiles. Enable it and point the API at it by setting `CHROMA_SERVER_URL=http://chroma:8000` in your `.env`, then start with the profile:

```bash
docker compose --profile chroma up
```

vLLM is no longer optional: the app will refuse to start unless the dependency is available and one of the approved model identifiers is provided (or the default is used).
