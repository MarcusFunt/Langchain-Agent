# Langchain-Agent

Langchain-Agent is a FastAPI chatbot powered by a LangChain pipeline, a Chroma vector store, and vLLM-hosted models. The stack is intended to run inside Docker, but local development is now a single path: run the installer, then start the app.

## Requirements
- Docker (mandatory runtime environment)
- Python 3.11+

## One-step setup
Run the installer once to create a virtual environment, install every dependency (including vLLM), and generate a default `.env`:

```bash
./install.sh
```

When the script finishes it will print the command to start the app. By default it creates:
- `.venv/` with all Python packages
- `data/` for your Markdown/Text/PDF sources
- `chroma_db/` for persisted vectors
- `.env` with sane defaults (edit it if you want different models or paths)

## Start the app
Activate the environment, load the `.env`, and launch the FastAPI server:

```bash
source .venv/bin/activate
set -a; source .env; set +a
python main.py
```

Open [http://localhost:8000](http://localhost:8000) to chat.

## Configuration (edit `.env`)
The installer writes a minimal `.env` file. Common tweaks:
- `VLLM_MODEL_ID`: Hugging Face model to serve (default quantized 1B model)
- `DATA_PATH`: Directory of documents to ingest (default `./data`)
- `CHROMA_PERSIST_DIR`: Where Chroma stores vectors locally (default `./chroma_db`)
- `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`: Embedding backend and model name (defaults to `sentence_transformer` / `all-MiniLM-L6-v2`)
- `RETRIEVER_K`: Number of documents injected into prompts (default `4`)
- `MEMORY_TOKEN_LIMIT`: Token budget for conversation history (default `2048`)

## Docker image
The project still ships a Dockerfile for production runs:

```bash
docker build -t langchain-agent .
docker run --rm -p 8000:8000 langchain-agent
```
