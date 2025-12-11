# Langchain-Agent

Langchain-Agent is a FastAPI chatbot powered by a LangChain pipeline, a Chroma vector store, and vLLM-hosted models. The stack runs entirely inside Docker: run the installer and the container will be built and started for you.

## Requirements
- Docker (mandatory runtime environment)
- Python 3.11+ (optional if you want to run tests outside Docker)

## One-step setup
Run the installer once to build the Docker image, generate a default `.env`, and start the container:

- Linux/macOS (Bash):

  ```bash
  ./install.sh
  ```

- Windows (PowerShell):

  ```powershell
  ./install.ps1
  ```

When the script finishes the container will already be running. By default it creates:
- `data/` for your Markdown/Text/PDF sources
- `chroma_db/` for persisted vectors
- `.env` with sane defaults (edit it if you want different models or paths)

If you leave `data/` empty, the app will pull the
[`MuskumPillerum/General-Knowledge`](https://huggingface.co/datasets/MuskumPillerum/General-Knowledge)
dataset so the Chroma store always has something to retrieve from.

## Start/stop and logs
The installer starts the container automatically. Use Docker to control it afterward:

- View logs: `docker logs -f langchain-agent`
- Stop the app: `docker rm -f langchain-agent`
- Restart after changes: rerun the installer to rebuild and relaunch.

Open [http://localhost:8000](http://localhost:8000) to chat.

## Running Tests
To run the automated tests with your local Python (outside Docker), create and activate a virtual environment, then run:

```bash
./run_tests.sh
```

### Frontend-only mode
If you only need the UI while iterating on styles or layout, skip the heavy
back-end dependencies by setting `FRONTEND_DEV_MODE=1` before launching the
server. The chatbot endpoint will stream a lightweight mock response so the
chat window still exercises its loading and rendering states.

## Configuration (edit `.env`)
The installer writes a minimal `.env` file. Common tweaks:
- `VLLM_MODEL_ID`: Hugging Face model to serve (default quantized 1B model)
- `DATA_PATH`: Directory of documents to ingest (default `./data`)
- `CHROMA_PERSIST_DIR`: Where Chroma stores vectors locally (default `./chroma_db`)
- `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL`: Embedding backend and model name (defaults to `sentence_transformer` / `all-MiniLM-L6-v2`)
- `RETRIEVER_K`: Number of documents injected into prompts (default `4`)
- `MEMORY_TOKEN_LIMIT`: Token budget for conversation history (default `2048`)

## Docker image
The project still ships a Dockerfile for production runs. The installer uses it automatically, but you can build manually if you prefer:

```bash
docker build -t langchain-agent .
docker run --rm -p 8000:8000 --env-file .env -v ./data:/app/data -v ./chroma_db:/app/chroma_db langchain-agent
```
