# Langchain-Agent

This repository now exposes a tiny chatbot UI powered by LangChain, Chroma, and an optional vLLM-backed model. It runs a FastAPI app that serves a minimal chat window and responds with answers grounded in a small in-memory vector store.

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
