# Langchain-Agent

This repository now exposes a tiny chatbot UI powered by LangChain, Chroma, and an optional vLLM-backed model. It runs a FastAPI app that serves a minimal chat window and responds with answers grounded in a small in-memory vector store.

## Requirements
- Docker (for container builds)
- Python 3.11+ with the packages from `requirements.txt`

## Run locally
Install dependencies and start the FastAPI server:

```bash
pip install -r requirements.txt
python main.py
```

Then open [http://localhost:8000](http://localhost:8000) to chat.

## Build and run with Docker

Build the container:

```bash
docker build -t langchain-agent .
```

Run it:

```bash
docker run --rm -p 8000:8000 langchain-agent
```

### Optional: load a model with vLLM

If you want to make a locally-hosted model available through vLLM, install the optional dependency and provide the model identifier via the `VLLM_MODEL_ID` environment variable:

```bash
pip install -r requirements-vllm.txt
VLLM_MODEL_ID="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" python main.py
```

When the variable is set, the app will instantiate a `VLLM` instance before running the vector store example. You should see a notice in the logs that the model loaded successfully.
