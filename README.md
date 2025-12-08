# Langchain-Agent

This repository includes a simple example that builds a Docker container with LangChain and Chroma installed. The default entrypoint runs a small script that seeds an in-memory Chroma vector store, performs a similarity search, and can optionally load a model through the vLLM integration.

## Requirements
- Docker

## Build and run

Build the container:

```bash
docker build -t langchain-agent .
```

Run it:

```bash
docker run --rm langchain-agent
```

### Optional: load a model with vLLM

If you want to make a locally-hosted model available through vLLM, install the optional dependency and provide the model identifier via the `VLLM_MODEL_ID` environment variable:

```bash
pip install -r requirements-vllm.txt
docker run --rm -e VLLM_MODEL_ID="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" langchain-agent
```

When the variable is set, the container will instantiate a `VLLM` instance before running the vector store example.

You should see output similar to:

```
Result 1: LangChain streamlines building LLM-powered apps.
```
