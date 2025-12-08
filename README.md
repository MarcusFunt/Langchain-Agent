# Langchain-Agent

This repository includes a simple example that builds a Docker container with LangChain and Chroma installed. The default entrypoint runs a small script that seeds an in-memory Chroma vector store and performs a similarity search.

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

You should see output similar to:

```
Result 1: LangChain streamlines building LLM-powered apps.
```
