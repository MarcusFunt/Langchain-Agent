#!/usr/bin/env bash
set -euo pipefail

# Installer to build and launch the Langchain-Agent Docker container on Linux/macOS.
# Usage: ./install.sh

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required to run this project. Please install Docker first." >&2
  exit 1
}

PROJECT_ROOT="$(pwd)"
IMAGE_NAME="langchain-agent:latest"
CONTAINER_NAME="langchain-agent"

mkdir -p data chroma_db

if [ ! -f .env ]; then
  cat <<'ENV' > .env
# Default configuration for Langchain-Agent
VLLM_MODEL_ID=meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8
DATA_PATH=./data
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_PROVIDER=sentence_transformer
EMBEDDING_MODEL=all-MiniLM-L6-v2
RETRIEVER_K=4
MEMORY_TOKEN_LIMIT=2048
ENV
  echo "Wrote default .env. Update values if you want different models or paths."
fi

echo "Building Docker image..."
docker build -t "$IMAGE_NAME" .

echo "Stopping any existing container named $CONTAINER_NAME..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Starting container..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --env-file .env \
  -p 8000:8000 \
  -v "$PROJECT_ROOT/data:/app/data" \
  -v "$PROJECT_ROOT/chroma_db:/app/chroma_db" \
  "$IMAGE_NAME"

echo "\nLangchain-Agent is running in Docker at http://localhost:8000"
echo "To view logs: docker logs -f $CONTAINER_NAME"
echo "To stop the app: docker rm -f $CONTAINER_NAME"
