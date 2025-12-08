#!/usr/bin/env bash
set -euo pipefail

# Simple installer to set up a local dev environment with vLLM-enabled FastAPI app.
# Usage: ./install.sh

PYTHON_BIN="${PYTHON:-python3}"

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required to run this project. Please install Docker first." >&2
  exit 1
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11+ is required. Set PYTHON=/path/to/python3.11 if needed." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    print("Python 3.11+ is required.", file=sys.stderr)
    sys.exit(1)
PY

if [ ! -d .venv ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

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

echo "\nEnvironment is ready. Activate with 'source .venv/bin/activate' and start the app with:\n"
echo "  set -a; source .env; set +a; python main.py"
