[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

Write-Host "Setting up Langchain-Agent in Docker for Windows..." -ForegroundColor Cyan

# Verify Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is required to run this project. Please install Docker first." -ErrorAction Stop
}

$imageName = "langchain-agent:latest"
$containerName = "langchain-agent"
$projectRoot = Get-Location

# Prepare folders
New-Item -ItemType Directory -Force -Path "data" | Out-Null
New-Item -ItemType Directory -Force -Path "chroma_db" | Out-Null

# Write default .env if missing
if (-not (Test-Path ".env")) {
    @"
# Default configuration for Langchain-Agent
VLLM_MODEL_ID=meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8
DATA_PATH=./data
CHROMA_PERSIST_DIR=./chroma_db
EMBEDDING_PROVIDER=sentence_transformer
EMBEDDING_MODEL=all-MiniLM-L6-v2
RETRIEVER_K=4
MEMORY_TOKEN_LIMIT=2048
"@ | Out-File -FilePath .env -Encoding UTF8 -NoNewline
    Write-Host "Wrote default .env. Update values if you want different models or paths."
}

Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t $imageName .

Write-Host "Stopping any existing container named $containerName..." -ForegroundColor Yellow
docker rm -f $containerName 2>$null | Out-Null

$dataPath = (Resolve-Path "$($projectRoot.Path)\data").Path
$chromaPath = (Resolve-Path "$($projectRoot.Path)\chroma_db").Path

Write-Host "Starting container..." -ForegroundColor Yellow
docker run -d `
    --name $containerName `
    --env-file .env `
    -p 8000:8000 `
    -v "$dataPath:/app/data" `
    -v "$chromaPath:/app/chroma_db" `
    $imageName | Out-Null

Write-Host "`nLangchain-Agent is running in Docker at http://localhost:8000" -ForegroundColor Green
Write-Host "To view logs: docker logs -f $containerName" -ForegroundColor DarkGray
Write-Host "To stop the app: docker rm -f $containerName" -ForegroundColor DarkGray
