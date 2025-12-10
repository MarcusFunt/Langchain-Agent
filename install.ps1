[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

Write-Host "Setting up Langchain-Agent for Windows..." -ForegroundColor Cyan

# Verify Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is required to run this project. Please install Docker first." -ErrorAction Stop
}

# Resolve Python executable
$pythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$pythonCmd = Get-Command $pythonBin -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "Python 3.11+ is required. Set the PYTHON environment variable to your Python path if needed." -ErrorAction Stop
}

# Check Python version and capture Python path
$versionScript = @'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required.")
print(sys.executable)
'@
$versionOutput = & $pythonCmd.Source -c $versionScript
$pythonPath = $versionOutput.Trim()

# Create venv if missing
if (-not (Test-Path ".venv")) {
    & $pythonPath -m venv .venv
}

$venvPython = Join-Path -Path ".venv" -ChildPath "Scripts/python.exe"

# Upgrade pip and install deps
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

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

Write-Host "`nEnvironment is ready. Activate with: .\\.venv\\Scripts\\Activate.ps1" -ForegroundColor Green
Write-Host "Then start the app with:" -ForegroundColor Green
Write-Host "  Get-Content .env | ForEach-Object { if ($_ -match '^(.*)=(.*)$') { Set-Item -Path \"Env:$($matches[1])\" -Value $matches[2] } }" -ForegroundColor DarkGray
Write-Host "  python main.py" -ForegroundColor DarkGray
