$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'
uv sync --extra dev
uv run drift-detect demo
uv run pytest
uv run ruff check .

Write-Host ''
Write-Host 'Etapa 1 concluída. Consulte artifacts/reports/ para as evidências.' -ForegroundColor Green

