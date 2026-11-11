$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:UV_CACHE_DIR = Join-Path $projectRoot '.uv-cache'

uv sync --extra dev
uv run drift-detect drift
uv run pytest
uv run ruff check .

Write-Host ''
Write-Host 'Etapa 2 concluída. Abra artifacts/reports/drift_report.html.' -ForegroundColor Green
