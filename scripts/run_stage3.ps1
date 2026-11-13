$ErrorActionPreference = "Stop"

$env:UV_CACHE_DIR = Join-Path $PSScriptRoot "..\.uv-cache"

uv sync --extra dev
uv run drift-detect monitor

Write-Host "Etapa 3 concluida. Abra artifacts/reports/monitoring_dashboard.html"
