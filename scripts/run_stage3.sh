#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export UV_CACHE_DIR="$SCRIPT_DIR/../.uv-cache"

uv sync --extra dev
uv run drift-detect monitor

echo "Etapa 3 concluida. Abra artifacts/reports/monitoring_dashboard.html"
