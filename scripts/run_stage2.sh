#!/usr/bin/env sh
set -eu

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$PROJECT_ROOT"
export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"
uv sync --extra dev
uv run drift-detect drift
uv run pytest
uv run ruff check .

printf '\nEtapa 2 concluída. Abra artifacts/reports/drift_report.html.\n'
