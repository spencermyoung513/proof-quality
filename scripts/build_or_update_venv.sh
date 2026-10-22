#!/bin/bash
# Build or update a virtual environment for this repo in the Phoenix scratch environment.
#   bash scripts/build_or_update_venv.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRATCH="/storage/home/hcoda1/1/syoung328/scratch"
VENV="${SCRATCH}/venvs/.proof-quality-venv"

export UV_CACHE_DIR="${SCRATCH}/.cache/uv"
export UV_PYTHON_INSTALL_DIR="${SCRATCH}/uv/python"
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not on PATH. Install it once, then rerun this script." >&2
    exit 1
fi

mkdir -p "$(dirname "$VENV")"

if [[ -d "$VENV" ]] && ! "$VENV/bin/python" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "Removing ${VENV}: bin/python does not run Python >= 3.10."
    rm -rf "$VENV"
fi

echo "Syncing ${VENV}"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen --directory "$REPO_ROOT"
"$VENV/bin/python" -c "import sys, torch; print(f'{sys.executable}: Python {sys.version.split()[0]}, torch {torch.__version__}')"
echo "Environment ready: ${VENV}"
