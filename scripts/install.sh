#!/usr/bin/env bash
# Run from the repository root: bash scripts/install.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
if [[ ! -d .venv ]]; then
    uv venv --python "${REPEAT_TASK_PYTHON:-3.15t}"
fi
uv pip install --python .venv/bin/python -e ".[dev,examples]"
echo "Installed. Activate with: source .venv/bin/activate"
