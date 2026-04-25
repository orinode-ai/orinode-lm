#!/usr/bin/env bash
# Run Vite dev server (port 5173) and uvicorn (port 7860) in parallel.
# Kill both when either exits or Ctrl-C is pressed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${ROOT}/workspace/cache/huggingface"

cleanup() {
  kill "${UVICORN_PID:-}" "${VITE_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting uvicorn on 127.0.0.1:7860 ..."
cd "${ROOT}"
uv run uvicorn orinode.ui.server:app \
  --host 127.0.0.1 \
  --port 7860 \
  --reload \
  --log-level info &
UVICORN_PID=$!

echo "Starting Vite dev server on :5173 ..."
cd "${ROOT}/ui/frontend"
npm run dev &
VITE_PID=$!

wait
