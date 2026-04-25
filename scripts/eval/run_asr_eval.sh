#!/usr/bin/env bash
# Run ASR evaluation for a given run ID.
# Usage: bash scripts/eval/run_asr_eval.sh <run_id>
set -euo pipefail

RUN_ID="${1:?Usage: $0 <run_id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export HF_HOME="${ROOT}/workspace/cache/huggingface"

cd "${ROOT}"
exec uv run python scripts/eval/run_eval.py --run-id "${RUN_ID}" --mode asr
