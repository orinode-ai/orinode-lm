#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${ROOT}/workspace/cache/huggingface"
export TRANSFORMERS_CACHE="${ROOT}/workspace/cache/transformers"

cd "${ROOT}"

NUM_GPUS="${NUM_GPUS:-1}"
CONFIG="${CONFIG:-configs/training/stage3.yaml}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  exec accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision bf16 \
    -m orinode.training.stage3_speech_llm \
    --config "${CONFIG}"
else
  exec uv run python -m orinode.training.stage3_speech_llm --config "${CONFIG}"
fi
