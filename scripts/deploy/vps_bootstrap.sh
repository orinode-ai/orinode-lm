#!/usr/bin/env bash
set -euo pipefail

echo "=== Orinode-LM VPS bootstrap ==="

echo "[1/8] System packages"
sudo apt-get update -qq
sudo apt-get install -y curl build-essential ffmpeg sox libsndfile1 \
  python3-dev ufw apache2-utils

if ! command -v uv &> /dev/null; then
    echo "[2/8] Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[2/8] uv already installed: $(uv --version)"
fi

if ! command -v node &> /dev/null; then
    echo "[3/8] Installing Node.js 20"
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    echo "[3/8] Node.js already installed: $(node --version)"
fi

echo "[4/8] GPU check"
nvidia-smi || { echo "ERROR: nvidia-smi failed — no GPU or driver not loaded"; exit 1; }

echo "[5/8] Workspace + datasets dir"
make init-workspace
DATASETS_DIR="${ORINODE_DATASETS_DIR:-$HOME/datasets}"
    mkdir -p "$DATASETS_DIR"

echo "[6/8] Python deps"
export PATH="$HOME/.local/bin:$PATH"
uv sync

echo "[7/8] PyTorch CUDA check"
uv run python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after uv sync'
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda}')
print(f'  Device:  {torch.cuda.get_device_name(0)}')
print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo "[8/8] Firewall"
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing
SSH_PORT="${SSH_PORT:-22}"
sudo ufw allow ${SSH_PORT}/tcp comment 'SSH'
# Note: adjust SSH_PORT env var if your VPS uses a non-standard port
sudo ufw allow 7860/tcp  comment 'Orinode UI'
# Note: 7860 is the UI dashboard. Restrict to your IP only in production.
sudo ufw --force enable
sudo ufw status verbose

echo ""
echo "=== Bootstrap complete ==="
