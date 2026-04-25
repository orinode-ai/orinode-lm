#!/usr/bin/env bash
# Symlink ${ORINODE_DATASETS_DIR:-$HOME/datasets}/* into workspace/data/raw/ so build_manifests.py
# can find corpora without duplicating data.
set -euo pipefail

DATASETS_DIR="${ORINODE_DATASETS_DIR:-$HOME/datasets}"
RAW_DIR="$(pwd)/workspace/data/raw"

if [ ! -d "$DATASETS_DIR" ]; then
    echo "ERROR: $DATASETS_DIR does not exist. Run download scripts first."
    exit 1
fi

mkdir -p "$RAW_DIR"

link_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -d "$src" ]; then
        ln -sfn "$src" "$dst"
        echo "  linked: $src → $dst"
    else
        echo "  skip (not found): $src"
    fi
}

echo "=== Linking datasets → workspace/data/raw/ ==="
link_if_exists "$DATASETS_DIR/afrispeech-200"        "$RAW_DIR/afrispeech_200"
link_if_exists "$DATASETS_DIR/naijavoices"            "$RAW_DIR/naijavoices"
link_if_exists "$DATASETS_DIR/bibletts"               "$RAW_DIR/bibletts"
link_if_exists "$DATASETS_DIR/common_voice"           "$RAW_DIR/common_voice"
link_if_exists "$DATASETS_DIR/crowdsourced_cs"        "$RAW_DIR/crowdsourced_cs"
echo "Done."
