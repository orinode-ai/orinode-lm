#!/usr/bin/env bash
# Build the React frontend and output to src/orinode/ui/static/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Installing frontend dependencies..."
cd "${ROOT}/ui/frontend"
npm ci

echo "Building frontend..."
npm run build

echo "Frontend built → ${ROOT}/src/orinode/ui/static/"
