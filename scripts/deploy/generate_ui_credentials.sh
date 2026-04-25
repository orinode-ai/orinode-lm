#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ] && grep -q "ORINODE_UI_PASS=" .env && ! grep -q "CHANGE_ME_BEFORE" .env; then
    echo ".env already has credentials. To regenerate, delete ORINODE_UI_* lines and rerun."
    exit 0
fi

USER="orinode"
PASS=$(openssl rand -base64 24 | tr -d '=+/' | cut -c1-24)

if [ -f .env ]; then
    grep -v "^ORINODE_UI_" .env > .env.tmp && mv .env.tmp .env
fi

cat >> .env <<EOF
ORINODE_UI_USER=${USER}
ORINODE_UI_PASS=${PASS}
ORINODE_UI_HOST=0.0.0.0
ORINODE_UI_PORT=7860
EOF

chmod 600 .env

PUBLIC_IP=$(curl -s --max-time 5 ifconfig.me 2>/dev/null || echo "YOUR_VPS_IP")

echo ""
echo "===================================="
echo "UI credentials generated:"
echo "  URL:      http://${PUBLIC_IP}:7860"
echo "  Username: ${USER}"
echo "  Password: ${PASS}"
echo "===================================="
echo ""
echo "SAVE THE PASSWORD. Also stored in .env (mode 600)."
echo "Retrieve later: grep ORINODE_UI_PASS .env"
