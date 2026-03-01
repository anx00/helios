#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-main}"
APP_DIR="/opt/helios"
VENV_BIN="$APP_DIR/venv/bin/activate"
SERVICE_NAME="helios.service"

cd "$APP_DIR"

echo "[helios] Fetching origin/$BRANCH"
git fetch origin "$BRANCH"

echo "[helios] Resetting tracked files to origin/$BRANCH"
git reset --hard "origin/$BRANCH"

echo "[helios] Installing dependencies"
source "$VENV_BIN"
pip install -r requirements.txt

echo "[helios] Restarting $SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
sudo systemctl status "$SERVICE_NAME" --no-pager
