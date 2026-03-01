#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-main}"
APP_DIR="/opt/helios"
VENV_BIN="$APP_DIR/venv/bin/activate"
SERVICE_NAME="helios.service"
RETENTION_DAYS="${HELIOS_RETENTION_DAYS:-3}"
JOURNAL_RETENTION="${HELIOS_JOURNAL_RETENTION:-7d}"

cd "$APP_DIR"

echo "[helios] Housekeeping old logs and recordings"
/usr/bin/python3 "$APP_DIR/scripts/helios_housekeeping.py" --repo-root "$APP_DIR" --retention-days "$RETENTION_DAYS" || true
sudo journalctl --vacuum-time="$JOURNAL_RETENTION" >/dev/null 2>&1 || true
rm -f "$APP_DIR/.git/index.lock"

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
