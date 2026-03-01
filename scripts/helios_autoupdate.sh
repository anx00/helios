#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-main}"
APP_DIR="${APP_DIR:-/opt/helios}"

cd "$APP_DIR"

echo "[helios-autoupdate] Fetching origin/$BRANCH"
git fetch origin "$BRANCH"

LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse "origin/$BRANCH")"

if [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
  echo "[helios-autoupdate] Already up to date at $LOCAL_SHA"
  exit 0
fi

echo "[helios-autoupdate] Updating $LOCAL_SHA -> $REMOTE_SHA"
"$APP_DIR/start_helios.sh" "$BRANCH"
