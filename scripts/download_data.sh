#!/usr/bin/env bash
# Download FREUID competition data into ./data (which is gitignored).
# Requires the Kaggle API token at ~/.config/kaggle/kaggle.json (or ~/.kaggle/kaggle.json).
#   https://www.kaggle.com/settings -> Create New Token
set -euo pipefail

COMP="the-freuid-challenge-2026-ijcai-ecai"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DEST"

echo "Downloading '${COMP}' into ${DEST} ..."
uv run kaggle competitions download -c "$COMP" -p "$DEST"

echo "Unzipping ..."
for z in "$DEST"/*.zip; do
    [ -e "$z" ] || continue
    unzip -o -q "$z" -d "$DEST" && rm -f "$z"
done

echo "Done. Contents:"
ls -la "$DEST"
