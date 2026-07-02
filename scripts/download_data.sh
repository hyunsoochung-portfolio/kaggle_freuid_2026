#!/usr/bin/env bash
# Download FREUID competition data into ./data (which is gitignored).
# Requires the Kaggle API token at ~/.config/kaggle/kaggle.json (or ~/.kaggle/kaggle.json).
#   https://www.kaggle.com/settings -> Create New Token
set -euo pipefail

COMP="the-freuid-challenge-2026-ijcai-ecai"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$DEST"

# Use the kaggle CLI directly if it's on PATH (e.g. after `pip install -e .` on a cloud box),
# otherwise fall back to `uv run kaggle` for the local uv-managed dev env.
if command -v kaggle >/dev/null 2>&1; then
    KAGGLE=(kaggle)
elif command -v uv >/dev/null 2>&1; then
    KAGGLE=(uv run kaggle)
else
    echo "kaggle CLI not found — run 'pip install -e .' or 'uv sync' first." >&2
    exit 1
fi

echo "Downloading '${COMP}' into ${DEST} ..."
"${KAGGLE[@]}" competitions download -c "$COMP" -p "$DEST"

echo "Unzipping ..."
for z in "$DEST"/*.zip; do
    [ -e "$z" ] || continue
    if command -v unzip >/dev/null 2>&1; then
        unzip -o -q "$z" -d "$DEST" && rm -f "$z"
    else
        # some cloud images ship without unzip; fall back to Python's zipfile
        python -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$z" "$DEST" \
            && rm -f "$z"
    fi
done

echo "Done. Contents:"
ls -la "$DEST"
