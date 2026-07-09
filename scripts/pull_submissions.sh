#!/usr/bin/env bash
# Copy Colab-generated submission CSVs from the synced Google Drive folder into ./submissions.
#
# Why: when you train/infer remotely on Colab, the submission csv is saved to Google Drive
# (MyDrive/freuid/submissions). With Google Drive desktop on your Mac it syncs locally, but to
# the CloudStorage path — NOT into this repo's submissions/. This copies it in.
#
# The repo's submissions/ is a plain (gitignored-content) dir, so copied csvs stay local and are
# never committed. macOS + Google Drive desktop only. See notebooks/run_baseline.ipynb (STEP 3).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/submissions"

# 1) Find the Google Drive desktop mount (any signed-in account).
DRIVE_ROOT=""
for d in "$HOME"/Library/CloudStorage/GoogleDrive-*; do
    [ -d "$d" ] || continue
    DRIVE_ROOT="$d"
    break
done
if [ -z "$DRIVE_ROOT" ]; then
    echo "✗ Google Drive mount not found under ~/Library/CloudStorage/."
    echo "  Install Google Drive desktop and sign in first (brew install --cask google-drive)."
    exit 1
fi

# 2) The My Drive folder name is localized: 'My Drive' (en) or '내 드라이브' (ko).
SRC=""
for name in "My Drive" "내 드라이브"; do
    cand="$DRIVE_ROOT/$name/freuid/submissions"
    if [ -d "$cand" ]; then
        SRC="$cand"
        break
    fi
done
if [ -z "$SRC" ]; then
    echo "✗ freuid/submissions not found in $DRIVE_ROOT."
    echo "  Run a Colab inference first (it creates MyDrive/freuid/submissions), then retry."
    exit 1
fi

# 3) Copy any csvs in.
mkdir -p "$DEST"
shopt -s nullglob
csvs=("$SRC"/*.csv)
if [ ${#csvs[@]} -eq 0 ]; then
    echo "(no .csv files in $SRC yet)"
    exit 0
fi
echo "Copying ${#csvs[@]} csv(s):"
echo "  from: $SRC"
echo "  to:   $DEST"
cp -v "${csvs[@]}" "$DEST"/
echo "Done. (copied csvs are gitignored — they won't be committed)"
