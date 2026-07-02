#!/usr/bin/env bash
# Pull training/inference artifacts from the VESSL GPU box down to this repo.
#
#   ./scripts/pull_from_vessl.sh
#
# The box's /root is wiped when the workspace is *terminated* (pause keeps it), so
# checkpoints and submissions live only on the box until you sync them here. This
# mirrors the box's checkpoints/, submissions/ and training logs into the local repo.
# Re-run any time; rsync only transfers what changed. Nothing pulled is committed
# (checkpoints/ and submissions/ are gitignored).
#
# Config via env vars:
#   VESSL_HOST        ssh host/alias        (default: freuid-a100, from ~/.ssh/config)
#   VESSL_REMOTE_DIR  repo path on the box  (default: /root/kaggle_freuid_2026)
set -euo pipefail

HOST="${VESSL_HOST:-freuid-a100}"
REMOTE_DIR="${VESSL_REMOTE_DIR:-/root/kaggle_freuid_2026}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# rsync on both ends is the fast path (incremental); fall back to scp otherwise.
have_rsync() { command -v rsync >/dev/null 2>&1 && ssh "$HOST" 'command -v rsync >/dev/null 2>&1'; }

pull() {  # pull <remote-subpath> <local-subpath>
    local remote="$1" local="$2"
    mkdir -p "$LOCAL_DIR/$local"
    echo "→ $HOST:$remote  ->  $local/"
    if have_rsync; then
        rsync -az --progress "$HOST:$remote/" "$LOCAL_DIR/$local/"
    else
        scp -rq "$HOST:$remote/." "$LOCAL_DIR/$local/" 2>/dev/null || echo "  (nothing there yet)"
    fi
}

pull "$REMOTE_DIR/checkpoints" checkpoints    # trained model weights (*.pt)
pull "$REMOTE_DIR/submissions" submissions    # inference outputs / submission csvs

# Training logs live at /root/*.log (from `tee`); keep them under logs/ locally.
mkdir -p "$LOCAL_DIR/logs"
echo "→ $HOST:/root/*.log  ->  logs/"
scp -q "$HOST:/root/*.log" "$LOCAL_DIR/logs/" 2>/dev/null || echo "  (no logs yet)"

echo
echo "Done. Local artifacts:"
ls -lh "$LOCAL_DIR/checkpoints" "$LOCAL_DIR/submissions" "$LOCAL_DIR/logs" 2>/dev/null || true
