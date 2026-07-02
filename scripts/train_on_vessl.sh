#!/usr/bin/env bash
# Train on the VESSL GPU box, then auto-pull the results down to this repo.
#
#   ./scripts/train_on_vessl.sh configs/baseline_xdomain.yaml
#
# One command from your Mac: it streams training live from the box, and the moment
# training exits it runs scripts/pull_from_vessl.sh to sync checkpoints/submissions/
# logs back here (checkpoints/ is a Drive symlink, so they're backed up to Drive too).
# Artifacts are pulled even if training crashes, so you always get the log.
#
# NOTE: training runs in the foreground of this ssh session — if your laptop sleeps
# or wifi drops, the run is killed. For long unattended runs, start it in tmux on the
# box yourself and run scripts/pull_from_vessl.sh afterwards.
#
# Config via env vars:
#   VESSL_HOST        ssh host/alias        (default: freuid-a100, from ~/.ssh/config)
#   VESSL_REMOTE_DIR  repo path on the box  (default: /root/kaggle_freuid_2026)
set -uo pipefail   # not -e: a training crash must still fall through to the pull

CONFIG="${1:?usage: $0 <config-path>   e.g. configs/baseline_xdomain.yaml}"
HOST="${VESSL_HOST:-freuid-a100}"
REMOTE_DIR="${VESSL_REMOTE_DIR:-/root/kaggle_freuid_2026}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STEM="$(basename "${CONFIG%.*}")"
LOG="/root/train_${STEM}.log"

echo "== [$HOST] train '$CONFIG'  (box log: $LOG) =="
# -t gives the remote a tty so tqdm renders live. `tee` keeps a copy on the box.
# git pull keeps the box on the latest committed code (best-effort — a dirty box
# just warns and runs anyway). exit ${PIPESTATUS[0]} propagates python's real exit
# code through the tee pipe, so a crash surfaces here instead of tee's 0.
ssh -t "$HOST" "cd '$REMOTE_DIR' && { git pull --ff-only || echo '[warn] git pull skipped'; } && \
    PYTHONUNBUFFERED=1 python -u -m freuid.train --config '$CONFIG' 2>&1 | tee '$LOG'; exit \${PIPESTATUS[0]}"
rc=$?

echo
echo "== training exited (rc=$rc) — pulling artifacts to $LOCAL_DIR =="
"$LOCAL_DIR/scripts/pull_from_vessl.sh"
exit "$rc"
