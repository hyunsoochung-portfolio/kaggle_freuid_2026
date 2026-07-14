#!/usr/bin/env bash
# Reproduce our ranked FREUID submission from the baked-in checkpoint, fully offline.
#
#   env / mounts:
#     DATA_DIR (default /data) : Kaggle test tree -> public_test/public_test/<id>.jpeg
#                                + sample_submission.csv (the full test-id list)
#     OUT      (default /out/submission.csv) : where the id,label CSV is written
#
# The model was fine-tuned end-to-end, so the checkpoint holds every weight; the
# backbone is created with pretrained=False and nothing is downloaded. TTA scales,
# normalization and the missing-id fallback (0.5) all come from the checkpoint config.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
OUT="${OUT:-/out/submission.csv}"

echo "[run] FREUID offline inference"
echo "[run]   data_dir = ${DATA_DIR}"
echo "[run]   out      = ${OUT}"
mkdir -p "$(dirname "${OUT}")"

python -m freuid.infer \
    --checkpoint /app/model/synth_tamper_v1_last.pt \
    --config     /app/configs/synth_tamper_v1.yaml \
    --data-dir   "${DATA_DIR}" \
    --out        "${OUT}"

echo "[run] done -> ${OUT}"
