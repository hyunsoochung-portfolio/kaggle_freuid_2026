# syntax=docker/dockerfile:1
# ─── FREUID Challenge 2026 — reproducible OFFLINE inference (organizer sandbox) ───
# Implements the organizer's sandbox contract exactly:
#
#   docker build -t freuid-repro:local .
#   docker run --network none \
#       -v /path/to/flat/test/images:/data:ro \
#       -v "$(pwd)/out:/submissions" \
#       freuid-repro:local
#   # -> writes /submissions/submission.csv   (id,label ; label = fraud score, higher = more fraud)
#
# /data  : flat directory of image files only ({id}.{jpeg|jpg|png|webp|bmp|tif|tiff}); id = filename stem.
# Output : exactly one row per input image, no missing/extra ids.
# No network: the fine-tuned checkpoint carries ALL weights, so the backbone is built with
# pretrained=False and nothing is downloaded. GPU optional (--gpus all); CPU fallback is automatic.

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# cv2 (pulled by albumentations) needs libGL/glib at import; harmless for inference.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Python deps in their own layer (torch 2.3.1 already present -> pip keeps it).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# 2) Bake in the exact ranked model + the offline inference entrypoint.
COPY docker/model/synth_recapture_v1_ep11.pt ./model/synth_recapture_v1_ep11.pt
COPY docker/prepare_submission.py ./prepare_submission.py

# 3) Offline defaults + belt-and-suspenders: forbid any accidental phone-home.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    CKPT=/app/model/synth_recapture_v1_ep11.pt \
    DATA_DIR=/data \
    OUT=/submissions/submission.csv

ENTRYPOINT ["python", "/app/prepare_submission.py"]
