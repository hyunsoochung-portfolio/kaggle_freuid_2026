# syntax=docker/dockerfile:1
# ─── FREUID Challenge 2026 — reproducible OFFLINE inference ──────────────────────
# Self-contained image that reproduces our ranked submission with NO network at run
# time. Our fine-tuned checkpoint carries ALL weights, so the backbone is built with
# pretrained=False (no HuggingFace/timm download) — see src/freuid/infer.py.
#
#   Build  (needs network, once):
#     # 1. put the trained checkpoint where the build can see it:
#     #    cp checkpoints/synth_tamper_v1_last.pt docker/model/
#     docker build -t freuid-submission .
#
#   Run  (NO network; mount the Kaggle test tree read-only + an output dir):
#     docker run --rm --network none \
#         -v /path/to/kaggle_data:/data:ro -v /path/to/out:/out \
#         freuid-submission
#     # -> writes /out/submission.csv  (columns: id,label ; label = P(fraud) in [0,1])
#
#   GPU (optional, ~10x faster): add `--gpus all`. Without it, inference falls back
#   to CPU automatically (freuid.utils.pick_device).
#
# /data must contain the Kaggle layout:  public_test/public_test/<id>.jpeg  +  sample_submission.csv

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# cv2 (pulled by albumentations) needs libGL/glib at import; harmless for inference.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Python deps in their own layer (torch 2.3.1 already present -> pip keeps it).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# 2) Bake in the exact ranked model + the config that produced it.
COPY configs/synth_tamper_v1.yaml ./configs/synth_tamper_v1.yaml
COPY docker/model/synth_tamper_v1_last.pt ./model/synth_tamper_v1_last.pt

# 3) Belt-and-suspenders: forbid any accidental phone-home at run time.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

COPY docker/run_inference.sh ./run_inference.sh
RUN chmod +x ./run_inference.sh

ENTRYPOINT ["./run_inference.sh"]
