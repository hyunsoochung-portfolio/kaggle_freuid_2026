#!/usr/bin/env python3
"""FREUID reproducibility entrypoint — organizer sandbox contract.

Reads a FLAT directory of test images from $DATA_DIR (default /data), scores each
with the baked-in fine-tuned checkpoint, and writes $OUT (default
/submissions/submission.csv) with exactly one `id,label` row per image
(id = filename without extension; label = fraud score, higher = more fraud).

Fully offline: the checkpoint carries all weights, so the backbone is built with
pretrained=False and nothing is downloaded. Scoring reuses the exact training-repo
inference (multi-scale TTA + rank-average via freuid.infer), so the outputs match
our ranked Kaggle submission.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from freuid.infer import _rank_normalize  # exact rank formula used for the submission
from freuid.models import build_model
from freuid.transforms import build_transforms, resolve_data_config

CKPT = os.environ.get("CKPT", "/app/model/synth_recapture_v1_ep11.pt")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
OUT = os.environ.get("OUT", "/submissions/submission.csv")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
# 0 = load in the main process: no worker subprocesses, so no dependency on Docker's
# (small, 64MB default) /dev/shm. The organizer runs a fixed `docker run` with no
# --shm-size, so workers>0 would crash with a Bus error. num_workers doesn't affect
# scores (shuffle=False), only load speed.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))

_EXTS = {".jpeg", ".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class _FlatImageDataset(Dataset):
    """Every image file directly under DATA_DIR; id = filename stem."""

    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), i


@torch.no_grad()
def _score_at_scale(model, paths, device, mean, std, scale) -> list[float]:
    """Sigmoid fraud probability per image at one TTA scale (dataset order)."""
    tf = build_transforms(scale, False, mean, std)
    loader = DataLoader(
        _FlatImageDataset(paths, tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    scores = [0.0] * len(paths)
    for imgs, idx in loader:
        logits = model(imgs.to(device))                 # [B, 1] raw logits
        probs = torch.sigmoid(logits).squeeze(1).cpu()  # matches freuid.infer.predict_scores
        for p, i in zip(probs.tolist(), idx.tolist(), strict=True):
            scores[i] = float(p)
    return scores


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    backbone = cfg["backbone"]
    image_size = int(cfg["image_size"])
    extra = cfg.get("extra", {})

    # pretrained=False: fine-tuned weights below carry everything -> no network.
    model = build_model(
        backbone, pretrained=False,
        pool=extra.get("pool"),
        head_dropout=float(extra.get("head_dropout", 0.0)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    data_cfg = resolve_data_config(backbone, image_size)
    mean, std = data_cfg["mean"], data_cfg["std"]
    tta = extra.get("tta") or [image_size]
    scales = [int(s) for s in tta] if isinstance(tta, (list, tuple)) else [image_size]

    paths = sorted(
        p for p in Path(DATA_DIR).iterdir()
        if p.is_file() and p.suffix.lower() in _EXTS
    )
    print(f"[repro] {len(paths)} images in {DATA_DIR} | backbone={backbone} "
          f"image_size={image_size} device={device} tta={scales}")

    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    if not paths:
        Path(OUT).write_text("id,label\n")
        print(f"[repro] no images found -> wrote empty {OUT}")
        return

    # Multi-scale TTA, rank-averaged across scales (invariant to per-scale calibration;
    # AuDET is a rank metric). Identical to freuid.infer.predict_scores_tta.
    avg_ranks = np.zeros(len(paths), dtype=np.float64)
    for scale in scales:
        scores = _score_at_scale(model, paths, device, mean, std, scale)
        avg_ranks += np.array(_rank_normalize(scores))
    avg_ranks /= len(scales)

    ids = [p.stem for p in paths]
    with open(OUT, "w") as f:
        f.write("id,label\n")
        for i, s in zip(ids, avg_ranks.tolist(), strict=True):
            f.write(f"{i},{s}\n")

    lo, hi = float(avg_ranks.min()), float(avg_ranks.max())
    print(f"[repro] wrote {len(ids)} rows -> {OUT} | unique={len(set(avg_ranks.tolist()))} "
          f"min={lo:.6f} max={hi:.6f} exact_zeros={int((avg_ranks == 0.0).sum())}")


if __name__ == "__main__":
    main()
