"""Score the 20 real non-digital (print-and-capture) training images with our trained
checkpoints -- the only real analog-hole ground truth in the training set (everything else
that looks "recaptured" is synthetic augmentation). Read-only: no training, no src/freuid
changes, single-scale inference (no TTA) at each checkpoint's native trained resolution.

Usage: python scripts/analysis/nondigital_probe.py [--checkpoints name=path ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import build_finetuned_model, eval_transform, load_checkpoint, score_paths  # noqa: E402
from freuid.data import load_labels  # noqa: E402
from freuid.utils import pick_device  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports"
DEFAULT_CHECKPOINTS = {
    "finetune_v0": REPO_ROOT / "checkpoints" / "finetune_v0.pt",
    "finetune_v1": REPO_ROOT / "checkpoints" / "finetune_v1.pt",
    "finetune_v2": REPO_ROOT / "checkpoints" / "finetune_v2.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    df = load_labels(args.data_root, "train")
    nondig = df[df["is_digital"] == False].copy()  # noqa: E712
    nondig = nondig.sort_values(["label", "id"]).reset_index(drop=True)
    print(f"[nondigital_probe] {len(nondig)} non-digital images "
          f"({(nondig['label'] == 1).sum()} fraud, {(nondig['label'] == 0).sum()} bona-fide)")

    results = nondig[["id", "type", "label"]].copy()

    for name, ckpt_path in DEFAULT_CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            print(f"[nondigital_probe] {name}: checkpoint not found at {ckpt_path} -- skipping")
            continue
        cfg, state = load_checkpoint(ckpt_path)
        model = build_finetuned_model(cfg, state, device)
        tf, data_cfg = eval_transform(cfg)
        scores = score_paths(model, nondig["path"].tolist(), tf, device)
        results[f"score_{name}"] = scores
        pred = (scores >= 0.5).astype(int)
        results[f"correct_{name}"] = (pred == nondig["label"].to_numpy()).astype(int)
        acc = results[f"correct_{name}"].mean()
        print(f"[nondigital_probe] {name} (image_size={cfg.image_size}): "
              f"accuracy={acc:.2%} ({results[f'correct_{name}'].sum()}/{len(results)})")
        del model

    csv_path = OUT_DIR / "nondigital_probe.csv"
    results.to_csv(csv_path, index=False)
    print(f"[nondigital_probe] wrote {csv_path}")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
