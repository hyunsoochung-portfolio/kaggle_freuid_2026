"""Evaluate a checkpoint on the local labeled validation split (AuDET / APCER@1%BPCER).

    uv run python -m freuid.eval --checkpoint checkpoints/baseline_colab.pt

infer.py scores the *unlabeled* public_test, so it cannot produce a metric. This reuses the
train split's held-out validation ids — the only locally labeled data — to reproduce the offline
AuDET and print a per-class score distribution to sanity-check a suspiciously good number.
"""

from __future__ import annotations

import argparse
from dataclasses import fields

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freuid.config import Config, load_config
from freuid.data import FreuidDataset, domain_holdout_split, stratified_split
from freuid.metrics import evaluate
from freuid.models import build_model
from freuid.transforms import build_transforms, resolve_data_config
from freuid.utils import pick_device, seed_everything


@torch.no_grad()
def collect_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, labels) over the loader; scores = P(fraud)."""
    scores, labels = [], []
    for imgs, ys in tqdm(loader, leave=False):
        logits = model(imgs.to(device))
        scores.append(torch.sigmoid(logits).squeeze(1).cpu().numpy())
        labels.append(ys.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default=None,
        help="optional; backbone/image_size come from the checkpoint",
    )
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    ckpt_cfg = state.get("config", {})
    if args.config:
        cfg = load_config(args.config)
    elif ckpt_cfg:
        known = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in ckpt_cfg.items() if k in known})
    else:
        raise SystemExit("checkpoint has no stored config — pass --config explicitly")
    if ckpt_cfg:  # model-defining params always from the checkpoint
        cfg.backbone = ckpt_cfg.get("backbone", cfg.backbone)
        cfg.image_size = ckpt_cfg.get("image_size", cfg.image_size)

    seed_everything(cfg.seed)
    device = pick_device()
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)

    # Same held-out validation ids as training (cross-domain holdout if val_types is set).
    if cfg.val_types:
        _, val_ids = domain_holdout_split(cfg.data_dir, cfg.val_types)
    else:
        _, val_ids = stratified_split(cfg.data_dir, cfg.val_fraction, cfg.seed)
    transform = build_transforms(data_cfg["image_size"], False, data_cfg["mean"], data_cfg["std"])
    ds = FreuidDataset(cfg.data_dir, "train", transform, ids=val_ids)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    print(f"[eval] device={device} | backbone={cfg.backbone} | val n={len(ds)}")

    model = build_model(cfg.backbone, pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    scores, labels = collect_scores(model, loader, device)
    m = evaluate(scores, labels)
    print(f"[eval] AuDET={m['audet']:.4f}  APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.4f}")

    # Diagnostic: if the two classes' score ranges don't overlap, separation is trivial.
    for lab, name in [(0, "bona-fide"), (1, "fraud")]:
        s = scores[labels == lab]
        if s.size:
            print(f"  {name:9s} n={s.size:5d} mean={s.mean():.3f} "
                  f"min={s.min():.3f} max={s.max():.3f}")


if __name__ == "__main__":
    main()
