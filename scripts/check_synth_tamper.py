"""Visualisation check for synthetic-tamper augmentation.

Saves 8 images (≥2 of each edit type) that have been tampered and then
passed through recapture_transforms — i.e. exactly what the model sees
during training when synth_tamper_prob > 0.

Usage (run from repo root on the VESSL workspace):
    python scripts/check_synth_tamper.py --config configs/baseline_v0.yaml

Output: synth_samples/<edit_type>_<n>.png  (denormalized, visually inspectable)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from freuid.augment import (
    _copy_move,
    _field_smudge,
    _local_splice,
    recapture_transforms,
)
from freuid.config import load_config
from freuid.data import FreuidDataset
from freuid.transforms import resolve_data_config


def _denorm(tensor, mean, std):
    """CHW float tensor → HWC uint8 numpy array (undo ImageNet normalization)."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (tensor * s + m).clamp(0, 1).permute(1, 2, 0).mul(255).byte().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_v0.yaml")
    parser.add_argument("--data-dir", default=None, help="override data_dir from config")
    parser.add_argument("--out-dir", default="synth_samples")
    parser.add_argument("--n-per-type", type=int, default=3,
                        help="images per edit type (3 types × n = total saved)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg.data_dir
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    size, mean, std = data_cfg["image_size"], data_cfg["mean"], data_cfg["std"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load bona-fide images for tampering and as splice donors
    ds = FreuidDataset(data_dir, "train", transform=None)
    bona_fide = [s for s in ds.samples if s.label == 0]
    if len(bona_fide) < args.n_per_type * 2:
        raise RuntimeError(f"Not enough bona-fide images in {data_dir!r}")

    rng = np.random.default_rng(42)
    rng.shuffle(bona_fide)
    tamper_tf = recapture_transforms(size, mean, std)

    saved: dict[str, int] = {"copy_move": 0, "local_splice": 0, "field_smudge": 0}
    idx = 0

    # Cycle through edit types explicitly to guarantee coverage
    edit_order = (
        ["copy_move"] * args.n_per_type
        + ["local_splice"] * args.n_per_type
        + ["field_smudge"] * args.n_per_type
    )

    for edit_name in edit_order:
        sample = bona_fide[idx % len(bona_fide)]
        idx += 1
        arr = np.array(Image.open(sample.path).convert("RGB"))

        if edit_name == "copy_move":
            tampered = _copy_move(arr, rng)
        elif edit_name == "local_splice":
            donor_sample = bona_fide[(idx + 10) % len(bona_fide)]
            donor_arr = np.array(Image.open(donor_sample.path).convert("RGB"))
            tampered = _local_splice(arr, donor_arr, rng)
        else:
            tampered = _field_smudge(arr, rng)

        # Pass through analog degradation (same as seen during training)
        tensor = tamper_tf(Image.fromarray(tampered))
        img_arr = _denorm(tensor, mean, std)

        n = saved[edit_name]
        out_path = out_dir / f"{edit_name}_{n}.png"
        Image.fromarray(img_arr).save(out_path)
        saved[edit_name] += 1
        print(f"  label=1  edit={edit_name:<14s}  -> {out_path}")

    total = sum(saved.values())
    print(f"\n{total} synthetic-tamper samples saved to {out_dir}/")
    print("All labeled 1 (fraud), analog-degraded via recapture_transforms, no horizontal flip.")


if __name__ == "__main__":
    main()
