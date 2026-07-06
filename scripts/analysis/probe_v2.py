"""Score each checkpoint's own held-out val split under the probe_v2 degradation chain
(`freuid.probe_v2_augment`) -- an independently-designed print-and-capture simulation disjoint
from the training-time `recapture_transforms`, meant to catch the blind spot that let
bayar_dinov2_v0's exact-0.0 recapture probe miss a real ~2.9x public-LB regression.

Read-only: no training, no checkpoint or src/freuid modification. Degrades every val image
ONCE with a fixed seed (so every checkpoint is scored against the identical degraded set --
differences are then attributable to the checkpoint, not to random re-draws), then scores
each checkpoint single-scale (no TTA) at its own trained resolution, same convention as
nondigital_probe.py.

VESSL-off note: this runs on CPU against the local `data/raw/` copy. A full ~10%-of-69352 val
split is too slow for a CPU smoke test -- use --max-images to subsample; run the full split on
VESSL once the workspace is back on.

Usage: python scripts/analysis/probe_v2.py [--max-images 500] [--seed 42]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from common import build_finetuned_model, get_split_ids, load_checkpoint, split_dataframe  # noqa: E402
from freuid.metrics import evaluate  # noqa: E402
from freuid.probe_v2_augment import probe_v2_transforms  # noqa: E402
from freuid.transforms import resolve_data_config  # noqa: E402
from freuid.utils import pick_device  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports"


def resolve_data_root(data_root: Path) -> Path:
    """Same local-checkout quirk handled in resolution_census.py: images may live under
    data/raw/... locally instead of directly under data/... (the VESSL layout)."""
    if (data_root / "train" / "train").is_dir():
        return data_root
    if (data_root / "raw" / "train" / "train").is_dir():
        return data_root / "raw"
    raise SystemExit(f"could not find train/train under {data_root} or {data_root / 'raw'}")


DEFAULT_CHECKPOINTS = {
    "finetune_v0": REPO_ROOT / "checkpoints" / "finetune_v0.pt",
    "finetune_v1": REPO_ROOT / "checkpoints" / "finetune_v1.pt",
    "finetune_v2": REPO_ROOT / "checkpoints" / "finetune_v2.pt",
    "bayar_dinov2_v0": REPO_ROOT / "checkpoints" / "bayar_dinov2_v0.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--max-images", type=int, default=None,
                         help="Subsample the val split for a fast local CPU run.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Shared seed for both the val subsample and the degradation draw.")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    results_rows = []
    for name, ckpt_path in DEFAULT_CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            print(f"[probe_v2] {name}: checkpoint not found at {ckpt_path} -- skipping")
            continue

        cfg, state = load_checkpoint(ckpt_path)
        cfg.data_dir = str(resolve_data_root(Path(args.data_root)))
        _, val_ids = get_split_ids(cfg)
        df = split_dataframe(cfg, val_ids)
        if args.max_images is not None and len(df) > args.max_images:
            df = df.sample(n=args.max_images, random_state=args.seed).sort_values("id").reset_index(drop=True)

        data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
        transform = probe_v2_transforms(data_cfg["image_size"], data_cfg["mean"], data_cfg["std"])

        # Fixed seed -> identical degraded images across checkpoints (fair comparison), and
        # reproducible across script runs, matching freuid.train._run_probe's convention.
        random.seed(args.seed)
        np.random.seed(args.seed)

        model = build_finetuned_model(cfg, state, device)
        scores = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(df), args.batch_size):
                batch_paths = df["path"].iloc[i:i + args.batch_size].tolist()
                imgs = torch.stack(
                    [transform(Image.open(p).convert("RGB")) for p in batch_paths]
                ).to(device)
                logits = model(imgs)
                scores.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
        scores = np.concatenate(scores) if scores else np.array([])
        labels = df["label"].to_numpy()
        elapsed = time.time() - t0

        m = evaluate(scores, labels)
        n_fraud, n_bona = int((labels == 1).sum()), int((labels == 0).sum())
        print(f"[probe_v2] {name} (n={len(df)}, {n_fraud} fraud / {n_bona} bona-fide, "
              f"{elapsed:.0f}s): probe_v2_AuDET={m['audet']:.6f} "
              f"probe_v2_APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.6f}")
        results_rows.append({
            "checkpoint": name, "n": len(df), "n_fraud": n_fraud, "n_bona_fide": n_bona,
            "probe_v2_audet": m["audet"], "probe_v2_apcer_at_1pct_bpcer": m["apcer_at_1pct_bpcer"],
            "seconds": elapsed,
        })
        del model

    if not results_rows:
        print("[probe_v2] no checkpoints found -- nothing to report")
        return

    results = pd.DataFrame(results_rows)
    csv_path = OUT_DIR / "probe_v2.csv"
    results.to_csv(csv_path, index=False)
    print(f"[probe_v2] wrote {csv_path}")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
