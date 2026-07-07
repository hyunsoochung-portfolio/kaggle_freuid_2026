"""Score each checkpoint's own held-out val split under the probe_v2 degradation chain
(`freuid.probe_v2_augment`) -- an independently-designed print-and-capture simulation disjoint
from the training-time `recapture_transforms`, meant to catch the blind spot that let
bayar_dinov2_v0's exact-0.0 recapture probe miss a real ~2.9x public-LB regression.

Model-type agnostic: builds whatever architecture the checkpoint's own config specifies
(`freuid.models.build_model_for_config`, the same dispatch train.py/infer.py use) and scores it
through `freuid.data.FreuidDataset` + `unpack_and_move`/`forward_with_extras` -- the same
generic plumbing infer.py's predict_scores() uses -- so `model_type="bayar_fusion"` (needs a
face_crop + face_meta) or a future model_type work without this script special-casing them.
For bayar_fusion, the face-crop stream is degraded through the same probe_v2 chain (its own
Probe2Transform instance, at the crop's own size) rather than left pristine, mirroring how
training degrades both streams for that model.

Read-only: no training, no checkpoint or src/freuid modification. Degrades every val image
ONCE with a fixed seed (so every checkpoint is scored against the identical degraded set --
differences are then attributable to the checkpoint, not to random re-draws), then scores
each checkpoint single-scale (no TTA) at its own trained resolution, same convention as
nondigital_probe.py.

VESSL-off note: this runs on CPU against the local `data/raw/` copy. A full ~10%-of-69352 val
split is too slow for a CPU smoke test -- use --max-images to subsample; run the full split on
VESSL once the workspace is back on.

Usage: python scripts/analysis/probe_v2.py [--max-images 500] [--seed 42]
       python scripts/analysis/probe_v2.py --checkpoint checkpoints/finetune_v3.pt
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
from common import build_finetuned_model, get_split_ids, load_checkpoint  # noqa: E402
from freuid.data import FreuidDataset, forward_with_extras, unpack_and_move  # noqa: E402
from freuid.metrics import evaluate  # noqa: E402
from freuid.probe_v2_augment import probe_v2_transforms  # noqa: E402
from freuid.transforms import resolve_data_config  # noqa: E402
from freuid.utils import pick_device  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def resolve_data_root(data_root: Path) -> Path:
    """Same local-checkout quirk handled in resolution_census.py: images may live under
    data/raw/... locally instead of directly under data/... (the VESSL layout)."""
    if (data_root / "train" / "train").is_dir():
        return data_root
    if (data_root / "raw" / "train" / "train").is_dir():
        return data_root / "raw"
    raise SystemExit(f"could not find train/train under {data_root} or {data_root / 'raw'}")


def discover_checkpoints(explicit: list[str] | None) -> dict[str, Path]:
    """Name -> path for every checkpoint to score.

    Defaults to every `checkpoints/*.pt` on disk (so a newly trained checkpoint, e.g.
    finetune_v3, is picked up with no code edit) rather than a hardcoded list.
    """
    if explicit:
        return {Path(p).stem: Path(p) for p in explicit}
    return {p.stem: p for p in sorted(CHECKPOINT_DIR.glob("*.pt"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--checkpoint", action="append", default=None,
                         help="Checkpoint path to score; repeatable. Default: every "
                              "checkpoints/*.pt on disk.")
    parser.add_argument("--max-images", type=int, default=None,
                         help="Subsample the val split for a fast local CPU run.")
    parser.add_argument("--seed", type=int, default=42,
                         help="Shared seed for both the val subsample and the degradation draw.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0,
                         help="Keep at 0 (default): >0 forks workers with their own RNG "
                              "stream, breaking the fixed-seed identical-degraded-set "
                              "guarantee this probe relies on for cross-checkpoint comparison.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    checkpoints = discover_checkpoints(args.checkpoint)
    if not checkpoints:
        print(f"[probe_v2] no checkpoints found under {CHECKPOINT_DIR} -- nothing to score")
        return

    results_rows = []
    for name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"[probe_v2] {name}: checkpoint not found at {ckpt_path} -- skipping")
            continue

        cfg, state = load_checkpoint(ckpt_path)
        cfg.data_dir = str(resolve_data_root(Path(args.data_root)))
        _, val_ids = get_split_ids(cfg)
        if args.max_images is not None and len(val_ids) > args.max_images:
            sampled = pd.Series(sorted(val_ids)).sample(n=args.max_images, random_state=args.seed)
            val_ids = set(sampled)

        model_type = cfg.extra.get("model_type", "baseline")
        data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
        mean, std = data_cfg["mean"], data_cfg["std"]
        transform = probe_v2_transforms(data_cfg["image_size"], mean, std)

        # Regions cache: needed whenever the checkpoint's own recipe used it (card
        # rectification and/or bayar_fusion's face crop) -- same condition infer.py gates on.
        regions_dir = None
        if cfg.extra.get("use_rectify", False) or model_type == "bayar_fusion":
            from freuid.preprocess import regions_dir as _get_rdir
            _rdir = _get_rdir(cfg.data_dir)
            if _rdir.exists():
                regions_dir = _rdir
            else:
                print(f"[probe_v2] {name}: regions cache not found at {_rdir}; "
                      "face-crop stream (if any) will be all-invalid placeholders")

        overlay_cfg = cfg.extra.get("overlay", {}) if model_type == "bayar_fusion" else {}
        return_face_crop = model_type == "bayar_fusion"
        return_face_meta = model_type == "consistency" and bool(cfg.extra.get("use_face_region", False))
        face_crop_size = int(overlay_cfg.get("crop_size", 224))
        face_crop_margin = float(overlay_cfg.get("crop_margin", 0.75))
        # Degrade the face-crop stream through the same probe_v2 chain (its own instance, at
        # the crop's own size) rather than leaving it pristine -- bayar_fusion trained with
        # recapture augmentation applied to both streams, so this stays a fair test of it.
        face_crop_transform = probe_v2_transforms(face_crop_size, mean, std) if return_face_crop else None

        # Fixed seed -> identical degraded images across checkpoints (fair comparison), and
        # reproducible across script runs, matching freuid.train._run_probe's convention.
        random.seed(args.seed)
        np.random.seed(args.seed)

        dataset = FreuidDataset(
            cfg.data_dir, "train", transform=transform, ids=val_ids,
            regions_dir=regions_dir,
            return_face_meta=return_face_meta, return_face_crop=return_face_crop,
            face_crop_size=face_crop_size, face_crop_margin=face_crop_margin,
            face_crop_transform=face_crop_transform,
            use_rectified_as_main=cfg.extra.get("use_rectify", False),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

        model = build_finetuned_model(cfg, state, device)
        scores, labels_all = [], []
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                imgs, labels, face_meta, face_crop = unpack_and_move(batch, device)
                logits = forward_with_extras(model, imgs, face_meta, face_crop)
                scores.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
                labels_all.append(labels.numpy())
        scores = np.concatenate(scores) if scores else np.array([])
        labels = np.concatenate(labels_all) if labels_all else np.array([])
        elapsed = time.time() - t0

        m = evaluate(scores, labels)
        n_fraud, n_bona = int((labels == 1).sum()), int((labels == 0).sum())
        print(f"[probe_v2] {name} (model_type={model_type}, n={len(labels)}, "
              f"{n_fraud} fraud / {n_bona} bona-fide, {elapsed:.0f}s): "
              f"probe_v2_AuDET={m['audet']:.6f} probe_v2_APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.6f}")
        results_rows.append({
            "checkpoint": name, "model_type": model_type, "n": len(labels),
            "n_fraud": n_fraud, "n_bona_fide": n_bona,
            "probe_v2_audet": m["audet"], "probe_v2_apcer_at_1pct_bpcer": m["apcer_at_1pct_bpcer"],
            "seconds": elapsed,
        })
        del model

    if not results_rows:
        print("[probe_v2] no checkpoints scored -- nothing to report")
        return

    results = pd.DataFrame(results_rows)
    csv_path = OUT_DIR / "probe_v2.csv"
    results.to_csv(csv_path, index=False)
    print(f"[probe_v2] wrote {csv_path}")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
