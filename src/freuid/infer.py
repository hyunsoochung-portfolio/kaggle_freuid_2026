"""Inference → Kaggle submission csv.

    uv run python -m freuid.infer --checkpoint checkpoints/baseline.pt \
        --out submissions/baseline.csv

Model-defining params (``backbone``, ``image_size``) are read from the checkpoint itself
so preprocessing + architecture always match the trained weights — no way to silently
mismatch them. ``--config`` is optional and only supplies runtime/environment params; the
CLI flags ``--data-dir`` / ``--batch-size`` / ``--num-workers`` override those per machine.

Submission format (from sample_submission.csv): columns ``id,label`` where label is the
predicted fraud score in [0, 1] (the DET metrics need a continuous score, not a hard 0/1).

This is a code competition: sample_submission.csv lists the FULL test set (~142.8k ids) but
only the public subset (~7.8k) of images ships in the download. We score every id whose image
is present locally and default the rest to 0.0; on Kaggle's grading run all images are present,
so every id gets a real score.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freuid.config import Config, load_config
from freuid.data import FreuidDataset, OverlayDataset, load_labels, precache_crops, resolve_cache_dir
from freuid.models import build_model
from freuid.models.overlay import build_overlay_model
from freuid.transforms import build_transforms, get_overlay_val_transforms, resolve_data_config
from freuid.utils import pick_device, seed_everything

ID_COLUMN = "id"
SCORE_COLUMN = "label"
MISSING_DEFAULT = 0.0  # ids whose image is absent locally (filled for real on Kaggle)


@torch.no_grad()
def predict_scores(model, loader, device) -> list[float]:
    """Fraud scores in dataset order (loader must be shuffle=False)."""
    scores: list[float] = []
    for imgs, _ in tqdm(loader, leave=False):
        logits = model(imgs.to(device))
        scores.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())
    return scores


def resolve_config(args) -> tuple[Config, dict]:
    """Build the run config and return it alongside the loaded checkpoint.

    ``backbone`` and ``image_size`` always come from the checkpoint's stored config so the
    model matches its weights. ``--config`` (optional) seeds the rest; CLI flags override
    runtime/environment params (data dir, batch size, workers).
    """
    state = torch.load(args.checkpoint, map_location="cpu")
    ckpt_cfg = state.get("config", {})

    if args.config:
        cfg = load_config(args.config)
    elif ckpt_cfg:
        known = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in ckpt_cfg.items() if k in known})
    else:
        raise SystemExit("checkpoint has no stored config — pass --config explicitly")

    # Model-defining params ALWAYS come from the checkpoint (guarantees the weights match).
    if ckpt_cfg:
        cfg.backbone = ckpt_cfg.get("backbone", cfg.backbone)
        cfg.image_size = ckpt_cfg.get("image_size", cfg.image_size)

    # Runtime / environment overrides.
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    return cfg, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default=None,
        help="optional; backbone/image_size still come from the checkpoint",
    )
    parser.add_argument("--out", default="submissions/submission.csv")
    parser.add_argument(
        "--data-dir", default=None,
        help="override data dir (default: from checkpoint config)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    # state = torch.load(args.checkpoint, map_location="cpu")
    # checkpoint 불러오기. state는 checkpoint의 내용이 담긴 딕셔너리.
    # model 가중치 + config + epoch + metrics 등이 들어있음.
    cfg, state = resolve_config(args)
    seed_everything(cfg.seed)
    device = pick_device()
    is_overlay = cfg.extra.get("model_type") == "overlay"

    if is_overlay:
        print(
            f"[infer] model=overlay image_size={cfg.image_size or 224} (from checkpoint) | "
            f"device={device} | data_dir={cfg.data_dir}"
        )
        model = build_overlay_model(cfg).to(device)
    else:
        print(
            f"[infer] backbone={cfg.backbone} image_size={cfg.image_size} (from checkpoint) | "
            f"device={device} | data_dir={cfg.data_dir}"
        )
        model = build_model(cfg.backbone, pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    submission = load_labels(cfg.data_dir, "public_test")
    present_mask = submission["path"].map(lambda p: Path(p).exists())
    present_ids = set(submission.loc[present_mask, "id"])
    print(f"[infer] {len(submission)} ids total; {len(present_ids)} images present locally")

    if is_overlay:
        precache_crops(cfg, splits=("public_test",))
        ov = cfg.extra.get("overlay", {})
        ds = OverlayDataset(
            cfg.data_dir, "public_test",
            get_overlay_val_transforms(cfg.image_size or 224),
            ids=present_ids,
            crop_margin=ov.get("crop_margin", 0.75),
            cache_dir=resolve_cache_dir(cfg),
            detect_long_side=ov.get("detect_long_side", 1024),
            min_face_size=ov.get("min_face_size", 60),
        )
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        scores = predict_scores(model, loader, device)
        id_to_score = dict(zip((s[0] for s in ds.samples), scores, strict=True))
    else:
        data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
        transform = build_transforms(data_cfg["image_size"], False, data_cfg["mean"], data_cfg["std"])
        ds = FreuidDataset(cfg.data_dir, "public_test", transform, ids=present_ids)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        scores = predict_scores(model, loader, device)
        id_to_score = dict(zip((s.id for s in ds.samples), scores, strict=True))

    submission[SCORE_COLUMN] = submission[ID_COLUMN].map(
        lambda i: id_to_score.get(i, MISSING_DEFAULT)
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission[[ID_COLUMN, SCORE_COLUMN]].to_csv(args.out, index=False)
    print(f"[infer] wrote {len(submission)} rows -> {args.out} "
          f"({len(id_to_score)} scored, {len(submission) - len(id_to_score)} defaulted)")


if __name__ == "__main__":
    main()
