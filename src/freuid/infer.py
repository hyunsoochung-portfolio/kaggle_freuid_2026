"""Inference -> Kaggle submission csv.

    uv run python -m freuid.infer --checkpoint checkpoints/baseline.pt \
        --out submissions/baseline.csv

Model-defining params (``backbone``, ``image_size``) are read from the checkpoint itself
so preprocessing + architecture always match the trained weights -- no way to silently
mismatch them. ``--config`` is optional and only supplies runtime/environment params; the
CLI flags ``--data-dir`` / ``--batch-size`` / ``--num-workers`` override those per machine.

Submission format (from sample_submission.csv): columns ``id,label`` where label is the
predicted fraud score in [0, 1] (the DET metrics need a continuous score, not a hard 0/1).

This is a code competition: sample_submission.csv lists the FULL test set (~142.8k ids) but
only the public subset (~7.8k) of images ships in the download. We score every id whose image
is present locally and default the rest to ``extra.missing_id_score`` (default 0.5 -- never
0.0, which would silently tank AuDET if any genuinely-absent id happens to be fraud). On
Kaggle's grading run all images are present so every id gets a real score.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freuid.config import Config, load_config
from freuid.data import FreuidDataset, load_labels
from freuid.models import build_model
from freuid.transforms import build_transforms, resolve_data_config
from freuid.utils import pick_device, seed_everything

ID_COLUMN = "id"
SCORE_COLUMN = "label"
_MISSING_FALLBACK = 0.5  # module-level fallback; overridden by extra.missing_id_score


@torch.no_grad()
def predict_scores(model, loader, device) -> list[float]:
    """Fraud scores in dataset order (loader must be shuffle=False)."""
    scores: list[float] = []
    for imgs, _ in tqdm(loader, leave=False):
        logits = model(imgs.to(device))
        scores.extend(torch.sigmoid(logits).squeeze(1).cpu().tolist())
    return scores


def _rank_normalize(arr) -> list[float]:
    """Convert a score array to fractional ranks in (0, 1).

    Uses average rank for ties. Result is in (0, 1) — never exactly 0.0
    (guarding the submission integrity check).
    """
    import numpy as np
    a = np.asarray(arr, dtype=np.float64)
    n = len(a)
    if n == 0:
        return []
    order = np.argsort(a, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    # average ties: find runs of equal values and replace their ranks with the mean
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j > i + 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j
    # map to (epsilon, 1-epsilon) so no exact zeros reach the integrity check
    lo, hi = ranks.min(), ranks.max()
    if hi > lo:
        ranks = (ranks - lo) / (hi - lo) * (1 - 2e-7) + 1e-7
    return ranks.tolist()


def predict_scores_tta(
    model,
    device,
    data_dir: str,
    present_ids: set,
    batch_size: int,
    num_workers: int,
    mean,
    std,
    scales: list[int],
    regions_dir: Path | None = None,
) -> list[tuple[str, float]]:
    """Multi-scale TTA: run inference at each scale, rank-average, return (id, score) pairs.

    No horizontal flip — documents carry orientation.
    Rank-averaging is used instead of score-averaging because AuDET is a rank
    metric; averaging ranks is invariant to per-scale score calibration differences.
    """
    per_scale_scores: list[list[float]] = []
    sample_ids: list[str] | None = None

    for scale in scales:
        tf = build_transforms(scale, False, mean, std)
        ds = FreuidDataset(data_dir, "public_test", tf, ids=present_ids, regions_dir=regions_dir)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        if sample_ids is None:
            sample_ids = [s.id for s in ds.samples]
        scores = predict_scores(model, loader, device)
        per_scale_scores.append(scores)
        print(f"[tta] scale={scale}  scores: min={min(scores):.4f} max={max(scores):.4f}")

    # Rank-average across scales
    import numpy as np
    n = len(per_scale_scores[0])
    avg_ranks = np.zeros(n, dtype=np.float64)
    for scores in per_scale_scores:
        ranked = _rank_normalize(scores)
        avg_ranks += np.array(ranked)
    avg_ranks /= len(per_scale_scores)

    return list(zip(sample_ids or [], avg_ranks.tolist()))


def check_submission(path: str | Path) -> None:
    """Print an integrity report for a finished submission CSV.

    Catches the most dangerous silent failure: exact-zero scores for missing ids
    (a 0.0 fraud score on a genuine fraud sample collapses AuDET).
    """
    df = pd.read_csv(path)
    scores = df[SCORE_COLUMN]
    n = len(df)
    n_zeros = int((scores == 0.0).sum())
    pct_zeros = 100.0 * n_zeros / max(n, 1)
    print(
        f"[infer] integrity: rows={n} unique_scores={scores.nunique()} "
        f"exact_zeros={n_zeros} ({pct_zeros:.2f}%) "
        f"min={scores.min():.6f} max={scores.max():.6f}"
    )
    if n_zeros > 0:
        print(
            f"[WARNING] {n_zeros} exact-zero score(s) in submission -- "
            "if any of those ids are fraud, AuDET will be severely penalised."
        )


def resolve_config(args) -> tuple[Config, dict]:
    """Build the run config and return it alongside the loaded checkpoint.

    ``backbone`` and ``image_size`` always come from the checkpoint's stored config so the
    model matches its weights. ``--config`` (optional) seeds the rest; CLI flags override
    runtime/environment params (data dir, batch size, workers).
    """
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = state.get("config", {})

    if args.config:
        cfg = load_config(args.config)
    elif ckpt_cfg:
        known = {f.name for f in fields(Config)}
        cfg = Config(**{k: v for k, v in ckpt_cfg.items() if k in known})
    else:
        raise SystemExit("checkpoint has no stored config -- pass --config explicitly")

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

    cfg, state = resolve_config(args)
    seed_everything(cfg.seed)
    device = pick_device()

    # backbone and image_size are sourced from the checkpoint -- logged here for audit.
    print(
        f"[infer] backbone={cfg.backbone} image_size={cfg.image_size} (from checkpoint) | "
        f"device={device} | data_dir={cfg.data_dir}"
    )

    # model_type dispatch: add new model types here (e.g. model_type="consistency")
    model_type = cfg.extra.get("model_type", "baseline")
    if model_type == "consistency":
        from freuid.models import build_consistency_model
        model = build_consistency_model(cfg).to(device)
    else:
        model = build_model(cfg.backbone, pretrained=False).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # Full test-id list from sample_submission.csv; score all present, fill the rest.
    missing_score = cfg.extra.get("missing_id_score", _MISSING_FALLBACK)
    submission = load_labels(cfg.data_dir, "public_test")
    present_mask = submission["path"].map(lambda p: Path(p).exists())
    present_ids = set(submission.loc[present_mask, ID_COLUMN])
    n_missing = len(submission) - len(present_ids)
    print(
        f"[infer] {len(submission)} ids total | "
        f"{len(present_ids)} images present | "
        f"{n_missing} missing (will be scored {missing_score})"
    )
    if n_missing > 0:
        print(
            f"[WARNING] {n_missing} test id(s) have no local image -- "
            f"writing missing_id_score={missing_score} for those rows. "
            "On Kaggle's grading server all images are present; this is expected locally."
        )

    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    base_size = data_cfg["image_size"]
    mean, std = data_cfg["mean"], data_cfg["std"]

    # Regions cache: used when extra.use_rectify=True (card-rectification path).
    _rdir: Path | None = None
    if cfg.extra.get("use_rectify", False):
        from freuid.preprocess import regions_dir as _get_rdir
        _rdir = _get_rdir(cfg.data_dir)
        if not _rdir.exists():
            print(f"[infer] WARNING: use_rectify=True but cache not found at {_rdir}; using raw images")
            _rdir = None
        else:
            print(f"[infer] use_rectify=True → loading from {_rdir}")

    tta_cfg = cfg.extra.get("tta", False)
    if tta_cfg:
        if isinstance(tta_cfg, list):
            tta_scales = [int(s) for s in tta_cfg]
        else:
            # default: ±64px around the trained resolution (stays divisible by 32)
            step = max(32, (base_size // 6) & ~31)
            tta_scales = [base_size - step, base_size, base_size + step]
        print(f"[tta] scales={tta_scales}")
        id_score_pairs = predict_scores_tta(
            model, device,
            data_dir=cfg.data_dir,
            present_ids=present_ids,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            mean=mean, std=std,
            scales=tta_scales,
            regions_dir=_rdir,
        )
        id_to_score: dict[str, float] = dict(id_score_pairs)
    else:
        transform = build_transforms(base_size, False, mean, std)
        ds = FreuidDataset(cfg.data_dir, "public_test", transform, ids=present_ids, regions_dir=_rdir)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
        scores = predict_scores(model, loader, device)
        id_to_score = dict(zip((s.id for s in ds.samples), scores, strict=True))

    submission[SCORE_COLUMN] = submission[ID_COLUMN].map(
        lambda i: id_to_score.get(i, missing_score)
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission[[ID_COLUMN, SCORE_COLUMN]].to_csv(out_path, index=False)
    print(f"[infer] wrote {len(submission)} rows -> {out_path}")

    check_submission(out_path)


if __name__ == "__main__":
    main()
