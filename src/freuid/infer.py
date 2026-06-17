"""Inference → Kaggle submission csv.

    uv run python -m freuid.infer --config configs/baseline.yaml \
        --checkpoint checkpoints/baseline.pt --out submissions/baseline.csv

Submission format (from sample_submission.csv): columns ``id,label`` where label is the
predicted fraud score in [0, 1] (the DET metrics need a continuous score, not a hard 0/1).

This is a code competition: sample_submission.csv lists the FULL test set (~142.8k ids) but
only the public subset (~7.8k) of images ships in the download. We score every id whose image
is present locally and default the rest to 0.0; on Kaggle's grading run all images are present,
so every id gets a real score.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freuid.config import load_config
from freuid.data import FreuidDataset, load_labels
from freuid.models import build_model
from freuid.transforms import build_transforms
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="submissions/submission.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = pick_device()

    model = build_model(cfg.backbone, pretrained=False).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    submission = load_labels(cfg.data_dir, "public_test")  # all ids, label = -1
    present_mask = submission["path"].map(lambda p: Path(p).exists())
    present_ids = set(submission.loc[present_mask, "id"])
    print(f"[infer] {len(submission)} ids total; {len(present_ids)} images present locally")

    ds = FreuidDataset(cfg.data_dir, "public_test", build_transforms(cfg.image_size, False), ids=present_ids)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    scores = predict_scores(model, loader, device)
    id_to_score = dict(zip((s.id for s in ds.samples), scores))

    submission[SCORE_COLUMN] = submission[ID_COLUMN].map(lambda i: id_to_score.get(i, MISSING_DEFAULT))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    submission[[ID_COLUMN, SCORE_COLUMN]].to_csv(args.out, index=False)
    print(f"[infer] wrote {len(submission)} rows -> {args.out} "
          f"({len(id_to_score)} scored, {len(submission) - len(id_to_score)} defaulted)")


if __name__ == "__main__":
    main()
