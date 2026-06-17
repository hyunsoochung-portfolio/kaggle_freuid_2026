"""Inference → Kaggle submission csv.

    uv run python -m freuid.infer --config configs/baseline.yaml \
        --checkpoint checkpoints/baseline.pt --out submissions/baseline.csv

STUB: the exact submission columns are not published yet (full dataset June 2026).
Until then we emit the most likely shape: one row per test image with a continuous
fraud score. Adjust column names to match the official sample_submission.csv.
"""

from __future__ import annotations

import argparse

import pandas as pd

from freuid.config import load_config
from freuid.utils import seed_everything

# TODO(team): confirm against official sample_submission.csv on release.
ID_COLUMN = "image_id"
SCORE_COLUMN = "fraud_score"


def write_submission(ids: list[str], scores: list[float], out_path: str) -> None:
    df = pd.DataFrame({ID_COLUMN: ids, SCORE_COLUMN: scores})
    df.to_csv(out_path, index=False)
    print(f"[infer] wrote {len(df)} rows -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="submissions/submission.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    # TODO(team): load model + checkpoint, run test loader, sigmoid -> fraud score.
    raise NotImplementedError("Inference pending dataset + checkpoint wiring.")


if __name__ == "__main__":
    main()
