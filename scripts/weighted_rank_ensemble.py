"""Weighted rank-average ensemble: W_V0 * finetune_v0 + W_OVERLAY * overlay_colab, combined via
fractional-rank averaging across the whole present public-test set (project convention -- see
CLAUDE.md: "TTA and ensembling combine by rank-averaging, never raw-score averaging"). Uses the
same `_rank_normalize` helper `src/freuid/infer.py` uses for multi-scale TTA, for bit-identical
tie-handling. Missing/placeholder ids (no local image) keep finetune_v0's 0.5 fill untouched --
overlay_colab's file fills those with 0.0, which must never leak into the output.

Usage: python scripts/weighted_rank_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from freuid.infer import _rank_normalize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
V0_PATH = REPO_ROOT / "submissions" / "finetune_v0.csv"
OVERLAY_PATH = REPO_ROOT / "submissions" / "overlay_colab.csv"
PRESENT_IDS_PATH = REPO_ROOT / "reports" / "public_test_scores_all.csv"
OUT_PATH = REPO_ROOT / "submissions" / "weighted_rank_v0_overlay.csv"

W_V0 = 0.8
W_OVERLAY = 0.2


def check_submission(df: pd.DataFrame) -> None:
    scores = df["label"]
    n = len(df)
    n_zeros = int((scores == 0.0).sum())
    print(
        f"[weighted_rank] integrity: rows={n} unique_scores={scores.nunique()} "
        f"exact_zeros={n_zeros} ({100.0 * n_zeros / max(n, 1):.2f}%) "
        f"min={scores.min():.6f} max={scores.max():.6f}"
    )
    if n_zeros > 0:
        print(f"[WARNING] {n_zeros} exact-zero score(s) -- investigate before submitting")


def main() -> None:
    v0 = pd.read_csv(V0_PATH, dtype={"id": str})
    overlay = pd.read_csv(OVERLAY_PATH, dtype={"id": str}).rename(columns={"label": "overlay_score"})
    present_ids = set(pd.read_csv(PRESENT_IDS_PATH, dtype={"id": str})["id"])
    print(f"[weighted_rank] {len(v0)} total ids, {len(present_ids)} present, "
          f"weights v0={W_V0} overlay={W_OVERLAY}")

    merged = v0.merge(overlay, on="id", how="left")
    present_mask = merged["id"].isin(present_ids)
    present = merged.loc[present_mask].reset_index(drop=True)

    v0_ranks = np.array(_rank_normalize(present["label"].to_numpy()))
    overlay_ranks = np.array(_rank_normalize(present["overlay_score"].to_numpy()))
    combined_present = W_V0 * v0_ranks + W_OVERLAY * overlay_ranks

    out_scores = merged["label"].to_numpy(copy=True)
    out_scores[present_mask.to_numpy()] = combined_present
    out = pd.DataFrame({"id": merged["id"], "label": out_scores})

    check_submission(out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[weighted_rank] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
