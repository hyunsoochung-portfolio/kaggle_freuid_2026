"""Weighted rank-average ensemble restricted to the 300 hesitant ids only: for those 300,
combine W_V0 * rank(finetune_v0) + W_OVERLAY * rank(overlay_colab), where ranks are computed
within just that 300-id subset (not the full present set). All other ids (present or missing)
keep finetune_v0's original score untouched. A middle ground between the two prior experiments:
- gated_ensemble.py: hard override with overlay_colab's raw score on ~305 hesitant ids -> 0.01101
- weighted_rank_ensemble.py: 0.8/0.2 blend across all 7821 present ids -> 0.05319
This restricts the blend (not override) to just the hesitant 300, to see if a softer combination
narrowly scoped to the known blind spot does better than either prior extreme.

Usage: python scripts/weighted_rank_ensemble_hesitant.py
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
OUT_PATH = REPO_ROOT / "submissions" / "weighted_rank_hesitant300_v0_overlay.csv"

W_V0 = 0.8
W_OVERLAY = 0.2
N_HESITANT = 300


def check_submission(df: pd.DataFrame) -> None:
    scores = df["label"]
    n = len(df)
    n_zeros = int((scores == 0.0).sum())
    print(
        f"[weighted_rank_hesitant] integrity: rows={n} unique_scores={scores.nunique()} "
        f"exact_zeros={n_zeros} ({100.0 * n_zeros / max(n, 1):.2f}%) "
        f"min={scores.min():.6f} max={scores.max():.6f}"
    )
    if n_zeros > 0:
        print(f"[WARNING] {n_zeros} exact-zero score(s) -- investigate before submitting")


def main() -> None:
    v0 = pd.read_csv(V0_PATH, dtype={"id": str})
    overlay = pd.read_csv(OVERLAY_PATH, dtype={"id": str}).rename(columns={"label": "overlay_score"})
    present_ids = set(pd.read_csv(PRESENT_IDS_PATH, dtype={"id": str})["id"])

    merged = v0.merge(overlay, on="id", how="left")
    present = merged[merged["id"].isin(present_ids)].copy()
    present["dist_from_half"] = (present["label"] - 0.5).abs()
    hesitant = present.nsmallest(N_HESITANT, "dist_from_half")
    hesitant_ids = set(hesitant["id"])
    print(f"[weighted_rank_hesitant] {len(v0)} total ids, {len(present_ids)} present, "
          f"{len(hesitant_ids)} hesitant (weights v0={W_V0} overlay={W_OVERLAY})")

    v0_ranks = np.array(_rank_normalize(hesitant["label"].to_numpy()))
    overlay_ranks = np.array(_rank_normalize(hesitant["overlay_score"].to_numpy()))
    combined = W_V0 * v0_ranks + W_OVERLAY * overlay_ranks

    out_scores = merged["label"].to_numpy(copy=True)
    hesitant_mask = merged["id"].isin(hesitant_ids).to_numpy()
    # combined is in hesitant's row order; map back via id to be safe against any reordering
    combined_by_id = dict(zip(hesitant["id"], combined))
    out_scores[hesitant_mask] = [combined_by_id[i] for i in merged.loc[hesitant_mask, "id"]]

    out = pd.DataFrame({"id": merged["id"], "label": out_scores})
    check_submission(out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[weighted_rank_hesitant] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
