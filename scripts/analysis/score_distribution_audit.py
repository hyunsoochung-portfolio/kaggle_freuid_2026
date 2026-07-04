"""Score-distribution audit on the real finetune_v0 Kaggle submission.

Loads submissions/finetune_v0.csv, restricts to the ids with a locally-present test image
(the rest are code-competition placeholders scored at extra.missing_id_score and carry no
information), plots the score histogram, and copies the 30 most "uncertain" images (score
closest to 0.5, the rank-normalized midpoint) to reports/analysis_v0/borderline/ for manual
inspection.

Note: submission scores are RANK-AVERAGED across TTA scales (see infer.py's
predict_scores_tta / _rank_normalize), not raw sigmoid probabilities -- "closest to 0.5"
here means closest to the middle of the rank distribution, not necessarily the model's most
uncertain raw logit, though the two are highly correlated in practice.

Usage: python scripts/analysis/score_distribution_audit.py [--checkpoint PATH] [--submission PATH]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_CHECKPOINT, REPO_ROOT, ensure_report_dir, load_checkpoint  # noqa: E402
from freuid.data import load_labels  # noqa: E402

DEFAULT_SUBMISSION = REPO_ROOT / "submissions" / "finetune_v0.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--submission", default=None)
    parser.add_argument("--n-borderline", type=int, default=30)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    sub_path = Path(args.submission) if args.submission else DEFAULT_SUBMISSION

    cfg, _ = load_checkpoint(ckpt_path)
    test_meta = load_labels(cfg.data_dir, "public_test")
    present_mask = test_meta["path"].map(lambda p: Path(p).exists())
    present_ids = set(test_meta.loc[present_mask, "id"])
    print(f"[score_audit] {len(test_meta)} test ids total, {len(present_ids)} locally present")

    sub = pd.read_csv(sub_path, dtype={"id": str})
    sub_present = sub[sub["id"].isin(present_ids)].merge(
        test_meta[["id", "path"]], on="id", how="left")
    print(f"[score_audit] {len(sub_present)} present-id rows loaded from {sub_path}")

    out_dir = ensure_report_dir()
    sub_present[["id", "label", "path"]].rename(columns={"label": "score"}).to_csv(
        out_dir / "present_scores.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(sub_present["label"], bins=60, color="#4477AA", edgecolor="white", linewidth=0.3)
        ax.axvline(0.5, color="#CC3311", linestyle="--", linewidth=1, label="rank midpoint (0.5)")
        ax.set_xlabel("rank-averaged fraud score")
        ax.set_ylabel("count")
        ax.set_title(f"finetune_v0 submission score distribution (n={len(sub_present)} present ids)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "score_distribution.png", dpi=150)
        print(f"[score_audit] wrote {out_dir / 'score_distribution.png'}")
    except ImportError:
        print("[score_audit] matplotlib not available -- skipped plot")

    sub_present = sub_present.copy()
    sub_present["dist_from_half"] = (sub_present["label"] - 0.5).abs()
    borderline = sub_present.nsmallest(args.n_borderline, "dist_from_half")
    borderline[["id", "label", "dist_from_half", "path"]].rename(columns={"label": "score"}).to_csv(
        out_dir / "borderline_ids.csv", index=False)

    border_dir = out_dir / "borderline"
    border_dir.mkdir(parents=True, exist_ok=True)
    for row in borderline.itertuples(index=False):
        src = Path(row.path)
        if src.exists():
            shutil.copy(src, border_dir / f"{row.id}_score{row.label:.4f}.jpeg")
    print(f"[score_audit] copied {len(borderline)} borderline images -> {border_dir}")
    print(f"[score_audit] wrote {out_dir / 'borderline_ids.csv'}, present_scores.csv")


if __name__ == "__main__":
    main()
