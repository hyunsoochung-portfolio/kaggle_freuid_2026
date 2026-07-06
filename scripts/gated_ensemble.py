"""Gated combination: finetune_v0 everywhere, except deferring to overlay_colab (a
BayarConv2d-based forensic-noise + face-region model, feat/overlay-detector branch) on the
narrow band where finetune_v0 is uncertain (score near 0.5).

Motivation (see conversation/CLAUDE.md): overlay_colab's standalone public LB is awful (0.377,
~50x worse than finetune_v0's 0.00744) -- almost certainly because forensic-noise residuals get
erased by print-and-capture, which dominates the test set. But on finetune_v0's ~300 most
uncertain present-test predictions, overlay_colab is decisive essentially 100% of the time
(~91-95% confident-fraud, ~5-9% confident-genuine, ~0% still-unsure in every 100-wide band out
to rank 300) -- a validated, non-decaying pattern, not a top-100 fluke. This is a one-shot
empirical test: if this submission's public LB improves on 0.00744, the combination gets ported
into the actual inference pipeline (src/freuid/infer.py); if not, this stays a one-off csv.

Gate threshold: |finetune_v0_score - 0.5| <= 0.01794623662616024, the exact dist-from-half
boundary at rank 300 among the 7821 present test ids -- i.e. only the empirically-validated zone,
not an extrapolated guess. Missing/placeholder ids (no local image) always keep finetune_v0's
0.5 fill; overlay_colab's file fills those with 0.0, which must NOT leak into the output (that's
exactly the "never 0.0 for missing" invariant this project cares about).

Usage: python scripts/gated_ensemble.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
V0_PATH = REPO_ROOT / "submissions" / "finetune_v0.csv"
OVERLAY_PATH = REPO_ROOT / "submissions" / "overlay_colab.csv"
PRESENT_IDS_PATH = REPO_ROOT / "reports" / "public_test_scores_all.csv"
OUT_PATH = REPO_ROOT / "submissions" / "gated_v0_overlay.csv"

GATE_THRESHOLD = 0.01794623662616024  # dist-from-half boundary at rank 300 (validated zone)


def check_submission(df: pd.DataFrame) -> None:
    scores = df["label"]
    n = len(df)
    n_zeros = int((scores == 0.0).sum())
    print(
        f"[gated_ensemble] integrity: rows={n} unique_scores={scores.nunique()} "
        f"exact_zeros={n_zeros} ({100.0 * n_zeros / max(n, 1):.2f}%) "
        f"min={scores.min():.6f} max={scores.max():.6f}"
    )
    if n_zeros > 0:
        print(f"[WARNING] {n_zeros} exact-zero score(s) -- investigate before submitting")


def main() -> None:
    v0 = pd.read_csv(V0_PATH, dtype={"id": str})
    overlay = pd.read_csv(OVERLAY_PATH, dtype={"id": str}).rename(columns={"label": "overlay_score"})
    present_ids = set(pd.read_csv(PRESENT_IDS_PATH, dtype={"id": str})["id"])
    print(f"[gated_ensemble] {len(v0)} total ids, {len(present_ids)} present")

    merged = v0.merge(overlay, on="id", how="left")
    dist_from_half = (merged["label"] - 0.5).abs()
    is_present = merged["id"].isin(present_ids)
    gated = is_present & (dist_from_half <= GATE_THRESHOLD)
    print(f"[gated_ensemble] gated (present + uncertain): {gated.sum()} ids")

    combined = merged["label"].where(~gated, merged["overlay_score"])
    out = pd.DataFrame({"id": merged["id"], "label": combined})

    check_submission(out)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"[gated_ensemble] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
