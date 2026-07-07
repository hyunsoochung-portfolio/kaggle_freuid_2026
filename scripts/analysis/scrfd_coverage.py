"""Audit how often SCRFD actually detects a face vs. falls back, for the regions cache that
`bayar_dinov2_v0`'s overlay branch depends on.

`freuid.preprocess.detect_face_box` always writes a face.json -- even on failure -- via a
center-square fallback with `score: 0.0` (see preprocess.py). That score field is the ONLY
signal distinguishing a real SCRFD detection from the fallback; face_crop_image itself doesn't
check it (it crops whatever box is on disk, real or fallback), so a silently-all-fallback cache
would look identical to a healthy one unless someone explicitly reads `score`. CLAUDE.md cites
~21.5% (train) / ~16.4% (public_test) real-detection rates from an earlier one-off check with no
saved script -- this reproduces that number and goes further:

  1. real-detection rate (score > 0) per split, overall and by document `type` and `is_digital`
     -- tells us whether misses are a systematic technical failure (SCRFD/insightface never
     loading -- rate would be ~0% everywhere) or a content-driven pattern (some doc types
     legitimately have no visible portrait, or non-digital images are harder to detect faces on).
  2. score distribution for real detections (not just the >0/==0 split) -- a pile of very-low
     nonzero scores would suggest low-confidence, possibly-wrong detections rather than a clean
     signal.
  3. box-size sanity: a "detected" box covering most of the 512x512 rectified card is more likely
     a false positive (matched some other card feature) than a genuine tight face crop -- flags
     boxes above an area-fraction threshold for manual review.

Read-only: opens face.json files already on disk, does not run SCRFD, does not touch training
code or checkpoints.

Usage: python scripts/analysis/scrfd_coverage.py --data-dir data [--splits train public_test]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.data import load_labels  # noqa: E402
from freuid.preprocess import regions_dir  # noqa: E402

CARD_SIZE = 512  # freuid.preprocess._CARD_SIZE
LARGE_BOX_AREA_FRACTION = 0.5  # flag boxes covering >50% of the card as likely false positives


def audit_split(data_dir: str, split: str) -> pd.DataFrame:
    meta = load_labels(data_dir, split)
    cache_root = regions_dir(data_dir)

    rows = []
    for r in meta.itertuples(index=False):
        face_path = cache_root / str(r.id) / "face.json"
        if not face_path.exists():
            rows.append({"id": r.id, "cached": False, "score": None, "area_frac": None})
            continue
        try:
            fb = json.loads(face_path.read_text())
        except Exception:
            rows.append({"id": r.id, "cached": False, "score": None, "area_frac": None})
            continue
        area = max(0, fb["x2"] - fb["x1"]) * max(0, fb["y2"] - fb["y1"])
        rows.append({
            "id": r.id,
            "cached": True,
            "score": float(fb.get("score", 0.0)),
            "area_frac": area / (CARD_SIZE * CARD_SIZE),
            "type": getattr(r, "type", None),
            "is_digital": getattr(r, "is_digital", None),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, split: str) -> None:
    n = len(df)
    n_cached = int(df["cached"].sum())
    print(f"\n=== {split}: {n} ids, {n_cached} with a cached face.json "
          f"({n - n_cached} not yet precached) ===")
    if n_cached == 0:
        return

    cached = df[df["cached"]].copy()
    real = cached[cached["score"] > 0.0]
    fallback = cached[cached["score"] <= 0.0]
    print(f"real SCRFD detection (score>0): {len(real)}/{n_cached} ({100 * len(real) / n_cached:.1f}%)")
    print(f"center-square fallback (score==0): {len(fallback)}/{n_cached} ({100 * len(fallback) / n_cached:.1f}%)")

    if len(real):
        print(f"real-detection score distribution: "
              f"min={real['score'].min():.3f} p25={real['score'].quantile(.25):.3f} "
              f"median={real['score'].median():.3f} p75={real['score'].quantile(.75):.3f} "
              f"max={real['score'].max():.3f}")
        large = real[real["area_frac"] > LARGE_BOX_AREA_FRACTION]
        print(f"real detections with box area > {int(LARGE_BOX_AREA_FRACTION * 100)}% of card "
              f"(likely false positives, not a tight face crop): {len(large)}/{len(real)} "
              f"({100 * len(large) / max(1, len(real)):.1f}%)")

    if "type" in cached.columns and cached["type"].notna().any():
        rate_by_type = (
            cached.assign(real=cached["score"] > 0.0)
            .groupby("type")["real"].agg(["mean", "count"])
            .sort_values("count", ascending=False)
        )
        print("\nreal-detection rate by document type (top 15 by volume):")
        print(rate_by_type.head(15).to_string(float_format=lambda x: f"{x:.3f}"))

    if "is_digital" in cached.columns and cached["is_digital"].notna().any():
        rate_by_digital = (
            cached.assign(real=cached["score"] > 0.0)
            .groupby("is_digital")["real"].agg(["mean", "count"])
        )
        print("\nreal-detection rate by is_digital:")
        print(rate_by_digital.to_string(float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--splits", nargs="+", default=["train", "public_test"])
    parser.add_argument("--out", default="reports/scrfd_coverage.csv")
    args = parser.parse_args()

    all_rows = []
    for split in args.splits:
        df = audit_split(args.data_dir, split)
        df["split"] = split
        summarize(df, split)
        all_rows.append(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(all_rows, ignore_index=True).to_csv(out_path, index=False)
    print(f"\n[scrfd_coverage] wrote per-id detail -> {out_path}")


if __name__ == "__main__":
    main()
