"""Extract the N public-test images whose rank-averaged fraud score sits closest to 0.5 --
the "hesitation zone" where a rank metric's remaining errors most plausibly live. Excludes
the ~135k code-competition placeholder ids (fixed at exactly extra.missing_id_score, usually
0.5) that have no local image -- those aren't "hesitant", they're just unscored.

Read-only. Requires the submission csv and the public_test image directory to both be
reachable from wherever this runs (i.e. run this on the box that has both).

Usage: python scripts/analysis/hesitant_test_images.py --submission submissions/finetune_v0.csv \
    --data-dir data --n 100 --out reports/hesitant_test_100.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.data import load_labels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    test_meta = load_labels(args.data_dir, "public_test")
    present_mask = test_meta["path"].map(lambda p: Path(p).exists())
    present_ids = set(test_meta.loc[present_mask, "id"])
    print(f"[hesitant] {len(test_meta)} ids total, {len(present_ids)} locally present")

    sub = pd.read_csv(args.submission, dtype={"id": str})
    sub_present = sub[sub["id"].isin(present_ids)].copy()
    print(f"[hesitant] {len(sub_present)} present-id rows loaded from {args.submission}")

    sub_present = sub_present.rename(columns={"label": "score"})
    sub_present["dist_from_half"] = (sub_present["score"] - 0.5).abs()
    top = sub_present.nsmallest(args.n, "dist_from_half")[["id", "score", "dist_from_half"]]
    top = top.sort_values("dist_from_half").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_path, index=False)
    print(f"[hesitant] wrote {len(top)} rows -> {out_path}")
    print(top.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
