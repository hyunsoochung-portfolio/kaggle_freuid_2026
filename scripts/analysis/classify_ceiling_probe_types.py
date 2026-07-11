"""Fills in `doc_type_proxy` for `data/probes/ceiling_frauds_sample_ids.csv` (200 ids, all
NaN as shipped -- unlike deep/boundary/clean_floor, which already carry it from earlier manual
review passes). Needed for the per-template ceiling-budget breakdown in
`src/freuid/photosub/probes.py`'s `run_probe_hooks` (each template's contribution to the
bona-fide budget).

Reuses `movement_census.py`'s color-histogram classifier verbatim (NOT the embedding-KNN one --
see that module's docstring for why embedding-based classification is unreliable specifically in
the saturated ceiling zone: mean pairwise cosine similarity ~0.999, confirmed via
`hesitant_clusters.embedding_collapse_diagnostic`). Color histograms don't touch the model's
representation at all, so they aren't subject to that collapse.

Read-only against data/, writes only to the probe CSV. Run once; the result is frozen into the
CSV like `logit`/`stratum` already are, not recomputed per training epoch.

Usage: python scripts/analysis/classify_ceiling_probe_types.py [--data-root data/raw]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from movement_census import build_color_hist_reference, color_hist_classify  # noqa: E402

from freuid.data import load_labels  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_CSV = REPO_ROOT / "data" / "probes" / "ceiling_frauds_sample_ids.csv"


def resolve_data_root(data_root: Path) -> Path:
    if (data_root / "train" / "train").is_dir():
        return data_root
    if (data_root / "raw" / "train" / "train").is_dir():
        return data_root / "raw"
    raise SystemExit(f"could not find train/train under {data_root} or {data_root / 'raw'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = parser.parse_args()

    data_dir = resolve_data_root(Path(args.data_root))
    df = pd.read_csv(PROBE_CSV, dtype={"id": str})
    print(f"[classify_ceiling] {len(df)} ceiling probe ids, "
          f"{df['doc_type_proxy'].notna().sum()} already typed")

    test_meta = load_labels(str(data_dir), "public_test").set_index("id")
    missing = [i for i in df["id"] if i not in test_meta.index]
    if missing:
        raise SystemExit(f"{len(missing)} probe ids not found in public_test, e.g. {missing[:3]!r}")
    paths = [test_meta.loc[i, "path"] for i in df["id"]]

    print("[classify_ceiling] building color-histogram reference from TRAIN...")
    centroids = build_color_hist_reference(str(data_dir))
    preds, margins = color_hist_classify(paths, centroids)

    df["doc_type_proxy"] = preds
    df["type_proxy_margin"] = margins
    df.to_csv(PROBE_CSV, index=False)
    print(f"[classify_ceiling] wrote {PROBE_CSV}")
    print(df["doc_type_proxy"].value_counts())
    print(f"[classify_ceiling] margin stats: min={min(margins):.3f} "
          f"median={sorted(margins)[len(margins)//2]:.3f}")


if __name__ == "__main__":
    main()
