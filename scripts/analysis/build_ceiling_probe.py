"""Freezes a `ceiling_frauds_sample_ids.csv` probe: ~200 public-test ids the finetune_v0
checkpoint already scores deep in the ceiling mode (see logit_census_report.md: ceiling mode
at raw logit ~12.502, block covers ~48% of the public-test corpus).

Motivation (CLAUDE.md's photosub_v0 pre-registered gates): a photo-substitution fix must not
regress the model's EXISTING detection capability while it improves the deep/boundary misses.
There is no ground truth for public_test, so this is a self-consistency regression guard, not
an accuracy check against labels: these ids are the model's own most-confident fraud calls, and
the gate is "the retrained checkpoint's mean logit here should stay saturated near the ceiling
mode, not silently give ground back to fix the misses elsewhere."

Pure pandas over the already-computed scripts/analysis/logit_census_raw.csv -- no GPU, no
checkpoint needed. Sampling is seeded (SEED=2, distinct from freeze_probes' SURVEY_SEED=1 and
deep_miss_dossiers' MODE_ASSIGNMENTS provenance) for reproducibility.

``doc_type_proxy`` is a CLASSIFIER OUTPUT, not a ground-truth label: public_test's own ``type``
column is None for all 142,818 rows (document type is never revealed for the test split -- see
CLAUDE.md's "document types not seen in training" risk), so a naive merge against
freuid.data.load_labels' ``type`` column silently produces 100% NaN (a real bug this file used to
have -- caught by a photosub_v1 smoke-test run finding train.py's per-template ceiling-exposure
line never printed). Reuses movement_census.py's ``color_hist_classify`` /
``build_color_hist_reference`` (CPU-only, no checkpoint, nearest-centroid over a 5-known-type RGB
histogram built from labeled TRAIN images) -- the same working proxy classifier movement_census
Part 2 already validated for this exact "classify a ceiling-zone public_test id with no ground
truth" problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from movement_census import build_color_hist_reference, color_hist_classify  # noqa: E402

from freuid.data import load_labels  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CENSUS_CSV = Path(__file__).resolve().parent / "logit_census_raw.csv"
DEFAULT_OUT_CSV = REPO_ROOT / "data" / "probes" / "ceiling_frauds_sample_ids.csv"
DEFAULT_DATA_DIR = REPO_ROOT / "data"

CEILING_MODE = 12.502  # scripts/analysis/logit_census_report.md
CEILING_TOL = 0.5      # same tolerance as that report's headline ceiling-block figure
N_SAMPLE = 200
SEED = 2


def build(
    census_csv: Path = DEFAULT_CENSUS_CSV,
    out_csv: Path = DEFAULT_OUT_CSV,
    data_dir: Path = DEFAULT_DATA_DIR,
    n_sample: int = N_SAMPLE,
    seed: int = SEED,
) -> pd.DataFrame:
    census = pd.read_csv(census_csv, dtype={"id": str})
    in_block = census[(census["mean_logit"] - CEILING_MODE).abs() <= CEILING_TOL]
    if len(in_block) < n_sample:
        raise RuntimeError(
            f"ceiling block only has {len(in_block)} ids at tol={CEILING_TOL}, need {n_sample}"
        )
    rng = np.random.default_rng(seed)
    sample = in_block.sample(n=n_sample, random_state=rng.integers(2**31 - 1)).copy()

    test_meta = load_labels(data_dir, "public_test")[["id", "path"]]
    sample = sample.merge(test_meta, on="id", how="left")
    have_path = sample["path"].notna() & sample["path"].map(lambda p: Path(p).exists())
    centroids = build_color_hist_reference(data_dir)
    preds, _margins = color_hist_classify(sample.loc[have_path, "path"].tolist(), centroids)
    sample["doc_type_proxy"] = None
    sample.loc[have_path, "doc_type_proxy"] = preds
    sample = sample.rename(columns={"mean_logit": "logit"})
    sample["stratum"] = "CEILING"
    out = sample[["id", "stratum", "logit", "doc_type_proxy"]].sort_values("logit", ascending=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[build_ceiling_probe] wrote {len(out)} ids -> {out_csv}")
    return out


if __name__ == "__main__":
    build()
