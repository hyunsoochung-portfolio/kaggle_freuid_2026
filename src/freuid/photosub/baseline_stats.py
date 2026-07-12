"""Natural intra-card baseline for the MODE_C/D "not a local-statistics outlier" acceptance
criterion.

Eyeballing render sheets doesn't tell you whether a MODE_C/D composite is genuinely
indistinguishable from its surroundings by cheap forensic stats -- it only tells you whether it
LOOKS plausible. This module answers the actual question empirically: how much do
blur_laplacian_var / moire_fft_score / blockiness_score (freuid.photosub.stats) naturally vary
between two random same-sized regions on the SAME untouched bona-fide card, with zero tampering
at all? That natural intra-card spread is the honest yardstick -- a tampered region whose
delta-from-its-own-original-content falls within, say, the 90th percentile of that natural
spread is not distinguishable from ordinary document-to-document (or region-to-region)
variation; a tampered region far outside it IS a local-statistics outlier, no matter how
plausible the render looks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from freuid.photosub.stats import STAT_COLS, region_stats
from freuid.photosub.template_regions import frame_box_from_face

BASELINE_SEED = 7
DEFAULT_N_CARDS = 200


def _random_placement(img_w: int, img_h: int, box_w: int, box_h: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    box_w, box_h = min(box_w, img_w), min(box_h, img_h)
    x1 = int(rng.integers(0, img_w - box_w + 1)) if img_w > box_w else 0
    y1 = int(rng.integers(0, img_h - box_h + 1)) if img_h > box_h else 0
    return (x1, y1, x1 + box_w, y1 + box_h)


def natural_pair_delta(img: Image.Image, box_w: int, box_h: int, rng: np.random.Generator) -> dict:
    """abs delta of the 3 cheap stats between TWO random same-sized regions on the SAME
    untouched card -- how much stats naturally vary within one document, with zero tampering."""
    box1 = _random_placement(img.width, img.height, box_w, box_h, rng)
    box2 = _random_placement(img.width, img.height, box_w, box_h, rng)
    s1 = region_stats(img.crop(box1))
    s2 = region_stats(img.crop(box2))
    return {c: abs(s1[c] - s2[c]) for c in STAT_COLS}


def build_natural_baseline(
    rows: list,  # bona-fide TRAIN rows (e.g. load_labels(...).itertuples()) with .id/.path
    regions_dir_path: str | Path,
    n_cards: int = DEFAULT_N_CARDS,
    seed: int = BASELINE_SEED,
    progress_every: int | None = 50,
) -> pd.DataFrame:
    """One row per sampled card: abs delta of the 3 cheap stats between two random same-sized
    (this card's own face-derived frame_box size) regions. Skips rows without a real cached
    SCRFD detection, same convention as freuid.photosub.donor_pool.build_donor_index."""
    import time

    rng = np.random.default_rng(seed)
    records = []
    last_printed = 0
    t0 = time.monotonic()
    for row in rows:
        if len(records) >= n_cards:
            break
        if progress_every and len(records) >= last_printed + progress_every:
            last_printed = len(records)
            print(f"[natural_baseline] {len(records)}/{n_cards} cards "
                  f"(elapsed {time.monotonic() - t0:.0f}s)")
        face_path = Path(regions_dir_path) / str(row.id) / "face.json"
        if not face_path.exists() or not Path(str(row.path)).exists():
            continue
        try:
            face_box = json.loads(face_path.read_text())
        except Exception:
            continue
        if float(face_box.get("score", 0.0)) <= 0.0:
            continue
        with Image.open(row.path) as img:
            img = img.convert("RGB")
            fb = frame_box_from_face(face_box, img.width, img.height)
            box_w, box_h = fb["x2"] - fb["x1"], fb["y2"] - fb["y1"]
            if box_w < 8 or box_h < 8:
                continue
            delta = natural_pair_delta(img, box_w, box_h, rng)
        delta["id"] = str(row.id)
        records.append(delta)
    if progress_every:
        print(f"[natural_baseline] done: {len(records)} cards in {time.monotonic() - t0:.0f}s")
    return pd.DataFrame(records)


def compute_p90_thresholds(baseline_df: pd.DataFrame) -> dict[str, float]:
    """Per-stat 90th-percentile of the natural intra-card delta distribution -- the pass/fail
    line for MODE_C/D composites."""
    return {c: float(baseline_df[c].quantile(0.90)) for c in STAT_COLS}
