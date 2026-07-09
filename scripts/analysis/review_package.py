"""Manual-review package for measuring bona-fide contamination of finetune_v0's ceiling block.

Working theory under test (see logit_census_report.md, occlusion_v2_report.md): the ceiling
block (48.3% of the public-test corpus at tolerance=0.5, per the full-corpus logit census) is
not pure fraud -- some meaningful fraction is bona-fide cards carrying innocent local
disruptions (glare, a finger, dirt, physical damage) that the checkpoint's `any-anomaly`
shortcut (confirmed in occlusion_test_v2.py) mistakes for tamper evidence. This script builds
the human-review package needed to measure that contamination BY EYE -- no ML verdicts here,
only sorting, rendering, tallying, and arithmetic.

Two stages:

  --stage build    Samples BOTTOM/MIDDLE/TOP strata (150 ids each, uniform within stratum) from
                    inside the ceiling block, plus FLOOR/TRANSITIONAL calibration sets (100 ids
                    each), sorts each stratum's sheet order by PIXEL-space template similarity
                    (greedy nearest-neighbor walk over downscaled thumbnails -- explicitly NOT
                    model embeddings, which hesitant_clusters.py already found collapsed and
                    non-discriminative for this near-median population), renders 5x5 grid sheets
                    + a per-stratum "most common templates" summary sheet, and emits a blank
                    review CSV (id, stratum, sheet, cell, logit, verdict, anomaly_type) for
                    manual filling. Needs the regions cache (SCRFD boxes) -> VESSL.

  --stage ingest    Takes the FILLED review CSV, computes per-stratum contamination rates
                    ((A+B)/judgeable, with Wilson score confidence intervals), extrapolates to
                    the full ceiling block (states the uniform-sampling assumption explicitly),
                    and combines with an AuDET-improvement-bound calculation (mirroring
                    hesitant_clusters.py's `audet_bound_table`, but for the OPPOSITE failure
                    direction -- see `audet_bound_ceiling_contamination`'s docstring for why the
                    formula differs) to estimate recoverable AuDET if the A-category ids were
                    correctly ranked below true frauds. Prints the pre-registered CONFIRMED
                    (>~10%) / REFUTED (<~3%) / in-between decision rule. Pure CSV arithmetic --
                    no GPU, no checkpoint, can run anywhere.

No training, no submissions, no modification of existing source files or the regions cache.

Usage:
    python scripts/analysis/review_package.py --stage build --data-dir data
    # ... manually fill the verdict/anomaly_type columns in the emitted CSV ...
    python scripts/analysis/review_package.py --stage ingest --filled-csv <your filled csv>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.data import load_labels  # noqa: E402
from freuid.preprocess import regions_dir  # noqa: E402
from freuid.transforms import build_transforms, resolve_data_config  # noqa: E402
from freuid.utils import pick_device  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_CHECKPOINT, build_finetuned_model, df_to_md, load_checkpoint  # noqa: E402
from hesitant_clusters import (  # noqa: E402
    KNOWN_TYPES,
    build_type_reference,
    embed_paths,
    pixel_stats,
    type_proxy_knn,
)
from logit_census import find_modes  # noqa: E402
from occlusion_test import hesitant_ranked_ids, read_face_box  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGIT_CENSUS_CSV = Path(__file__).resolve().parent / "logit_census_raw.csv"
DEFAULT_HESITANT_CSV = Path(__file__).resolve().parent / "hesitant_results.csv"
DEFAULT_SUBMISSION_CSV = REPO_ROOT / "submissions" / "finetune_v0.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "review_package_out"
DEFAULT_REVIEW_CSV = Path(__file__).resolve().parent / "review_package.csv"
DEFAULT_META_JSON = Path(__file__).resolve().parent / "review_package_meta.json"
DEFAULT_BUILD_REPORT = Path(__file__).resolve().parent / "review_package_report.md"
DEFAULT_INGEST_REPORT = Path(__file__).resolve().parent / "review_package_ingest_report.md"

DEFAULT_DEEP_OUT_DIR = Path(__file__).resolve().parent / "review_package_deep_out"
DEFAULT_DEEP_CSV = Path(__file__).resolve().parent / "review_package_deep.csv"
DEFAULT_DEEP_META_JSON = Path(__file__).resolve().parent / "review_package_deep_meta.json"
DEFAULT_DEEP_REPORT = Path(__file__).resolve().parent / "review_package_deep_report.md"

CEILING_STRATA = ("BOTTOM", "MIDDLE", "TOP")
CALIBRATION_GROUPS = ("FLOOR", "TRANSITIONAL")
ALL_GROUPS = CEILING_STRATA + CALIBRATION_GROUPS
VERDICT_LEVELS = ("F", "A", "B", "U", "?")
CONTAMINATION_LEVELS = ("A", "B")
JUDGEABLE_LEVELS = ("F", "A", "B")

# Deep-review (transitional-zone + floor-deep) groups: 5 equal-count transitional strata
# (TRANS_1 = lowest logit / nearest floor, TRANS_5 = highest logit / nearest ceiling) plus two
# floor-deep subgroups. Sheets are rendered highest-logit-first, i.e. reversed order below.
N_TRANSITIONAL_STRATA = 5
TRANSITIONAL_STRATA = tuple(f"TRANS_{i}" for i in range(1, N_TRANSITIONAL_STRATA + 1))
FLOOR_DEEP_GROUPS = ("FLOOR_DEEP_TOP", "FLOOR_DEEP_REST")
ALL_DEEP_GROUPS = TRANSITIONAL_STRATA + FLOOR_DEEP_GROUPS


# ---------------------------------------------------------------------------
# Ceiling block / stratification (build stage)
# ---------------------------------------------------------------------------

def load_census_with_modes(logit_census_csv: Path) -> tuple[pd.DataFrame, float, float]:
    df = pd.read_csv(logit_census_csv, dtype={"id": str})
    if "mean_logit" not in df.columns:
        scale_cols = [c for c in df.columns if c.startswith("logit_")]
        df["mean_logit"] = df[scale_cols].mean(axis=1)
    modes = find_modes(df["mean_logit"].to_numpy())
    if len(modes) < 2:
        raise SystemExit(
            "logit_census_raw.csv did not yield a bimodal split -- cannot define the ceiling "
            "block. Re-check logit_census_report.md's own verdict before proceeding."
        )
    floor_mode, ceiling_mode = modes[0][0], modes[-1][0]
    return df, floor_mode, ceiling_mode


def block_masks(df: pd.DataFrame, floor_mode: float, ceiling_mode: float, tol: float):
    ceiling_mask = (df["mean_logit"] - ceiling_mode).abs() <= tol
    floor_mask = (df["mean_logit"] - floor_mode).abs() <= tol
    neither_mask = ~ceiling_mask & ~floor_mask
    return ceiling_mask, floor_mask, neither_mask


def stratify_ceiling_block(
    ceiling_df: pd.DataFrame, n_per_stratum: int, seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Rank ids by logit within the ceiling block, split into three contiguous rank-thirds
    (BOTTOM = lowest logits in the block, i.e. least-extreme -- where the hesitant-500 live;
    TOP = most extreme), then sample n_per_stratum uniformly at random WITHIN each third
    (not just take the extremes of each third)."""
    sorted_df = ceiling_df.sort_values("mean_logit", kind="stable").reset_index(drop=True)
    n = len(sorted_df)
    third = n // 3
    pools = {
        "BOTTOM": sorted_df.iloc[:third],
        "MIDDLE": sorted_df.iloc[third:2 * third],
        "TOP": sorted_df.iloc[2 * third:],
    }
    rng = np.random.default_rng(seed)
    samples, pool_sizes = {}, {}
    for name, pool in pools.items():
        pool_sizes[name] = len(pool)
        if len(pool) <= n_per_stratum:
            samples[name] = pool.copy()
        else:
            idx = np.sort(rng.choice(len(pool), size=n_per_stratum, replace=False))
            samples[name] = pool.iloc[idx].copy()
    return samples, pool_sizes


def sample_uniform(df_subset: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df_subset) <= n:
        return df_subset.copy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df_subset), size=n, replace=False))
    return df_subset.iloc[idx].copy()


def _exclude(df_subset: pd.DataFrame, exclude_ids: set[str] | None) -> pd.DataFrame:
    if not exclude_ids:
        return df_subset
    return df_subset[~df_subset["id"].isin(exclude_ids)]


def stratify_transitional_by_logit(
    transitional_df: pd.DataFrame, n_strata: int, n_per_stratum: int, seed: int,
    exclude_ids: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Split the transitional zone into `n_strata` EQUAL-COUNT contiguous bands by raw logit
    (TRANS_1 = lowest logit / nearest the floor mode, TRANS_{n_strata} = highest logit / nearest
    the ceiling mode), then sample n_per_stratum uniformly at random WITHIN each band.

    Strata boundaries are computed from the FULL transitional pool before any exclusion, so a
    second-pass re-render (exclude_ids set) keeps the same band definitions as the first pass --
    only the sampling pool within a band shrinks.
    """
    sorted_df = transitional_df.sort_values("mean_logit", kind="stable").reset_index(drop=True)
    n = len(sorted_df)
    edges = np.linspace(0, n, n_strata + 1).astype(int)
    rng = np.random.default_rng(seed)
    samples, meta = {}, {}
    for i in range(n_strata):
        name = f"TRANS_{i + 1}"
        band = sorted_df.iloc[edges[i]:edges[i + 1]]
        pool = _exclude(band, exclude_ids)
        meta[name] = {
            "logit_min": float(band["mean_logit"].min()) if len(band) else float("nan"),
            "logit_max": float(band["mean_logit"].max()) if len(band) else float("nan"),
            "band_total_count": int(len(band)),
            "pool_after_exclude": int(len(pool)),
        }
        if len(pool) <= n_per_stratum:
            samples[name] = pool.copy()
        else:
            idx = np.sort(rng.choice(len(pool), size=n_per_stratum, replace=False))
            samples[name] = pool.iloc[idx].copy()
    return samples, meta


def sample_floor_deep(
    floor_df: pd.DataFrame, n_top: int, n_rest: int, seed: int,
    exclude_ids: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """5 equal-count quintiles of the floor block by raw logit; TOP quintile = least-negative
    logit (closest to the transitional boundary -- where a fraud that fell all the way to the
    floor would most plausibly land). Samples n_top uniformly from the top quintile and n_rest
    uniformly from the union of the other 4 quintiles."""
    sorted_df = floor_df.sort_values("mean_logit", kind="stable").reset_index(drop=True)
    n = len(sorted_df)
    edges = np.linspace(0, n, 6).astype(int)  # 5 quintiles
    top_quintile = sorted_df.iloc[edges[4]:edges[5]]
    rest = sorted_df.iloc[:edges[4]]

    rng = np.random.default_rng(seed)
    samples, meta = {}, {}
    for name, pool_full, n_want in (
        ("FLOOR_DEEP_TOP", top_quintile, n_top), ("FLOOR_DEEP_REST", rest, n_rest),
    ):
        pool = _exclude(pool_full, exclude_ids)
        meta[name] = {
            "logit_min": float(pool_full["mean_logit"].min()) if len(pool_full) else float("nan"),
            "logit_max": float(pool_full["mean_logit"].max()) if len(pool_full) else float("nan"),
            "pool_total_count": int(len(pool_full)),
            "pool_after_exclude": int(len(pool)),
        }
        if len(pool) <= n_want:
            samples[name] = pool.copy()
        else:
            idx = np.sort(rng.choice(len(pool), size=n_want, replace=False))
            samples[name] = pool.iloc[idx].copy()
    return samples, meta


def compute_pct_rank(df: pd.DataFrame, submission_csv: Path, data_dir: str) -> pd.DataFrame:
    """Merge in each id's final-submission rank percentile, restricted to PRESENT ids only (the
    full submission file is dominated by missing_id_score=0.5 placeholder rows for undownloaded
    private-test ids -- ranking against those would silently understate/misplace every
    percentile, the same bug already found and fixed in logit_census.py)."""
    test_meta = load_labels(data_dir, "public_test")
    present_mask = test_meta["path"].map(lambda p: Path(p).exists())
    present_ids = set(test_meta.loc[present_mask, "id"])
    sub = pd.read_csv(submission_csv, dtype={"id": str})
    sub_present = sub[sub["id"].isin(present_ids)].copy()
    sub_present["pct_rank"] = sub_present["label"].rank(pct=True) * 100.0
    return df.merge(sub_present[["id", "pct_rank"]], on="id", how="left")


# ---------------------------------------------------------------------------
# Pixel-space (NOT embedding-space) similarity: greedy walk + template clusters
# ---------------------------------------------------------------------------

def downscale_vector(path: Path, size: int = 32) -> np.ndarray:
    """Zero-mean, L2-normalized flattened grayscale thumbnail -- a cheap, embedding-free
    similarity feature. Zero-mean + L2-norm makes cosine similarity between two vectors
    invariant to overall brightness/contrast offsets, so it groups by LAYOUT/PATTERN rather
    than by lighting."""
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float64).flatten()
    arr = arr - arr.mean()
    norm = np.linalg.norm(arr)
    if norm > 1e-8:
        arr = arr / norm
    return arr


def build_thumbnail_vectors(paths: list[Path], size: int = 32) -> np.ndarray:
    return np.stack([downscale_vector(p, size) for p in paths]) if paths else np.zeros((0, size * size))


def greedy_nn_order(vectors: np.ndarray) -> list[int]:
    """Greedy nearest-neighbor tour over cosine similarity (vectors are zero-mean, L2-normalized
    -> dot product = cosine similarity): start at index 0, repeatedly append the most-similar
    not-yet-visited vector. Produces a sheet order where visually similar templates cluster
    together, purely from pixel statistics -- no model embeddings."""
    n = len(vectors)
    if n <= 1:
        return list(range(n))
    sims = vectors @ vectors.T
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = order[-1]
        row = sims[last].copy()
        row[visited] = -np.inf
        nxt = int(np.argmax(row))
        order.append(nxt)
        visited[nxt] = True
    return order


def greedy_template_clusters(vectors: np.ndarray, order: list[int], sim_threshold: float) -> list[dict]:
    """Online greedy clustering in the given (nn-walk) order: assign each vector to the nearest
    existing cluster centroid if similarity >= sim_threshold, else start a new cluster. Rough
    and pixel-only by design (not ML) -- purely to surface "these N cards share a visual layout"
    for the summary sheet."""
    clusters: list[dict] = []
    for i in order:
        v = vectors[i]
        best_idx, best_sim = -1, -np.inf
        for ci, c in enumerate(clusters):
            sim = float(np.dot(v, c["centroid"]))
            if sim > best_sim:
                best_sim, best_idx = sim, ci
        if best_idx >= 0 and best_sim >= sim_threshold:
            clusters[best_idx]["indices"].append(i)
            idxs = clusters[best_idx]["indices"]
            new_centroid = vectors[idxs].mean(axis=0)
            norm = np.linalg.norm(new_centroid)
            clusters[best_idx]["centroid"] = new_centroid / norm if norm > 1e-8 else new_centroid
        else:
            clusters.append({"centroid": v.copy(), "indices": [i]})
    return clusters


def top_template_representatives(vectors: np.ndarray, order: list[int], sim_threshold: float, top_k: int = 15):
    clusters = greedy_template_clusters(vectors, order, sim_threshold)
    clusters.sort(key=lambda c: len(c["indices"]), reverse=True)
    reps = []
    for c in clusters[:top_k]:
        idxs = c["indices"]
        sub = vectors[idxs]
        sims_to_centroid = sub @ c["centroid"]
        medoid = idxs[int(np.argmax(sims_to_centroid))]
        reps.append({"medoid_idx": medoid, "count": len(idxs)})
    return reps


# ---------------------------------------------------------------------------
# Deep-review feature computation (embeddings, type proxy, degradation stats) -- reuses
# hesitant_clusters.py's machinery rather than duplicating it. GPU/checkpoint required.
# ---------------------------------------------------------------------------

def load_hesitant_probe_embeddings(
    hesitant_csv: Path, submission_csv: Path, data_dir: str, model, device, transform, n: int = 500,
) -> np.ndarray:
    """The frozen hesitant-500 probe set's L2-normalized penultimate embeddings -- prefers the
    already-materialized hesitant_results.csv id list (exact match to hesitant_clusters.py's own
    run); falls back to occlusion_test.hesitant_ranked_ids' fresh top-k-by-|score-0.5| selection
    if that artifact isn't present."""
    if hesitant_csv.exists():
        hes = pd.read_csv(hesitant_csv, dtype={"id": str})
        ids = hes.loc[hes["set"] == "HESITANT", "id"].tolist()
    else:
        print(f"[review_package] {hesitant_csv} not found -- falling back to a fresh hesitant_ranked_ids() selection")
        ids = hesitant_ranked_ids(submission_csv, data_dir, n)
    test_meta = load_labels(data_dir, "public_test").set_index("id")
    paths = [Path(test_meta.loc[i, "path"]) for i in ids if i in test_meta.index]
    print(f"[review_package] embedding {len(paths)} frozen hesitant-probe ids")
    return embed_paths(model, device, paths, transform)


def compute_deep_features(
    df: pd.DataFrame, cfg, model, device, transform, rdir: Path, probe_embs: np.ndarray,
    n_ref_per_type: int = 40, seed: int = 42,
) -> pd.DataFrame:
    """Adds: face_score (SCRFD), blur/moire/blockiness/min_side (cheap pixel stats), type_proxy
    + type_proxy_similarity (5-NN vs a labeled-train reference set), and hesitant_cosine_mean
    (mean cosine similarity of this id's own penultimate embedding against the frozen
    hesitant-500 probe set -- answers whether the embedding-collapse finding from
    hesitant_report.md extends into the transitional zone)."""
    stat_rows = []
    for row in df.itertuples():
        pil_img = Image.open(row.path).convert("RGB")
        stats = pixel_stats(pil_img)
        fb = read_face_box(rdir, row.id)
        stats["face_score"] = float(fb.get("score", 0.0)) if fb is not None else 0.0
        stats["id"] = row.id
        stat_rows.append(stats)
    df = df.merge(pd.DataFrame(stat_rows), on="id")

    embs = embed_paths(model, device, [Path(p) for p in df["path"]], transform)
    ref_embs, ref_types = build_type_reference(cfg, model, device, transform, n_ref_per_type, seed)
    type_proxy, type_proxy_sim = type_proxy_knn(embs, ref_embs, ref_types, k=5)
    df["type_proxy"] = type_proxy
    df["type_proxy_similarity"] = type_proxy_sim
    df["hesitant_cosine_mean"] = (embs @ probe_embs.T).mean(axis=1) if len(probe_embs) else np.nan
    return df


# ---------------------------------------------------------------------------
# Rendering (PIL composite cells; no matplotlib for cells -- need multi-panel cells)
# ---------------------------------------------------------------------------

def _load_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_cell(
    row, rdir: Path, cell_width: int = 340, thumb_height: int = 220, caption_height: int = 60,
    zoom: float = 2.0, show_face_crop: bool = True,
) -> Image.Image:
    """One review-grid cell: full card (with SCRFD box drawn), optionally alongside a 2x zoom
    face crop (show_face_crop=False renders the card alone, full cell width), with an id/logit/
    stratum caption below. Falls back to a gray 'NO FACE' placeholder crop when there's no valid
    SCRFD detection for this id (only relevant when show_face_crop=True)."""
    img = Image.open(row["path"]).convert("RGB")
    w, h = img.size
    fb = read_face_box(rdir, row["id"])
    has_face = fb is not None and float(fb.get("score", 0.0)) > 0.0

    card = img.copy()
    if has_face:
        draw = ImageDraw.Draw(card)
        box = (int(fb["x1"]), int(fb["y1"]), int(fb["x2"]), int(fb["y2"]))
        draw.rectangle(box, outline=(0, 255, 0), width=max(2, w // 250))

    card_w = (cell_width // 2 - 4) if show_face_crop else cell_width
    # Letterbox (never crop) -- these are photos of a card, not the card itself, and are often
    # portrait-oriented with the card only filling part of the frame; a vertical center-crop
    # would silently hide real content (tested and confirmed: it truncated card text at the top
    # in several real images during dry-run review).
    scale = min(card_w / w, thumb_height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    card_resized = card.resize((new_w, new_h))
    card_thumb = Image.new("RGB", (card_w, thumb_height), (30, 30, 30))
    card_thumb.paste(card_resized, ((card_w - new_w) // 2, (thumb_height - new_h) // 2))

    crop_thumb = None
    if show_face_crop:
        half_w = card_w
        if has_face:
            x1, y1, x2, y2 = int(fb["x1"]), int(fb["y1"]), int(fb["x2"]), int(fb["y2"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop = img.crop((x1, y1, x2, y2))
                zoom_w, zoom_h = int(crop.width * zoom), int(crop.height * zoom)
                crop_thumb = crop.resize((max(1, zoom_w), max(1, zoom_h)))

        if crop_thumb is None:
            crop_thumb = Image.new("RGB", (half_w, thumb_height), (90, 90, 90))
            d = ImageDraw.Draw(crop_thumb)
            d.text((10, thumb_height // 2 - 8), "NO FACE", fill=(255, 255, 255))
        else:
            # letterbox into the same half_w x thumb_height cell
            scale = min(half_w / crop_thumb.width, thumb_height / crop_thumb.height)
            new_w, new_h = max(1, int(crop_thumb.width * scale)), max(1, int(crop_thumb.height * scale))
            crop_thumb = crop_thumb.resize((new_w, new_h))
            pad = Image.new("RGB", (half_w, thumb_height), (30, 30, 30))
            pad.paste(crop_thumb, ((half_w - new_w) // 2, (thumb_height - new_h) // 2))
            crop_thumb = pad

    cell = Image.new("RGB", (cell_width, thumb_height + caption_height), (255, 255, 255))
    cell.paste(card_thumb, (0, 0))
    if crop_thumb is not None:
        cell.paste(crop_thumb, (card_w + 8, 0))
    draw = ImageDraw.Draw(cell)
    font = _load_font(13)
    caption = f"{row['id'][:10]}\nlogit={row['mean_logit']:.2f}  {row['stratum']}"
    if "pct_rank" in row.index and pd.notna(row["pct_rank"]):
        caption += f"\npct={row['pct_rank']:.1f}"
    draw.multiline_text((4, thumb_height + 4), caption, fill=(0, 0, 0), font=font)
    return cell


def render_sheet(cells: list[Image.Image], out_path: Path, ncols: int = 5, nrows: int = 5) -> None:
    if not cells:
        return
    cw, ch = cells[0].width, cells[0].height
    canvas = Image.new("RGB", (ncols * cw, nrows * ch), (255, 255, 255))
    for i, cell in enumerate(cells[:ncols * nrows]):
        r, c = divmod(i, ncols)
        canvas.paste(cell, (c * cw, r * ch))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render_template_summary(
    group_df: pd.DataFrame, vectors: np.ndarray, order: list[int], sim_threshold: float,
    out_path: Path, thumb_size: int = 160, top_k: int = 15,
) -> list[dict]:
    reps = top_template_representatives(vectors, order, sim_threshold, top_k)
    thumbs = []
    for r in reps:
        row = group_df.iloc[r["medoid_idx"]]
        img = Image.open(row["path"]).convert("RGB").resize((thumb_size, thumb_size))
        canvas = Image.new("RGB", (thumb_size, thumb_size + 24), (255, 255, 255))
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((4, thumb_size + 2), f"n={r['count']}", fill=(0, 0, 0), font=_load_font(13))
        thumbs.append(canvas)
    if thumbs:
        strip = Image.new("RGB", (thumb_size * len(thumbs), thumb_size + 24), (255, 255, 255))
        for i, t in enumerate(thumbs):
            strip.paste(t, (i * thumb_size, 0))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        strip.save(out_path)
    return reps


# ---------------------------------------------------------------------------
# Build stage
# ---------------------------------------------------------------------------

def build(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    census, floor_mode, ceiling_mode = load_census_with_modes(Path(args.logit_census_csv))
    # logit_census_raw.csv has no image path column -- merge it in from the public_test labels.
    test_meta = load_labels(args.data_dir, "public_test")[["id", "path"]]
    census = census.merge(test_meta, on="id", how="inner")
    ceiling_mask, floor_mask, neither_mask = block_masks(census, floor_mode, ceiling_mode, args.tol)
    ceiling_df, floor_df, transitional_df = census[ceiling_mask], census[floor_mask], census[neither_mask]
    print(
        f"[review_package] ceiling_mode={ceiling_mode:.3f} floor_mode={floor_mode:.3f} "
        f"| block sizes: ceiling={len(ceiling_df)} floor={len(floor_df)} transitional={len(transitional_df)} "
        f"(of {len(census)} total)"
    )

    strata, pool_sizes = stratify_ceiling_block(ceiling_df, args.n_per_stratum, args.seed)
    floor_sample = sample_uniform(floor_df, args.n_calibration, args.seed)
    transitional_sample = sample_uniform(transitional_df, args.n_calibration, args.seed)
    groups = {**strata, "FLOOR": floor_sample, "TRANSITIONAL": transitional_sample}
    for name in ALL_GROUPS:
        groups[name] = groups[name].assign(stratum=name)

    # Overlap check against the hesitant-500.
    hesitant_csv = Path(args.hesitant_csv)
    overlap_note = "hesitant_results.csv not found -- skipped overlap check."
    if hesitant_csv.exists():
        hes = pd.read_csv(hesitant_csv, dtype={"id": str})
        hesitant_ids = set(hes.loc[hes["set"] == "HESITANT", "id"])
        bottom_ids = set(groups["BOTTOM"]["id"])
        block_ids = set(ceiling_df["id"])
        overlap_bottom = hesitant_ids & bottom_ids
        overlap_block = hesitant_ids & block_ids
        overlap_note = (
            f"hesitant-500 overlap: {len(overlap_block)}/{len(hesitant_ids)} "
            f"({len(overlap_block) / max(1, len(hesitant_ids)) * 100:.1f}%) of the hesitant-500 "
            f"sit anywhere in the ceiling block; {len(overlap_bottom)}/{len(hesitant_ids)} "
            f"({len(overlap_bottom) / max(1, len(hesitant_ids)) * 100:.1f}%) specifically land in "
            "this run's BOTTOM stratum sample (a 150-id sample of the block's own bottom third, "
            "not the full bottom third, so this is a lower bound on true overlap with that third)."
        )
    print(f"[review_package] {overlap_note}")

    # Regions cache is required for face boxes.
    rdir = regions_dir(args.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this needs VESSL")

    review_rows = []
    template_summaries = {}
    for name in ALL_GROUPS:
        g = groups[name].reset_index(drop=True)
        if len(g) == 0:
            print(f"[review_package] WARNING: {name} has 0 ids -- skipping")
            continue
        vectors = build_thumbnail_vectors([Path(p) for p in g["path"]], size=args.thumb_size)
        order = greedy_nn_order(vectors)

        cells, positions = [], []
        for rank, idx in enumerate(order):
            row = g.iloc[idx]
            cell = render_cell(row, rdir, cell_width=args.cell_width)
            cells.append(cell)
            sheet = rank // 25 + 1
            cell_num = rank % 25 + 1
            positions.append((row["id"], sheet, cell_num, float(row["mean_logit"])))

        n_sheets = (len(cells) + 24) // 25
        for s in range(n_sheets):
            sheet_cells = cells[s * 25:(s + 1) * 25]
            render_sheet(sheet_cells, out_dir / f"{name}_sheet{s + 1:02d}.png")
        print(f"[review_package] {name}: {len(g)} ids -> {n_sheets} sheet(s)")

        reps = render_template_summary(
            g, vectors, order, args.template_sim_threshold, out_dir / f"{name}_templates.png",
        )
        template_summaries[name] = reps

        for id_, sheet, cell_num, logit in positions:
            review_rows.append({
                "id": id_, "stratum": name, "sheet": sheet, "cell": cell_num,
                "logit": logit, "verdict": "", "anomaly_type": "",
            })

    review_df = pd.DataFrame(review_rows)
    review_csv_path = Path(args.csv_out)
    review_df.to_csv(review_csv_path, index=False)
    print(f"[review_package] wrote blank review csv -> {review_csv_path} ({len(review_df)} rows)")

    meta = {
        "ceiling_mode": ceiling_mode, "floor_mode": floor_mode, "tol": args.tol,
        "block_size": len(ceiling_df), "n_present": len(census),
        "stratum_pool_sizes": pool_sizes,
        "n_per_stratum": args.n_per_stratum, "n_calibration": args.n_calibration,
        "seed": args.seed, "current_audet": args.current_audet,
        "assumed_fraud_rate": args.assumed_fraud_rate,
    }
    Path(args.meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[review_package] wrote metadata -> {args.meta_out}")

    write_build_report(
        Path(args.report), meta, pool_sizes, overlap_note, template_summaries, groups, out_dir,
    )


def _relpath_or_abs(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_build_report(
    report_path: Path, meta: dict, pool_sizes: dict, overlap_note: str,
    template_summaries: dict, groups: dict, out_dir: Path,
) -> None:
    lines = ["# Review-package build report\n"]
    lines.append(
        "Sampling + rendering log for the ceiling-block contamination review package. This is "
        "NOT a verdict report -- no images have been manually reviewed yet. Fill in "
        f"`{DEFAULT_REVIEW_CSV.name}`'s `verdict`/`anomaly_type` columns, then run "
        "`--stage ingest` to get the actual contamination estimate.\n"
    )
    lines.append("## Ceiling block definition\n")
    lines.append(
        f"- Ceiling mode: **{meta['ceiling_mode']:.3f}**, floor mode: **{meta['floor_mode']:.3f}** "
        f"(re-derived from `logit_census_raw.csv` via the same KDE mode-finder as "
        "`logit_census.py`/`occlusion_test_v2.py` -- not hardcoded)."
    )
    lines.append(
        f"- Tolerance: {meta['tol']}. Ceiling block size: **{meta['block_size']}** of "
        f"{meta['n_present']} present public-test ids."
    )
    lines.append(
        "- Ceiling-block rank-thirds (full pool sizes, before sampling down to "
        f"{meta['n_per_stratum']}/stratum): "
        + ", ".join(f"{k}={v}" for k, v in pool_sizes.items())
    )
    lines.append(f"- {overlap_note}\n")

    lines.append("## Groups sampled\n")
    rows = [{"group": name, "n": len(g)} for name, g in groups.items()]
    lines.append(df_to_md(pd.DataFrame(rows), float_fmt="{:.0f}"))
    lines.append("")

    lines.append("## Template summary sheets (most common visual layouts per group)\n")
    lines.append(
        "Pixel-only greedy clustering on downscaled (zero-mean, L2-normalized) grayscale "
        "thumbnails -- NOT model embeddings (those are collapsed/non-discriminative for this "
        "population per `hesitant_report.md`). Approximate by construction; a visual aid, not a "
        "rigorous grouping.\n"
    )
    for name, reps in template_summaries.items():
        img_path = out_dir / f"{name}_templates.png"
        lines.append(f"**{name}** ({len(reps)} template groups shown, sizes: {[r['count'] for r in reps]}):")
        lines.append(f"![{name} templates]({_relpath_or_abs(img_path)})\n")

    lines.append("## Review sheets\n")
    for name in groups:
        n_sheets = (len(groups[name]) + 24) // 25
        paths = [_relpath_or_abs(out_dir / f"{name}_sheet{s + 1:02d}.png") for s in range(n_sheets)]
        lines.append(f"**{name}** ({n_sheets} sheets): " + ", ".join(paths))
    lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[review_package] wrote build report -> {report_path}")


# ---------------------------------------------------------------------------
# Aggregate figures (deep-review stage)
# ---------------------------------------------------------------------------

def plot_transitional_histogram(
    transitional_df: pd.DataFrame, floor_mode: float, ceiling_mode: float, out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(transitional_df["mean_logit"], bins=60, color="steelblue", alpha=0.85)
    ax.axvline(floor_mode, color="tab:blue", linestyle="--", label=f"floor mode ({floor_mode:.2f})")
    ax.axvline(ceiling_mode, color="tab:red", linestyle="--", label=f"ceiling mode ({ceiling_mode:.2f})")
    ax.set_xlabel("mean_logit")
    ax.set_ylabel("count")
    ax.set_title(f"Transitional zone logit distribution (n={len(transitional_df)})")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_doctype_composition(group_type_proxy: dict[str, pd.Series], out_path: Path) -> None:
    """Grouped bar chart of type_proxy fractional composition across zones. `type_proxy` is a
    5-NN majority vote against only the 5 known training types (see hesitant_clusters.py) -- it
    structurally cannot say 'unseen type', so a group dominated by one label here is a hint, not
    a ground-truth composition. That caveat is stamped directly on the figure."""
    groups = list(group_type_proxy.keys())
    fracs = {t: [group_type_proxy[g].value_counts(normalize=True).get(t, 0.0) for g in groups] for t in KNOWN_TYPES}

    x = np.arange(len(groups))
    width = 0.8 / len(KNOWN_TYPES)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, t in enumerate(KNOWN_TYPES):
        ax.bar(x + i * width, fracs[t], width, label=t)
    ax.set_xticks(x + width * (len(KNOWN_TYPES) - 1) / 2)
    ax.set_xticklabels(groups)
    ax.set_ylabel("fraction of group (type_proxy)")
    ax.set_title("Document-type-proxy composition by zone")
    ax.legend(fontsize=8)
    fig.text(
        0.5, -0.02,
        "CAVEAT: type_proxy is a 5-NN vote against only 5 known training types -- it cannot "
        "recognize a genuinely unseen document type; treat as a hint, not ground truth.",
        ha="center", fontsize=8, style="italic", wrap=True,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Deep-review build stage (TRANSITIONAL stratified sample + FLOOR_DEEP)
# ---------------------------------------------------------------------------

def build_deep(args) -> None:
    out_dir = Path(args.deep_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    census, floor_mode, ceiling_mode = load_census_with_modes(Path(args.logit_census_csv))
    test_meta = load_labels(args.data_dir, "public_test")[["id", "path"]]
    census = census.merge(test_meta, on="id", how="inner")
    census = compute_pct_rank(census, Path(args.submission_csv), args.data_dir)
    ceiling_mask, floor_mask, neither_mask = block_masks(census, floor_mode, ceiling_mode, args.tol)
    ceiling_df, floor_df, transitional_df = census[ceiling_mask], census[floor_mask], census[neither_mask]
    print(
        f"[review_package] deep stage: ceiling_mode={ceiling_mode:.3f} floor_mode={floor_mode:.3f} "
        f"| block sizes: ceiling={len(ceiling_df)} floor={len(floor_df)} transitional={len(transitional_df)}"
    )

    requested = set(args.strata.split(",")) if args.strata and args.strata != "ALL" else set(ALL_DEEP_GROUPS)
    unknown_requested = requested - set(ALL_DEEP_GROUPS)
    if unknown_requested:
        raise SystemExit(f"--strata contains unknown group name(s): {unknown_requested} (valid: {ALL_DEEP_GROUPS})")

    exclude_ids: set[str] = set()
    for p in (args.exclude_csv.split(",") if args.exclude_csv else []):
        p = p.strip()
        if not p:
            continue
        prior = pd.read_csv(p, dtype={"id": str})
        exclude_ids |= set(prior["id"])
    if exclude_ids:
        print(f"[review_package] excluding {len(exclude_ids)} already-sampled ids (from --exclude-csv)")

    trans_samples, trans_meta = stratify_transitional_by_logit(
        transitional_df, N_TRANSITIONAL_STRATA, args.n_per_trans_stratum, args.seed, exclude_ids,
    )
    floor_samples, floor_meta = sample_floor_deep(
        floor_df, args.n_floor_top, args.n_floor_rest, args.seed, exclude_ids,
    )
    groups = {**trans_samples, **floor_samples}
    groups = {name: g for name, g in groups.items() if name in requested}
    for name in list(groups):
        groups[name] = groups[name].assign(stratum=name)
        if len(groups[name]) == 0:
            print(f"[review_package] WARNING: {name} sampled 0 ids (empty pool after exclusion?) -- skipping")
            del groups[name]

    # --- GPU-dependent feature computation ---
    ckpt_path = args.checkpoint or DEFAULT_CHECKPOINT
    cfg, state = load_checkpoint(ckpt_path)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    device = pick_device()
    model = build_finetuned_model(cfg, state, device)
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    transform = build_transforms(data_cfg["image_size"], False, data_cfg["mean"], data_cfg["std"])

    rdir = regions_dir(args.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this needs VESSL")

    t0 = time.time()
    probe_embs = load_hesitant_probe_embeddings(
        Path(args.hesitant_csv), Path(args.submission_csv), args.data_dir, model, device, transform,
    )
    print(f"[review_package] hesitant-probe embeddings ready ({time.time() - t0:.1f}s)")

    review_rows = []
    for name in ALL_DEEP_GROUPS:
        if name not in groups:
            continue
        g = groups[name].reset_index(drop=True)
        g = compute_deep_features(g, cfg, model, device, transform, rdir, probe_embs, args.n_ref_per_type, args.seed)
        groups[name] = g
        print(f"[review_package] {name}: {len(g)} ids, features computed ({time.time() - t0:.1f}s elapsed)")

        if name.startswith("TRANS_"):
            # Logit-descending order, NO pixel-similarity sort (per spec -- a top-down truncated
            # or pixel-clustered order would obscure which end of the band an id sits in).
            order = list(g.sort_values("mean_logit", ascending=False).index)
        else:
            vectors = build_thumbnail_vectors([Path(p) for p in g["path"]], size=args.thumb_size)
            order = greedy_nn_order(vectors)

        cells = []
        for rank, idx in enumerate(order):
            row = g.iloc[idx]
            cells.append(render_cell(row, rdir, cell_width=args.cell_width, show_face_crop=not args.card_only))
            sheet, cell_num = rank // 25 + 1, rank % 25 + 1
            review_rows.append({
                "id": row["id"], "stratum": name, "sheet": sheet, "cell": cell_num,
                "logit": float(row["mean_logit"]), "pct_rank": float(row["pct_rank"]) if pd.notna(row["pct_rank"]) else None,
                "doc_type_proxy": row["type_proxy"], "type_proxy_similarity": float(row["type_proxy_similarity"]),
                "face_score": float(row["face_score"]), "blur_laplacian_var": float(row["blur_laplacian_var"]),
                "moire_fft_score": float(row["moire_fft_score"]), "blockiness_score": float(row["blockiness_score"]),
                "min_side_px": float(row["min_side_px"]), "hesitant_cosine_mean": float(row["hesitant_cosine_mean"]),
                "verdict": "", "note": "",
            })

        n_sheets = (len(cells) + 24) // 25
        for s in range(n_sheets):
            render_sheet(cells[s * 25:(s + 1) * 25], out_dir / f"{name}_sheet{s + 1:02d}.png")
        print(f"[review_package] {name}: -> {n_sheets} sheet(s)")

    review_df = pd.DataFrame(review_rows)
    review_df.to_csv(Path(args.deep_csv_out), index=False)
    print(f"[review_package] wrote deep review csv -> {args.deep_csv_out} ({len(review_df)} rows)")

    # --- Aggregate figures ---
    hist_path = out_dir / "transitional_histogram.png"
    plot_transitional_histogram(transitional_df, floor_mode, ceiling_mode, hist_path)

    group_type_proxy = {}
    trans_groups = [groups[n] for n in groups if n.startswith("TRANS_")]
    if trans_groups:
        group_type_proxy["TRANSITIONAL"] = pd.concat(trans_groups, ignore_index=True)["type_proxy"]

    prior_csv = Path(args.prior_review_csv)
    if prior_csv.exists():
        prior = pd.read_csv(prior_csv, dtype={"id": str})
        n_recomputed = 0
        for name in ("BOTTOM", "MIDDLE", "TOP", "FLOOR"):
            ids = prior.loc[prior["stratum"] == name, "id"].tolist()
            if not ids:
                continue
            sub = census[census["id"].isin(ids)].copy()
            sub = compute_deep_features(sub, cfg, model, device, transform, rdir, probe_embs, args.n_ref_per_type, args.seed)
            group_type_proxy[name] = sub["type_proxy"]
            n_recomputed += len(sub)
        print(f"[review_package] recomputed type_proxy for {n_recomputed} prior-sampled ids (from {prior_csv})")
    else:
        print(f"[review_package] {prior_csv} not found -- doc-type composition chart will only show TRANSITIONAL")

    composition_path = out_dir / "doctype_composition.png"
    if group_type_proxy:
        plot_doctype_composition(group_type_proxy, composition_path)
    else:
        composition_path = None

    meta = {
        "ceiling_mode": ceiling_mode, "floor_mode": floor_mode, "tol": args.tol,
        "transitional_size": len(transitional_df), "floor_size": len(floor_df),
        "transitional_strata_meta": trans_meta, "floor_deep_meta": floor_meta,
        "n_per_trans_stratum": args.n_per_trans_stratum,
        "n_floor_top": args.n_floor_top, "n_floor_rest": args.n_floor_rest,
        "seed": args.seed, "requested_strata": sorted(requested),
        "n_excluded": len(exclude_ids),
    }
    Path(args.deep_meta_out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[review_package] wrote deep metadata -> {args.deep_meta_out}")

    write_deep_build_report(
        Path(args.deep_report), meta, groups, hist_path, composition_path, out_dir,
    )


def write_deep_build_report(
    report_path: Path, meta: dict, groups: dict, hist_path: Path, composition_path: Path | None,
    out_dir: Path,
) -> None:
    lines = ["# Review-package DEEP build report -- transitional zone + floor-deep\n"]
    lines.append(
        "Follow-up to `review_package_report.md`: the ceiling block review (see "
        "`review_package_ingest_report.md`) found the ceiling block is ~99% genuine fraud, not "
        "bona-fide contamination -- so the AuDET-relevant errors must live in low-ranked frauds "
        "hiding in the TRANSITIONAL zone or the FLOOR. This is NOT a verdict report -- fill in "
        f"`{Path(DEFAULT_DEEP_CSV).name}`'s `verdict`/`note` columns next.\n"
    )
    lines.append("## Sampling definitions\n")
    lines.append(
        f"- Ceiling/floor modes: {meta['ceiling_mode']:.3f} / {meta['floor_mode']:.3f} "
        f"(tol={meta['tol']}). Transitional zone size: {meta['transitional_size']}. "
        f"Floor block size: {meta['floor_size']}."
    )
    lines.append(
        "- TRANSITIONAL: 5 equal-count logit strata (TRANS_1=lowest logit/nearest floor .. "
        f"TRANS_5=highest logit/nearest ceiling), {meta['n_per_trans_stratum']}/stratum, seed={meta['seed']}."
    )
    trans_rows = [{"stratum": s, **meta["transitional_strata_meta"][s]} for s in TRANSITIONAL_STRATA if s in meta["transitional_strata_meta"]]
    lines.append(df_to_md(pd.DataFrame(trans_rows), float_fmt="{:.3f}"))
    lines.append(
        f"\n- FLOOR_DEEP: FLOOR_DEEP_TOP (top logit quintile of the floor block, n={meta['n_floor_top']}) "
        f"+ FLOOR_DEEP_REST (remaining 4 quintiles, n={meta['n_floor_rest']})."
    )
    floor_rows = [{"stratum": s, **meta["floor_deep_meta"][s]} for s in FLOOR_DEEP_GROUPS if s in meta["floor_deep_meta"]]
    lines.append(df_to_md(pd.DataFrame(floor_rows), float_fmt="{:.3f}"))
    if meta["n_excluded"]:
        lines.append(f"\n- Second-pass run: {meta['n_excluded']} previously-sampled ids excluded from the pool.")
    lines.append(f"- Groups actually rendered this run: {meta['requested_strata']}\n")

    lines.append("## Groups sampled\n")
    rows = [{"group": name, "n": len(g)} for name, g in groups.items()]
    lines.append(df_to_md(pd.DataFrame(rows), float_fmt="{:.0f}"))
    lines.append("")

    lines.append("## Aggregate figures\n")
    lines.append(
        "Transitional-zone logit histogram with floor/ceiling mode positions marked:\n"
        f"![transitional histogram]({_relpath_or_abs(hist_path)})\n"
    )
    if composition_path is not None:
        lines.append(
            "Document-type-proxy composition across zones (see figure for the type_proxy "
            "unreliability caveat):\n"
            f"![doctype composition]({_relpath_or_abs(composition_path)})\n"
        )
    else:
        lines.append(
            "Doc-type composition chart skipped -- no prior review CSV found to source "
            "BOTTOM/MIDDLE/TOP/FLOOR ids from.\n"
        )

    lines.append("## Review sheets\n")
    lines.append(
        "TRANSITIONAL sheets are ordered stratum TRANS_5 -> TRANS_1 (highest logit first), "
        "logit-descending within each stratum, pixel-similarity sort OFF. FLOOR_DEEP sheets use "
        "the same pixel-similarity greedy-walk ordering as the ceiling-block groups.\n"
    )
    for name in ALL_DEEP_GROUPS:
        if name not in groups:
            continue
        n_sheets = (len(groups[name]) + 24) // 25
        paths = [_relpath_or_abs(out_dir / f"{name}_sheet{s + 1:02d}.png") for s in range(n_sheets)]
        lines.append(f"**{name}** (n={len(groups[name])}, {n_sheets} sheets): " + ", ".join(paths))
    lines.append("")

    lines.append("## Second-pass hook\n")
    lines.append(
        "To pull a targeted follow-up tranche from specific strata (e.g. if F verdicts "
        "concentrate in TRANS_1/TRANS_2), re-run with `--strata TRANS_1,TRANS_2 "
        f"--exclude-csv {Path(DEFAULT_DEEP_CSV).name} --deep-csv-out review_package_deep_pass2.csv` "
        "-- strata boundaries stay fixed (computed from the full pool before exclusion), only "
        "already-sampled ids are excluded from the new draw.\n"
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[review_package] wrote deep build report -> {report_path}")


# ---------------------------------------------------------------------------
# Re-render stage: re-draw already-sampled deep-review sheets with a different cell layout
# (e.g. card-only) WITHOUT re-sampling or re-running the GPU feature computation. Reads the
# existing deep CSV's own id/stratum/sheet/cell assignments -- the sheet order is unchanged,
# only the per-cell image composition is.
# ---------------------------------------------------------------------------

def rerender_deep(args) -> None:
    filled = pd.read_csv(args.deep_csv_out, dtype={"id": str})
    test_meta = load_labels(args.data_dir, "public_test")[["id", "path"]]
    filled = filled.merge(test_meta, on="id", how="inner")
    # render_cell expects the ceiling-block dataframes' column name (mean_logit); the deep CSV
    # stores the same value under `logit` instead -- alias it rather than touching the CSV schema.
    filled = filled.rename(columns={"logit": "mean_logit"})

    rdir = regions_dir(args.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this needs VESSL")

    out_dir = Path(args.deep_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, g in filled.groupby("stratum"):
        g = g.sort_values(["sheet", "cell"])
        cells = [render_cell(row, rdir, cell_width=args.cell_width, show_face_crop=not args.card_only) for _, row in g.iterrows()]
        n_sheets = (len(cells) + 24) // 25
        for s in range(n_sheets):
            render_sheet(cells[s * 25:(s + 1) * 25], out_dir / f"{name}_sheet{s + 1:02d}.png")
        print(f"[review_package] re-rendered {name}: {len(cells)} cells -> {n_sheets} sheet(s)")


# ---------------------------------------------------------------------------
# Ingest stage
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    # Wilson intervals are mathematically bounded to [0, 1], but floating-point subtraction can
    # leave a -1e-18-scale residue at k=0 that prints as "-0.0000" -- clip it away.
    lo, hi = (center - margin) / denom, (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def stratum_tally(
    filled: pd.DataFrame, stratum: str, target_levels: tuple[str, ...] = CONTAMINATION_LEVELS,
) -> dict:
    """Generic per-stratum verdict tally: `contamination_rate` is the fraction of judgeable
    (F/A/B) verdicts falling in `target_levels` -- (A, B) for the ceiling-block bona-fide-
    contamination question, (F,) for the transitional/floor misranked-fraud question. Same
    Wilson-CI machinery either way; only which verdicts count as the "hit" differs."""
    sub = filled[filled["stratum"] == stratum]
    counts = sub["verdict"].value_counts()
    judgeable = sub[sub["verdict"].isin(JUDGEABLE_LEVELS)]
    n = len(judgeable)
    k_target = int(judgeable["verdict"].isin(target_levels).sum())
    k_a = int((judgeable["verdict"] == "A").sum())
    rate = k_target / n if n else float("nan")
    a_rate = k_a / n if n else float("nan")
    lo, hi = wilson_ci(k_target, n)
    return {
        "stratum": stratum, "n_total": len(sub), "n_judgeable": n,
        "n_F": int(counts.get("F", 0)), "n_A": k_a, "n_B": int(counts.get("B", 0)),
        "n_U": int(counts.get("U", 0)), "n_unsure": int(counts.get("?", 0)),
        "contamination_rate": rate, "ci_lo": lo, "ci_hi": hi, "a_rate": a_rate,
    }


def audet_bound_ceiling_contamination(
    current_audet: float, n_present: int, assumed_fraud_rate: float, k_values: list[float],
) -> pd.DataFrame:
    """Mirror image of hesitant_clusters.py's `audet_bound_table`, for the OPPOSITE failure mode.

    That function models k currently-median-rank FRAUD ids that are truly fraud but misranked --
    each is assumed discordant with ~50% of BONA-FIDE pairs (roughly tied with the whole score
    distribution's median), so removed_discordant = k * 0.5 * n_bonafide.

    This one models k BONA-FIDE ids sitting at the ceiling (the "A" category: bona-fide with an
    innocent anomaly the model mistakes for tamper evidence) -- each of THOSE is interspersed
    with the genuine-fraud cluster at the ceiling, not with the whole distribution's median, so
    for each genuine fraud id it's roughly a coin flip whether this misplaced bona-fide currently
    ranks above (discordant) or below (concordant) it. That gives removed_discordant =
    k * 0.5 * n_fraud -- n_FRAUD, not n_bonafide, because the "other side" of each discordant
    pair here is a fraud id, not a bona-fide one. Getting this backwards would silently reuse a
    formula built for a different mechanism; kept as a separate function rather than a flag on
    the original so the difference is impossible to miss in a diff.
    """
    n_fraud = round(n_present * assumed_fraud_rate)
    n_bonafide = n_present - n_fraud
    total_pairs = n_fraud * n_bonafide
    current_discordant = current_audet * total_pairs

    rows = []
    for k in k_values:
        removed = min(current_discordant, k * 0.5 * n_fraud)
        new_discordant = current_discordant - removed
        new_audet = new_discordant / total_pairs
        improvement_pct = (current_audet - new_audet) / current_audet * 100.0 if current_audet > 0 else float("nan")
        rows.append({"k_ids": k, "new_audet_bound": new_audet, "improvement_pct": improvement_pct})
    df = pd.DataFrame(rows)
    df.attrs["n_fraud"] = n_fraud
    df.attrs["n_bonafide"] = n_bonafide
    df.attrs["total_pairs"] = total_pairs
    df.attrs["current_discordant"] = current_discordant
    return df


def ingest(args) -> None:
    filled = pd.read_csv(args.filled_csv, dtype={"id": str, "verdict": str, "anomaly_type": str})
    filled["verdict"] = filled["verdict"].fillna("").str.strip()
    unknown = set(filled.loc[filled["verdict"] != "", "verdict"]) - set(VERDICT_LEVELS)
    if unknown:
        print(f"[review_package] WARNING: unrecognized verdict values found and ignored in tallies: {unknown}")

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    pool_sizes = meta["stratum_pool_sizes"]

    ceiling_tallies = {s: stratum_tally(filled, s) for s in CEILING_STRATA}
    calibration_tallies = {s: stratum_tally(filled, s) for s in CALIBRATION_GROUPS}

    for s, t in ceiling_tallies.items():
        if t["n_judgeable"] == 0:
            print(f"[review_package] WARNING: {s} has 0 judgeable (F/A/B) verdicts -- did you fill in the CSV?")

    total_pool = sum(pool_sizes.values())
    weights = {s: pool_sizes[s] / total_pool for s in CEILING_STRATA}

    def se_from_ci(t):
        if np.isnan(t["ci_lo"]) or np.isnan(t["ci_hi"]):
            return float("nan")
        return (t["ci_hi"] - t["ci_lo"]) / (2 * 1.96)

    combined_rate = sum(weights[s] * ceiling_tallies[s]["contamination_rate"] for s in CEILING_STRATA)
    combined_a_rate = sum(weights[s] * ceiling_tallies[s]["a_rate"] for s in CEILING_STRATA)
    combined_var = sum((weights[s] ** 2) * (se_from_ci(ceiling_tallies[s]) ** 2) for s in CEILING_STRATA)
    combined_se = float(np.sqrt(combined_var)) if not np.isnan(combined_var) else float("nan")
    # Clip to [0, 1] -- this is a proportion; the normal-approximation variance propagation used
    # here (unlike each stratum's own Wilson interval) doesn't respect that floor/ceiling on its
    # own, and produces a visibly nonsensical negative lower bound when the point estimate is
    # close to 0 (confirmed happens in practice, not just in theory).
    combined_lo = max(0.0, combined_rate - 1.96 * combined_se)
    combined_hi = min(1.0, combined_rate + 1.96 * combined_se)

    block_size = meta["block_size"]
    n_contam_est = block_size * combined_rate
    n_a_est = block_size * combined_a_rate
    n_a_lo = block_size * max(0.0, combined_lo) * (combined_a_rate / combined_rate if combined_rate > 0 else 0)
    n_a_hi = block_size * min(1.0, combined_hi) * (combined_a_rate / combined_rate if combined_rate > 0 else 0)

    k_values = sorted(set([max(1, round(v)) for v in (n_a_lo, n_a_est, n_a_hi)]))
    audet_table = audet_bound_ceiling_contamination(
        args.current_audet, meta["n_present"], args.assumed_fraud_rate, k_values,
    )

    if combined_rate > 0.10:
        verdict = (
            f"**CONFIRMED and sized**: block-wide extrapolated (A+B) contamination "
            f"{combined_rate * 100:.1f}% exceeds the ~10% pre-registered threshold. The "
            "hard-negative-mining strategy (train against innocent-anomaly bona-fide examples "
            "specifically) is worth pursuing at this scale."
        )
    elif combined_rate < 0.03:
        verdict = (
            f"**REFUTED**: block-wide extrapolated (A+B) contamination {combined_rate * 100:.1f}% "
            "is below the ~3% pre-registered threshold. The bona-fide-contamination theory does "
            "not explain a meaningful share of the ceiling block -- re-aim follow-up work at the "
            "TRANSITIONAL/FLOOR populations instead."
        )
    else:
        verdict = (
            f"**In between** ({combined_rate * 100:.1f}%, between the 3%/10% thresholds) -- "
            "reporting this straight per the pre-registered instruction not to force a verdict "
            "here. This is a judgment call weighing effort against the estimated recoverable "
            "AuDET below."
        )

    write_ingest_report(
        Path(args.report_out), ceiling_tallies, calibration_tallies, weights, combined_rate,
        combined_a_rate, combined_lo, combined_hi, block_size, n_contam_est, n_a_est,
        (n_a_lo, n_a_hi), audet_table, verdict, meta, args,
    )


def write_ingest_report(
    report_path: Path, ceiling_tallies: dict, calibration_tallies: dict, weights: dict,
    combined_rate: float, combined_a_rate: float, combined_lo: float, combined_hi: float,
    block_size: int, n_contam_est: float, n_a_est: float, n_a_range: tuple,
    audet_table: pd.DataFrame, verdict: str, meta: dict, args,
) -> None:
    lines = ["# Review-package ingest report -- ceiling-block contamination estimate\n"]

    lines.append("## Per-stratum contamination (A+B / judgeable), Wilson 95% CI\n")
    tally_rows = []
    for s in CEILING_STRATA:
        t = ceiling_tallies[s]
        tally_rows.append({
            "stratum": s, "n_total": t["n_total"], "n_judgeable": t["n_judgeable"],
            "n_F": t["n_F"], "n_A": t["n_A"], "n_B": t["n_B"], "n_U": t["n_U"], "n_unsure": t["n_unsure"],
            "contamination_rate": t["contamination_rate"], "ci_lo": t["ci_lo"], "ci_hi": t["ci_hi"],
        })
    lines.append(df_to_md(pd.DataFrame(tally_rows), float_fmt="{:.4f}"))
    lines.append("")

    lines.append("## Block-wide extrapolation\n")
    lines.append(
        "**Assumption stated explicitly**: each stratum's 150-id sample is uniformly drawn from "
        "its full rank-third of the ceiling block, so the sample's contamination rate is treated "
        "as an unbiased estimate of that third's true rate. The block-wide estimate below "
        "weights each stratum's rate by its full rank-third's population size (not by the "
        f"150-id sample size). Rank-third pool sizes: {meta['stratum_pool_sizes']} "
        f"(weights: {', '.join(f'{s}={w:.3f}' for s, w in weights.items())})."
    )
    lines.append(
        f"\n- Block-wide (A+B) contamination: **{combined_rate * 100:.1f}%** "
        f"(approx. 95% CI: {combined_lo * 100:.1f}%-{combined_hi * 100:.1f}%, propagated from the "
        "per-stratum Wilson intervals assuming independence across strata)."
    )
    lines.append(
        f"- Of that, A-only (bona-fide WITH an innocent anomaly -- the actionable category): "
        f"**{combined_a_rate * 100:.1f}%**."
    )
    lines.append(
        f"- Extrapolated to the full ceiling block ({block_size} ids): "
        f"~{n_contam_est:.0f} contaminated (A+B), ~{n_a_est:.0f} specifically A-category "
        f"(range ~{n_a_range[0]:.0f}-{n_a_range[1]:.0f} from the CI).\n"
    )

    lines.append("## Calibration sets (informational only -- not part of the contamination estimate)\n")
    lines.append(
        "FLOOR should be almost entirely B (clean bona-fide) if your own sense of 'clean' "
        "matches the model's floor pole; TRANSITIONAL is expected to be a genuine mixed bag. If "
        "FLOOR shows meaningful F/A, recalibrate before trusting the ceiling-block numbers above.\n"
    )
    calib_rows = [
        {"group": s, **{k: v for k, v in calibration_tallies[s].items() if k != "stratum"}}
        for s in CALIBRATION_GROUPS
    ]
    lines.append(df_to_md(pd.DataFrame(calib_rows), float_fmt="{:.4f}"))
    lines.append("")

    lines.append("## Estimated recoverable AuDET (A-category only)\n")
    lines.append(
        "Uses `audet_bound_ceiling_contamination` -- the mirror image of "
        "`hesitant_clusters.py`'s `audet_bound_table`, appropriate for bona-fide ids "
        "contaminating the ceiling rather than fraud ids stuck at median rank (see that "
        "function's docstring for why the formula differs: n_fraud, not n_bonafide, in the "
        "removed-discordant-pairs term). Scoped to the A category only -- B (clean bona-fide "
        "scored as fraud, no visible anomaly) isn't something a fix targeting innocent-anomaly "
        "robustness would plausibly resolve, so it's excluded from this specific bound, even "
        "though it counts toward the (A+B) contamination rate and decision rule above."
    )
    lines.append(
        f"\nn_fraud≈{audet_table.attrs['n_fraud']}, n_bonafide≈{audet_table.attrs['n_bonafide']}, "
        f"total pairs≈{audet_table.attrs['total_pairs']:,}, current discordant pairs≈"
        f"{audet_table.attrs['current_discordant']:,.0f}.\n"
    )
    lines.append(df_to_md(audet_table.round(6)))
    lines.append(
        "\nRows correspond to the low/point/high estimates of the A-category count above -- "
        "this is a **bound under best-case assumptions** (each fixed id assumed to move from "
        "fully discordant with the fraud cluster to fully concordant, ids treated "
        "independently), not a prediction.\n"
    )

    lines.append("## Pre-registered decision rule\n")
    lines.append(verdict)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[review_package] wrote ingest report -> {report_path}")
    print(f"[review_package] {verdict}")


# ---------------------------------------------------------------------------
# Deep-review ingest stage -- the OPPOSITE question from the ceiling-block ingest above: not
# "how much bona-fide contamination sits at the ceiling" but "how many genuine frauds are
# hiding, misranked, in the TRANSITIONAL zone or the FLOOR block". Pure CSV arithmetic, no GPU.
# ---------------------------------------------------------------------------

def _deep_stratum_pool_size(meta: dict, stratum: str) -> int:
    if stratum in meta["transitional_strata_meta"]:
        return meta["transitional_strata_meta"][stratum]["band_total_count"]
    return meta["floor_deep_meta"][stratum]["pool_total_count"]


def audet_bound_observed_rank_misrank(
    filled: pd.DataFrame, meta: dict, strata: list[str], current_audet: float,
    n_present: int, assumed_fraud_rate: float,
) -> pd.DataFrame:
    """For each stratum, use each F-verdict id's own OBSERVED pct_rank (not an assumed median)
    to estimate the bona-fide pairs it's currently discordant with (~pct_rank% of n_bonafide),
    then scale the sample's total by pool_size/n_sampled to extrapolate to the full
    stratum/pool. Deliberately NOT a reuse of hesitant_clusters.py's `audet_bound_table` (which
    assumes every flagged id sits at ~the 50th percentile) -- these ids span the full 0-100th
    percentile range (TRANS_5 in particular sits at 95-100th, i.e. already nearly perfectly
    ranked), so that coarser assumption would badly overstate the improvement available from
    TRANS_5 and understate it for the genuinely hidden TRANS_1-4/FLOOR_DEEP cases. Using each
    id's real rank is possible here specifically because pct_rank was precomputed per id at
    build time (the abstract k-flagged-images bound in hesitant_clusters.py had no such data)."""
    n_fraud = round(n_present * assumed_fraud_rate)
    n_bonafide = n_present - n_fraud
    total_pairs = n_fraud * n_bonafide
    current_discordant = current_audet * total_pairs

    rows = []
    total_extrapolated_discordant = 0.0
    for s in strata:
        sub = filled[filled["stratum"] == s]
        n_sampled = len(sub)
        pool_size = _deep_stratum_pool_size(meta, s)
        f_rows = sub[sub["verdict"] == "F"]
        n_f = len(f_rows)
        # A fraud id currently at percentile pct_rank ranks ABOVE pct_rank% of the population
        # already -- it's discordant only with the (100-pct_rank)% that still outrank it, not
        # with pct_rank% of it. (Caught via a real inversion: this originally used
        # pct_rank/100, which would have made TRANS_5 -- ids already sitting at the 95th-100th
        # percentile, i.e. nearly perfectly placed -- look like the DOMINANT contributor, when
        # it should be the smallest.)
        sample_discordant = float(((100.0 - f_rows["pct_rank"]) / 100.0).sum()) * n_bonafide
        scale = pool_size / n_sampled if n_sampled else 0.0
        extrapolated_discordant_s = sample_discordant * scale
        total_extrapolated_discordant += extrapolated_discordant_s
        rows.append({
            "stratum": s, "n_sampled": n_sampled, "pool_size": pool_size, "n_F_sampled": n_f,
            "mean_pct_rank_of_F": float(f_rows["pct_rank"].mean()) if n_f else float("nan"),
            "extrapolated_F_count": n_f * scale,
            "extrapolated_discordant_pairs": extrapolated_discordant_s,
        })

    removed = min(current_discordant, total_extrapolated_discordant)
    new_discordant = current_discordant - removed
    new_audet = new_discordant / total_pairs
    improvement_pct = (current_audet - new_audet) / current_audet * 100.0 if current_audet > 0 else float("nan")

    detail_df = pd.DataFrame(rows)
    detail_df.attrs.update({
        "n_fraud": n_fraud, "n_bonafide": n_bonafide, "total_pairs": total_pairs,
        "current_discordant": current_discordant,
        "total_extrapolated_discordant": total_extrapolated_discordant,
        "new_audet": new_audet, "improvement_pct": improvement_pct,
    })
    return detail_df


def ingest_deep(args) -> None:
    filled = pd.read_csv(args.filled_deep_csv, dtype={"id": str, "verdict": str, "note": str})
    filled["verdict"] = filled["verdict"].fillna("").str.strip()
    unknown = set(filled.loc[filled["verdict"] != "", "verdict"]) - set(VERDICT_LEVELS)
    if unknown:
        print(f"[review_package] WARNING: unrecognized verdict values found and ignored in tallies: {unknown}")

    meta = json.loads(Path(args.deep_meta).read_text(encoding="utf-8"))
    present_strata = [s for s in ALL_DEEP_GROUPS if s in set(filled["stratum"])]
    tallies = {s: stratum_tally(filled, s, target_levels=("F",)) for s in present_strata}
    for s, t in tallies.items():
        if t["n_judgeable"] == 0:
            print(f"[review_package] WARNING: {s} has 0 judgeable (F/A/B) verdicts -- did you fill in the CSV?")

    audet_detail = audet_bound_observed_rank_misrank(
        filled, meta, present_strata, args.current_audet, args.n_present, args.assumed_fraud_rate,
    )

    write_deep_ingest_report(Path(args.deep_report_out), tallies, audet_detail, meta, args)


def write_deep_ingest_report(
    report_path: Path, tallies: dict, audet_detail: pd.DataFrame, meta: dict, args,
) -> None:
    lines = ["# Review-package DEEP ingest report -- misranked-fraud sizing (transitional + floor)\n"]
    lines.append(
        "Answers the opposite question from `review_package_ingest_report.md`: not bona-fide "
        "contamination of the ceiling, but genuine FRAUD hiding, misranked, in the TRANSITIONAL "
        "zone or FLOOR block. No pre-registered CONFIRMED/REFUTED threshold was set for this "
        "task (unlike the ceiling-block review) -- numbers are reported straight below.\n"
    )

    lines.append("## Per-stratum misrank rate (F / judgeable), Wilson 95% CI\n")
    tally_rows = [
        {
            "stratum": s, "n_total": t["n_total"], "n_judgeable": t["n_judgeable"], "n_F": t["n_F"],
            "n_B": t["n_B"], "n_U": t["n_U"], "n_unsure": t["n_unsure"],
            "misrank_rate": t["contamination_rate"], "ci_lo": t["ci_lo"], "ci_hi": t["ci_hi"],
        }
        for s, t in tallies.items()
    ]
    lines.append(df_to_md(pd.DataFrame(tally_rows), float_fmt="{:.4f}"))
    lines.append(
        "\nTRANS_5 sits immediately adjacent to the ceiling mode (see "
        "`review_package_deep_report.md`'s stratum boundaries) -- a high misrank rate there is "
        "the EXPECTED boundary effect, not a novel finding. TRANS_1-4 and FLOOR_DEEP_TOP/REST "
        "are the genuinely-hidden-misrank question this survey was built to answer.\n"
    )

    lines.append("## Extrapolated misranked-fraud count and AuDET-discordant-pair contribution, by stratum\n")
    lines.append(
        "Each stratum's F-verdict ids are extrapolated to their full pool "
        f"({', '.join(f'{s}={_deep_stratum_pool_size(meta, s)}' for s in tallies)}) by "
        "pool_size/n_sampled; the discordant-pair estimate uses each F id's OWN observed "
        "pct_rank (not an assumed median -- see `audet_bound_observed_rank_misrank`'s "
        "docstring).\n"
    )
    lines.append(df_to_md(audet_detail.round(4)))

    trans_1_4 = audet_detail[audet_detail["stratum"].isin([s for s in TRANSITIONAL_STRATA if s != "TRANS_5"])]
    trans_5 = audet_detail[audet_detail["stratum"] == "TRANS_5"]
    floor_deep = audet_detail[audet_detail["stratum"].isin(FLOOR_DEEP_GROUPS)]
    lines.append(
        f"\n- TRANS_1-4 (genuinely-hidden zone) extrapolated F count: "
        f"~{trans_1_4['extrapolated_F_count'].sum():.0f}"
    )
    lines.append(f"- TRANS_5 (boundary-adjacent, expected) extrapolated F count: ~{trans_5['extrapolated_F_count'].sum():.0f}")
    lines.append(f"- FLOOR_DEEP (TOP+REST) extrapolated F count: ~{floor_deep['extrapolated_F_count'].sum():.0f}\n")

    lines.append("## Estimated recoverable AuDET (all sampled strata combined)\n")
    lines.append(
        f"n_fraud≈{audet_detail.attrs['n_fraud']}, n_bonafide≈{audet_detail.attrs['n_bonafide']}, "
        f"total pairs≈{audet_detail.attrs['total_pairs']:,}, current discordant pairs≈"
        f"{audet_detail.attrs['current_discordant']:,.0f}, extrapolated discordant pairs "
        f"attributable to found misranked frauds≈{audet_detail.attrs['total_extrapolated_discordant']:,.0f}."
    )
    lines.append(
        f"\n- New AuDET bound if every found misranked fraud were perfectly re-ranked: "
        f"**{audet_detail.attrs['new_audet']:.6f}** "
        f"({audet_detail.attrs['improvement_pct']:.2f}% improvement over current "
        f"{args.current_audet}).\n"
        "\nThis is a **best-case bound** (each id assumed moved from its current rank to fully "
        "concordant with all bona-fide, ids treated independently) -- not a prediction. Because "
        "it uses each id's own observed rank rather than an assumed median, TRANS_5's contribution "
        "is correctly modest relative to its extrapolated count (those ids are already ranked "
        "near the top, so 'fixing' each one recovers only a small number of discordant pairs) "
        "while TRANS_1-4/FLOOR_DEEP ids sitting at mid-percentile ranks contribute more per id.\n"
    )
    if audet_detail.attrs["total_extrapolated_discordant"] >= audet_detail.attrs["current_discordant"]:
        lines.append(
            f"\n**Note the saturation** (same phenomenon flagged in `hesitant_report.md`): the "
            f"extrapolated discordant-pair total ({audet_detail.attrs['total_extrapolated_discordant']:,.0f}) "
            f"already meets or exceeds the entire current gap "
            f"({audet_detail.attrs['current_discordant']:,.0f} discordant pairs), so the bound reads "
            "a flat 100% -- not because these ids are unusually decisive, but because AuDET is "
            "computed over ~14.9M pairs while the current gap itself is numerically tiny. Read "
            "this as **the misranked frauds found here are, under this bound's best-case "
            "assumptions, large enough in aggregate to plausibly explain the entire current AuDET "
            "gap** -- a genuinely different outcome from the ceiling-block review (which was "
            "REFUTED at 0.9% contamination): this survey found real, spread-out signal (TRANS_1/3/4 "
            "each independently show a non-zero misrank rate, not concentrated in one stratum), "
            "not a null result. It is still a bound, not proof that fixing these specific 38-ish "
            "extrapolated TRANS_1-4 ids would fully close the LB gap.\n"
        )

    lines.append("## Second-pass hook\n")
    lines.append(
        "If a specific stratum's misrank rate or extrapolated count looks worth confirming with "
        "a larger sample, pull more ids from just that stratum with `--stage build_deep --strata "
        f"<name> --exclude-csv {Path(DEFAULT_DEEP_CSV).name} --deep-csv-out "
        "review_package_deep_pass2.csv` (see `review_package_deep_report.md`'s own hook section).\n"
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[review_package] wrote deep ingest report -> {report_path}")
    print(
        f"[review_package] extrapolated misranked-fraud AuDET bound: {audet_detail.attrs['new_audet']:.6f} "
        f"({audet_detail.attrs['improvement_pct']:.2f}% improvement)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["build", "ingest", "build_deep", "rerender_deep", "ingest_deep"], required=True)
    # build args
    parser.add_argument("--logit-census-csv", default=str(DEFAULT_LOGIT_CENSUS_CSV))
    parser.add_argument("--hesitant-csv", default=str(DEFAULT_HESITANT_CSV))
    parser.add_argument("--submission-csv", default=str(DEFAULT_SUBMISSION_CSV))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tol", type=float, default=0.5)
    parser.add_argument("--n-per-stratum", type=int, default=150)
    parser.add_argument("--n-calibration", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thumb-size", type=int, default=32)
    parser.add_argument("--template-sim-threshold", type=float, default=0.85)
    parser.add_argument("--cell-width", type=int, default=340)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--csv-out", default=str(DEFAULT_REVIEW_CSV))
    parser.add_argument("--meta-out", default=str(DEFAULT_META_JSON))
    parser.add_argument("--report", default=str(DEFAULT_BUILD_REPORT))
    # ingest args
    parser.add_argument("--filled-csv", default=str(DEFAULT_REVIEW_CSV))
    parser.add_argument("--meta", default=str(DEFAULT_META_JSON))
    parser.add_argument("--report-out", default=str(DEFAULT_INGEST_REPORT))
    # build_deep args
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-per-trans-stratum", type=int, default=80)
    parser.add_argument("--n-floor-top", type=int, default=200)
    parser.add_argument("--n-floor-rest", type=int, default=100)
    parser.add_argument("--n-ref-per-type", type=int, default=40)
    parser.add_argument(
        "--card-only", action="store_true",
        help="Render only the full card (drop the 2x zoom face-crop panel) in build_deep review sheets.",
    )
    parser.add_argument(
        "--strata", default="ALL",
        help="Comma-separated subset of {%s} for a targeted second pass; 'ALL' (default) builds every group." % ", ".join(ALL_DEEP_GROUPS),
    )
    parser.add_argument(
        "--exclude-csv", default=None,
        help="Comma-separated prior deep-review CSV path(s) whose ids to exclude from the sampling pool "
             "(second-pass hook -- strata boundaries stay fixed, only the draw shrinks).",
    )
    parser.add_argument("--prior-review-csv", default=str(DEFAULT_REVIEW_CSV))
    parser.add_argument("--deep-out-dir", default=str(DEFAULT_DEEP_OUT_DIR))
    parser.add_argument("--deep-csv-out", default=str(DEFAULT_DEEP_CSV))
    parser.add_argument("--deep-meta-out", default=str(DEFAULT_DEEP_META_JSON))
    parser.add_argument("--deep-report", default=str(DEFAULT_DEEP_REPORT))
    # ingest_deep args
    parser.add_argument("--filled-deep-csv", default=str(DEFAULT_DEEP_CSV))
    parser.add_argument("--deep-meta", default=str(DEFAULT_DEEP_META_JSON))
    parser.add_argument("--deep-report-out", default=str(Path(__file__).resolve().parent / "review_package_deep_ingest_report.md"))
    parser.add_argument(
        "--n-present", type=int, default=7821,
        help="Present public-test id count (stable across this project's analysis scripts; the deep-build meta doesn't store it).",
    )
    # shared
    parser.add_argument("--current-audet", type=float, default=0.00744)
    parser.add_argument("--assumed-fraud-rate", type=float, default=0.42)
    args = parser.parse_args()

    if args.stage == "build":
        build(args)
    elif args.stage == "build_deep":
        build_deep(args)
    elif args.stage == "rerender_deep":
        rerender_deep(args)
    elif args.stage == "ingest_deep":
        ingest_deep(args)
    else:
        ingest(args)


if __name__ == "__main__":
    main()
