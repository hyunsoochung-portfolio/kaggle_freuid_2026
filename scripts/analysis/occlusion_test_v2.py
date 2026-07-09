"""Face-occlusion sensitivity test v2 -- removes the two confounds v1 documented.

v1 (`occlusion_test.py` / `occlusion_report.md`) found two things that block a clean read of
"does finetune_v0 read face-region evidence":

  (a) TRAIN_FRAUD_FACE and HESITANT_TEST are ~93-99% logit-saturated at the model's apparent
      ceiling (~+12.5) on the CLEAN image already -- there's no headroom left for an occlusion
      to move the logit further, informative or not.
  (b) v1's only fill method (local ring mean color) is structurally near-identical to
      `_field_smudge`'s third branch (one of the synthetic-tamper ops finetune_v0 trains
      against): "fill a rectangular region with its own local mean color." TRAIN_BONAFIDE (the
      one group with real headroom) reacted almost identically to ANY flat patch regardless of
      location -- consistent with a location-agnostic "is there an artificially flat patch here"
      shortcut rather than face-specific reading.

This script re-runs the experiment with both confounds addressed:
  1. Four fill methods per occlusion site (mean_color = v1's original method, kept as a direct
     comparison arm; texture_clone, noise_matched, shuffle -- see the FILL_METHODS section).
  2. Splits TRAIN_FRAUD_FACE and HESITANT_TEST into SAT (ceiling-saturated, within 1.0 logit of
     the ceiling mode from `logit_census_report.md`) and NONSAT (real headroom) subsets, and
     runs the full experiment on both separately.
  3. A dedicated flat-patch-vs-any-anomaly measurement on TRAIN_BONAFIDE's control-site deltas.
  4. Face-vs-control comparison per fill method, per group -- especially the NONSAT subsets,
     which is where v1 had no way to see an effect at all.

Reuses occlusion_test.py's sampling/scoring machinery directly (imported, not duplicated):
OcclusionSample, read_face_box, expand_box, boxes_overlap, pick_control_box, ring_mean_color,
occlude, sample_valid_face_ids_random, filter_valid_face_ids_ordered, hesitant_ranked_ids,
build_group, score_batch. Only the fill methods, the saturation split, and the report are new.

Read-only / inference-only. Does NOT train, does NOT touch src/freuid/*, does NOT touch the
regions cache, does NOT write submissions.

Usage (VESSL, GPU + checkpoint + regions cache + logit_census_raw.csv required):
    python scripts/analysis/occlusion_test_v2.py \\
        --checkpoint checkpoints/finetune_v0.pt --data-dir data \\
        --submission submissions/finetune_v0.csv \\
        --logit-census-csv scripts/analysis/logit_census_raw.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.data import load_labels  # noqa: E402
from freuid.preprocess import regions_dir  # noqa: E402
from freuid.transforms import build_transforms, resolve_data_config  # noqa: E402
from freuid.utils import pick_device  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_finetuned_model, load_checkpoint  # noqa: E402
from logit_census import find_modes  # noqa: E402
from occlusion_test import (  # noqa: E402
    OcclusionSample,
    REPO_ROOT,
    build_group,
    filter_valid_face_ids_ordered,
    hesitant_ranked_ids,
    occlude,
    pick_control_box,
    ring_mean_color,
    sample_valid_face_ids_random,
    score_batch,
)

DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "finetune_v0.pt"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "occlusion_v2_output"
DEFAULT_REPORT = Path(__file__).resolve().parent / "occlusion_v2_report.md"
DEFAULT_CSV = Path(__file__).resolve().parent / "occlusion_v2_results.csv"
DEFAULT_V1_CSV = Path(__file__).resolve().parent / "occlusion_results.csv"
DEFAULT_LOGIT_CENSUS_CSV = Path(__file__).resolve().parent / "logit_census_raw.csv"

FILL_METHODS = ("mean_color", "texture_clone", "noise_matched", "shuffle")
SITES = ("face", "control")
SAT_TOLERANCE = 1.0  # logits; matches logit_census_report.md's tolerance-sensitivity convention


def _stable_seed(*parts) -> int:
    """Deterministic int seed from arbitrary (str/int) parts -- np.random.default_rng only
    accepts int/array-of-int/SeedSequence, not tuples containing strings, so every stochastic
    fill draw that needs to key off a sample id goes through this."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def variant_name(site: str, method: str) -> str:
    return f"{site}__{method}"


ALL_VARIANTS = ["clean"] + [variant_name(s, m) for s in SITES for m in FILL_METHODS]


# ---------------------------------------------------------------------------
# Fill methods (new: everything below is new code, not present in v1)
# ---------------------------------------------------------------------------

def _ring_pixels(arr: np.ndarray, box: tuple[int, int, int, int], ring_frac: float = 0.25) -> np.ndarray:
    """Pixel values in the ring surrounding ``box`` (mirrors ring_mean_color's own ring
    geometry so noise/variance stats are computed over the exact same neighborhood v1's
    mean-color fill used -- but exposes the raw pixels, which ring_mean_color doesn't)."""
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    rx, ry = max(1, int(bw * ring_frac)), max(1, int(bh * ring_frac))
    rx1, ry1 = max(0, x1 - rx), max(0, y1 - ry)
    rx2, ry2 = min(w, x2 + rx), min(h, y2 + ry)
    region = arr[ry1:ry2, rx1:rx2]
    mask = np.ones(region.shape[:2], dtype=bool)
    iy1, ix1 = max(0, y1 - ry1), max(0, x1 - rx1)
    iy2, ix2 = min(mask.shape[0], iy1 + bh), min(mask.shape[1], ix1 + bw)
    mask[iy1:iy2, ix1:ix2] = False
    pixels = region[mask]
    if pixels.size == 0:
        full_mask = np.ones((h, w), dtype=bool)
        full_mask[y1:y2, x1:x2] = False
        pixels = arr[full_mask]
    return pixels.reshape(-1, arr.shape[2])


def ring_mean_and_std(arr: np.ndarray, box, ring_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    pixels = _ring_pixels(arr, box, ring_frac).astype(np.float64)
    return pixels.mean(axis=0), pixels.std(axis=0)


def ring_local_variance(arr: np.ndarray, box, ring_frac: float = 0.25) -> float:
    pixels = _ring_pixels(arr, box, ring_frac).astype(np.float64)
    return float(pixels.var(axis=0).mean())


def local_variance(arr: np.ndarray, box) -> float:
    x1, y1, x2, y2 = box
    region = arr[y1:y2, x1:x2].astype(np.float64)
    return float(region.var(axis=(0, 1)).mean())


def fill_mean_color(arr: np.ndarray, box, ring_frac: float = 0.25) -> np.ndarray:
    """v1's original fill method, kept as the 4th direct-comparison arm."""
    color = ring_mean_color(arr, box, ring_frac)
    return occlude(arr, box, color)


def fill_noise_matched(arr: np.ndarray, box, rng: np.random.Generator, ring_frac: float = 0.25) -> np.ndarray:
    """Mean-color fill PLUS noise matched to the surrounding ring's own per-channel std, so
    the patch isn't anomalously flat the way v1's fill was."""
    mean, std = ring_mean_and_std(arr, box, ring_frac)
    x1, y1, x2, y2 = box
    bh, bw = y2 - y1, x2 - x1
    noise = rng.normal(0.0, std[None, None, :], size=(bh, bw, arr.shape[2]))
    patch = np.clip(mean[None, None, :] + noise, 0, 255).astype(np.uint8)
    out = arr.copy()
    out[y1:y2, x1:x2] = patch
    return out


def pick_texture_donor(
    rng: np.random.Generator, arr: np.ndarray, target_box, exclude_boxes: list, w: int, h: int,
    n_candidates: int = 20, max_tries: int = 200,
):
    """A same-sized donor box elsewhere on the card, chosen so its own local variance is
    closest to the variance of the ring surrounding ``target_box`` -- i.e. a texture-matched
    donor, not just any random patch."""
    from occlusion_test import boxes_overlap

    tx1, ty1, tx2, ty2 = target_box
    bw, bh = max(1, tx2 - tx1), max(1, ty2 - ty1)
    target_var = ring_local_variance(arr, target_box)

    candidates = []
    tries = 0
    while len(candidates) < n_candidates and tries < max_tries:
        tries += 1
        x1 = int(rng.integers(0, max(1, w - bw + 1)))
        y1 = int(rng.integers(0, max(1, h - bh + 1)))
        cand = (x1, y1, x1 + bw, y1 + bh)
        if any(boxes_overlap(cand, ex) for ex in exclude_boxes):
            continue
        candidates.append(cand)

    if not candidates:
        # Deterministic fallback, mirrors pick_control_box's own fallback convention.
        x1 = max(0, w - bw) if tx1 < w / 2 else 0
        y1 = max(0, h - bh) if ty1 < h / 2 else 0
        return (x1, y1, x1 + bw, y1 + bh)

    variances = [local_variance(arr, c) for c in candidates]
    best = int(np.argmin([abs(v - target_var) for v in variances]))
    return candidates[best]


def fill_texture_clone(arr: np.ndarray, box, donor_box, feather_px: int = 8) -> np.ndarray:
    """Paste the donor patch into ``box``, feathered over ``feather_px`` at the border so
    there's no hard rectangular edge (a hard edge would itself be a location-agnostic
    anomaly cue, the exact thing this fill method is trying to avoid introducing)."""
    x1, y1, x2, y2 = box
    dx1, dy1, dx2, dy2 = donor_box
    bh, bw = y2 - y1, x2 - x1
    donor = arr[dy1:dy2, dx1:dx2]
    if donor.shape[0] != bh or donor.shape[1] != bw:
        donor = np.array(Image.fromarray(donor).resize((bw, bh), Image.BILINEAR))

    yy, xx = np.mgrid[0:bh, 0:bw]
    dist_to_edge = np.minimum(np.minimum(xx + 1, bw - xx), np.minimum(yy + 1, bh - yy)).astype(np.float64)
    fpx = max(1, feather_px)
    mask = np.clip(dist_to_edge / fpx, 0.0, 1.0)

    out = arr.copy()
    region = out[y1:y2, x1:x2].astype(np.float64)
    blended = mask[..., None] * donor.astype(np.float64) + (1 - mask[..., None]) * region
    out[y1:y2, x1:x2] = blended.astype(np.uint8)
    return out


def fill_shuffle(arr: np.ndarray, box, rng: np.random.Generator, grid: int = 4) -> np.ndarray:
    """Permute a grid x grid tiling of the box in place -- destroys spatial content/edges
    while preserving the box's own local color statistics (no new colors introduced at all,
    unlike the other three methods)."""
    x1, y1, x2, y2 = box
    region = arr[y1:y2, x1:x2].copy()
    bh, bw = region.shape[:2]
    ys = np.linspace(0, bh, grid + 1).astype(int)
    xs = np.linspace(0, bw, grid + 1).astype(int)

    coords, tiles = [], []
    for i in range(grid):
        for j in range(grid):
            coords.append((ys[i], ys[i + 1], xs[j], xs[j + 1]))
            tiles.append(region[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].copy())

    order = rng.permutation(len(tiles))
    out_region = region.copy()
    for dst, src_idx in zip(coords, order):
        dy0, dy1, dx0, dx1 = dst
        tile = tiles[src_idx]
        th, tw = dy1 - dy0, dx1 - dx0
        if tile.shape[0] != th or tile.shape[1] != tw:
            tile = np.array(Image.fromarray(tile).resize((tw, th)))
        out_region[dy0:dy1, dx0:dx1] = tile

    out = arr.copy()
    out[y1:y2, x1:x2] = out_region
    return out


def build_variant_image_v2(
    sample: OcclusionSample, variant: str, rng: np.random.Generator,
    ring_frac: float = 0.25, feather_px: int = 8,
) -> Image.Image:
    img = Image.open(sample.path).convert("RGB")
    if variant == "clean":
        return img
    site, method = variant.split("__", 1)
    box = sample.face_box_exp if site == "face" else sample.control_box
    arr = np.array(img)
    w, h = img.size

    if method == "mean_color":
        out = fill_mean_color(arr, box, ring_frac)
    elif method == "noise_matched":
        out = fill_noise_matched(arr, box, rng, ring_frac)
    elif method == "texture_clone":
        exclude = [sample.face_box_exp, sample.control_box]
        donor_box = pick_texture_donor(rng, arr, box, exclude, w, h)
        out = fill_texture_clone(arr, box, donor_box, feather_px)
    elif method == "shuffle":
        out = fill_shuffle(arr, box, rng)
    else:
        raise ValueError(f"unknown fill method {method!r}")
    return Image.fromarray(out)


# ---------------------------------------------------------------------------
# Saturation split
# ---------------------------------------------------------------------------

def compute_ceiling_mode(logit_census_csv: Path) -> float:
    """Re-derives the ceiling mode from logit_census_raw.csv via the SAME KDE mode-finder
    logit_census.py itself uses, rather than hardcoding the report's printed number (avoids
    drift if the census is ever re-run with different data)."""
    df = pd.read_csv(logit_census_csv, dtype={"id": str})
    scale_cols = [c for c in df.columns if c.startswith("logit_")]
    mean_logit = df[scale_cols].mean(axis=1).to_numpy() if "mean_logit" not in df.columns else df["mean_logit"].to_numpy()
    modes = find_modes(mean_logit)
    if len(modes) < 2:
        raise SystemExit(
            "logit_census_raw.csv did not yield a bimodal split -- cannot determine a ceiling "
            "mode to define the saturation split. Re-check logit_census_report.md's own verdict."
        )
    return max(m[0] for m in modes)


def score_clean_logits(
    model, device, samples: list[OcclusionSample], tta_scales: list[int], mean, std, batch_size: int,
) -> dict[str, float]:
    """Screening pass: CLEAN variant only (no occlusion), 3-scale mean logit -- used purely to
    determine each candidate's saturation status before committing to the full 9-variant scoring."""
    import torch

    logits_per_scale: dict[str, list[float]] = {s.id: [] for s in samples}
    for scale in tta_scales:
        tf = build_transforms(scale, False, mean, std)
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            imgs = [Image.open(s.path).convert("RGB") for s in batch]
            with torch.no_grad():
                logits = score_batch(model, device, imgs, tf)
            for s, lg in zip(batch, logits, strict=True):
                logits_per_scale[s.id].append(float(lg))
    return {sid: float(np.mean(v)) for sid, v in logits_per_scale.items()}


def split_by_saturation(ids_logits: dict[str, float], ceiling_mode: float, tol: float = SAT_TOLERANCE):
    sat = [sid for sid, lg in ids_logits.items() if abs(lg - ceiling_mode) <= tol]
    nonsat = [sid for sid, lg in ids_logits.items() if abs(lg - ceiling_mode) > tol]
    return sat, nonsat


# ---------------------------------------------------------------------------
# Scoring (v2): precompute all variant images once per sample, then TTA-scale
# ---------------------------------------------------------------------------

def run_scoring_v2(
    model, device, samples: list[OcclusionSample], variants: list[str], tta_scales: list[int],
    mean, std, batch_size: int, ring_frac: float, feather_px: int, seed: int, chunk_size: int = 32,
) -> dict[tuple[str, str], dict[int, float]]:
    """Chunked over samples so at most ``chunk_size`` images' worth of full-resolution variant
    renders (``len(variants)`` each) are held in memory at once. Variant images are built ONCE
    per sample (not once per TTA scale) so the stochastic fill methods (texture_clone,
    noise_matched, shuffle) see the identical occluded image at every scale -- only the resize
    differs, matching what TTA is supposed to measure."""
    import torch

    results: dict[tuple[str, str], dict[int, float]] = {}
    transforms_by_scale = {scale: build_transforms(scale, False, mean, std) for scale in tta_scales}

    for chunk_start in range(0, len(samples), chunk_size):
        chunk = samples[chunk_start:chunk_start + chunk_size]
        # Deterministic per-chunk rng: seeded from the global seed + chunk start index, so
        # re-running the script reproduces identical stochastic fills regardless of chunk_size.
        rng = np.random.default_rng(_stable_seed(seed, "chunk", chunk_start))
        cache: dict[tuple[str, str], Image.Image] = {}
        for s in chunk:
            for v in variants:
                cache[(s.id, v)] = build_variant_image_v2(s, v, rng, ring_frac, feather_px)

        tasks = [(s.id, v) for s in chunk for v in variants]
        for scale in tta_scales:
            tf = transforms_by_scale[scale]
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                imgs = [cache[key] for key in batch]
                with torch.no_grad():
                    logits = score_batch(model, device, imgs, tf)
                for key, lg in zip(batch, logits, strict=True):
                    results.setdefault(key, {})[scale] = float(lg)
        del cache
    return results


def calibrate_v2(
    model, device, samples: list[OcclusionSample], variants: list[str], tta_scales: list[int],
    mean, std, batch_size: int, ring_frac: float, feather_px: int, seed: int,
) -> float:
    """Time one chunk (build + score at the first scale only), project total wall-clock."""
    if not samples:
        return 0.0
    probe_chunk = samples[:min(len(samples), 8)]
    rng = np.random.default_rng(_stable_seed(seed, "calib"))
    t0 = time.time()
    cache = {}
    for s in probe_chunk:
        for v in variants:
            cache[(s.id, v)] = build_variant_image_v2(s, v, rng, ring_frac, feather_px)
    tf = build_transforms(tta_scales[0], False, mean, std)
    tasks = list(cache.keys())
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        imgs = [cache[k] for k in batch]
        _ = score_batch(model, device, imgs, tf)
    elapsed = time.time() - t0
    per_task = elapsed / max(1, len(tasks))
    return per_task


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------

def render_fill_examples(
    samples_by_group: dict[str, list[OcclusionSample]], variants: list[str],
    rng_seed: int, ring_frac: float, feather_px: int, out_dir: Path, per_method: int = 10,
) -> dict[str, list[Path]]:
    """10 example renders per fill method: clean | face_<method> | control_<method> side by
    side, drawn round-robin across groups so a bad fill (e.g. a glitchy texture_clone) is
    visible regardless of which group happened to trigger it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_samples = [s for grp in samples_by_group.values() for s in grp]
    paths_by_method: dict[str, list[Path]] = {m: [] for m in FILL_METHODS}

    for method in FILL_METHODS:
        picks = all_samples[:per_method]
        for idx, s in enumerate(picks):
            rng = np.random.default_rng(_stable_seed(rng_seed, "render", method, s.id))
            clean = Image.open(s.path).convert("RGB")
            face_img = build_variant_image_v2(s, variant_name("face", method), rng, ring_frac, feather_px)
            ctrl_img = build_variant_image_v2(s, variant_name("control", method), rng, ring_frac, feather_px)

            h = 400
            def _thumb(im):
                w = int(im.width * h / im.height)
                return im.resize((w, h))
            panels = [_thumb(clean), _thumb(face_img), _thumb(ctrl_img)]
            total_w = sum(p.width for p in panels) + 20
            canvas = Image.new("RGB", (total_w, h), (255, 255, 255))
            x = 0
            for p in panels:
                canvas.paste(p, (x, 0))
                x += p.width + 10
            out_path = out_dir / f"{method}_{idx:02d}_{s.group}_{s.id}.png"
            canvas.save(out_path)
            paths_by_method[method].append(out_path)
    return paths_by_method


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def build_results_df(samples: list[OcclusionSample], results: dict[tuple[str, str], dict[int, float]]) -> pd.DataFrame:
    rows = []
    for s in samples:
        logit = {}
        for v in ALL_VARIANTS:
            key = (s.id, v)
            if key not in results:
                continue
            logit[v] = float(np.mean(list(results[key].values())))
        if "clean" not in logit:
            continue
        row = {
            "id": s.id, "group": s.group, "label": s.label, "is_digital": s.is_digital,
            "doc_type": s.doc_type, "face_score": s.face_score, "logit_clean": logit["clean"],
        }
        for site in SITES:
            for method in FILL_METHODS:
                v = variant_name(site, method)
                if v not in logit:
                    continue
                delta = logit[v] - logit["clean"]
                row[f"logit_{v}"] = logit[v]
                row[f"delta_{v}"] = delta
                row[f"abs_delta_{v}"] = abs(delta)
        rows.append(row)
    return pd.DataFrame(rows)


def per_fill_method_stats(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    rows = []
    for group, g in df.groupby("group"):
        for method in FILL_METHODS:
            face_col, ctrl_col = f"abs_delta_face__{method}", f"abs_delta_control__{method}"
            if face_col not in g.columns or ctrl_col not in g.columns:
                continue
            face = g[face_col].dropna().to_numpy()
            ctrl = g[ctrl_col].dropna().to_numpy()
            n = min(len(face), len(ctrl))
            if n == 0:
                continue
            face, ctrl = face[:n], ctrl[:n]
            try:
                stat, p = wilcoxon(face, ctrl, zero_method="pratt")
            except ValueError:
                stat, p = float("nan"), float("nan")
            rows.append({
                "group": group, "fill_method": method, "n": n,
                "mean_abs_face": float(np.mean(face)), "mean_abs_control": float(np.mean(ctrl)),
                "mean_paired_diff": float(np.mean(face - ctrl)),
                "median_paired_diff": float(np.median(face - ctrl)),
                "wilcoxon_p": float(p),
            })
    return pd.DataFrame(rows)


def flat_patch_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """TRAIN_BONAFIDE's control-site |delta| per fill method -- the dedicated flat-patch-vs-
    any-anomaly measurement (item 3)."""
    g = df[df["group"] == "TRAIN_BONAFIDE"]
    rows = []
    for method in FILL_METHODS:
        col = f"abs_delta_control__{method}"
        if col not in g.columns:
            continue
        vals = g[col].dropna().to_numpy()
        rows.append({
            "fill_method": method, "n": len(vals),
            "mean_abs_delta_control": float(np.mean(vals)) if len(vals) else float("nan"),
            "median_abs_delta_control": float(np.median(vals)) if len(vals) else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def write_report(
    report_path: Path, stats_df: pd.DataFrame, flat_df: pd.DataFrame,
    v1_csv: Path, group_sizes: dict, ceiling_mode: float, render_paths: dict[str, list[Path]],
    checkpoint_path: Path, tta_scales: list[int],
) -> None:
    lines = []
    lines.append("# Face-occlusion sensitivity report v2 -- non-flat fills + saturation split\n")
    lines.append(
        "Re-runs `occlusion_test.py`'s experiment with the two confounds it documented removed: "
        "logit-ceiling saturation (split SAT/NONSAT per group using the ceiling mode from "
        "`logit_census_report.md`) and the mean-color fill's resemblance to the "
        "`_field_smudge` synthetic-tamper op (three additional, non-flat fill methods added: "
        "texture_clone, noise_matched, shuffle; mean_color kept as a direct v1-comparison arm).\n"
    )
    lines.append("## Methodology\n")
    lines.append(f"- Checkpoint: `{checkpoint_path}` | TTA scales: `{tta_scales}` (mean logit across scales, same convention as v1).")
    lines.append(
        f"- Ceiling mode used for the saturation split: **{ceiling_mode:.3f}** (re-derived here via "
        "`logit_census.find_modes` on `logit_census_raw.csv`'s mean_logit column -- not hardcoded "
        "from the report text, so it can't drift if the census is re-run)."
    )
    lines.append(
        f"- Saturation split tolerance: |logit - ceiling| <= {SAT_TOLERANCE} logits -> SAT; "
        "everything else -> NONSAT. HESITANT_TEST's split uses `logit_census_raw.csv` directly "
        "(already covers every present public_test id -- no extra GPU pass needed). "
        "TRAIN_FRAUD_FACE's split needed a dedicated clean-logit screening pass (train ids "
        "aren't in the public-test-only logit census)."
    )
    lines.append(
        "- Fill methods: **mean_color** (v1's original: ring mean color, no noise -- kept for "
        "direct comparison); **noise_matched** (mean_color + gaussian noise matched to the "
        "surrounding ring's own per-channel std); **texture_clone** (a same-sized donor patch "
        "elsewhere on the card, chosen so its local variance matches the target's surrounding "
        "ring, feathered 8px at the border); **shuffle** (4x4 grid-tile permutation in place -- "
        "no new pixel values introduced at all)."
    )
    lines.append(
        "- Groups actually scored (n): " + ", ".join(f"{g}={n}" for g, n in group_sizes.items()) + "\n"
    )

    lines.append("## v1 vs v2 comparison (mean_color arm only, aggregate per-group means)\n")
    if v1_csv.exists():
        v1 = pd.read_csv(v1_csv, dtype={"id": str})
        v1_stats = v1.groupby("group").agg(
            v1_mean_abs_face=("abs_delta_face", "mean"),
            v1_mean_abs_control=("abs_delta_control", "mean"),
        ).reset_index()
        v2_mean_arm = stats_df[stats_df["fill_method"] == "mean_color"][
            ["group", "mean_abs_face", "mean_abs_control"]
        ].rename(columns={"mean_abs_face": "v2_mean_abs_face", "mean_abs_control": "v2_mean_abs_control"})
        cmp = v1_stats.merge(v2_mean_arm, on="group", how="outer")
        lines.append(df_to_md(cmp))
        lines.append(
            "\nNote: v1's groups (TRAIN_FRAUD_FACE, HESITANT_TEST) are NOT saturation-split and "
            "used a different, generally smaller/differently-seeded sample than v2's SAT/NONSAT "
            "subsets -- this is an aggregate sanity check that the mean_color arm reproduces "
            "roughly the same magnitude of effect as v1, not an exact paired-id comparison.\n"
        )
    else:
        lines.append(f"`{v1_csv}` not found -- skipped.\n")

    lines.append("## Per-group x per-fill-method face-vs-control stats\n")
    lines.append(df_to_md(stats_df))
    lines.append("")
    lines.append(
        "`mean_paired_diff` = mean(|Δface| - |Δcontrol|) per image; positive means face "
        "occlusion moved the logit more than control occlusion on average. Wilcoxon p is the "
        "paired signed-rank test on |Δface| vs |Δcontrol|.\n"
    )

    lines.append("## Flat-patch-shortcut measurement (TRAIN_BONAFIDE, control-site only)\n")
    lines.append(df_to_md(flat_df))
    lines.append("")
    if len(flat_df) == len(FILL_METHODS) and not flat_df["mean_abs_delta_control"].isna().any():
        mean_row = flat_df[flat_df["fill_method"] == "mean_color"]["mean_abs_delta_control"]
        others = flat_df[flat_df["fill_method"] != "mean_color"]["mean_abs_delta_control"]
        if len(mean_row) and len(others) and others.mean() > 0:
            ratio = float(mean_row.iloc[0] / others.mean())
            lines.append(
                f"mean_color / mean(other three methods) ratio = **{ratio:.2f}x**. Pre-registered "
                "read: ratio >> 1 confirms and quantifies the flat-patch-detector theory (fill "
                "realism specifically matters); ratio ~= 1 (all methods produce comparably large "
                "deltas) means the model has a broader any-local-anomaly trigger, and fill realism "
                "alone won't resolve the shortcut in the tamper-synthesis work."
            )
            if ratio > 2.0:
                lines.append(
                    "\n**Verdict: flat-patch theory CONFIRMED and quantified** "
                    f"({ratio:.2f}x) -- mean_color's deltas are substantially larger than the "
                    "other three non-flat fills'."
                )
            elif ratio < 1.5:
                lines.append(
                    "\n**Verdict: any-anomaly theory** -- all four fill methods (including the "
                    "non-flat ones) produce comparably large control-site deltas on TRAIN_BONAFIDE. "
                    "The model reacts to ANY local disruption, not specifically to flatness. Fill "
                    "realism alone will not fix this in the tamper-synthesis augmentation."
                )
            else:
                lines.append(
                    f"\n**Verdict: in between** ({ratio:.2f}x) -- some but not overwhelming "
                    "sensitivity to fill flatness specifically. Report this straight rather than "
                    "forcing either pre-registered read."
                )
    lines.append("")

    lines.append("## Face-vs-control verdict per group (does face occlusion beat control, per fill method)\n")
    for group in stats_df["group"].unique():
        g = stats_df[stats_df["group"] == group]
        lines.append(f"**{group}** (n={group_sizes.get(group, '?')}):")
        if group_sizes.get(group, 0) < 30 and "NONSAT" in group:
            lines.append(
                f"- **UNDERPOWERED**: only {group_sizes.get(group)} non-saturated ids found. "
                "Numbers below are reported for completeness but should not be treated as "
                "conclusive on their own."
            )
        for _, r in g.iterrows():
            beats = r["wilcoxon_p"] < 0.05 and r["mean_paired_diff"] > 0
            lines.append(
                f"- {r['fill_method']}: mean|Δface|={r['mean_abs_face']:.4f} "
                f"mean|Δcontrol|={r['mean_abs_control']:.4f} paired_diff={r['mean_paired_diff']:.4f} "
                f"p={r['wilcoxon_p']:.4g} -> face {'BEATS' if beats else 'does NOT beat'} control"
            )
        lines.append("")

    lines.append("## Three pre-registered verdicts\n")
    lines.append("**1. Flat-patch theory** (from the TRAIN_BONAFIDE control-site measurement above): see verdict in that section.\n")
    lines.append(
        "**2. Any-anomaly theory**: the complement of (1) -- if all fill methods produced "
        "comparably large TRAIN_BONAFIDE control-site deltas, fill realism alone will not "
        "resolve the shortcut; the tamper-synthesis augmentation would need to stop relying on "
        "location-agnostic local-disruption cues entirely, not just make them look more realistic.\n"
    )
    lines.append(
        "**3. Face-specific reliance**: read the NONSAT rows in the face-vs-control section "
        "above (TRAIN_FRAUD_FACE_NONSAT, HESITANT_TEST_NONSAT) for the non-flat fill methods "
        "(texture_clone, noise_matched, shuffle) specifically -- mean_color's NONSAT numbers are "
        "still confounded by the flat-patch shortcut even outside the ceiling. If face occlusion "
        "beats control on the non-flat methods in the NONSAT subsets, that's the first clean "
        "evidence either way on the original question this whole line of experiments was meant "
        "to answer; if not, the honest read is still inconclusive, now for lack of a large enough "
        "non-saturated sample rather than for the two confounds v1 identified.\n"
    )

    lines.append("## Fill-quality example renders\n")
    for method, paths in render_paths.items():
        lines.append(f"**{method}** ({len(paths)} examples):")
        for p in paths:
            try:
                rel = p.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                rel = p.as_posix()
            lines.append(f"- ![{method}]({rel})")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[occlusion_v2] wrote report -> {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--submission", default="submissions/finetune_v0.csv")
    parser.add_argument("--logit-census-csv", default=str(DEFAULT_LOGIT_CENSUS_CSV))
    parser.add_argument("--v1-csv", default=str(DEFAULT_V1_CSV))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--expand-frac", type=float, default=0.10)
    parser.add_argument("--ring-frac", type=float, default=0.25)
    parser.add_argument("--feather-px", type=int, default=8)
    parser.add_argument("--n-bonafide", type=int, default=200)
    parser.add_argument("--n-sat-target", type=int, default=200)
    parser.add_argument("--n-nonsat-cap", type=int, default=150)
    parser.add_argument("--n-screen-fraud", type=int, default=3000, help="candidate pool size to screen for TRAIN_FRAUD_FACE saturation status")
    parser.add_argument("--k-hesitant-screen", type=int, default=3000, help="hesitant-rank candidate pool size for HESITANT_TEST saturation status")
    parser.add_argument("--time-guard-minutes", type=float, default=45.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--csv-out", default=str(DEFAULT_CSV))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cfg, state = load_checkpoint(args.checkpoint)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    device = pick_device()
    model = build_finetuned_model(cfg, state, device)
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    mean, std = data_cfg["mean"], data_cfg["std"]
    tta_cfg = cfg.extra.get("tta")
    tta_scales = [int(s) for s in tta_cfg] if isinstance(tta_cfg, list) else [data_cfg["image_size"]]
    print(f"[occlusion_v2] backbone={cfg.backbone} tta_scales={tta_scales} device={device}")

    rdir = regions_dir(cfg.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this experiment needs it (VESSL)")

    logit_census_csv = Path(args.logit_census_csv)
    if not logit_census_csv.exists():
        raise SystemExit(f"{logit_census_csv} not found -- run logit_census.py --stage infer first")
    ceiling_mode = compute_ceiling_mode(logit_census_csv)
    print(f"[occlusion_v2] ceiling mode (re-derived from logit census): {ceiling_mode:.3f}")

    train_df = load_labels(cfg.data_dir, "train")
    test_df = load_labels(cfg.data_dir, "public_test")
    fraud_ids = train_df.loc[train_df["label"] == 1, "id"].tolist()
    bonafide_ids = train_df.loc[train_df["label"] == 0, "id"].tolist()

    # --- TRAIN_BONAFIDE: unchanged from v1, no saturation split needed (already has headroom) ---
    bonafide_valid = sample_valid_face_ids_random(bonafide_ids, rdir, rng, args.n_bonafide)
    bonafide_samples = build_group(bonafide_valid, "TRAIN_BONAFIDE", train_df, args.expand_frac, rng)
    print(f"[occlusion_v2] TRAIN_BONAFIDE: {len(bonafide_samples)}/{args.n_bonafide}")

    # --- TRAIN_FRAUD_FACE: screen a large candidate pool for clean logit, split SAT/NONSAT ---
    fraud_screen_valid = sample_valid_face_ids_random(fraud_ids, rdir, rng, args.n_screen_fraud)
    fraud_screen_samples = build_group(fraud_screen_valid, "TRAIN_FRAUD_FACE_SCREEN", train_df, args.expand_frac, rng)
    print(f"[occlusion_v2] screening {len(fraud_screen_samples)} TRAIN_FRAUD_FACE candidates for clean logit...")
    t0 = time.time()
    fraud_clean_logits = score_clean_logits(model, device, fraud_screen_samples, tta_scales, mean, std, args.batch_size)
    print(f"[occlusion_v2] fraud screening done in {(time.time() - t0) / 60:.1f} min")
    fraud_sat_ids, fraud_nonsat_ids = split_by_saturation(fraud_clean_logits, ceiling_mode)
    print(f"[occlusion_v2] TRAIN_FRAUD_FACE: {len(fraud_sat_ids)} SAT / {len(fraud_nonsat_ids)} NONSAT (of {len(fraud_screen_samples)} screened)")

    fraud_sat_ids = fraud_sat_ids[:args.n_sat_target]
    fraud_nonsat_ids = fraud_nonsat_ids[:args.n_nonsat_cap]
    by_id = {s.id: s for s in fraud_screen_samples}
    fraud_sat_samples = [by_id[i] for i in fraud_sat_ids]
    fraud_nonsat_samples = [by_id[i] for i in fraud_nonsat_ids]
    for s in fraud_sat_samples:
        s.group = "TRAIN_FRAUD_FACE_SAT"
    for s in fraud_nonsat_samples:
        s.group = "TRAIN_FRAUD_FACE_NONSAT"

    # --- HESITANT_TEST: split via logit_census_raw.csv directly, no extra GPU pass ---
    census = pd.read_csv(logit_census_csv, dtype={"id": str}).set_index("id")
    census_logit = census["mean_logit"] if "mean_logit" in census.columns else census[[c for c in census.columns if c.startswith("logit_")]].mean(axis=1)

    hes_ranked = hesitant_ranked_ids(args.submission, cfg.data_dir, args.k_hesitant_screen)
    hes_valid = filter_valid_face_ids_ordered(hes_ranked, rdir, n=len(hes_ranked))
    hes_ids_in_census = [(cid, fb) for cid, fb in hes_valid if cid in census_logit.index]
    print(f"[occlusion_v2] HESITANT_TEST: {len(hes_ids_in_census)}/{len(hes_valid)} valid-face candidates found in logit census")

    hes_logits = {cid: float(census_logit.loc[cid]) for cid, _ in hes_ids_in_census}
    hes_sat_ids, hes_nonsat_ids = split_by_saturation(hes_logits, ceiling_mode)
    print(f"[occlusion_v2] HESITANT_TEST: {len(hes_sat_ids)} SAT / {len(hes_nonsat_ids)} NONSAT (of {len(hes_ids_in_census)})")

    hes_sat_ids = hes_sat_ids[:args.n_sat_target]
    hes_nonsat_ids = hes_nonsat_ids[:args.n_nonsat_cap]
    hes_boxes = dict(hes_ids_in_census)
    hes_sat_samples = build_group([(i, hes_boxes[i]) for i in hes_sat_ids], "HESITANT_TEST_SAT", test_df, args.expand_frac, rng)
    hes_nonsat_samples = build_group([(i, hes_boxes[i]) for i in hes_nonsat_ids], "HESITANT_TEST_NONSAT", test_df, args.expand_frac, rng)

    all_samples = (
        bonafide_samples + fraud_sat_samples + fraud_nonsat_samples + hes_sat_samples + hes_nonsat_samples
    )
    group_sizes = {
        "TRAIN_BONAFIDE": len(bonafide_samples),
        "TRAIN_FRAUD_FACE_SAT": len(fraud_sat_samples),
        "TRAIN_FRAUD_FACE_NONSAT": len(fraud_nonsat_samples),
        "HESITANT_TEST_SAT": len(hes_sat_samples),
        "HESITANT_TEST_NONSAT": len(hes_nonsat_samples),
    }
    print(f"[occlusion_v2] group sizes: {group_sizes}")
    for g in ("TRAIN_FRAUD_FACE_NONSAT", "HESITANT_TEST_NONSAT"):
        if group_sizes[g] < 30:
            print(f"[occlusion_v2] WARNING: {g} n={group_sizes[g]} < 30 -- will be marked UNDERPOWERED in the report")

    # --- Runtime guard on the main 9-variant scoring pass ---
    per_task = calibrate_v2(model, device, all_samples[:8], ALL_VARIANTS, tta_scales, mean, std, args.batch_size, args.ring_frac, args.feather_px, args.seed)
    total_tasks = len(all_samples) * len(ALL_VARIANTS) * len(tta_scales)
    projected_s = per_task * total_tasks
    print(f"[occlusion_v2] calibration: {per_task * 1000:.1f} ms/task -> projected {projected_s / 60:.1f} min for {total_tasks} tasks")
    if projected_s > args.time_guard_minutes * 60 and not args.force:
        print(f"[occlusion_v2] ABORTING: projected {projected_s/60:.1f} min exceeds the {args.time_guard_minutes}-min guard. Re-run with --force.")
        raise SystemExit(1)

    render_paths = render_fill_examples(
        {g: s for g, s in [
            ("TRAIN_BONAFIDE", bonafide_samples), ("TRAIN_FRAUD_FACE_SAT", fraud_sat_samples),
            ("TRAIN_FRAUD_FACE_NONSAT", fraud_nonsat_samples), ("HESITANT_TEST_SAT", hes_sat_samples),
            ("HESITANT_TEST_NONSAT", hes_nonsat_samples),
        ]},
        ALL_VARIANTS, args.seed, args.ring_frac, args.feather_px, out_dir,
    )

    print(f"[occlusion_v2] scoring {len(all_samples)} images x {len(ALL_VARIANTS)} variants x {len(tta_scales)} scales")
    t0 = time.time()
    results = run_scoring_v2(
        model, device, all_samples, ALL_VARIANTS, tta_scales, mean, std,
        args.batch_size, args.ring_frac, args.feather_px, args.seed, args.chunk_size,
    )
    print(f"[occlusion_v2] scoring done in {(time.time() - t0) / 60:.1f} min")

    df = build_results_df(all_samples, results)
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[occlusion_v2] wrote per-image csv -> {csv_path} ({len(df)} rows)")

    stats_df = per_fill_method_stats(df)
    flat_df = flat_patch_analysis(df)
    write_report(
        Path(args.report), stats_df, flat_df, Path(args.v1_csv), group_sizes, ceiling_mode,
        render_paths, Path(args.checkpoint), tta_scales,
    )


if __name__ == "__main__":
    main()
