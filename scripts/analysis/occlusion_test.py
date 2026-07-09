"""Face-occlusion sensitivity test for the finetune_v0 checkpoint.

Hypothesis under test: finetune_v0 learned to score training-distribution face-edit frauds
via tool-fingerprint shortcuts (compression/resampling artefacts left by the editing tool)
rather than by actually reading tamper evidence inside the face region. If that's true,
occluding the face region should barely move its fraud logit relative to occluding an
equally-sized, equally-filled patch somewhere else on the card (the "control" occlusion).

Read-only / inference-only: loads the finetune_v0 checkpoint and the SCRFD regions cache,
runs forward passes, writes a report + csv. Does NOT train, does NOT touch
`src/freuid/*`, does NOT regenerate or write anything under the regions cache.

Methodology notes (read before trusting the numbers):

* No tamper-type/category field exists anywhere in `train_labels.csv` (columns are only
  `id, image_path, label, is_digital, type` -- `type` is a COUNTRY/DOCTYPE domain field, not
  a tamper-location/category field). The only bounding-box tracking in this repo
  (`scripts/analysis/tamper_bbox.py`) is for *synthetically* injected tampers, not real fraud
  rows. So TRAIN_FRAUD_FACE, per the task's own fallback instruction, is really "all real
  fraud rows" (label==1) -- diluted by whatever fraction of real fraud is NOT a face-region
  edit (unknown; could be field/MRZ/barcode edits, whole-document reprints, etc.). Every group
  is additionally restricted to samples with a REAL SCRFD detection (regions cache
  `face.json` `score > 0`, i.e. not the center-square fallback box) -- without a real
  detection there is no face region to occlude in the first place. Post the cbe7fe2 fix this
  excludes only a negligible fraction (~100% train / ~99.9% public_test detection rate).
* Scoring reuses `freuid.transforms.build_transforms` / `resolve_data_config` (the same
  functions `infer.py` calls) at the checkpoint's own 3 TTA scales -- transforms are not
  reimplemented. Combination across scales, however, is a plain mean of raw logits, NOT
  infer.py's sigmoid + rank-average: rank-averaging is a corpus-wide relative operation (a
  score's rank depends on the ~7.8k-image population it's computed against) and isn't a
  meaningful unit for a single image's clean-vs-occluded delta on a custom ~200-image
  subsample. Mean-logit is the natural analog for a paired per-image comparison and is what
  the task explicitly asked for ("work in raw logit space for deltas, not post-sigmoid
  scores").
* "control_occluded" fills its patch with the mean color of ITS OWN local surroundings (not
  the face patch's fill color) -- using a mismatched color at the control location would
  paint an unnatural-looking patch that could itself read as a tamper cue, confounding "any
  occlusion" with "an oddly-colored patch". Filling from local context isolates the effect of
  removing information at a plain, blended patch.

Usage (VESSL, GPU + checkpoint + regions cache required):
    python scripts/analysis/occlusion_test.py \\
        --checkpoint checkpoints/finetune_v0.pt --data-dir data \\
        --submission submissions/finetune_v0.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from freuid.data import forward_with_extras, load_labels  # noqa: E402
from freuid.preprocess import regions_dir  # noqa: E402
from freuid.transforms import build_transforms, resolve_data_config  # noqa: E402
from freuid.utils import pick_device  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_finetuned_model, load_checkpoint  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "finetune_v0.pt"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "occlusion_output"
DEFAULT_REPORT = Path(__file__).resolve().parent / "occlusion_report.md"
DEFAULT_CSV = Path(__file__).resolve().parent / "occlusion_results.csv"

VARIANTS = ("clean", "face_occluded", "control_occluded")


@dataclass
class OcclusionSample:
    id: str
    group: str
    path: Path
    label: int
    is_digital: bool | None
    doc_type: str | None
    face_score: float
    face_box_raw: tuple[int, int, int, int]   # as detected, no expansion
    face_box_exp: tuple[int, int, int, int]   # expanded 10% per side
    control_box: tuple[int, int, int, int]    # same size as face_box_exp, elsewhere


# ---------------------------------------------------------------------------
# Face box / regions cache helpers
# ---------------------------------------------------------------------------

def read_face_box(rdir: Path, id_: str) -> dict | None:
    p = rdir / id_ / "face.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def expand_box(box: dict, frac: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * frac, bh * frac
    nx1 = max(0, int(round(x1 - mx)))
    ny1 = max(0, int(round(y1 - my)))
    nx2 = min(w, int(round(x2 + mx)))
    ny2 = min(h, int(round(y2 + my)))
    return (nx1, ny1, nx2, ny2)


def boxes_overlap(a: tuple, b: tuple) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def pick_control_box(
    rng: np.random.Generator, face_box: tuple[int, int, int, int], w: int, h: int,
    max_tries: int = 200,
) -> tuple[int, int, int, int]:
    fx1, fy1, fx2, fy2 = face_box
    bw = max(1, min(fx2 - fx1, w))
    bh = max(1, min(fy2 - fy1, h))
    for _ in range(max_tries):
        x1 = int(rng.integers(0, max(1, w - bw + 1)))
        y1 = int(rng.integers(0, max(1, h - bh + 1)))
        cand = (x1, y1, x1 + bw, y1 + bh)
        if not boxes_overlap(cand, face_box):
            return cand
    # Deterministic fallback if no non-overlapping placement was found in max_tries
    # (only possible for boxes covering close to half the image): mirror to the
    # opposite half.
    x1 = max(0, w - bw) if fx1 < w / 2 else 0
    y1 = max(0, h - bh) if fy1 < h / 2 else 0
    return (x1, y1, x1 + bw, y1 + bh)


def ring_mean_color(arr: np.ndarray, box: tuple[int, int, int, int], ring_frac: float = 0.25) -> np.ndarray:
    """Mean RGB of a ring immediately surrounding ``box`` (falls back to whole-image-minus-box)."""
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
    ring_pixels = region[mask]
    if ring_pixels.size == 0:
        full_mask = np.ones((h, w), dtype=bool)
        full_mask[y1:y2, x1:x2] = False
        ring_pixels = arr[full_mask]
    return ring_pixels.reshape(-1, arr.shape[2]).astype(np.float64).mean(axis=0)


def occlude(arr: np.ndarray, box: tuple[int, int, int, int], color: np.ndarray) -> np.ndarray:
    out = arr.copy()
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = color
    return out


def build_variant_image(sample: OcclusionSample, variant: str, ring_frac: float) -> Image.Image:
    img = Image.open(sample.path).convert("RGB")
    if variant == "clean":
        return img
    arr = np.array(img)
    box = sample.face_box_exp if variant == "face_occluded" else sample.control_box
    color = ring_mean_color(arr, box, ring_frac)
    return Image.fromarray(occlude(arr, box, color))


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def sample_valid_face_ids_random(
    candidate_ids: list[str], rdir: Path, rng: np.random.Generator, n: int,
) -> list[tuple[str, dict]]:
    order = rng.permutation(len(candidate_ids))
    valid: list[tuple[str, dict]] = []
    for idx in order:
        cid = candidate_ids[idx]
        fb = read_face_box(rdir, cid)
        if fb is not None and float(fb.get("score", 0.0)) > 0.0:
            valid.append((cid, fb))
            if len(valid) >= n:
                break
    return valid


def filter_valid_face_ids_ordered(
    ordered_ids: list[str], rdir: Path, n: int,
) -> list[tuple[str, dict]]:
    valid: list[tuple[str, dict]] = []
    for cid in ordered_ids:
        fb = read_face_box(rdir, cid)
        if fb is not None and float(fb.get("score", 0.0)) > 0.0:
            valid.append((cid, fb))
            if len(valid) >= n:
                break
    return valid


def hesitant_ranked_ids(submission_csv: str | Path, data_dir: str | Path, k: int) -> list[str]:
    """Mirrors hesitant_test_images.py's ranking exactly (top-k by |score - 0.5|),
    computed fresh here to get ``k`` candidates rather than reusing the existing
    100-row reports/hesitant_test_100.csv artifact."""
    test_meta = load_labels(data_dir, "public_test")
    present_mask = test_meta["path"].map(lambda p: Path(p).exists())
    present_ids = set(test_meta.loc[present_mask, "id"])
    sub = pd.read_csv(submission_csv, dtype={"id": str})
    sub_present = sub[sub["id"].isin(present_ids)].copy()
    sub_present = sub_present.rename(columns={"label": "score"})
    sub_present["dist_from_half"] = (sub_present["score"] - 0.5).abs()
    top = sub_present.nsmallest(k, "dist_from_half")
    return top["id"].tolist()


def build_group(
    ids_boxes: list[tuple[str, dict]], group: str, df: pd.DataFrame, expand_frac: float,
    rng: np.random.Generator,
) -> list[OcclusionSample]:
    df_idx = df.set_index("id")
    out = []
    for cid, fb in ids_boxes:
        row = df_idx.loc[cid]
        path = Path(row["path"])
        with Image.open(path) as im:
            w, h = im.size
        raw_box = (int(fb["x1"]), int(fb["y1"]), int(fb["x2"]), int(fb["y2"]))
        exp_box = expand_box(fb, expand_frac, w, h)
        ctrl_box = pick_control_box(rng, exp_box, w, h)
        out.append(OcclusionSample(
            id=cid,
            group=group,
            path=path,
            label=int(row["label"]),
            is_digital=bool(row["is_digital"]) if pd.notna(row.get("is_digital")) else None,
            doc_type=row.get("type") if pd.notna(row.get("type")) else None,
            face_score=float(fb.get("score", 0.0)),
            face_box_raw=raw_box,
            face_box_exp=exp_box,
            control_box=ctrl_box,
        ))
    return out


# ---------------------------------------------------------------------------
# Spot-check rendering
# ---------------------------------------------------------------------------

def spot_render(samples: list[OcclusionSample], out_dir: Path, n: int = 5) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = samples[:n]
    for s in picks:
        img = Image.open(s.path).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle(s.face_box_raw, outline=(0, 255, 0), width=4)   # raw SCRFD box
        draw.rectangle(s.face_box_exp, outline=(255, 0, 0), width=4)   # expanded occlusion box
        draw.rectangle(s.control_box, outline=(0, 128, 255), width=4)  # control box
        out_path = out_dir / f"spotcheck_{s.group}_{s.id}.png"
        img.save(out_path)
        print(f"[occlusion] spot-check saved: {out_path} (green=raw SCRFD, red=expanded/occluded, blue=control)")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_batch(model, device, pil_images: list[Image.Image], tf) -> np.ndarray:
    imgs = torch.stack([tf(im) for im in pil_images]).to(device)
    logits = forward_with_extras(model, imgs)
    return logits.squeeze(1).float().cpu().numpy()


def run_scoring(
    model, device, tasks: list[tuple[OcclusionSample, str]], tta_scales: list[int],
    mean, std, batch_size: int, ring_frac: float,
) -> dict[tuple[str, str], dict[int, float]]:
    results: dict[tuple[str, str], dict[int, float]] = {}
    for scale in tta_scales:
        tf = build_transforms(scale, False, mean, std)
        t_scale_start = time.time()
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            imgs = [build_variant_image(s, variant, ring_frac) for s, variant in batch]
            logits = score_batch(model, device, imgs, tf)
            for (s, variant), lg in zip(batch, logits, strict=True):
                results.setdefault((s.id, variant), {})[scale] = float(lg)
        print(f"[occlusion] scale={scale}: {len(tasks)} tasks in {time.time() - t_scale_start:.1f}s")
    return results


def calibrate_runtime(
    model, device, tasks: list[tuple[OcclusionSample, str]], tta_scales: list[int],
    mean, std, batch_size: int, ring_frac: float,
) -> float:
    """Time one batch at the first TTA scale; project total runtime across all scales."""
    calib = tasks[:min(batch_size, len(tasks))]
    tf = build_transforms(tta_scales[0], False, mean, std)
    t0 = time.time()
    imgs = [build_variant_image(s, variant, ring_frac) for s, variant in calib]
    _ = score_batch(model, device, imgs, tf)
    elapsed = time.time() - t0
    per_task = elapsed / max(1, len(calib))
    projected_total_s = per_task * len(tasks) * len(tta_scales)
    print(
        f"[occlusion] calibration: {len(calib)} tasks in {elapsed:.2f}s "
        f"({per_task * 1000:.1f} ms/task) -> projected total for {len(tasks)} tasks x "
        f"{len(tta_scales)} scales = {projected_total_s / 60:.1f} min"
    )
    return projected_total_s


# ---------------------------------------------------------------------------
# Stats + report
# ---------------------------------------------------------------------------

def compute_group_stats(df: pd.DataFrame) -> dict:
    from scipy.stats import wilcoxon

    stats = {}
    for group, g in df.groupby("group"):
        abs_face = g["abs_delta_face"].to_numpy()
        abs_ctrl = g["abs_delta_control"].to_numpy()
        paired_diff = abs_face - abs_ctrl
        try:
            stat, p = wilcoxon(abs_face, abs_ctrl, zero_method="pratt")
        except ValueError:
            stat, p = float("nan"), float("nan")
        stats[group] = {
            "n": len(g),
            "mean_abs_face": float(np.mean(abs_face)),
            "median_abs_face": float(np.median(abs_face)),
            "mean_abs_control": float(np.mean(abs_ctrl)),
            "median_abs_control": float(np.median(abs_ctrl)),
            "mean_paired_diff": float(np.mean(paired_diff)),
            "median_paired_diff": float(np.median(paired_diff)),
            "wilcoxon_stat": float(stat),
            "wilcoxon_p": float(p),
        }
    return stats


def headline_fraction(df: pd.DataFrame) -> tuple[float, float, int]:
    """TRAIN_FRAUD_FACE only: fraction of images whose fraud logit DROPS (clean -> face_occluded)
    by more than the group's own control-occlusion drop 90th percentile."""
    g = df[df["group"] == "TRAIN_FRAUD_FACE"]
    if g.empty:
        return float("nan"), float("nan"), 0
    drop_face = -g["delta_face"].to_numpy()       # positive = logit dropped
    drop_control = -g["delta_control"].to_numpy()
    p90 = float(np.percentile(drop_control, 90))
    frac = float(np.mean(drop_face > p90))
    return frac, p90, len(g)


def write_report(
    report_path: Path, stats: dict, headline: tuple[float, float, int],
    n_requested: int, n_actual: dict[str, int], spot_check_dir: Path,
    tta_scales: list[int], checkpoint_path: Path, fields_available: list[str],
) -> None:
    frac, p90, n_face = headline
    lines = []
    lines.append("# Face-occlusion sensitivity report\n")
    lines.append(
        "Tests whether `finetune_v0` relies on visible face-region tamper evidence for its "
        "fraud calls, or on some correlate of the editing tool that survives regardless of "
        "whether the face region itself is occluded.\n"
    )
    lines.append("## Methodology\n")
    lines.append(f"- Checkpoint: `{checkpoint_path}`")
    lines.append(f"- TTA scales (from checkpoint config): `{tta_scales}` — combined by **mean logit** across scales, not sigmoid+rank-average (see script docstring for why).")
    lines.append(
        "- `train_labels.csv` fields available: "
        f"`{fields_available}` — no tamper-type/category or tamper-location field exists. "
        "**TRAIN_FRAUD_FACE is therefore all real fraud rows (label==1), not verified "
        "face-region edits** — diluted by whatever fraction of real fraud is a non-face edit "
        "(unknown proportion)."
    )
    lines.append(
        "- All three groups restricted to samples with a real SCRFD detection in the regions "
        "cache (`face.json` `score > 0`) — required so there's an actual face region to "
        "occlude; excludes only a small fraction given ~100%/99.9% post-fix detection rates."
    )
    lines.append(
        "- `face_occluded`: face box expanded 10% per side, filled with the mean color of its "
        "own local surrounding ring. `control_occluded`: a same-sized box at a random "
        "non-overlapping location on the same image, filled the same way from ITS OWN local "
        "surroundings (not the face patch's color — avoids an unnaturally-colored patch acting "
        "as its own tamper cue)."
    )
    lines.append(
        f"- Requested n={n_requested}/group; actual: "
        + ", ".join(f"{g}={n}" for g, n in n_actual.items())
    )
    lines.append(f"- Spot-check renders (5 images, before the full run): `{spot_check_dir}`\n")

    lines.append("## Per-group results (logit space)\n")
    lines.append("| group | n | mean\\|Δface\\| | median\\|Δface\\| | mean\\|Δcontrol\\| | median\\|Δcontrol\\| | mean paired diff (face−control) | median paired diff | Wilcoxon p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for group, s in stats.items():
        lines.append(
            f"| {group} | {s['n']} | {s['mean_abs_face']:.4f} | {s['median_abs_face']:.4f} | "
            f"{s['mean_abs_control']:.4f} | {s['median_abs_control']:.4f} | "
            f"{s['mean_paired_diff']:.4f} | {s['median_paired_diff']:.4f} | {s['wilcoxon_p']:.4g} |"
        )

    lines.append("\n## Headline number (TRAIN_FRAUD_FACE)\n")
    lines.append(
        f"Of {n_face} TRAIN_FRAUD_FACE images, **{frac * 100:.1f}%** drop their fraud logit by "
        f"more than the group's own control-occlusion 90th-percentile drop "
        f"(p90 = {p90:.4f} logits) when the face region is occluded."
    )

    lines.append("\n## Interpretation guide\n")
    lines.append(
        "- **CONFIRMED** (shortcut hypothesis): face occlusion on TRAIN_FRAUD_FACE moves logits "
        "about as much as control occlusion (small/insignificant paired diff, Wilcoxon p not "
        "small, headline fraction close to 10% — i.e. no better than chance against the "
        "control's own 90th percentile). The model isn't reading the face for its fraud calls "
        "on this population."
    )
    lines.append(
        "- **REJECTED** (face IS read, but test-time artifacts are OOD): face occlusion "
        "collapses fraud logits on TRAIN_FRAUD_FACE (large paired diff, small Wilcoxon p, "
        "headline fraction well above ~10%) while HESITANT_TEST shows small \\|Δface\\| AND "
        "small \\|Δcontrol\\| overall (little logit movement either way — consistent with "
        "near-zero-evidence scoring on real test-time images, not with reading a face that "
        "isn't there or doesn't look like the training-domain tamper pattern)."
    )
    lines.append(
        "- Either outcome is informative. State plainly which one this run's numbers support, "
        "and the caveats above (fraud-type dilution, and that this is diagnostic of the "
        "*trained checkpoint's* actual behavior, not a claim about what would happen if "
        "retrained differently).\n"
    )

    lines.append("## Data-driven read of this run\n")
    face_stats = stats.get("TRAIN_FRAUD_FACE")
    hesitant_stats = stats.get("HESITANT_TEST")
    if face_stats is not None:
        close = face_stats["wilcoxon_p"] > 0.05 or abs(face_stats["mean_paired_diff"]) < 0.1
        big_gap = face_stats["wilcoxon_p"] < 0.05 and face_stats["mean_paired_diff"] > 0.1
        if close and not big_gap:
            verdict = (
                "Numbers lean **CONFIRMED**: face-occlusion effect on TRAIN_FRAUD_FACE is not "
                "meaningfully larger than the control-occlusion effect."
            )
        elif big_gap and hesitant_stats is not None and hesitant_stats["mean_abs_face"] < 0.5 and hesitant_stats["mean_abs_control"] < 0.5:
            verdict = (
                "Numbers lean **REJECTED**: face occlusion moves TRAIN_FRAUD_FACE logits "
                "substantially more than control occlusion, while HESITANT_TEST barely moves "
                "under either occlusion — consistent with the model reading real face evidence "
                "on training-distribution frauds, but public-test images sitting in a "
                "low-evidence regime for this model regardless of occlusion."
            )
        else:
            verdict = "Mixed / does not cleanly match either interpretation-guide pattern — inspect the per-group table and spot-check renders directly before concluding."
        lines.append(verdict)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[occlusion] wrote report -> {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--submission", default="submissions/finetune_v0.csv")
    parser.add_argument("--n-per-group", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expand-frac", type=float, default=0.10)
    parser.add_argument("--ring-frac", type=float, default=0.25)
    parser.add_argument("--time-guard-minutes", type=float, default=45.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--csv-out", default=str(DEFAULT_CSV))
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
    print(f"[occlusion] backbone={cfg.backbone} tta_scales={tta_scales} device={device}")

    rdir = regions_dir(cfg.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this experiment needs it (VESSL)")

    train_df = load_labels(cfg.data_dir, "train")
    fields_available = list(train_df.columns)
    print(f"[occlusion] train_labels.csv fields: {fields_available} (no tamper-type/category field)")

    fraud_ids = train_df.loc[train_df["label"] == 1, "id"].tolist()
    bonafide_ids = train_df.loc[train_df["label"] == 0, "id"].tolist()
    print(
        f"[occlusion] TRAIN_FRAUD_FACE fallback: sampling from all {len(fraud_ids)} real fraud "
        "rows (no face-region tamper-type field exists to filter on) -- see report for the "
        "dilution caveat."
    )

    n = args.n_per_group
    fraud_valid = sample_valid_face_ids_random(fraud_ids, rdir, rng, n)
    bonafide_valid = sample_valid_face_ids_random(bonafide_ids, rdir, rng, n)
    hesitant_ranked = hesitant_ranked_ids(args.submission, cfg.data_dir, n * 3)
    hesitant_valid = filter_valid_face_ids_ordered(hesitant_ranked, rdir, n)

    print(
        f"[occlusion] valid-face samples found: TRAIN_FRAUD_FACE={len(fraud_valid)}/{n}, "
        f"TRAIN_BONAFIDE={len(bonafide_valid)}/{n}, HESITANT_TEST={len(hesitant_valid)}/{n}"
    )

    test_df = load_labels(cfg.data_dir, "public_test")

    samples: list[OcclusionSample] = []
    samples += build_group(fraud_valid, "TRAIN_FRAUD_FACE", train_df, args.expand_frac, rng)
    samples += build_group(bonafide_valid, "TRAIN_BONAFIDE", train_df, args.expand_frac, rng)
    samples += build_group(hesitant_valid, "HESITANT_TEST", test_df, args.expand_frac, rng)

    n_actual = {g: sum(1 for s in samples if s.group == g) for g in ("TRAIN_FRAUD_FACE", "TRAIN_BONAFIDE", "HESITANT_TEST")}

    spot_render(samples, out_dir, n=5)

    tasks = [(s, v) for s in samples for v in VARIANTS]
    projected_s = calibrate_runtime(model, device, tasks, tta_scales, mean, std, args.batch_size, args.ring_frac)
    if projected_s > args.time_guard_minutes * 60:
        print(
            f"[occlusion] projected runtime ({projected_s / 60:.1f} min) exceeds guard "
            f"({args.time_guard_minutes:.0f} min) -- halving group sizes and rebuilding"
        )
        half = max(1, n // 2)
        fraud_valid, bonafide_valid = fraud_valid[:half], bonafide_valid[:half]
        hesitant_valid = hesitant_valid[:half]
        rng = np.random.default_rng(args.seed)  # reset for reproducible re-derivation
        samples = []
        samples += build_group(fraud_valid, "TRAIN_FRAUD_FACE", train_df, args.expand_frac, rng)
        samples += build_group(bonafide_valid, "TRAIN_BONAFIDE", train_df, args.expand_frac, rng)
        samples += build_group(hesitant_valid, "HESITANT_TEST", test_df, args.expand_frac, rng)
        n_actual = {g: sum(1 for s in samples if s.group == g) for g in ("TRAIN_FRAUD_FACE", "TRAIN_BONAFIDE", "HESITANT_TEST")}
        tasks = [(s, v) for s in samples for v in VARIANTS]

    print(f"[occlusion] scoring {len(samples)} images x {len(VARIANTS)} variants x {len(tta_scales)} scales")
    t_start = time.time()
    results = run_scoring(model, device, tasks, tta_scales, mean, std, args.batch_size, args.ring_frac)
    print(f"[occlusion] scoring done in {(time.time() - t_start) / 60:.1f} min")

    rows = []
    for s in samples:
        logit = {v: float(np.mean(list(results[(s.id, v)].values()))) for v in VARIANTS}
        delta_face = logit["face_occluded"] - logit["clean"]
        delta_control = logit["control_occluded"] - logit["clean"]
        rows.append({
            "id": s.id,
            "group": s.group,
            "label": s.label,
            "is_digital": s.is_digital,
            "doc_type": s.doc_type,
            "face_score": s.face_score,
            "logit_clean": logit["clean"],
            "logit_face_occluded": logit["face_occluded"],
            "logit_control_occluded": logit["control_occluded"],
            "delta_face": delta_face,
            "delta_control": delta_control,
            "abs_delta_face": abs(delta_face),
            "abs_delta_control": abs(delta_control),
        })
    df = pd.DataFrame(rows)
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[occlusion] wrote per-image csv -> {csv_path} ({len(df)} rows)")

    stats = compute_group_stats(df)
    headline = headline_fraction(df)
    write_report(
        Path(args.report), stats, headline, n, n_actual, out_dir, tta_scales,
        Path(args.checkpoint), fields_available,
    )

    for group, s in stats.items():
        print(
            f"[occlusion] {group}: n={s['n']} mean|Δface|={s['mean_abs_face']:.4f} "
            f"mean|Δcontrol|={s['mean_abs_control']:.4f} paired_diff={s['mean_paired_diff']:.4f} "
            f"p={s['wilcoxon_p']:.4g}"
        )
    frac, p90, n_face = headline
    print(f"[occlusion] headline: {frac * 100:.1f}% of TRAIN_FRAUD_FACE drop > control p90 ({p90:.4f})")


if __name__ == "__main__":
    main()
