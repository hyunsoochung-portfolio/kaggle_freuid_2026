"""Census of finetune_v0's raw pre-sigmoid logits over the full public test corpus.

Motivation (see occlusion_report.md / hesitant_report.md): the occlusion + hesitant-cluster
experiments found finetune_v0's outputs are bimodal (a bona-fide floor ~-6.7 and a fraud
ceiling ~+12.5 in raw logit space) and that mid-RANK "hesitant" test images are mostly
logit-SATURATED at the ceiling -- i.e. the submission's rank ordering inside that saturated
block may be near-noise. This script measures how big that block is across the entire public
test corpus (not just the ~600-image samples those experiments used), and whether within-block
ordering is distinguishable from per-scale TTA noise.

Two stages, run separately since only the first needs a GPU:

    --stage infer   Rebuilds finetune_v0's exact 3-scale TTA inference (same transforms /
                     dataset / model as infer.py) but skips the sigmoid + rank-average step,
                     saving each scale's RAW LOGIT per present test id. Needs the finetuned
                     checkpoint + GPU (or a lot of patience on CPU -- see the runtime guard).
    --stage report  Pure pandas/matplotlib/scipy over the saved raw-logit CSV. No GPU, no
                     checkpoint needed. Produces logit_census_report.md + plots.

Nothing under src/freuid/ is touched; no training; no submissions written.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    REPO_ROOT,
    DEFAULT_CHECKPOINT,
    df_to_md,
    load_checkpoint,
    build_finetuned_model,
    device_and_seed,
)

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_CSV = OUT_DIR / "logit_census_raw.csv"
DEFAULT_REPORT_MD = OUT_DIR / "logit_census_report.md"
DEFAULT_PLOT_DIR = OUT_DIR / "logit_census_out"
DEFAULT_SUBMISSION_CSV = REPO_ROOT / "submissions" / "finetune_v0.csv"
DEFAULT_HESITANT_CSV = OUT_DIR / "hesitant_results.csv"
TOLERANCES = (0.5, 1.0, 2.0)


# --------------------------------------------------------------------------------------
# Stage: infer (GPU)
# --------------------------------------------------------------------------------------

def get_present_ids(cfg) -> set[str]:
    from freuid.data import load_labels
    submission = load_labels(cfg.data_dir, "public_test")
    present_mask = submission["path"].map(lambda p: Path(p).exists())
    return set(submission.loc[present_mask, "id"])


def resolve_tta_scales(cfg) -> list[int]:
    tta_cfg = cfg.extra.get("tta", False)
    if isinstance(tta_cfg, list):
        return [int(s) for s in tta_cfg]
    base_size = cfg.image_size
    step = max(32, (base_size // 6) & ~31)
    return [base_size - step, base_size, base_size + step]


def _score_scale_raw_logits(model, loader, device) -> list[float]:
    import torch
    from freuid.data import unpack_and_move, forward_with_extras
    logits: list[float] = []
    with torch.no_grad():
        for batch in loader:
            imgs, _, face_meta, face_crop = unpack_and_move(batch, device)
            out = forward_with_extras(model, imgs, face_meta, face_crop)
            logits.extend(out.squeeze(1).float().cpu().tolist())
    return logits


def run_inference_stage(args) -> Path:
    import torch
    from freuid.data import FreuidDataset
    from freuid.transforms import build_transforms, resolve_data_config
    from torch.utils.data import DataLoader

    cfg, state = load_checkpoint(args.checkpoint)
    device = device_and_seed(cfg)
    model = build_finetuned_model(cfg, state, device)
    present_ids = get_present_ids(cfg)
    print(f"[logit_census] {len(present_ids)} present public_test ids | device={device}")

    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    mean, std = data_cfg["mean"], data_cfg["std"]
    scales = args.scales if args.scales else resolve_tta_scales(cfg)
    print(f"[logit_census] scales={scales}")

    ids_order: list[str] | None = None
    per_scale_logits: dict[int, list[float]] = {}
    total_batches_all_scales = None
    t_start_all = time.time()

    for si, scale in enumerate(scales):
        tf = build_transforms(scale, False, mean, std)
        ds = FreuidDataset(cfg.data_dir, "public_test", tf, ids=present_ids)
        if ids_order is None:
            ids_order = [s.id for s in ds.samples]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        n_batches = len(loader)

        if si == 0:
            # Calibrate on the SECOND batch of the first scale, not the first: batch 0 pays a
            # one-time CUDA/cuDNN warmup + dataloader-worker-startup cost that is not
            # representative of steady-state throughput (measured 3.6s for batch 0 vs. a flat
            # ~1.15s for every batch after it on the A100 -- a >3x overestimate that wrongly
            # tripped the time guard on the first attempt). Batch 0 still gets scored, just
            # not timed for the projection.
            it = iter(loader)
            import torch as _torch
            from freuid.data import unpack_and_move, forward_with_extras

            def _score_one(batch):
                with _torch.no_grad():
                    imgs, _, face_meta, face_crop = unpack_and_move(batch, device)
                    out = forward_with_extras(model, imgs, face_meta, face_crop)
                    logits = out.squeeze(1).float().cpu().tolist()
                if device.type == "cuda":
                    _torch.cuda.synchronize()
                return logits

            warmup_logits = _score_one(next(it))
            t0 = time.time()
            calib_logits = _score_one(next(it))
            t_calib = time.time() - t0

            total_batches_all_scales = n_batches * len(scales)
            # batch 0 (warmup) and batch 1 (calib) both already scored -> 2 fewer remain.
            remaining_after_calib = total_batches_all_scales - 2
            projected_seconds = t_calib * remaining_after_calib
            print(
                f"[logit_census] warmup batch (untimed) + calibration batch: {t_calib:.2f}s for "
                f"{len(calib_logits)} images | projected wall-clock for the remaining "
                f"{remaining_after_calib} of {total_batches_all_scales} total batches "
                f"({len(scales)} scales x ~{n_batches} batches): {projected_seconds/60:.1f} min"
            )
            if projected_seconds > args.time_guard_minutes * 60 and not args.force:
                print(
                    f"[logit_census] ABORTING: projected {projected_seconds/60:.1f} min exceeds "
                    f"the {args.time_guard_minutes}-min guard. Re-run with --force to proceed "
                    "anyway, or run on a faster device (VESSL GPU)."
                )
                raise SystemExit(1)

            scale_logits = list(warmup_logits) + list(calib_logits)
            scale_logits.extend(_score_scale_raw_logits(model, it, device))
        else:
            scale_logits = _score_scale_raw_logits(model, loader, device)

        per_scale_logits[scale] = scale_logits
        print(f"[logit_census] scale={scale} done | n={len(scale_logits)} | elapsed={time.time()-t_start_all:.1f}s")

    elapsed = time.time() - t_start_all
    print(f"[logit_census] inference complete in {elapsed/60:.1f} min")

    df = pd.DataFrame({"id": ids_order})
    scale_cols = []
    for scale in scales:
        col = f"logit_{scale}"
        df[col] = per_scale_logits[scale]
        scale_cols.append(col)
    df["mean_logit"] = df[scale_cols].mean(axis=1)
    df["std_logit"] = df[scale_cols].std(axis=1, ddof=0)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[logit_census] wrote {len(df)} rows -> {args.out_csv}")
    return args.out_csv


# --------------------------------------------------------------------------------------
# Stage: report (CPU only)
# --------------------------------------------------------------------------------------

def find_modes(
    values: np.ndarray,
    grid_points: int = 1000,
    top_k: int = 2,
    relative_height_threshold: float = 0.01,
    min_relative_distance: float = 0.05,
    bandwidth_multiplier: float = 2.5,
):
    """KDE-based mode finder: fit a Gaussian KDE, deliberately oversmoothed
    (``bandwidth_multiplier`` x Scott's-rule bandwidth -- plain Scott's-rule bandwidth
    reliably manufactures a spurious close second lobe out of pure sampling noise even for
    a genuinely unimodal n~1000 Gaussian, confirmed empirically; 2.5x removes that artifact
    in repeated trials while a true, well-separated bimodal split some 15+ std-devs apart
    survives easily), evaluate it on a fine grid, and find local maxima via
    scipy.signal.find_peaks. Returns up to ``top_k`` peaks sorted by position, each as
    (position, density). ``min_relative_distance`` (as a fraction of the grid) already
    empirically eliminates every secondary peak for confirmed-unimodal data at n=1000-7821
    (measured max secondary/primary height ratio across 90 unimodal trials: 0.0 -- distance
    alone does the filtering work here), while a genuinely real minority mode down to ~2% of
    the population survives with height ratio >=0.02; ``relative_height_threshold`` is kept
    low (0.01) so it acts only as a second line of defense, not the primary filter, and won't
    swallow a real-but-small ceiling/floor cluster the way a higher threshold (e.g. 0.10) did
    in testing. Falls back to the single global-max grid point if find_peaks detects nothing
    at all."""
    from scipy.signal import find_peaks
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=np.float64)
    kde = gaussian_kde(values)
    kde.set_bandwidth(bw_method=kde.factor * bandwidth_multiplier)
    lo, hi = values.min(), values.max()
    grid = np.linspace(lo, hi, grid_points)
    density = kde(grid)

    distance = max(1, int(grid_points * min_relative_distance))
    peaks, _ = find_peaks(density, distance=distance)
    if len(peaks) == 0:
        i = int(np.argmax(density))
        return [(float(grid[i]), float(density[i]))]

    heights = density[peaks]
    max_height = heights.max()
    keep = heights >= relative_height_threshold * max_height
    peaks, heights = peaks[keep], heights[keep]

    order = np.argsort(heights)[::-1][:top_k]
    chosen = sorted(
        [(float(grid[p]), float(h)) for p, h in zip(peaks[order], heights[order])],
        key=lambda t: t[0],
    )
    return chosen


def block_membership(logit: np.ndarray, floor_mode: float, ceiling_mode: float, tol: float) -> np.ndarray:
    """Vectorized label: 'floor' / 'ceiling' / 'neither' for each value in ``logit``.

    If a value is within ``tol`` of both modes (only possible if the modes themselves are
    closer together than 2*tol), floor wins arbitrarily -- flagged separately by the caller
    if it ever happens, since it would indicate the tolerance is too large relative to the
    mode separation for this dataset.
    """
    logit = np.asarray(logit, dtype=np.float64)
    labels = np.full(logit.shape, "neither", dtype=object)
    labels[np.abs(logit - ceiling_mode) <= tol] = "ceiling"
    labels[np.abs(logit - floor_mode) <= tol] = "floor"
    return labels


def tolerance_table(logit: np.ndarray, floor_mode: float, ceiling_mode: float, tolerances=TOLERANCES) -> pd.DataFrame:
    n = len(logit)
    rows = []
    for tol in tolerances:
        labels = block_membership(logit, floor_mode, ceiling_mode, tol)
        n_floor = int((labels == "floor").sum())
        n_ceiling = int((labels == "ceiling").sum())
        n_neither = n - n_floor - n_ceiling
        rows.append({
            "tolerance": tol,
            "floor_n": n_floor, "floor_frac": n_floor / n,
            "ceiling_n": n_ceiling, "ceiling_frac": n_ceiling / n,
            "neither_n": n_neither, "neither_frac": n_neither / n,
        })
    return pd.DataFrame(rows)


def plot_histogram(logit: np.ndarray, floor_mode: float, ceiling_mode: float, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(logit, bins=200, color="steelblue", edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(floor_mode, color="tab:blue", linestyle="--", label=f"floor mode ~{floor_mode:.2f}")
    ax.axvline(ceiling_mode, color="tab:red", linestyle="--", label=f"ceiling mode ~{ceiling_mode:.2f}")
    ax.set_xlabel("raw mean logit (3-scale TTA, pre-sigmoid, pre-rank-average)")
    ax.set_ylabel("count (log scale)")
    ax.set_title("finetune_v0 public-test logit census (full corpus)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_vs_logit(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = df["block"].map({"floor": "tab:blue", "ceiling": "tab:red", "neither": "0.6"})
    ax.scatter(df["pct_rank"], df["mean_logit"], s=4, alpha=0.4, c=colors)
    ax.set_xlabel("final submission rank percentile (0-100)")
    ax.set_ylabel("raw mean logit")
    ax.set_title("Rank-vs-logit decoupling: does the ceiling block span a wide rank band?")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_scale_disagreement(df: pd.DataFrame, main_tol: float) -> dict:
    in_block = df["block"] == "ceiling"
    mean_std_in = float(df.loc[in_block, "std_logit"].mean()) if in_block.any() else float("nan")
    mean_std_out = float(df.loc[~in_block, "std_logit"].mean()) if (~in_block).any() else float("nan")
    between_id_spread_in = float(df.loc[in_block, "mean_logit"].std(ddof=1)) if in_block.sum() > 1 else float("nan")
    return {
        "tolerance_used": main_tol,
        "n_in_ceiling_block": int(in_block.sum()),
        "mean_per_id_scale_std_in_block": mean_std_in,
        "mean_per_id_scale_std_outside_block": mean_std_out,
        "between_id_logit_std_within_block": between_id_spread_in,
    }


def _relpath_or_abs(path: Path) -> str:
    """Path relative to REPO_ROOT for the markdown report, or the absolute path if ``path``
    lives outside the repo (e.g. an --out-dir pointed at a scratch/smoke-test location)."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_report(args) -> None:
    raw = pd.read_csv(args.raw_csv, dtype={"id": str})
    scale_cols = [c for c in raw.columns if c.startswith("logit_")]
    if "mean_logit" not in raw.columns:
        raw["mean_logit"] = raw[scale_cols].mean(axis=1)
    if "std_logit" not in raw.columns:
        raw["std_logit"] = raw[scale_cols].std(axis=1, ddof=0)

    modes = find_modes(raw["mean_logit"].to_numpy())
    bimodal = len(modes) >= 2
    if bimodal:
        floor_mode, ceiling_mode = modes[0][0], modes[-1][0]
    else:
        # Unimodal fallback: still need two reference points for the tolerance tables so the
        # rest of the pipeline runs, but this itself is the headline finding (see interpretation).
        floor_mode = ceiling_mode = modes[0][0]

    main_tol = args.tolerances[0]
    tol_table = tolerance_table(raw["mean_logit"].to_numpy(), floor_mode, ceiling_mode, args.tolerances)
    raw["block"] = block_membership(raw["mean_logit"].to_numpy(), floor_mode, ceiling_mode, main_tol)

    # Merge submission rank percentile for the decoupling scatter. sample_submission.csv (and
    # this exported copy of it) lists the FULL ~142.8k-id test set; only the ~7.8k present ids
    # have a real score, the rest are the missing_id_score=0.5 placeholder (see infer.py). Rank
    # percentile MUST be computed only among present ids -- ranking over the full file would let
    # ~135k tied placeholder rows dominate the middle of the distribution and squeeze every real
    # score to the extreme percentiles regardless of its actual value. Restricting to raw["id"]
    # (the present ids this census just scored) matches hesitant_clusters.py's convention.
    sub = pd.read_csv(args.submission_csv, dtype={"id": str})
    sub = sub[sub["id"].isin(set(raw["id"]))].copy()
    sub["pct_rank"] = sub["label"].rank(pct=True) * 100
    merged = raw.merge(sub[["id", "pct_rank"]], on="id", how="left")
    n_unmatched = int(merged["pct_rank"].isna().sum())

    # Hesitant-500 placement.
    hes_table = None
    if args.hesitant_csv.exists():
        hes = pd.read_csv(args.hesitant_csv, dtype={"id": str})
        hes = hes[hes["set"] == "HESITANT"][["id"]]
        hes_merged = hes.merge(raw[["id", "mean_logit"]], on="id", how="left")
        n_hes_total = len(hes_merged)
        n_hes_matched = int(hes_merged["mean_logit"].notna().sum())
        hes_rows = []
        for tol in args.tolerances:
            labels = block_membership(hes_merged["mean_logit"].dropna().to_numpy(), floor_mode, ceiling_mode, tol)
            n = len(labels)
            hes_rows.append({
                "tolerance": tol,
                "n_matched": n,
                "floor_frac": float((labels == "floor").sum()) / n if n else float("nan"),
                "ceiling_frac": float((labels == "ceiling").sum()) / n if n else float("nan"),
                "neither_frac": float((labels == "neither").sum()) / n if n else float("nan"),
            })
        hes_table = pd.DataFrame(hes_rows)
    else:
        n_hes_total = n_hes_matched = 0

    disagreement = per_scale_disagreement(merged, main_tol)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = args.out_dir / "logit_histogram.png"
    scatter_path = args.out_dir / "rank_vs_logit.png"
    plot_histogram(raw["mean_logit"].to_numpy(), floor_mode, ceiling_mode, hist_path)
    plot_rank_vs_logit(merged.dropna(subset=["pct_rank"]), scatter_path)

    ceiling_frac_main = float(tol_table.loc[tol_table["tolerance"] == main_tol, "ceiling_frac"].iloc[0])
    n_total = len(raw)
    n_ceiling_main = int(tol_table.loc[tol_table["tolerance"] == main_tol, "ceiling_n"].iloc[0])

    if not bimodal:
        verdict = (
            "**No second mode was detected** by the peak finder over the full corpus -- the "
            "bimodality seen in the ~600-image sample from the occlusion/hesitant experiments "
            "did NOT reproduce at full-corpus scale. This is the 'working theory failed' case "
            "from the pre-registered criteria: the noise-ordered-block theory should be treated "
            "as **not confirmed** by this run, and the hesitant-set behavior documented in "
            "occlusion_report.md / hesitant_report.md needs a different explanation than "
            "corpus-wide ceiling saturation. Flagging this loudly rather than reporting a single "
            "spurious mode as if it were meaningful."
        )
    elif ceiling_frac_main > 0.20:
        verdict = (
            f"**Ceiling block > 20% of the corpus** at tolerance={main_tol} "
            f"({ceiling_frac_main*100:.1f}%, {n_ceiling_main} of {n_total} ids). Per the "
            "pre-registered criteria, the noise-ordered-block theory is **CONFIRMED**: a "
            f"block of **{n_ceiling_main} ids** sits saturated at the ceiling mode "
            f"(~{ceiling_mode:.2f} raw logit), and per the per-scale-disagreement check below, "
            "this is the practical improvement budget for any fix that targets within-block "
            "ordering specifically (not a bound on total achievable AuDET improvement -- see "
            "hesitant_report.md's AuDET bound table for that separate calculation)."
        )
    elif ceiling_frac_main < 0.05:
        verdict = (
            f"**Ceiling block < 5% of the corpus** at tolerance={main_tol} "
            f"({ceiling_frac_main*100:.1f}%, {n_ceiling_main} of {n_total} ids). Per the "
            "pre-registered criteria: **the working theory failed this test.** The bimodality "
            "in the ~600-image sample used by the occlusion/hesitant experiments does not "
            "generalize to a large saturated block across the full public test corpus -- "
            "whatever is driving the hesitant-set's mid-rank/high-logit mismatch, it is NOT "
            "explained by 'most of the corpus sits in one saturated ceiling pile.' This should "
            "be read as a genuine negative result for that specific theory, not soft-pedaled."
        )
    else:
        verdict = (
            f"Ceiling block at tolerance={main_tol} is {ceiling_frac_main*100:.1f}% of the corpus "
            f"({n_ceiling_main} of {n_total} ids) -- in between the pre-registered 5%/20% "
            "thresholds. Reporting this straight, per the pre-registered instruction not to force "
            "a verdict in this band: see the tolerance-sensitivity table below for how much this "
            "conclusion moves with the block-width definition before drawing further conclusions."
        )

    lines = []
    lines.append("# Logit census report (full public test corpus)")
    lines.append("")
    lines.append(
        "Sizes the raw-logit ceiling/floor saturation found in the ~600-image occlusion/hesitant "
        "samples (see `occlusion_report.md`, `hesitant_report.md`) across all "
        f"{n_total} present public-test ids, to check whether the 'noise-ordered saturated block' "
        "theory holds at corpus scale rather than being a sample-specific artifact."
    )
    lines.append("")
    lines.append("## Modes detected")
    lines.append("")
    if bimodal:
        lines.append(f"- floor mode (bona-fide pole): **{floor_mode:.3f}**")
        lines.append(f"- ceiling mode (fraud pole): **{ceiling_mode:.3f}**")
    else:
        lines.append(
            f"- Only one mode detected at **{modes[0][0]:.3f}** -- see verdict below, this itself "
            "is the headline finding."
        )
    lines.append("")
    lines.append("## Full-corpus histogram")
    lines.append("")
    lines.append(f"![logit histogram]({_relpath_or_abs(hist_path)})")
    lines.append("")
    lines.append("Log-scaled y-axis so tail mass away from the two modes is visible.")
    lines.append("")
    lines.append("## Ceiling / floor block sizes (tolerance-sensitivity table)")
    lines.append("")
    lines.append(df_to_md(tol_table, float_fmt="{:.4f}"))
    lines.append("")
    lines.append(
        f"`floor`/`ceiling` = within `tolerance` logits of that mode; `neither` = everything "
        "else. Reported at 0.5/1.0/2.0 so the headline conclusion isn't an artifact of picking "
        "one threshold."
    )
    lines.append("")
    lines.append("## Where do the hesitant-500 ids sit?")
    lines.append("")
    if hes_table is not None:
        lines.append(
            f"{n_hes_matched}/{n_hes_total} hesitant-set ids (from `hesitant_clusters.py`'s "
            f"`set == 'HESITANT'` rows) matched to a raw logit in this run."
        )
        lines.append("")
        lines.append(df_to_md(hes_table, float_fmt="{:.4f}"))
    else:
        lines.append(f"`{args.hesitant_csv}` not found -- skipped this section.")
    lines.append("")
    lines.append("## Rank-vs-logit decoupling")
    lines.append("")
    if n_unmatched:
        lines.append(f"({n_unmatched} raw-logit ids had no matching row in `{args.submission_csv.name}` -- excluded from the scatter.)")
        lines.append("")
    lines.append(f"![rank vs logit]({_relpath_or_abs(scatter_path)})")
    lines.append("")
    lines.append(
        "Colored by block membership at tolerance="
        f"{main_tol} (red=ceiling, blue=floor, grey=neither). If the ceiling block spans a wide "
        "band of final rank percentiles rather than clustering at one end, that visually confirms "
        "ids inside it are being fine-ordered by something other than the raw logit itself."
    )
    lines.append("")
    lines.append("## Per-scale disagreement inside vs. outside the ceiling block")
    lines.append("")
    lines.append(f"(tolerance={disagreement['tolerance_used']}, n_in_block={disagreement['n_in_ceiling_block']})")
    lines.append("")
    lines.append(f"- mean per-id std across the 3 TTA scales, **inside** the ceiling block: **{disagreement['mean_per_id_scale_std_in_block']:.4f}**")
    lines.append(f"- mean per-id std across the 3 TTA scales, **outside** the ceiling block: **{disagreement['mean_per_id_scale_std_outside_block']:.4f}**")
    lines.append(f"- between-id spread of mean_logit **within** the ceiling block (population std across the {disagreement['n_in_ceiling_block']} block ids): **{disagreement['between_id_logit_std_within_block']:.4f}**")
    lines.append("")
    lines.append(
        "If within-block ordering is genuinely noise, the per-id scale-disagreement (how much a "
        "single id's own logit jitters across TTA scales) should be comparable to or larger than "
        "the between-id spread (how far apart two different ids' logits typically are) inside the "
        "same block -- i.e. an id's rank relative to its neighbors could plausibly flip just by "
        "changing which TTA scale you looked at."
    )
    lines.append("")
    lines.append("## Interpretation (pre-registered criteria)")
    lines.append("")
    lines.append(verdict)
    lines.append("")

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[logit_census] wrote report -> {args.report_md}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["infer", "report", "all"], default="all")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_RAW_CSV, help="raw per-scale logit CSV (infer stage output / report stage input)")
    p.add_argument("--raw-csv", type=Path, default=None, help="defaults to --out-csv")
    p.add_argument("--submission-csv", type=Path, default=DEFAULT_SUBMISSION_CSV)
    p.add_argument("--hesitant-csv", type=Path, default=DEFAULT_HESITANT_CSV)
    p.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_PLOT_DIR)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--scales", type=int, nargs="*", default=None, help="override TTA scales (default: from checkpoint config)")
    p.add_argument("--time-guard-minutes", type=float, default=10.0)
    p.add_argument("--force", action="store_true", help="proceed past the time guard anyway")
    p.add_argument("--tolerances", type=float, nargs="*", default=list(TOLERANCES))
    args = p.parse_args()
    if args.raw_csv is None:
        args.raw_csv = args.out_csv
    return args


def main():
    args = parse_args()
    if args.stage in ("infer", "all"):
        run_inference_stage(args)
    if args.stage in ("report", "all"):
        build_report(args)


if __name__ == "__main__":
    main()
