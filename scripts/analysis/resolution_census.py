"""Resolution / compression census for train and public_test images.

Read-only: walks every train (69,352) and public_test (7,821) image, reading
dimensions straight from the JPEG header (``Image.open`` without ``.load()`` --
no pixel decode) plus the embedded quantization tables as a JPEG-quality proxy.
Answers one question: can training above finetune_v0's 518px possibly help, or
is the source resolution already the ceiling? Never touches ``src/freuid/``,
never trains, writes only under ``reports/res_precheck/``.

Usage: python scripts/analysis/resolution_census.py [--data-root PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.data import load_labels  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = (518, 784, 1036)

# Standard IJG luminance quantization table (quality=50), natural (row-major) order,
# reordered into zigzag scan order to match Image.quantization[0]'s element order.
_ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]
_STD_LUMA_NATURAL = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]
_STD_LUMA_ZIGZAG = [_STD_LUMA_NATURAL[i] for i in _ZIGZAG]


def estimate_jpeg_quality(img: Image.Image) -> float | None:
    """Approximate the JPEG encode quality from the luminance quantization table.

    Inverts the standard IJG quality->quantization-table scaling formula. This is a
    proxy (encoders vary), good enough to flag "heavily recompressed" sources, not an
    exact quality readout.
    """
    quant = getattr(img, "quantization", None)
    if not quant or 0 not in quant or len(quant[0]) != 64:
        return None
    scales = [
        (qi * 100.0 - 50.0) / base
        for qi, base in zip(quant[0], _STD_LUMA_ZIGZAG)
        if base > 0
    ]
    if not scales:
        return None
    scale = float(np.median(scales))
    if scale <= 0:
        return 100.0
    quality = (200.0 - scale) / 2.0 if scale < 100 else 5000.0 / scale
    return float(np.clip(quality, 1.0, 100.0))


def resolve_data_root(data_root: Path) -> Path:
    """Handle the local-checkout quirk where images live under data/raw/... instead
    of directly under data/... (VESSL layout). Prefer the direct layout; fall back to
    a `raw/` subdirectory if that's where `train/train` actually is.
    """
    if (data_root / "train" / "train").is_dir():
        return data_root
    if (data_root / "raw" / "train" / "train").is_dir():
        return data_root / "raw"
    raise SystemExit(
        f"could not find train/train under {data_root} or {data_root / 'raw'}"
    )


def census_split(root: Path, split: str) -> tuple[pd.DataFrame, int]:
    """Per-image (id, type, label, width, height, min_side, file_bytes) for one split.

    Returns (dataframe, n_missing_or_corrupt). For public_test, restricts to ids with
    a locally-present file (sample_submission.csv also carries private-LB placeholder
    ids that have no image on disk).
    """
    df = load_labels(root, split)
    if split == "public_test":
        present = df["path"].map(lambda p: Path(p).exists())
        df = df[present]
    records = []
    n_bad = 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"census[{split}]"):
        path = Path(row.path)
        try:
            with Image.open(path) as img:
                w, h = img.size
                quality = estimate_jpeg_quality(img)
        except Exception:
            n_bad += 1
            continue
        records.append({
            "id": row.id,
            "type": getattr(row, "type", None),
            "label": int(row.label),
            "width": w,
            "height": h,
            "min_side": min(w, h),
            "file_bytes": path.stat().st_size,
            "quality": quality,
        })
    return pd.DataFrame.from_records(records), n_bad


def side_stats(s: pd.Series) -> dict:
    return {
        "min": float(s.min()),
        "median": float(s.median()),
        "p90": float(np.percentile(s, 90)),
        "max": float(s.max()),
    }


def threshold_fracs(min_side: pd.Series) -> dict:
    return {f">={t}px": float((min_side >= t).mean()) for t in THRESHOLDS}


def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def make_hist(census: dict[str, pd.DataFrame], out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"train": "#4477AA", "public_test": "#CC6677"}
    for split, df in census.items():
        ax.hist(df["min_side"], bins=60, alpha=0.55, label=f"{split} (n={len(df)})",
                 color=colors.get(split, "#666666"), edgecolor="white", linewidth=0.2)
    for t in THRESHOLDS:
        ax.axvline(t, color="black", linestyle="--", linewidth=0.8)
        ax.text(t, ax.get_ylim()[1] * 0.95, f"{t}px", rotation=90, va="top", ha="right", fontsize=8)
    ax.set_xlabel("min(width, height) in pixels")
    ax.set_ylabel("count")
    ax.set_title("min-side resolution distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def verdict_res(p90_train: float, p90_test: float) -> tuple[int, float]:
    p90_min = min(p90_train, p90_test)
    if p90_min < 700:
        return 518, p90_min
    if p90_min < 1000:
        return 784, p90_min
    return 1036, p90_min


def native_ceiling_caveat(census: dict[str, pd.DataFrame], verdict_px: int) -> str | None:
    """Catch the boundary case where p90 lands exactly on a bucket edge (e.g. p90==1000.0)
    but essentially no image natively reaches the bucket's resolution -- the mechanical
    threshold rule would call it "viable" when it's actually pure upsampling.
    """
    if verdict_px not in THRESHOLDS:
        return None
    fracs = {split: float((df["min_side"] >= verdict_px).mean()) for split, df in census.items()}
    worst = min(fracs.values())
    if worst >= 0.05:
        return None
    max_native = max(df["min_side"].max() for df in census.values())
    frac_str = ", ".join(f"{s}={f * 100:.2f}%" for s, f in fracs.items())
    return (
        f"**CAVEAT -- boundary artifact**: the p90-based rule buckets this as {verdict_px}px, "
        f"but only {frac_str} of images natively reach >= {verdict_px}px (native max across "
        f"both splits = {max_native:.0f}px). p90 landing exactly on the bucket edge does not "
        f"mean headroom above it exists. Training at {verdict_px}px would upsample essentially "
        f"the entire dataset beyond its native resolution, which cannot add real signal -- "
        "prefer the next lower tier that most of the data natively clears."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "res_precheck"))
    parser.add_argument("--quality-sample-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = resolve_data_root(Path(args.data_root))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[res_census] data root resolved to {root}")

    census: dict[str, pd.DataFrame] = {}
    n_bad_by_split: dict[str, int] = {}
    for split in ("train", "public_test"):
        df, n_bad = census_split(root, split)
        census[split] = df
        n_bad_by_split[split] = n_bad
        df.to_csv(out_dir / f"census_{split}.csv", index=False)
        print(f"[res_census] {split}: {len(df)} ok, {n_bad} missing/corrupt")

    # 2) overall stats + threshold fractions per split
    overall_rows = []
    for split, df in census.items():
        row = {"split": split, "n": len(df)}
        for dim in ("width", "height", "min_side"):
            for k, v in side_stats(df[dim]).items():
                row[f"{dim}_{k}"] = v
        row.update(threshold_fracs(df["min_side"]))
        row["median_file_kb"] = float(df["file_bytes"].median() / 1024)
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows)

    # 3) per-type breakdown (train only -- public_test has no type metadata)
    train_df = census["train"]
    type_df = (
        train_df.groupby("type", dropna=False)["min_side"]
        .agg(n="count", median="median", p90=lambda s: np.percentile(s, 90))
        .reset_index()
        .sort_values("n", ascending=False)
    )

    # per-label breakdown (train only) -- shortcut-risk flag
    label_df = (
        train_df.groupby("label")["min_side"]
        .agg(n="count", median="median", p90=lambda s: np.percentile(s, 90))
        .reset_index()
        .sort_values("label")
    )
    med0 = label_df.loc[label_df["label"] == 0, "median"]
    med1 = label_df.loc[label_df["label"] == 1, "median"]
    label_shortcut_flag = False
    label_diff_pct = 0.0
    if len(med0) and len(med1) and med0.iloc[0] > 0:
        label_diff_pct = abs(med1.iloc[0] - med0.iloc[0]) / med0.iloc[0] * 100
        label_shortcut_flag = label_diff_pct > 10

    # 4) JPEG quality proxy on a random sample per split
    quality_rows = []
    rng = np.random.default_rng(args.seed)
    for split, df in census.items():
        n = min(args.quality_sample_n, len(df))
        sample = df.iloc[rng.choice(len(df), size=n, replace=False)]
        q = sample["quality"].dropna()
        quality_rows.append({
            "split": split,
            "n_sampled": n,
            "n_with_quant_table": len(q),
            "quality_p10": float(np.percentile(q, 10)) if len(q) else float("nan"),
            "quality_median": float(q.median()) if len(q) else float("nan"),
            "quality_p90": float(np.percentile(q, 90)) if len(q) else float("nan"),
        })
    quality_df = pd.DataFrame(quality_rows)
    low_quality_flag = any(
        not np.isnan(r["quality_median"]) and r["quality_median"] < 70 for r in quality_rows
    )

    # full-population (not just the sample) mode check -- quantization tables are read for
    # every image as a side effect of header parsing, so we can report true uniformity here
    # rather than just percentiles of a 500-image sample.
    quality_mode_notes = []
    for split, df in census.items():
        q = df["quality"].dropna()
        if q.empty:
            continue
        top = q.round(3).value_counts(normalize=True).head(3)
        desc = ", ".join(f"{v:.1f} ({p * 100:.1f}%)" for v, p in top.items())
        quality_mode_notes.append(f"- **{split}** (n={len(q)}): top estimated-quality values -> {desc}")

    # histogram
    hist_path = out_dir / "min_side_hist.png"
    has_plot = make_hist(census, hist_path)

    # verdict
    p90_train = overall_df.loc[overall_df["split"] == "train", "min_side_p90"].iloc[0]
    p90_test = overall_df.loc[overall_df["split"] == "public_test", "min_side_p90"].iloc[0]
    verdict_px, p90_min = verdict_res(p90_train, p90_test)

    # write markdown report
    lines = []
    lines.append("# Resolution / compression census\n")
    lines.append(
        f"Train: {len(census['train'])} images censused "
        f"({n_bad_by_split['train']} missing/corrupt). "
        f"Public test: {len(census['public_test'])} locally-present images censused "
        f"({n_bad_by_split['public_test']} missing/corrupt).\n"
    )

    lines.append("## Overall stats (pixels, file size)\n")
    lines.append(df_to_md(overall_df, float_fmt="{:.1f}") + "\n")

    if has_plot:
        lines.append("## min-side histogram\n")
        lines.append(f"![min-side histogram]({hist_path.name})\n")
    else:
        lines.append("_matplotlib not available -- histogram skipped._\n")

    lines.append("## Per-document-type breakdown (train)\n")
    lines.append(df_to_md(type_df, float_fmt="{:.1f}") + "\n")

    lines.append("## Per-label breakdown (train) -- shortcut-risk check\n")
    lines.append(df_to_md(label_df, float_fmt="{:.1f}") + "\n")
    if label_shortcut_flag:
        lines.append(
            f"**FLAG: label/resolution shortcut risk** -- median min_side differs by "
            f"{label_diff_pct:.1f}% between label=0 and label=1 (>10% threshold). "
            "A model could learn to key on resolution as a fraud proxy instead of content.\n"
        )
    else:
        lines.append(
            f"No shortcut flag: median min_side differs by {label_diff_pct:.1f}% between "
            "labels (<=10% threshold).\n"
        )

    lines.append("## JPEG quality proxy (quantization-table estimate)\n")
    lines.append(df_to_md(quality_df, float_fmt="{:.1f}") + "\n")
    if quality_mode_notes:
        lines.append(
            "Full-population mode check (all images, not just the 500-sample -- quantization "
            "tables are read as a side effect of header parsing so this is free):\n"
        )
        lines.append("\n".join(quality_mode_notes) + "\n")
    if low_quality_flag:
        lines.append(
            "**FLAG: heavy recompression** -- median estimated quality < 70 for at least one "
            "split. Recompression can cap useful resolution well below the raw pixel dimensions.\n"
        )
    else:
        lines.append("No heavy-recompression flag: median estimated quality >= 70 for both splits.\n")

    lines.append("## Verdict\n")
    lines.append(
        f"train min_side p90 = {p90_train:.1f}px, public_test min_side p90 = {p90_test:.1f}px "
        f"-> using the lower ({p90_min:.1f}px) against the {{700, 1000}}px cutoffs:\n"
    )
    lines.append(f"**VERDICT: max_useful_train_res = {verdict_px}px**\n")
    justification = {
        518: "p90 min-side < 700px -- the high-res axis is dead; training above 518px cannot "
             "add real signal for the bulk of the distribution.",
        784: "p90 min-side in [700, 1000)px -- 784px is viable for most of the distribution; "
             "1036px would starve the top decile of any relevant training resolution.",
        1036: "p90 min-side >= 1000px -- 1036px is viable; the source resolution supports "
              "training above finetune_v0's 518px for the bulk of the data.",
    }[verdict_px]
    lines.append(justification + "\n")

    caveat = native_ceiling_caveat(census, verdict_px)
    if caveat:
        lines.append(caveat + "\n")

    report_path = out_dir / "resolution_census.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[res_census] wrote {report_path}")
    print(f"VERDICT: max_useful_train_res = {verdict_px}px")


if __name__ == "__main__":
    main()
