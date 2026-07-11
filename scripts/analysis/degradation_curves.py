"""Robustness-to-degradation curves for the finetune_v0 checkpoint.

Scores the same validation split under one input corruption at a time (never combined) and
reports AuDET per corruption family/level. No FastSAM/document-segmentation is available for
finetune_v0 (it doesn't use extra.use_rectify), so "card-only" and "center-card-masked" are
implemented as plain center-crop / center-mask approximations of "keep only the document
region" / "keep only the surrounding background" -- NOT true segmentation-based card
extraction. This is called out explicitly in the output so it isn't over-interpreted.

Interpretation guide (written into the report, not just this docstring):
  - If AuDET stays near 0 even at the smallest downscale (64px) or under center-card-masking
    (only background visible, document content hidden), the model is likely exploiting a
    global/background shortcut rather than local tamper evidence in the document itself.
  - If AuDET degrades sharply under blur/downscale/jpeg but stays low for card-only crops,
    that's consistent with the model actually using document-level forensic/content signal.

Resumable by design (the VESSL workspace's container can silently recycle mid-run, wiping
/tmp and the python env -- see project memory): each invocation processes ONE corruption
(or all not-yet-done ones if --corruption is omitted), and immediately upserts its result
into degradation_curves.csv before exiting, so a kill between corruptions loses at most one
corruption's ~4-5 min of work, not the whole sweep.

Usage:
    python scripts/analysis/degradation_curves.py --list
    python scripts/analysis/degradation_curves.py --corruption blur_sigma2
    python scripts/analysis/degradation_curves.py            # runs all not-yet-done, one by one
    python scripts/analysis/degradation_curves.py --plot-only  # just (re)draw the plot from the CSV
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    build_finetuned_model,
    device_and_seed,
    ensure_report_dir,
    eval_transform,
    get_split_ids,
    load_checkpoint,
    score_images,
    split_dataframe,
)
from freuid.metrics import evaluate  # noqa: E402

CARD_ONLY_NOTE = (
    "No document rectification/segmentation is available for finetune_v0 (extra.use_rectify "
    "is unset). 'card_only' and 'center_masked' below are plain center-crop / center-mask "
    "approximations, not true FastSAM-segmented card extraction -- treat them as coarse "
    "proxies for 'mostly document' vs 'mostly background', not exact isolations."
)
RESULT_CSV_NAME = "degradation_curves.csv"


def _downscale_then_up(img: Image.Image, size: int, target: int) -> Image.Image:
    small = img.resize((size, size), Image.BICUBIC)
    return small.resize((target, target), Image.BICUBIC)


def _jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")


def _blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def _center_masked(img: Image.Image, border_frac: float = 0.15) -> Image.Image:
    """Zero out the center, keep only a `border_frac`-wide border ring visible."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    by, bx = int(h * border_frac), int(w * border_frac)
    fill = arr.mean(axis=(0, 1)).astype(arr.dtype)
    out = arr.copy()
    out[by:h - by, bx:w - bx] = fill
    return Image.fromarray(out)


def _card_only(img: Image.Image, keep_frac: float = 0.70) -> Image.Image:
    """Center-crop to `keep_frac` of each side, then resize back up (approximates 'card only')."""
    w, h = img.size
    cw, ch = int(w * keep_frac), int(h * keep_frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    crop = img.crop((left, top, left + cw, top + ch))
    return crop.resize((w, h), Image.BICUBIC)


def build_corruptions(native_size: int) -> dict[str, callable]:
    corruptions: dict[str, callable] = {"clean": lambda im: im}
    for s in (64, 112, 224):
        corruptions[f"downscale_{s}px"] = lambda im, s=s: _downscale_then_up(im, s, native_size)
    for sigma in (1, 2, 4):
        corruptions[f"blur_sigma{sigma}"] = lambda im, sigma=sigma: _blur(im, sigma)
    corruptions["grayscale"] = _grayscale
    for q in (30, 50, 70):
        corruptions[f"jpeg_q{q}"] = lambda im, q=q: _jpeg_recompress(im, q)
    corruptions["center_masked_15pct_border"] = lambda im: _center_masked(im, 0.15)
    corruptions["card_only_70pct_crop"] = lambda im: _card_only(im, 0.70)
    return corruptions


def _load_existing(out_dir: Path) -> pd.DataFrame:
    path = out_dir / RESULT_CSV_NAME
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["corruption", "audet", "apcer_at_1pct_bpcer", "freuid"])


def _upsert(out_dir: Path, name: str, audet: float, apcer: float, freuid: float) -> pd.DataFrame:
    df = _load_existing(out_dir)
    df = df[df["corruption"] != name]
    df = pd.concat([df, pd.DataFrame([{"corruption": name, "audet": audet,
                                        "apcer_at_1pct_bpcer": apcer, "freuid": freuid}])],
                    ignore_index=True)
    df.to_csv(out_dir / RESULT_CSV_NAME, index=False)
    return df


def make_plot(result_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[degradation] matplotlib not available -- skipped plot")
        return
    result_df = result_df.sort_values("corruption")
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#4477AA" if r != "clean" else "#CC3311" for r in result_df["corruption"]]
    ax.bar(result_df["corruption"], result_df["audet"], color=colors)
    if "clean" in result_df["corruption"].values:
        ax.axhline(result_df.loc[result_df["corruption"] == "clean", "audet"].iloc[0],
                   color="#CC3311", linestyle="--", linewidth=1, label="clean baseline")
        ax.legend()
    ax.set_ylabel("AuDET (lower = better)")
    ax.set_title("finetune_v0: AuDET under single-corruption degradation")
    ax.tick_params(axis="x", rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / "degradation_curves.png", dpi=150)
    print(f"[degradation] wrote {out_dir / 'degradation_curves.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap val images for speed")
    parser.add_argument("--corruption", default=None, help="run only this corruption; omit to run all not-yet-done")
    parser.add_argument("--force", action="store_true", help="recompute even if already in the CSV")
    parser.add_argument("--list", action="store_true", help="print available corruption names and exit")
    parser.add_argument("--plot-only", action="store_true", help="just (re)draw the plot from the existing CSV")
    args = parser.parse_args()

    out_dir = ensure_report_dir()

    if args.list:
        print("\n".join(build_corruptions(0).keys()))
        return

    if args.plot_only:
        make_plot(_load_existing(out_dir), out_dir)
        return

    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    cfg, state = load_checkpoint(ckpt_path)

    existing = _load_existing(out_dir)
    done = set(existing["corruption"]) if not args.force else set()

    _, val_ids = get_split_ids(cfg)
    df = split_dataframe(cfg, val_ids)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=cfg.seed).reset_index(drop=True)

    transform, data_cfg = eval_transform(cfg)
    native_size = data_cfg["image_size"]
    all_corruptions = build_corruptions(native_size)

    if args.corruption:
        if args.corruption not in all_corruptions:
            raise SystemExit(f"unknown corruption {args.corruption!r}; --list for options")
        todo = {args.corruption: all_corruptions[args.corruption]}
    else:
        todo = {k: v for k, v in all_corruptions.items() if k not in done}

    if not todo:
        print("[degradation] nothing to do -- all corruptions already in the CSV (use --force to redo)")
        make_plot(_load_existing(out_dir), out_dir)
        return

    print(f"[degradation] val n={len(df)}  todo={list(todo)}  already_done={sorted(done)}")
    device = device_and_seed(cfg)
    model = build_finetuned_model(cfg, state, device)

    print("[degradation] loading source images into memory...")
    raw_images = [Image.open(p).convert("RGB") for p in df["path"]]
    labels = df["label"].to_numpy()

    for name, fn in todo.items():
        corrupted = [fn(im) for im in raw_images]
        scores = score_images(model, corrupted, transform, device, batch_size=32)
        m = evaluate(scores, labels)
        _upsert(out_dir, name, m["audet"], m["apcer_at_1pct_bpcer"], m["freuid"])
        print(f"[degradation] {name:28s} AuDET={m['audet']:.6f} APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.6f} "
              f"FREUID={m['freuid']:.6f} -- checkpointed to {RESULT_CSV_NAME}")

    make_plot(_load_existing(out_dir), out_dir)
    (out_dir / "degradation_curves_note.md").write_text(CARD_ONLY_NOTE + "\n", encoding="utf-8")
    print(f"[degradation] done. see {out_dir / RESULT_CSV_NAME}")


if __name__ == "__main__":
    main()
