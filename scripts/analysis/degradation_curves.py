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

Usage: python scripts/analysis/degradation_curves.py [--checkpoint PATH] [--limit N]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap val images for speed")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    cfg, state = load_checkpoint(ckpt_path)
    device = device_and_seed(cfg)

    _, val_ids = get_split_ids(cfg)
    df = split_dataframe(cfg, val_ids)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=cfg.seed).reset_index(drop=True)
    print(f"[degradation] val n={len(df)}")

    model = build_finetuned_model(cfg, state, device)
    transform, data_cfg = eval_transform(cfg)
    native_size = data_cfg["image_size"]

    print("[degradation] loading source images into memory once...")
    raw_images = [Image.open(p).convert("RGB") for p in df["path"]]
    labels = df["label"].to_numpy()

    corruptions = build_corruptions(native_size)
    rows = []
    for name, fn in corruptions.items():
        corrupted = [fn(im) for im in raw_images]
        scores = score_images(model, corrupted, transform, device, batch_size=32)
        m = evaluate(scores, labels)
        rows.append({"corruption": name, "audet": m["audet"], "apcer_at_1pct_bpcer": m["apcer_at_1pct_bpcer"]})
        print(f"[degradation] {name:28s} AuDET={m['audet']:.6f} APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.6f}")

    result_df = pd.DataFrame(rows)
    out_dir = ensure_report_dir()
    result_df.to_csv(out_dir / "degradation_curves.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        colors = ["#4477AA" if r != "clean" else "#CC3311" for r in result_df["corruption"]]
        ax.bar(result_df["corruption"], result_df["audet"], color=colors)
        ax.axhline(result_df.loc[result_df["corruption"] == "clean", "audet"].iloc[0],
                   color="#CC3311", linestyle="--", linewidth=1, label="clean baseline")
        ax.set_ylabel("AuDET (lower = better)")
        ax.set_title("finetune_v0: AuDET under single-corruption degradation")
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "degradation_curves.png", dpi=150)
        print(f"[degradation] wrote {out_dir / 'degradation_curves.png'}")
    except ImportError:
        print("[degradation] matplotlib not available -- skipped plot, CSV still written")

    (out_dir / "degradation_curves_note.md").write_text(CARD_ONLY_NOTE + "\n", encoding="utf-8")
    print(f"[degradation] wrote {out_dir / 'degradation_curves.csv'}")


if __name__ == "__main__":
    main()
