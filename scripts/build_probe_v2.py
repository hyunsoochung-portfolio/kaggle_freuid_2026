"""Build probe_v2: an independent, deterministic degraded copy of the recapture-probe split.

    python scripts/build_probe_v2.py --config configs/finetune_v0.yaml \\
        --out-dir data/processed/probe_v2

Why this exists: probe_v1 (the per-epoch recapture probe used for checkpointing, see
src/freuid/train.py) degrades images with the SAME recapture_transforms function used to
train arm A -- so a model that overfits to that exact augmentation's statistical
fingerprint would look good on probe_v1 for reasons that have nothing to do with real
analog-hole robustness. probe_v2 is a second, independent judge: a disjoint set of
degradation primitives (different library calls, different component families, only
partially-overlapping parameter ranges -- see reports/ab_recapture/probe_v2_spec.md),
generated ONCE to disk with a fixed seed so every model/arm evaluated against it sees
bit-identical files.

Degradation chain (per severity level: mild / default / harsh):
  1. ordered-dither halftone overlay (numpy Bayer-matrix threshold, blended) -- simulates
     print screening; recapture_transforms has nothing like this at all.
  2. downscale/upscale via PIL (NEAREST down, BILINEAR up) -- recapture_transforms uses
     albumentations' cv2-backed Downscale with INTER_CUBIC both ways.
  3. JPEG re-encode via PIL's Image.save(..., format="JPEG") -- albumentations'
     ImageCompression is cv2.imencode-backed; PIL is a fully separate codec path. A single
     pass (recapture does two).
  4. box or median blur via PIL ImageFilter -- recapture_transforms uses
     Gaussian/motion blur.
  5. radial vignette + linear illumination tilt (numpy) -- not present in
     recapture_transforms at all.

Output layout: <out-dir>/<severity>/<id>.png (lossless container for the already-baked
JPEG artifacts). Labels are not duplicated here -- evaluation scripts join back to
train_labels.csv by id. Never commit these; data/* is already gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

from freuid.config import load_config
from freuid.data import stratified_split

GLOBAL_SEED = 20260704  # fixed, independent of any training/probe_v1 seed

SEVERITIES: dict[str, dict] = {
    "mild": dict(
        halftone_alpha=(0.05, 0.10),
        downscale_range=(0.55, 0.70),
        jpeg_quality=(65, 80),
        blur_radius=(1, 2),
        vignette_strength=(0.05, 0.15),
    ),
    "default": dict(
        halftone_alpha=(0.12, 0.18),
        downscale_range=(0.40, 0.60),
        jpeg_quality=(45, 65),
        blur_radius=(2, 4),
        vignette_strength=(0.15, 0.30),
    ),
    "harsh": dict(
        halftone_alpha=(0.20, 0.30),
        downscale_range=(0.25, 0.45),
        jpeg_quality=(25, 45),
        blur_radius=(4, 7),
        vignette_strength=(0.30, 0.50),
    ),
}

_BAYER4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float64
) / 16.0


def _rng_for(sample_id: str, severity: str) -> np.random.Generator:
    """Deterministic per-(id, severity) RNG, independent of iteration order or worker count."""
    key = f"{GLOBAL_SEED}:{severity}:{sample_id}".encode()
    seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _ordered_dither_overlay(arr: np.ndarray, rng: np.random.Generator, alpha_range: tuple[float, float]) -> np.ndarray:
    """Blend a Bayer-matrix ordered-dither halftone pattern into each channel.

    Simulates the dot-screen texture of offset/inkjet printing -- a component family with
    no analogue in recapture_transforms (which has no dithering/halftone step at all).
    """
    h, w = arr.shape[:2]
    tile = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    alpha = float(rng.uniform(*alpha_range))
    out = arr.astype(np.float64)
    for c in range(arr.shape[2]):
        norm = out[:, :, c] / 255.0
        dithered = (norm > tile).astype(np.float64) * 255.0
        out[:, :, c] = (1 - alpha) * out[:, :, c] + alpha * dithered
    return np.clip(out, 0, 255).astype(np.uint8)


def _downscale_upscale_pil(img: Image.Image, rng: np.random.Generator, scale_range: tuple[float, float]) -> Image.Image:
    """PIL-only resample, different interpolation pair than training's cv2 CUBIC/CUBIC."""
    w, h = img.size
    scale = float(rng.uniform(*scale_range))
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
    return small.resize((w, h), Image.BILINEAR)


def _jpeg_pil(img: Image.Image, rng: np.random.Generator, quality_range: tuple[int, int]) -> Image.Image:
    q = int(rng.integers(quality_range[0], quality_range[1] + 1))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _box_or_median_blur(img: Image.Image, rng: np.random.Generator, radius_range: tuple[int, int]) -> Image.Image:
    radius = int(rng.integers(radius_range[0], radius_range[1] + 1))
    if rng.random() < 0.5:
        return img.filter(ImageFilter.BoxBlur(radius))
    # MedianFilter needs an odd kernel size
    k = radius * 2 + 1
    return img.filter(ImageFilter.MedianFilter(size=k))


def _vignette(arr: np.ndarray, rng: np.random.Generator, strength_range: tuple[float, float]) -> np.ndarray:
    """Radial vignette blended with a linear illumination tilt (numpy only)."""
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cy = h / 2 + rng.uniform(-0.1, 0.1) * h
    cx = w / 2 + rng.uniform(-0.1, 0.1) * w
    r = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    strength = float(rng.uniform(*strength_range))
    radial_mask = 1.0 - strength * np.clip(r, 0, 1) ** 2

    angle = rng.uniform(0, 2 * np.pi)
    tilt = (xx / w) * np.cos(angle) + (yy / h) * np.sin(angle)
    tilt = (tilt - tilt.min()) / (tilt.max() - tilt.min() + 1e-8)
    tilt_mask = 1.0 - 0.3 * strength * tilt

    mask = (radial_mask * tilt_mask)[:, :, None]
    return np.clip(arr.astype(np.float64) * mask, 0, 255).astype(np.uint8)


def degrade(img: Image.Image, sample_id: str, severity: str) -> Image.Image:
    """Apply the full probe_v2 chain for one severity level, deterministically."""
    params = SEVERITIES[severity]
    rng = _rng_for(sample_id, severity)

    arr = np.array(img.convert("RGB"))
    arr = _ordered_dither_overlay(arr, rng, params["halftone_alpha"])
    img2 = Image.fromarray(arr)
    img2 = _downscale_upscale_pil(img2, rng, params["downscale_range"])
    img2 = _jpeg_pil(img2, rng, params["jpeg_quality"])
    img2 = _box_or_median_blur(img2, rng, params["blur_radius"])
    arr2 = np.array(img2)
    arr2 = _vignette(arr2, rng, params["vignette_strength"])
    return Image.fromarray(arr2)


def _process_one(args: tuple[str, str, str, str]) -> tuple[str, str, bool]:
    """Worker fn (module-level so it's picklable for ProcessPoolExecutor)."""
    sid, severity, src, out_path = args
    src_p, out_p = Path(src), Path(out_path)
    if out_p.exists():
        return sid, severity, True
    if not src_p.exists():
        return sid, severity, False
    img = Image.open(src_p).convert("RGB")
    degraded = degrade(img, sid, severity)
    degraded.save(out_p)
    return sid, severity, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finetune_v0.yaml",
                         help="config to source data_dir/seed/val_fraction from (same probe split as probe_v1)")
    parser.add_argument("--out-dir", default="data/processed/probe_v2")
    parser.add_argument("--severities", nargs="+", default=list(SEVERITIES),
                         help=f"subset of {list(SEVERITIES)} to generate")
    parser.add_argument("--limit", type=int, default=None, help="cap ids for a quick smoke run")
    parser.add_argument("--workers", type=int, default=32, help="process pool size (CPU-bound PIL work)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _, val_ids = stratified_split(cfg.data_dir, cfg.val_fraction, cfg.seed)
    val_ids_sorted = sorted(val_ids)
    if args.limit:
        val_ids_sorted = val_ids_sorted[: args.limit]
    print(f"[build_probe_v2] {len(val_ids_sorted)} probe ids (same split as probe_v1, "
          f"seed={cfg.seed}, val_fraction={cfg.val_fraction})")

    from freuid.data import load_labels
    df = load_labels(cfg.data_dir, "train").set_index("id")

    out_root = Path(args.out_dir)
    work: list[tuple[str, str, str, str]] = []
    for severity in args.severities:
        out_dir = out_root / severity
        out_dir.mkdir(parents=True, exist_ok=True)
        for sid in val_ids_sorted:
            work.append((sid, severity, str(df.loc[sid, "path"]), str(out_dir / f"{sid}.png")))

    print(f"[build_probe_v2] {len(work)} (id, severity) pairs across {args.workers} workers")
    counts: dict[str, dict[str, int]] = {s: {"ok": 0, "missing": 0} for s in args.severities}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process_one, w) for w in work]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="probe_v2"):
            sid, severity, ok = fut.result()
            counts[severity]["ok" if ok else "missing"] += 1

    for severity, c in counts.items():
        print(f"[build_probe_v2] {severity}: {c['ok']} written/present, {c['missing']} source images missing")

    print(f"[build_probe_v2] done -> {out_root}/ (not committed; data/* is gitignored)")


if __name__ == "__main__":
    main()
