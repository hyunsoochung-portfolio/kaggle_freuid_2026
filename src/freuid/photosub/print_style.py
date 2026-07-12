"""Print-style transforms for MODE_A/B physical-paste generators.

Simulates what a hand-cut, re-printed photo looks like before it's taped/pasted onto a card and
recaptured: grayscale, a simple rotated-dot-screen halftone, sepia, contrast crush, and paper
grain. These are the "style-mismatched printed portrait" half of the MODE_A/B taxonomy (see
scripts/analysis/deep_miss_dossiers.py's module docstring); the physical rim/shadow/paste
geometry lives in generators.py.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

PRINT_STYLES = ("grayscale", "halftone", "sepia", "contrast_crush", "paper_grain")


def _halftone_dot_mask(gray01: np.ndarray, angle_deg: float, cell: float) -> np.ndarray:
    """Simple rotated-dot-screen: a grid of dots (rotated by ``angle_deg``, spaced ``cell``
    px apart) whose radius grows with local darkness. Returns a boolean array, True = ink
    (black) dot."""
    h, w = gray01.shape
    theta = np.deg2rad(angle_deg)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    px = np.mod(xr, cell) - cell / 2.0
    py = np.mod(yr, cell) - cell / 2.0
    dist = np.sqrt(px ** 2 + py ** 2)
    darkness = 1.0 - gray01  # gray01 in [0, 1], 0 = black
    radius = darkness * (cell / 2.0) * 1.35  # >1 lets near-black regions fully fill the cell
    return dist < radius


def apply_print_style(patch: Image.Image, style: str, rng: np.random.Generator) -> Image.Image:
    """Apply one print-style transform to an RGB PIL patch. Deterministic given ``rng``'s state."""
    if style == "grayscale":
        return patch.convert("L").convert("RGB")

    if style == "halftone":
        gray01 = np.asarray(patch.convert("L"), dtype=np.float64) / 255.0
        angle = float(rng.uniform(10.0, 50.0))
        cell = float(rng.uniform(4.0, 8.0))
        dots = _halftone_dot_mask(gray01, angle, cell)
        out = np.where(dots, 0, 255).astype(np.uint8)
        return Image.fromarray(out).convert("RGB")

    if style == "sepia":
        gray = np.asarray(patch.convert("L"), dtype=np.float64)
        r = np.clip(gray * 1.07 + 15.0, 0, 255)
        g = np.clip(gray * 0.94 + 5.0, 0, 255)
        b = np.clip(gray * 0.74, 0, 255)
        out = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return Image.fromarray(out)

    if style == "contrast_crush":
        arr = np.asarray(patch.convert("RGB"), dtype=np.float64)
        mean = arr.mean()
        factor = float(rng.uniform(1.6, 2.4))
        out = np.clip((arr - mean) * factor + mean, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    if style == "paper_grain":
        arr = np.asarray(patch.convert("RGB"), dtype=np.float64)
        noise = rng.normal(0.0, 10.0, size=arr.shape[:2])
        out = np.clip(arr + noise[..., None], 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    raise ValueError(f"unknown print style {style!r}; expected one of {PRINT_STYLES}")
