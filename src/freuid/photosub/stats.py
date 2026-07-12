"""Cheap, no-reference forensic stats -- the canonical copy used by degradation_match.py and
baseline_stats.py.

Deliberately duplicated from scripts/analysis/hesitant_clusters.py's
laplacian_blur_score/fft_periodic_peak_score/jpeg_blockiness_score (identical implementations,
verified byte-for-byte against that module) rather than imported, for the same reason
freuid.photosub.template_regions duplicates GHOST_TEMPLATES: src/freuid must not depend on
scripts/analysis (the reverse is this repo's existing layering throughout). Keeping this the
SAME implementation as scripts/analysis/photosub_render_sheets.py's stats_check matters: the
MODE_C/D acceptance criterion (freuid.photosub.baseline_stats) computes a natural-baseline
threshold with these exact functions, so any drift between "what computed the threshold" and
"what's compared against it" would silently invalidate the check.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

STAT_COLS = ("blur_laplacian_var", "moire_fft_score", "blockiness_score")


def laplacian_blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian -- lower means blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fft_periodic_peak_score(gray: np.ndarray) -> float:
    """Halftone/moire proxy: how much a single frequency band's peak magnitude stands out
    above the surrounding annulus's median, in a fixed-size FFT (256x256)."""
    g = cv2.resize(gray, (256, 256)).astype(np.float64)
    g = g - g.mean()
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h, w = mag.shape
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(h, w) / 2)
    annulus = mag[(r >= 0.06) & (r <= 0.6)]
    if annulus.size == 0:
        return 0.0
    med = np.median(annulus)
    return float(annulus.max() / (med + 1e-6))


def jpeg_blockiness_score(gray: np.ndarray, block: int = 8) -> float:
    """Mean |gradient| at 8-pixel block-boundary columns/rows minus mean |gradient| at
    mid-block columns/rows -- a simple no-reference blockiness proxy. Higher = more visible
    8x8 block edges (heavier / repeated JPEG compression)."""
    g = gray.astype(np.float64)

    def one_axis(a: np.ndarray) -> float:
        diffs = np.abs(np.diff(a, axis=1))  # (H, W-1)
        n = diffs.shape[1]
        cols = np.arange(n)
        boundary = cols % block == (block - 1)
        interior = cols % block == (block // 2)
        if not boundary.any() or not interior.any():
            return 0.0
        return float(diffs[:, boundary].mean() - diffs[:, interior].mean())

    return float((one_axis(g) + one_axis(g.T)) / 2)


def region_stats(img: Image.Image) -> dict:
    """The 3 cheap stats for one already-cropped region."""
    gray = np.asarray(img.convert("L"))
    return {
        "blur_laplacian_var": laplacian_blur_score(gray),
        "moire_fft_score": fft_periodic_peak_score(gray),
        "blockiness_score": jpeg_blockiness_score(gray),
    }
