"""Matches a donor patch's local sharpness/grain/JPEG statistics to the target region it is
being composited into.

MODE_C/D's entire evidence model depends on the composite NOT being locally distinguishable by
cheap forensic stats (blur, JPEG blockiness, periodic/moire texture) -- only by cross-region
semantic mismatch (a different person, a ghost that doesn't match). The first version of these
generators only color-matched the donor (freuid.photosub.generators._match_color); it did
nothing about sharpness, compression grain, or periodic texture, and the render-sheet self-check
(docs/photosub_renders/stats_check_report.md) confirmed this was a real, not theoretical,
problem. This module fixes that at composite time, before the donor patch is pasted:

  1. ``match_moire`` -- for a donor LESS periodic than the target, blends in a faint fixed-
     frequency dot texture; for a donor MORE periodic (the common case for a clean face crop --
     see match_moire's own docstring for why blur doesn't fix this), adds broadband noise.
  2. ``match_blur`` -- Gaussian-blur the donor until its Laplacian-variance blur score is no
     higher than the target region's own.
  3. ``match_jpeg_grain`` -- re-encodes through JPEG at the quality level whose blockiness score
     is closest to the target's, approximating the card's own compression history.

**Order matters and was tuned empirically, not assumed**: moire-matching's noise-injection
branch can blow up the Laplacian-variance blur score by ~1-2 orders of magnitude (broadband
noise IS high-frequency energy, which is exactly what that score measures) -- running
match_blur AFTER match_moire lets its own bisection re-correct that side effect (it always
re-measures against the target fresh, regardless of what upstream step caused the current
score), whereas running match_blur first and match_moire second left blur devastated with no
later step to fix it back. Verified directly on a real render example: blur-then-moire left a
final blur delta of ~10,670 (vs. a 1207 p90 threshold, a bad fail); moire-then-blur left final
deltas of ~23 (blur) and ~16 (moire), both comfortably inside their thresholds. The two
mechanisms still aren't perfectly orthogonal (a small amount of back-and-forth disturbance is
inherent to using overlapping frequency-domain tools for two different metrics), but this
ordering is the empirically-better one, not a theoretical guess.

All three searches are coarse (binary search / a small quality grid) rather than exact
optimization -- "no worse than the natural intra-card baseline" (freuid.photosub.baseline_stats)
is the bar that matters, not an exact numeric match.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

from freuid.photosub.stats import fft_periodic_peak_score, jpeg_blockiness_score, laplacian_blur_score

_MAX_BLUR_RADIUS = 6.0
_BLUR_SEARCH_ITERS = 10
_JPEG_QUALITY_RANGE = (35, 95)
_JPEG_QUALITY_STEP = 5
_MOIRE_MAX_STRENGTH = 30.0
_MOIRE_SEARCH_ITERS = 8
_MOIRE_CELL_PX = 3.0  # fine, high-frequency dot spacing -- a subtle texture nudge, not a visible pattern
_MOIRE_MAX_NOISE_STD = 60.0  # broadband noise std cap for the "donor too periodic" direction


def _gray_arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"))


def match_blur(donor: Image.Image, target_region: Image.Image, max_radius: float = _MAX_BLUR_RADIUS) -> Image.Image:
    """Blur ``donor`` down to (at most) ``target_region``'s own Laplacian-variance blur score.
    No-op if the donor is already at or below that score."""
    target_score = laplacian_blur_score(_gray_arr(target_region))
    donor_score = laplacian_blur_score(_gray_arr(donor))
    if donor_score <= target_score:
        return donor

    lo, hi = 0.0, max_radius
    best = donor.filter(ImageFilter.GaussianBlur(max_radius))
    for _ in range(_BLUR_SEARCH_ITERS):
        mid = (lo + hi) / 2.0
        candidate = donor.filter(ImageFilter.GaussianBlur(mid))
        score = laplacian_blur_score(_gray_arr(candidate))
        if score > target_score:
            lo = mid  # still too sharp -- blur more
        else:
            hi = mid
            best = candidate
    return best


def _fine_dot_texture(size: tuple[int, int], cell: float = _MOIRE_CELL_PX) -> np.ndarray:
    """A regular, fine-grained sinusoidal dot grid in [-1, 1] -- a periodic signal at a fixed
    small spacing, used only to nudge fft_periodic_peak_score, not to look like print halftone
    (print_style.py's halftone transform is a separate, much coarser, visibly-styled effect)."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.sin(2 * np.pi * xx / cell) * np.sin(2 * np.pi * yy / cell)


def match_moire(
    donor: Image.Image,
    target_region: Image.Image,
    max_strength: float = _MOIRE_MAX_STRENGTH,
    max_noise_std: float = _MOIRE_MAX_NOISE_STD,
) -> Image.Image:
    """Nudges ``donor``'s moire_fft_score toward ``target_region``'s own. Real render-sheet data
    showed this mismatch runs in BOTH directions, not just one: a diagnostic trace on an actual
    generated example found the color-matched donor patch scored ~126 against a target region
    scoring ~45 -- the donor was MORE periodic, not less (a face crop's skin/hair texture can
    carry a narrow-band frequency signature a blur pass doesn't remove, since Laplacian-variance
    blur and this FFT-annulus peak ratio are different axes: match_blur already ran and left this
    donor's blur score untouched). So:

      - donor LESS periodic than target: blend in a faint fixed-frequency dot texture
        (increasing strength) to raise the peak.
      - donor MORE periodic than target: add broadband (white) Gaussian pixel noise
        (increasing std). A first attempt at this direction tried MORE Gaussian BLUR, on the
        assumption blur would suppress periodicity the same way it suppresses Laplacian-variance
        sharpness -- measured directly and found to do the OPPOSITE (blurring this donor raised
        its moire score from 126 to 656 at radius 3.0): the FFT-annulus peak/median ratio isn't
        a "high-frequency energy" measure the way Laplacian variance is, so a low-pass filter
        doesn't reliably lower it. Broadband noise does: it raises the annulus's MEDIAN (roughly
        uniform-spectrum energy added across all bins) faster than it raises any single existing
        peak bin, verified to move this same donor from 126 down through the target's ~45 by
        std~30 (measured directly before committing to this mechanism).

    Both searches track the CLOSEST attempt seen (like match_jpeg_grain) rather than the first
    one that crosses the target, so an unreachable target within the search range still returns
    the best available approximation instead of silently falling back to the untouched donor."""
    target_score = fft_periodic_peak_score(_gray_arr(target_region))
    donor_score = fft_periodic_peak_score(_gray_arr(donor))
    if donor_score == target_score:
        return donor

    best_img, best_diff = donor, abs(donor_score - target_score)
    donor_arr = np.asarray(donor.convert("RGB"), dtype=np.float64)

    if donor_score < target_score:
        texture = _fine_dot_texture(donor.size)
        for i in range(1, _MOIRE_SEARCH_ITERS + 1):
            strength = max_strength * i / _MOIRE_SEARCH_ITERS
            candidate_arr = np.clip(donor_arr + texture[..., None] * strength, 0, 255).astype(np.uint8)
            candidate = Image.fromarray(candidate_arr)
            score = fft_periodic_peak_score(_gray_arr(candidate))
            diff = abs(score - target_score)
            if diff < best_diff:
                best_diff = diff
                best_img = candidate
    else:
        rng = np.random.default_rng(0)  # deterministic -- see module docstring
        for i in range(1, _MOIRE_SEARCH_ITERS + 1):
            std = max_noise_std * i / _MOIRE_SEARCH_ITERS
            noise = rng.normal(0.0, std, size=donor_arr.shape)
            candidate_arr = np.clip(donor_arr + noise, 0, 255).astype(np.uint8)
            candidate = Image.fromarray(candidate_arr)
            score = fft_periodic_peak_score(_gray_arr(candidate))
            diff = abs(score - target_score)
            if diff < best_diff:
                best_diff = diff
                best_img = candidate

    return best_img


def match_jpeg_grain(
    donor: Image.Image,
    target_region: Image.Image,
    quality_range: tuple[int, int] = _JPEG_QUALITY_RANGE,
    quality_step: int = _JPEG_QUALITY_STEP,
) -> Image.Image:
    """Re-encode ``donor`` through JPEG at whichever quality (searched over
    ``quality_range``) brings its blockiness score closest to ``target_region``'s own."""
    target_score = jpeg_blockiness_score(_gray_arr(target_region))
    best_img, best_diff = donor, float("inf")
    lo, hi = quality_range
    for q in range(hi, lo - 1, -quality_step):
        buf = io.BytesIO()
        donor.convert("RGB").save(buf, format="JPEG", quality=q)
        buf.seek(0)
        candidate = Image.open(buf).convert("RGB")
        score = jpeg_blockiness_score(_gray_arr(candidate))
        diff = abs(score - target_score)
        if diff < best_diff:
            best_diff = diff
            best_img = candidate
    return best_img


def match_degradation(donor: Image.Image, target_region: Image.Image) -> Image.Image:
    """Full pipeline: moire-match, then blur-match, then JPEG/grain-match ``donor`` to
    ``target_region``'s local degradation profile. Deterministic (no rng -- every sub-step is a
    closest-match search, not sampling). This ORDER is empirically tuned, not arbitrary -- see
    the module docstring for why blur must run AFTER moire, not before."""
    out = match_moire(donor, target_region)
    out = match_blur(out, target_region)
    out = match_jpeg_grain(out, target_region)
    return out
