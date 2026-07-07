"""A second, independently-designed degradation pipeline for the unseen-domain probe.

The per-epoch recapture probe (`freuid.augment.recapture_transforms`) tests a model against
the SAME perturbation family it was trained to survive -- it is circular by construction, and
has saturated: finetune_v0/v2/bayar_dinov2_v0 all hit ~0 or exact 0.0 on it, including the
bayar_dinov2_v0 checkpoint that then regressed ~2.9x on the public leaderboard. That checkpoint
proved the recapture probe isn't just insensitive between two good candidates (its known
limit) -- it is blind to real regressions.

probe_v2 exists to catch that blind spot. It degrades images through physical mechanisms that
recapture_transforms does not model at all, so a model that learned genuine print-and-capture
robustness should hold up here too, while a model that overfit to recapture_transforms's
specific perturbation family should score measurably worse. Deliberately disjoint from
recapture_transforms:

  - halftone/moire interference (print halftone screen aliasing against the sensor grid) --
    absent upstream entirely
  - vignetting (radial illumination falloff) -- upstream's RandomBrightnessContrast is spatially
    uniform, not radial
  - specular glare (flash reflection off a laminated card) -- absent upstream entirely
  - chromatic aberration (per-channel spatial shift) -- upstream's HueSaturationValue is a
    global colour shift, not a geometric one
  - barrel/pincushion lens distortion (radial remap) -- upstream only has affine
    Perspective + Rotate
  - Poisson shot noise + salt-and-pepper dust -- upstream only has additive Gaussian noise
  - posterization (bit-depth reduction) -- absent upstream
  - single-pass JPEG at a harsher, non-overlapping quality range -- upstream double-compresses
    at quality in [50,95] then [60,95]; here it's one pass in [15,50]

JPEG is the one mechanism that's unavoidably shared (every real digital capture is JPEG-coded
somewhere in the chain) -- its parameterisation is deliberately shifted, not just re-randomised,
to reduce distributional overlap with the training-time transform.

Reproducibility follows the same convention as `freuid.train._run_probe`: this module reads the
GLOBAL `random`/`np.random` state (not a private Generator), so seeding via `random.seed(seed)`
+ `np.random.seed(seed)` before scoring reproduces an identical degraded image set every run.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A


def _halftone_moire(arr: np.ndarray) -> np.ndarray:
    """Print halftone screen aliasing against the recapture sensor's pixel grid."""
    h, w = arr.shape[:2]
    angle = random.uniform(0, np.pi)
    freq = random.uniform(0.35, 0.55)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grid = 0.5 + 0.5 * np.sin(2 * np.pi * freq * (xx * np.cos(angle) + yy * np.sin(angle)))
    amp = random.uniform(0.05, 0.15)
    grid = (1.0 - amp) + amp * grid
    out = arr.astype(np.float32) * grid[..., None]
    # alias the grid: downsample then upsample at a mismatched ratio so a moire beat survives
    scale = random.uniform(0.55, 0.85)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    small = cv2.resize(out, (sw, sh), interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0, 255).astype(np.uint8)


def _vignette(arr: np.ndarray) -> np.ndarray:
    """Radial illumination falloff from off-axis or ambient lighting."""
    h, w = arr.shape[:2]
    cy = h / 2 + random.uniform(-0.1, 0.1) * h
    cx = w / 2 + random.uniform(-0.1, 0.1) * w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    strength = random.uniform(0.25, 0.5)
    mask = 1.0 - strength * np.clip(r / (r.max() + 1e-6), 0, 1) ** 2
    out = arr.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _specular_glare(arr: np.ndarray) -> np.ndarray:
    """Bright elliptical highlight, simulating a flash reflection off a laminated card."""
    h, w = arr.shape[:2]
    cy, cx = random.uniform(0.1, 0.9) * h, random.uniform(0.1, 0.9) * w
    ry = max(1.0, random.uniform(0.08, 0.22) * h)
    rx = max(1.0, random.uniform(0.08, 0.22) * w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    glare = np.clip(1.0 - d, 0, 1) ** 2
    strength = random.uniform(80, 180)
    out = arr.astype(np.float32) + glare[..., None] * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def _chromatic_aberration(arr: np.ndarray) -> np.ndarray:
    """Per-channel spatial shift, simulating lens dispersion at the sensor."""
    shift = random.randint(1, 3)
    out = arr.copy()
    out[:, :, 0] = np.roll(arr[:, :, 0], shift, axis=1)
    out[:, :, 2] = np.roll(arr[:, :, 2], -shift, axis=1)
    return out


def _barrel_distortion(arr: np.ndarray) -> np.ndarray:
    """Radial lens distortion (barrel or pincushion) -- a geometric mechanism distinct from
    recapture_transforms's affine Perspective/Rotate."""
    h, w = arr.shape[:2]
    k = random.uniform(-0.15, 0.15)
    camera_matrix = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k, 0, 0, 0, 0], dtype=np.float32)
    return cv2.undistort(arr, camera_matrix, dist_coeffs)


def _shot_and_dust(arr: np.ndarray) -> np.ndarray:
    """Poisson (shot) noise plus occasional salt-and-pepper dust specks."""
    scale = random.uniform(15, 40)
    noisy = np.random.poisson(arr.astype(np.float32) / 255.0 * scale) / scale * 255.0
    out = np.clip(noisy, 0, 255).astype(np.uint8)
    if random.random() < 0.4:
        h, w = out.shape[:2]
        n_specks = random.randint(20, 150)
        ys = np.random.randint(0, h, n_specks)
        xs = np.random.randint(0, w, n_specks)
        vals = np.random.choice([0, 255], n_specks)
        out[ys, xs] = vals[:, None]
    return out


def _posterize(arr: np.ndarray) -> np.ndarray:
    """Bit-depth reduction -- a colour-fidelity degradation distinct from HSV shift."""
    bits = random.randint(3, 5)
    shift = 8 - bits
    return ((arr >> shift) << shift).astype(np.uint8)


def _jpeg_single_pass(arr: np.ndarray) -> np.ndarray:
    """One harsh JPEG pass in a quality range that does not overlap the training-time chain's
    [50,95]+[60,95] double pass."""
    q = random.randint(15, 50)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


# Ordered to mirror the physical chain: lens geometry and print/sensor interaction happen at
# capture time, illumination and optics next, then noise and finally digitisation/storage.
_STEPS = [
    (_barrel_distortion, 0.6),
    (_halftone_moire, 0.6),
    (_vignette, 0.5),
    (_specular_glare, 0.35),
    (_chromatic_aberration, 0.5),
    (_shot_and_dust, 0.6),
    (_posterize, 0.4),
    (_jpeg_single_pass, 0.9),
]


class Probe2Transform:
    """PIL Image -> CHW tensor, applying the probe_v2 degradation chain.

    Same call contract as `freuid.augment._AlbumentationsTransform` (and therefore a drop-in
    replacement for `recapture_transforms` anywhere that expects a callable transform).

    ``normalize=True`` (default) ImageNet-normalizes the result, for the main DINOv2 input.
    ``normalize=False`` instead scales to a bare [0,1] float tensor with no mean/std shift --
    for bayar_fusion's face-crop stream, whose OverlayStream normalizes internally for its
    own RGB branch (see its docstring); normalizing here too would double-normalize it.
    """

    def __init__(self, image_size: int, mean: tuple[float, float, float],
                 std: tuple[float, float, float], normalize: bool = True) -> None:
        self.image_size = image_size
        if normalize:
            self._finalize = A.Compose([A.Normalize(mean=mean, std=std), ToTensorV2()])
        else:
            self._finalize = A.Compose([A.ToFloat(max_value=255.0), ToTensorV2()])

    def __call__(self, img) -> torch.Tensor:
        arr = np.array(img.convert("RGB"))
        arr = cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        for step, prob in _STEPS:
            if random.random() < prob:
                arr = step(arr)
        return self._finalize(image=arr)["image"]


def probe_v2_transforms(image_size: int, mean: tuple[float, float, float],
                         std: tuple[float, float, float], normalize: bool = True) -> Probe2Transform:
    """Build a probe_v2 transform -- mirrors `recapture_transforms`'s signature exactly."""
    return Probe2Transform(image_size, mean, std, normalize=normalize)
