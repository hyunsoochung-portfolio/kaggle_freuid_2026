"""Augmentation pipelines for training.

Albumentations-based; gated by cfg.extra["augment"] so existing configs are unaffected.
No horizontal flip anywhere — document text and orientation must be preserved.
"""

from __future__ import annotations

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


class _AlbumentationsTransform:
    """Wraps an albumentations Compose so it accepts PIL Images (same contract as
    torchvision transforms.Compose) and returns a CHW float32 tensor."""

    def __init__(self, pipeline: A.Compose) -> None:
        self.pipeline = pipeline

    def __call__(self, img):
        arr = np.array(img)  # PIL RGB → HWC uint8
        return self.pipeline(image=arr)["image"]  # CHW float32 tensor


def recapture_transforms(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    seed: int | None = None,
) -> _AlbumentationsTransform:
    """Print-and-recapture simulation.

    ``seed`` makes the degradation reproducible (albumentations 2.x seeds the pipeline, not
    the global RNG). Pass a seed to get a fixed, identical degradation -- used to keep the
    validation analog copies stable across epochs. Leave None for random train augmentation.

    Ordered to match the real analog degradation chain:
      1. spatial resize (to the network's input resolution)
      2. first JPEG encode (capture / upload to a system)
      3. downscale (lower-res sensor or scan at reduced DPI)
      4. second JPEG encode (double-compression artifact)
      5. optical: focus blur or motion blur
      6. sensor noise
      7. lighting & colour shifts
      8. mild perspective warp + small rotation (handling/scan tilt)
      9. ImageNet normalize + to tensor

    std_range for GaussNoise is fractional relative to uint8 max (255), so
    (0.01, 0.04) → ~2.5–10 pixel std — subtle but visible grain.
    """
    pipeline = A.Compose([
        A.Resize(image_size, image_size),
        # first JPEG compression: simulate saving/uploading the captured image
        A.ImageCompression(quality_range=(50, 95), p=0.9),
        # downscale: lower-resolution sensor or reduced-DPI scan, then bicubic back up
        A.Downscale(
            scale_range=(0.5, 0.85),
            interpolation_pair={"downscale": 2, "upscale": 2},  # INTER_CUBIC
            p=0.5,
        ),
        # second JPEG pass: double-compression creates distinctive block artifacts
        A.ImageCompression(quality_range=(60, 95), p=0.7),
        # optical degradation: defocus or motion blur from handheld capture
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.5),
        # sensor / scan noise
        A.GaussNoise(std_range=(0.01, 0.04), p=0.5),
        # lighting: exposure and contrast variation from ambient / scanner lamp
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # colour: white-balance shift from mixed lighting or scanner calibration
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
        # geometry: mild perspective from non-flat capture angle (NOT a flip)
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        # small rotation: phone tilt / document not perfectly square in scanner
        A.Rotate(limit=5, p=0.3),
        # ImageNet normalise + CHW tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], seed=seed)
    return _AlbumentationsTransform(pipeline)


# ---------------------------------------------------------------------------
# Synthetic tamper edits
# ---------------------------------------------------------------------------

def _copy_move(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Clone a rectangular patch from one location and paste it elsewhere."""
    h, w = arr.shape[:2]
    ph = rng.integers(max(1, h // 10), max(2, h // 4))
    pw = rng.integers(max(1, w // 10), max(2, w // 4))
    y1 = int(rng.integers(0, h - ph))
    x1 = int(rng.integers(0, w - pw))
    y2 = int(rng.integers(0, h - ph))
    x2 = int(rng.integers(0, w - pw))
    out = arr.copy()
    out[y2:y2 + ph, x2:x2 + pw] = arr[y1:y1 + ph, x1:x1 + pw]
    return out


def _local_splice(arr: np.ndarray, donor_arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paste a patch from a donor image into arr (simulates photo substitution)."""
    h, w = arr.shape[:2]
    # Resize donor to match target resolution so patch pixels are compatible
    donor_resized = np.array(Image.fromarray(donor_arr).resize((w, h), Image.BILINEAR))
    ph = rng.integers(max(1, h // 6), max(2, h // 3))
    pw = rng.integers(max(1, w // 6), max(2, w // 3))
    sy = int(rng.integers(0, h - ph))
    sx = int(rng.integers(0, w - pw))
    dy = int(rng.integers(0, h - ph))
    dx = int(rng.integers(0, w - pw))
    out = arr.copy()
    out[dy:dy + ph, dx:dx + pw] = donor_resized[sy:sy + ph, sx:sx + pw]
    return out


def _field_smudge(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blur, recolor, or fill a document-field-shaped region (simulates field alteration)."""
    h, w = arr.shape[:2]
    # Typical document field: wide and short
    fh = rng.integers(max(1, h // 15), max(2, h // 5))
    fw = rng.integers(max(1, w // 5), max(2, w // 2))
    y = int(rng.integers(0, h - fh))
    x = int(rng.integers(0, w - fw))
    out = arr.copy()
    region = out[y:y + fh, x:x + fw]

    edit = int(rng.integers(3))
    if edit == 0:
        # blur: simulate out-of-focus re-photograph of a re-printed field
        blurred = Image.fromarray(region).filter(ImageFilter.GaussianBlur(radius=4))
        out[y:y + fh, x:x + fw] = np.array(blurred)
    elif edit == 1:
        # recolor: shift brightness/saturation to simulate digitally edited text
        shift = rng.integers(-50, 50, size=3).astype(np.int16)
        out[y:y + fh, x:x + fw] = np.clip(region.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    else:
        # fill + noise: white-out and reprint simulation
        fill = region.mean(axis=(0, 1)).astype(np.int16)
        noise = rng.integers(-25, 25, size=region.shape).astype(np.int16)
        out[y:y + fh, x:x + fw] = np.clip(fill + noise, 0, 255).astype(np.uint8)
    return out


class AnalogDoubleDataset(Dataset):
    """Doubles the dataset with analog (recapture) copies of the digital images.

    The training set becomes {every image, clean transform} ∪ {every is_digital=True image
    again, recapture transform}. Each digital document is thus seen both as-is and as a
    simulated print-and-recapture, with its original label preserved -- teaching the model
    the digital AND the analog appearance (the digital→physical shift the hidden test probes).
    is_digital=False images (already physically captured) are not duplicated; is_digital=None
    (unknown provenance) is likewise left un-doubled.

    The base dataset must be built with transform=None so this wrapper owns all transforms.
    Yields (image, label); the consistency face-meta path is not supported here.

    ``make_analog`` is a factory ``seed -> transform`` (e.g. a partial of recapture_transforms).
    ``deterministic_seed``: set it for VALIDATION -- each analog copy is then built from a fresh
    pipeline seeded with (deterministic_seed + sample_index), so the recaptured val images are
    identical every epoch (albumentations 2.x seeds the pipeline, not the global RNG, so this is
    the only way to fix them). Leave None for training (one shared random pipeline).
    """

    def __init__(
        self, base: Dataset, clean_transform, make_analog, deterministic_seed: int | None = None
    ) -> None:
        self.base = base
        self.clean_tf = clean_transform
        self.make_analog = make_analog          # seed (int|None) -> analog transform
        self.det_seed = deterministic_seed
        self._analog_tf = make_analog(None)     # one random pipeline, reused for train copies
        # indices of the digital samples -- these get a second, recaptured copy appended.
        # `if s.is_digital` excludes both False and None (unknown provenance -> not doubled).
        self.analog_idx = [i for i, s in enumerate(base.samples) if s.is_digital]  # type: ignore[attr-defined]
        self.n = len(base.samples)  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return self.n + len(self.analog_idx)

    def __getitem__(self, idx: int):
        if idx < self.n:
            sample, tf = self.base.samples[idx], self.clean_tf  # type: ignore[attr-defined]
        else:
            orig_i = self.analog_idx[idx - self.n]
            sample = self.base.samples[orig_i]  # type: ignore[attr-defined]
            # val: fresh pipeline seeded per-sample -> identical every epoch.
            # train: the shared random pipeline -> fresh degradation each pass.
            tf = self.make_analog(self.det_seed + orig_i) if self.det_seed is not None \
                else self._analog_tf
        src = sample.card_path if sample.card_path is not None else sample.path
        img = Image.open(src).convert("RGB")
        return tf(img), sample.label
