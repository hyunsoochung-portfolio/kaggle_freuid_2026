"""Augmentation pipelines for training.

Albumentations-based; gated by cfg.extra["augment"] so existing configs are unaffected.
No horizontal flip anywhere — document text and orientation must be preserved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
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
) -> _AlbumentationsTransform:
    """Print-and-recapture simulation for TRAIN images.

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
    ])
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


_EDIT_NAMES = ("copy_move", "local_splice", "field_smudge")


def synth_tamper(
    arr: np.ndarray,
    rng: np.random.Generator,
    donor_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Apply one random tamper edit to a HWC uint8 image array.

    Returns (edited_array, edit_name). The caller is responsible for passing
    the result through recapture_transforms before feeding it to the model, so
    the tamper is always seen through the analog hole.

    Chooses uniformly among copy_move, field_smudge, and local_splice (only
    when donor_arr is provided). Falls back to copy_move if splice is drawn
    but no donor is available.
    """
    ops: list[str] = ["copy_move", "field_smudge"]
    if donor_arr is not None:
        ops.append("local_splice")
    name = ops[int(rng.integers(len(ops)))]
    if name == "copy_move":
        return _copy_move(arr, rng), name
    if name == "field_smudge":
        return _field_smudge(arr, rng), name
    # local_splice
    if donor_arr is not None:
        return _local_splice(arr, donor_arr, rng), name
    return _copy_move(arr, rng), "copy_move"  # fallback


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

_MAX_DONOR_POOL = 200  # pre-loaded bona-fide images for splice; ~100 MB at 384px


class SynthTamperWrapper(Dataset):
    """Wraps a FreuidDataset, converting bona-fide samples to synthetic fraud.

    For each batch item whose underlying label is 0 (bona-fide), applies
    synth_tamper with probability `prob` and returns label 1. All other items
    pass through with their original label and transform.

    The tampered path always uses `tamper_transform` (recapture_transforms) so
    the edit is seen through the analog hole before the model.  The clean path
    uses `clean_transform`.

    The base dataset must be created with transform=None so this wrapper owns
    all transform decisions.

    Note on multiprocessing: each DataLoader worker gets a forked copy of this
    wrapper including its rng state.  Workers therefore diverge independently,
    which is fine for training (approximate prob is correct) but means exact
    per-sample augmentation is not reproducible across num_workers settings.
    """

    def __init__(
        self,
        base: Dataset,
        clean_transform,
        tamper_transform,
        prob: float,
        seed: int = 0,
    ) -> None:
        self.base = base
        self.clean_tf = clean_transform
        self.tamper_tf = tamper_transform
        self.prob = prob
        self._rng = np.random.default_rng(seed)

        # Pre-load a donor pool of bona-fide images for splice operations.
        # Done once at init so workers share the read-only pool without extra I/O.
        # Prefer rectified card_path when available (use_rectify path).
        donor_paths: list[Path] = [
            s.card_path if s.card_path is not None else s.path
            for s in getattr(base, "samples", [])
            if s.label == 0
        ]
        self._rng.shuffle(donor_paths)  # type: ignore[arg-type]
        self._donor_pool: list[np.ndarray] = []
        for p in donor_paths[:_MAX_DONOR_POOL]:
            try:
                self._donor_pool.append(np.array(Image.open(p).convert("RGB")))
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int):
        sample = self.base.samples[idx]  # type: ignore[attr-defined]
        src = sample.card_path if sample.card_path is not None else sample.path
        img_pil = Image.open(src).convert("RGB")
        label = sample.label

        if label == 0 and self._rng.random() < self.prob:
            arr = np.array(img_pil)
            donor = None
            if self._donor_pool:
                donor = self._donor_pool[int(self._rng.integers(len(self._donor_pool)))]
            tampered, _ = synth_tamper(arr, self._rng, donor)
            img_out = self.tamper_tf(Image.fromarray(tampered))
            return img_out, 1

        if self.clean_tf is not None:
            img_pil = self.clean_tf(img_pil)
        return img_pil, label
