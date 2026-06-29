"""Augmentation pipelines for training.

Albumentations-based; gated by cfg.extra["augment"] so existing configs are unaffected.
No horizontal flip anywhere — document text and orientation must be preserved.
"""

from __future__ import annotations

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
