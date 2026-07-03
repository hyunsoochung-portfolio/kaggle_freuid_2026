"""Synthetic tamper generation WITH ground-truth region tracking.

Replicates `src/freuid/augment.py`'s `_copy_move` / `_field_smudge` / `_local_splice` /
`synth_tamper` logic verbatim (same distributions, same rng usage) rather than modifying
that file, per the task's explicit preference to avoid touching training code. The only
difference is each function also returns the pixel-space bounding box of the region it
edited, so downstream code can check whether an attribution map actually points at the
tampered area.

Also replicates `recapture_transforms` as an equivalent albumentations pipeline with
`bbox_params` attached, so the ground-truth bbox is transformed *jointly* with the image
through the full degradation chain -- including the geometric steps (perspective warp,
rotation) that a plain resize-ratio bbox rescale could not account for. This keeps the
attribution test faithful to the actual analog-hole distribution the model trained on
(tampered images are always seen through this pipeline, never raw), without needing to
change `augment.py` to accept bbox targets.
"""

from __future__ import annotations

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFilter

BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2), pixel coords, pascal_voc convention


def _copy_move(arr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, BBox]:
    h, w = arr.shape[:2]
    ph = rng.integers(max(1, h // 10), max(2, h // 4))
    pw = rng.integers(max(1, w // 10), max(2, w // 4))
    y1 = int(rng.integers(0, h - ph))
    x1 = int(rng.integers(0, w - pw))
    y2 = int(rng.integers(0, h - ph))
    x2 = int(rng.integers(0, w - pw))
    out = arr.copy()
    out[y2:y2 + ph, x2:x2 + pw] = arr[y1:y1 + ph, x1:x1 + pw]
    return out, (x2, y2, x2 + pw, y2 + ph)  # ground truth = the pasted-into (destination) region


def _local_splice(arr: np.ndarray, donor_arr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, BBox]:
    h, w = arr.shape[:2]
    donor_resized = np.array(Image.fromarray(donor_arr).resize((w, h), Image.BILINEAR))
    ph = rng.integers(max(1, h // 6), max(2, h // 3))
    pw = rng.integers(max(1, w // 6), max(2, w // 3))
    sy = int(rng.integers(0, h - ph))
    sx = int(rng.integers(0, w - pw))
    dy = int(rng.integers(0, h - ph))
    dx = int(rng.integers(0, w - pw))
    out = arr.copy()
    out[dy:dy + ph, dx:dx + pw] = donor_resized[sy:sy + ph, sx:sx + pw]
    return out, (dx, dy, dx + pw, dy + ph)


def _field_smudge(arr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, BBox]:
    h, w = arr.shape[:2]
    fh = rng.integers(max(1, h // 15), max(2, h // 5))
    fw = rng.integers(max(1, w // 5), max(2, w // 2))
    y = int(rng.integers(0, h - fh))
    x = int(rng.integers(0, w - fw))
    out = arr.copy()
    region = out[y:y + fh, x:x + fw]

    edit = int(rng.integers(3))
    if edit == 0:
        blurred = Image.fromarray(region).filter(ImageFilter.GaussianBlur(radius=4))
        out[y:y + fh, x:x + fw] = np.array(blurred)
    elif edit == 1:
        shift = rng.integers(-50, 50, size=3).astype(np.int16)
        out[y:y + fh, x:x + fw] = np.clip(region.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    else:
        fill = region.mean(axis=(0, 1)).astype(np.int16)
        noise = rng.integers(-25, 25, size=region.shape).astype(np.int16)
        out[y:y + fh, x:x + fw] = np.clip(fill + noise, 0, 255).astype(np.uint8)
    return out, (x, y, x + fw, y + fh)


def synth_tamper_bbox(
    arr: np.ndarray,
    rng: np.random.Generator,
    donor_arr: np.ndarray | None = None,
) -> tuple[np.ndarray, str, BBox]:
    """Same op selection as augment.py's synth_tamper; returns (edited_array, edit_name, bbox)."""
    ops: list[str] = ["copy_move", "field_smudge"]
    if donor_arr is not None:
        ops.append("local_splice")
    name = ops[int(rng.integers(len(ops)))]
    if name == "copy_move":
        arr2, bbox = _copy_move(arr, rng)
        return arr2, name, bbox
    if name == "field_smudge":
        arr2, bbox = _field_smudge(arr, rng)
        return arr2, name, bbox
    if donor_arr is not None:
        arr2, bbox = _local_splice(arr, donor_arr, rng)
        return arr2, name, bbox
    arr2, bbox = _copy_move(arr, rng)
    return arr2, "copy_move", bbox


def recapture_transforms_with_bbox(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> A.Compose:
    """Equivalent to augment.py's recapture_transforms, but configured with bbox_params so a
    ground-truth tamper bbox is transformed jointly with the image through every step,
    including the geometric ones (Perspective, Rotate).

    Call as: pipeline(image=arr, bboxes=[(x1,y1,x2,y2)], category_ids=[0]) -> dict with
    "image" (CHW tensor) and "bboxes" (list of possibly-clipped/dropped boxes post-transform).
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.ImageCompression(quality_range=(50, 95), p=0.9),
            A.Downscale(
                scale_range=(0.5, 0.85),
                interpolation_pair={"downscale": 2, "upscale": 2},
                p=0.5,
            ),
            A.ImageCompression(quality_range=(60, 95), p=0.7),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
            ], p=0.5),
            A.GaussNoise(std_range=(0.01, 0.04), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.Rotate(limit=5, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=0.1),
    )
