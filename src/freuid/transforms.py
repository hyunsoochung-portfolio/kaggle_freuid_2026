"""Image transforms. No horizontal flip — documents carry text/orientation, so
mirroring would create invalid samples. Train aug stays mild to add robustness
without destroying document semantics, targeted at FREUID's three fraud modalities
(see docs/competition.md §3):

- **Physical manipulations** (tampered substrate, photographed) -> ``RandomPerspective``
  (capture-angle skew) + ``ColorJitter`` (uneven lighting/glare on a handheld card).
- **GenAI multimodal edits** -> ``RandomJPEGCompression`` (the re-save cycle these tools'
  outputs typically go through); kept mild so it doesn't erase the genuine edit artifacts.
- **Print-and-capture / "analog hole"** (printed, then re-photographed/scanned) ->
  ``RandomDownscaleUpscale`` (resampling/moiré from the recapture) + ``GaussianBlur``
  (refocus) + ``RandomJPEGCompression`` (the scan/photo's own re-encode).

``RandomRotation`` (slight misalignment) and ``ColorJitter`` apply broadly across all
three. Each is probability-gated and mild — this is meant to model real capture/encode
degradation, not arbitrary noise.

Normalization stats are NOT hardcoded per run: different pretrained backbones expect
different mean/std (e.g. ImageNet CNNs use (0.485,0.456,0.406); ViT / tf_efficientnet
use (0.5,0.5,0.5)). ``resolve_data_config`` pulls the right values straight from the
backbone's own timm config so swapping ``backbone`` keeps preprocessing correct.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image
from torchvision import transforms

try:
    import albumentations as A
    _ALBU_AVAILABLE = True
except ImportError:
    _ALBU_AVAILABLE = False

# Fallback only — used when a backbone isn't registered in timm.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomJPEGCompression:
    """Re-encode through JPEG at a random quality.

    Simulates the recompression a GenAI-edited image picks up on re-save, and the
    encode a printed document acquires when re-photographed or scanned.
    """

    def __init__(self, quality_range: tuple[int, int] = (30, 90)) -> None:
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomDownscaleUpscale:
    """Downscale then upscale back to the original size.

    Approximates the resampling/moiré artifacts of photographing or scanning a
    printed document (the "analog hole") rather than reading the digital original.
    """

    def __init__(self, scale_range: tuple[float, float] = (0.5, 0.9)) -> None:
        self.scale_range = scale_range

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        small = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class GridDistortion:
    """Albumentations GridDistortion wrapped for PIL → PIL use.

    Simulates physical warping/bending of a document (print-and-capture, scanner
    bed distortion). Falls back to a no-op if albumentations is unavailable.
    """

    def __init__(self, p: float = 0.3) -> None:
        self.p = p
        self._aug = A.GridDistortion(p=1.0) if _ALBU_AVAILABLE else None

    def __call__(self, img: Image.Image) -> Image.Image:
        if self._aug is None or random.random() > self.p:
            return img
        arr = np.array(img)
        result = self._aug(image=arr)["image"]
        return Image.fromarray(result)


def resolve_data_config(backbone: str, image_size: int | None = None) -> dict:
    """Mean/std/input-size the given backbone was pretrained with.

    Reads the backbone's own timm pretrained config (no model instantiation needed).
    ``image_size``, if provided, overrides the model's native resolution; otherwise the
    native size is used. Falls back to ImageNet stats + 384px for unregistered names.
    """
    import timm

    try:
        cfg = timm.get_pretrained_cfg(backbone)
        mean, std = tuple(cfg.mean), tuple(cfg.std)
        native = cfg.input_size[-1]
    except (AttributeError, RuntimeError, KeyError):
        mean, std, native = IMAGENET_MEAN, IMAGENET_STD, 384
    return {"mean": mean, "std": std, "image_size": image_size or native}


def build_transforms(
    image_size: int = 384,
    train: bool = False,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps += [
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(5)], p=0.3),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
            transforms.RandomApply([RandomDownscaleUpscale((0.5, 0.9))], p=0.3),
            transforms.RandomApply([RandomJPEGCompression((30, 90))], p=0.5),
            GridDistortion(p=0.3),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if train:
        # Randomly erase small patches — forces model to rely on distributed fraud
        # cues rather than a single tell-tale region.
        steps.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)))
    return transforms.Compose(steps)
#   class Compose:
#      def __init__(self, transforms):
#          self.transforms = transforms          # 부품 리스트 저장
#      def __call__(self, img):
#          for t in self.transforms:             # 부품을 순서대로
#              img = t(img)                       # 출력이 다음 입력으로
#          return img

