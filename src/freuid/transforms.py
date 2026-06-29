"""Image transforms. No horizontal flip — documents carry text/orientation, so
mirroring would create invalid samples. Train aug stays mild (jitter + small
rotation) to add print/capture robustness without destroying document semantics.

Normalization stats are NOT hardcoded per run: different pretrained backbones expect
different mean/std (e.g. ImageNet CNNs use (0.485,0.456,0.406); ViT / tf_efficientnet
use (0.5,0.5,0.5)). ``resolve_data_config`` pulls the right values straight from the
backbone's own timm config so swapping ``backbone`` keeps preprocessing correct.
"""

from __future__ import annotations

from torchvision import transforms

# Fallback only — used when a backbone isn't registered in timm.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(steps)
#   class Compose:
#      def __init__(self, transforms):
#          self.transforms = transforms          # 부품 리스트 저장
#      def __call__(self, img):
#          for t in self.transforms:             # 부품을 순서대로
#              img = t(img)                       # 출력이 다음 입력으로
#          return img