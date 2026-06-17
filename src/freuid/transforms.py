"""Image transforms. No horizontal flip — documents carry text/orientation, so
mirroring would create invalid samples. Train aug stays mild (jitter + small
rotation) to add print/capture robustness without destroying document semantics.
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 384, train: bool = False):
    steps = [transforms.Resize((image_size, image_size))]
    if train:
        steps += [
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(5)], p=0.3),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)
