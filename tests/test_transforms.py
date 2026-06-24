"""Sanity tests for the fraud-aware training augmentations."""

import numpy as np
from PIL import Image

from freuid.transforms import RandomDownscaleUpscale, RandomJPEGCompression, build_transforms


def _dummy_image(w=220, h=300):
    return Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8"))


def test_train_transform_output_shape():
    tf = build_transforms(224, train=True)
    out = tf(_dummy_image())
    assert out.shape == (3, 224, 224)


def test_eval_transform_is_deterministic_in_shape():
    tf = build_transforms(224, train=False)
    out = tf(_dummy_image())
    assert out.shape == (3, 224, 224)


def test_random_jpeg_compression_preserves_size_and_mode():
    img = _dummy_image()
    out = RandomJPEGCompression((30, 90))(img)
    assert out.size == img.size
    assert out.mode == "RGB"


def test_random_downscale_upscale_preserves_size():
    img = _dummy_image()
    out = RandomDownscaleUpscale((0.5, 0.9))(img)
    assert out.size == img.size
