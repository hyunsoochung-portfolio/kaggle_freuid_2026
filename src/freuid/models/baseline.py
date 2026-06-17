"""Baseline: a timm backbone with a single fraud-probability head.

Binary forgery detection → 1 logit, sigmoid at inference for the fraud score.
Swap the backbone via config (``backbone``) without touching train/infer code.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build_model(backbone: str = "tf_efficientnetv2_s.in21k", pretrained: bool = True) -> nn.Module:
    """timm backbone with num_classes=1 (single fraud logit)."""
    return timm.create_model(backbone, pretrained=pretrained, num_classes=1)
