"""Baseline: a timm backbone with a single fraud-probability head.

Binary forgery detection → 1 logit, sigmoid at inference for the fraud score.
Swap the backbone via config (``backbone``) without touching train/infer code.
"""

from __future__ import annotations

import timm
import torch.nn as nn

from freuid.models.heads import BackboneWithHead, ForensicHead


def _create_backbone(backbone: str, pretrained: bool, num_classes: int, drop_path_rate: float | None):
    """timm.create_model with best-effort ``dynamic_img_size``/``drop_path_rate``.

    ``dynamic_img_size=True`` lets ViT-family models accept input resolutions other
    than their pretrained native size (e.g. multi-scale TTA) without a pos-embed
    shape mismatch; CNN backbones don't accept the kwarg. ``drop_path_rate`` (extra
    stochastic depth, guards against overfitting on a full fine-tune) isn't accepted
    by every architecture either. Both are dropped independently on ``TypeError`` so
    any timm backbone still loads -- no per-backbone special-casing needed.
    """
    kwargs = {"pretrained": pretrained, "num_classes": num_classes}
    if drop_path_rate is not None:
        kwargs["drop_path_rate"] = drop_path_rate
    try:
        return timm.create_model(backbone, dynamic_img_size=True, **kwargs)
    except TypeError:
        pass
    try:
        return timm.create_model(backbone, **kwargs)
    except TypeError:
        kwargs.pop("drop_path_rate", None)
        return timm.create_model(backbone, **kwargs)


def build_model(
    backbone: str = "tf_efficientnetv2_s.in21k",
    pretrained: bool = True,
    drop_path_rate: float | None = None,
    head_type: str = "gap",
) -> nn.Module:
    """timm backbone producing a single fraud logit.

    ``head_type``:
      - "gap" (default, unchanged behavior): timm's own classifier on its default
        pooled feature (zero-initialized so logit=0 / p=0.5 at the start of training).
      - "gem": bypasses the backbone's own pooling and applies ``ForensicHead``
        (GAP+GeM pooled, see ``models/heads.py``) on the raw, unpooled features --
        keeps localized forensic artifacts that plain average-pooling smooths away.
    """
    if head_type == "gap":
        model = _create_backbone(backbone, pretrained, num_classes=1, drop_path_rate=drop_path_rate)
        # Zero-init the head: pretrained backbone features are large enough that random
        # head weights produce extreme logits. Zero weight+bias guarantees logit=0 → p=0.5
        # at the start of training for any input.
        head = model.get_classifier()
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return model
    if head_type == "gem":
        backbone_model = _create_backbone(backbone, pretrained, num_classes=0, drop_path_rate=drop_path_rate)
        num_prefix_tokens = getattr(backbone_model, "num_prefix_tokens", 0)
        head = ForensicHead(backbone_model.num_features, num_prefix_tokens=num_prefix_tokens)
        nn.init.zeros_(head.fc.weight)
        nn.init.zeros_(head.fc.bias)
        return BackboneWithHead(backbone_model, head)
    raise ValueError(f"unknown head_type={head_type!r} (expected 'gap' or 'gem')")
