"""Baseline: a timm backbone with a single fraud-probability head.

Binary forgery detection → 1 logit, sigmoid at inference for the fraud score.
Swap the backbone via config (``backbone``) without touching train/infer code.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build_model(
    backbone: str = "tf_efficientnetv2_s.in21k",
    pretrained: bool = True,
    pool: str | None = None,
    head_dropout: float = 0.0,
) -> nn.Module:
    """timm backbone with num_classes=1 (single fraud logit).

    Tries ``dynamic_img_size=True`` first: ViT-family models need this to accept
    input resolutions other than their pretrained native size (e.g. multi-scale TTA)
    without a pos-embed shape mismatch. CNN backbones (ConvNeXt, EfficientNet, ...)
    don't accept this kwarg, so it's caught and dropped -- no behavior change for them.

    ``pool`` overrides the global pooling: e.g. "map" gives a learned attention-pool head
    (a query token attends to suspicious patches, vs averaging which dilutes a small forged
    region). ``head_dropout`` adds dropout before the final linear head. Both default to the
    backbone's timm defaults, so existing configs are byte-for-byte unaffected.
    """
    kwargs: dict = {"num_classes": 1, "drop_rate": head_dropout}
    if pool is not None:
        kwargs["global_pool"] = pool
    try:
        model = timm.create_model(backbone, pretrained=pretrained, dynamic_img_size=True, **kwargs)
    except TypeError:
        model = timm.create_model(backbone, pretrained=pretrained, **kwargs)
    # Zero-init the head: pretrained backbone features are large enough that random
    # head weights produce extreme logits. Zero weight+bias guarantees logit=0 → p=0.5
    # at the start of training for any input.
    head = model.get_classifier()
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return model
