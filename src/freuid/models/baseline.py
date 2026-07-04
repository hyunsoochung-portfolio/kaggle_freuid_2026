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
    head_dropout: float = 0.0,
    pool: str | None = None,
    image_size: int | None = None,
) -> nn.Module:
    """timm backbone with num_classes=1 (single fraud logit).

    ``head_dropout`` inserts dropout before the final linear head (timm's ``drop_rate``);
    ``pool`` overrides the global pooling (e.g. "avg" to mean-pool ViT patch tokens instead
    of using the CLS token — often better for spatially-spread manipulation artifacts).
    ``image_size`` is forwarded as ``img_size`` for ViTs (which bake a fixed input size into
    patch/pos embeds and would otherwise assert); CNNs are resolution-agnostic and ignore it.
    """
    kwargs: dict = {"num_classes": 1, "drop_rate": head_dropout}
    if pool is not None:
        kwargs["global_pool"] = pool
    if image_size is not None:
        try:
            return timm.create_model(backbone, pretrained=pretrained, img_size=image_size, **kwargs)
        except TypeError:
            pass  # backbone doesn't take img_size (e.g. EfficientNet) — safe to omit
    return timm.create_model(backbone, pretrained=pretrained, **kwargs)


def llrd_param_groups(
    model: nn.Module, base_lr: float, weight_decay: float, decay: float
) -> list[dict]:
    """Build AdamW param groups with layer-wise LR decay (LLRD) for a ViT.

    Earlier (more generic) layers get a smaller LR so pretrained low-level features are barely
    disturbed, while the head/last blocks adapt fast. lr = base_lr * decay**(top - layer_id):
    the head gets base_lr; the patch-embed gets the smallest lr. Biases, norms and pos/cls
    tokens get weight_decay=0 (standard transformer fine-tuning recipe).

    Falls back to a single uniform group for non-ViT backbones (no ``.blocks``).
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:  # not a plain ViT — nothing to decay layer-wise
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params, "lr": base_lr, "weight_decay": weight_decay}]

    top = len(blocks) + 1  # depth ids: embeddings=0, blocks=1..N, head/norm=N+1
    no_wd_names = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()

    def layer_id(name: str) -> int:
        name = name.removeprefix("_orig_mod.")  # tolerate torch.compile-wrapped param names
        if name.startswith("blocks."):
            return int(name.split(".")[1]) + 1
        if name.startswith(("patch_embed", "cls_token", "pos_embed", "reg_token", "mask_token")):
            return 0
        return top  # final norm, head, etc.

    groups: dict[tuple[int, bool], dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = layer_id(name)
        no_wd = p.ndim <= 1 or name in no_wd_names or name.endswith(".bias")
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * (decay ** (top - lid)),
                "weight_decay": 0.0 if no_wd else weight_decay,
            }
        groups[key]["params"].append(p)
    return list(groups.values())
