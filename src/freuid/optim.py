"""Layer-wise LR decay (LLRD) fine-tuning helpers for timm ViT backbones.

Gated behind ``cfg.extra.llrd.enabled`` -- only used by configs that opt in (e.g.
``finetune_v0.yaml``). Every function here assumes the model exposes a ``.blocks``
attribute (an indexable sequence of transformer blocks), which every timm
``VisionTransformer`` does; nothing here touches the frozen-backbone consistency path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def freeze_all_but_last_k_blocks(model: nn.Module, k: int) -> None:
    """Freeze every param except the last ``k`` transformer blocks + head/norm.

    Fallback for when a full fine-tune doesn't fit in memory. ``model`` must expose
    ``.blocks`` (indexable sequence of transformer blocks) -- true for timm ViT models.
    """
    if not hasattr(model, "blocks"):
        raise AttributeError(
            f"{type(model).__name__} has no .blocks attribute -- train_last_k_blocks "
            "only supports timm ViT-style models"
        )
    n_blocks = len(model.blocks)
    keep_from = max(0, n_blocks - k)
    for p in model.parameters():
        p.requires_grad = False
    for i, block in enumerate(model.blocks):
        if i >= keep_from:
            for p in block.parameters():
                p.requires_grad = True
    for attr in ("head", "norm", "fc_norm", "attn_pool"):
        module = getattr(model, attr, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[optim] train_last_k_blocks={k}: unfroze blocks[{keep_from}:{n_blocks}] + head/norm "
        f"-- trainable={n_trainable:,}/{n_total:,}"
    )


def build_llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    decay: float = 0.7,
) -> list[dict]:
    """Per-block layer-wise-LR-decay param groups for AdamW.

    Groups by transformer depth: head/final-norm at ``base_lr``, block ``i`` at
    ``base_lr * decay**(num_blocks - i)``, patch embed / pos embed / cls / register
    tokens at the deepest-decayed rate (``base_lr * decay**(num_blocks + 1)``). Within
    every depth group, params are further split into decay / no_decay -- LayerNorm
    weights, all biases, and anything in ``model.no_weight_decay()`` get
    ``weight_decay=0.0``, per standard practice.

    Only trainable (``requires_grad=True``) params are included, so this composes with
    ``freeze_all_but_last_k_blocks`` -- frozen params are simply absent from the optimizer.
    """
    if not hasattr(model, "blocks"):
        raise AttributeError(
            f"{type(model).__name__} has no .blocks attribute -- LLRD only "
            "supports timm ViT-style models"
        )
    n_blocks = len(model.blocks)
    no_decay_names: set[str] = set()
    if hasattr(model, "no_weight_decay"):
        no_decay_names = set(model.no_weight_decay())

    def depth_of(name: str) -> int:
        if name.startswith("blocks."):
            block_idx = int(name.split(".")[1])
            return n_blocks - block_idx
        if name.startswith("patch_embed.") or name.split(".")[0] in (
            "pos_embed", "cls_token", "reg_token", "dist_token",
        ):
            return n_blocks + 1
        return 0  # head, norm, fc_norm, attn_pool, or anything else -> base LR

    def is_no_decay(name: str, param: torch.Tensor) -> bool:
        top = name.split(".")[0]
        return param.ndim <= 1 or name.endswith(".bias") or top in no_decay_names

    buckets: dict[tuple[int, bool], list[torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        buckets.setdefault((depth_of(name), is_no_decay(name, param)), []).append(param)

    groups = []
    for (depth, no_decay), params in sorted(buckets.items()):
        groups.append({
            "params": params,
            "lr": base_lr * (decay ** depth),
            "weight_decay": 0.0 if no_decay else weight_decay,
        })

    depths = sorted({d for d, _ in buckets})
    lrs = [base_lr * (decay ** d) for d in depths]
    print(
        f"[optim] LLRD: {len(groups)} param groups over depths {depths} "
        f"(decay={decay}) -- lr range [{min(lrs):.2e}, {max(lrs):.2e}]"
    )
    return groups


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 2,
    warmup_start_factor: float = 1e-2,
):
    """Linear warmup for ``warmup_epochs`` epochs, then cosine decay over the rest.

    Stepped once per epoch (matches ``train.py``'s per-epoch ``scheduler.step()``), so
    ``warmup_epochs=2`` means exactly 2 epoch-level warmup steps. Falls back to plain
    cosine annealing (S1's schedule) if ``epochs`` is too small to fit a warmup phase.
    """
    warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )
