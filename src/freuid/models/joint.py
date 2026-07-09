"""Joint attention-pool + consistency model (model_type="joint").

The winning full-fine-tuned attention-pool baseline (public 0.03935) is kept EXACTLY as-is
as the global branch; a second, parallel consistency branch reads the same backbone patch
tokens and looks for local inconsistency (splice/tamper) anywhere on the document. The two
logits are fused additively:

    final_logit = logit_global  +  logit_consist          (consist output zero-init)

Safe warm start: the consistency branch's final layer is zero-init, so at init the model is
bit-for-bit the attention-pool baseline (init BCE ~= ln2); additive (not multiply-by-a-zero-
scale) fusion means the branch still receives gradient from step 1 and only "opens up" as it
lowers the loss.

Consistency-branch variants (extra.consist_type):
  - "conv"  (v2, DEFAULT): position-INVARIANT. A SHARED conv stack over the patch-feature
            grid (same filters at every location -> translation-equivariant, a learned
            seam/Sobel detector) scores each cell, then GLOBAL MAX-pool over space
            (translation-invariant): a tamper anywhere -> high score, regardless of where.
  - "patch" (v1): a learned outlier query over the patch tokens (position-AWARE because the
            tokens carry position embeddings; kept only for ablation).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from freuid.config import Config
from freuid.consistency_model import PatchConsistencyHead
from freuid.models.baseline import build_model


class ConvConsistencyHead(nn.Module):
    """Position-invariant local-inconsistency detector (v2).

    patch tokens -> (H, W) feature grid -> shared conv (translation-equivariant seam
    detector) -> per-cell anomaly -> [global max, global mean] -> zero-init Linear -> logit.
    Weight-sharing + global max make it insensitive to WHERE the tamper sits.
    """

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(dim, hidden, kernel_size=1)
        self.conv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout)
        self.score = nn.Conv2d(hidden, 1, kernel_size=1)   # per-cell anomaly score
        self.fc = nn.Linear(2, 1)                          # [max, mean] -> logit
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:  # patch: (B, N, D) -> (B, 1)
        b, n, d = patch.shape
        h = int(math.isqrt(n))
        assert h * h == n, f"ConvConsistencyHead expects a square patch grid, got N={n}"
        x = patch.transpose(1, 2).reshape(b, d, h, h)        # (B, D, H, W)
        x = self.act(self.reduce(x))
        x = self.drop(self.act(self.conv(x)))
        s = self.score(x).flatten(2)                          # (B, 1, H*W) per-cell scores
        feat = torch.cat([s.amax(-1), s.mean(-1)], dim=-1)    # (B, 2)  global max + mean
        return self.fc(feat)                                  # (B, 1)


class _PatchConsist(nn.Module):
    """v1 (position-aware) consistency: PatchConsistencyHead embedding -> zero-init Linear."""

    def __init__(self, dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.head = PatchConsistencyHead(dim, num_layers=layers, num_heads=heads, dropout=dropout)
        self.fc = nn.Linear(dim, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.fc(self.head(patch))


class JointConsistencyModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        head_dropout: float = 0.0,
        consist_type: str = "conv",
        conv_hidden: int = 256,
        conv_dropout: float = 0.1,
        patch_layers: int = 2,
        patch_heads: int = 8,
        patch_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # global branch = the winning attention-pool model (num_classes=1, zero-init head)
        self.net = build_model(backbone, pretrained, pool="map", head_dropout=head_dropout)
        dim = self.net.num_features
        self.num_prefix = int(getattr(self.net, "num_prefix_tokens", 1))

        if consist_type == "conv":
            self.consist: nn.Module = ConvConsistencyHead(
                dim, hidden=conv_hidden, dropout=conv_dropout
            )
        elif consist_type == "patch":
            self.consist = _PatchConsist(dim, patch_layers, patch_heads, patch_dropout)
        else:
            raise ValueError(f"unknown consist_type {consist_type!r} (expected 'conv' or 'patch')")
        self.consist_type = consist_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.net.forward_features(x)              # (B, N_all, D)
        logit_global = self.net.forward_head(tokens)       # (B, 1)  attn-pool + head
        patch = tokens[:, self.num_prefix:, :]             # (B, N, D) patch tokens only
        return logit_global + self.consist(patch)          # additive fusion


def build_joint_model(cfg: Config) -> JointConsistencyModel:
    consist_type = str(cfg.extra.get("consist_type", "conv"))
    model = JointConsistencyModel(
        cfg.backbone,
        cfg.pretrained,
        head_dropout=float(cfg.extra.get("head_dropout", 0.0)),
        consist_type=consist_type,
        conv_hidden=int(cfg.extra.get("conv_hidden", 256)),
        conv_dropout=float(cfg.extra.get("conv_dropout", 0.1)),
        patch_layers=int(cfg.extra.get("patch_consistency_layers", 2)),
        patch_heads=int(cfg.extra.get("patch_consistency_heads", 8)),
        patch_dropout=float(cfg.extra.get("patch_consistency_dropout", 0.1)),
    )
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[joint] {cfg.backbone}: attn-pool(global) + {consist_type}-consistency(patch), "
        f"additive fuse (consist zero-init -> starts == baseline) | trainable={n_train:,}"
    )
    return model
