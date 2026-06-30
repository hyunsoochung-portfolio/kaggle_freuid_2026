"""Consistency model components.

ConsistencyNet = frozen DINO backbone (external) + GlobalHead on the CLS token.

The backbone is always passed in from outside so it can be shared / substituted
without touching this file.  Patch self-consistency and face-region heads will be
added here in S3 as additional nn.Module attributes on ConsistencyNet.

Checkpoint discipline: state_dict / load_state_dict expose only head weights
(~150 KB).  The ~330 MB backbone is always reloaded from torch.hub at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from freuid.backbone import _FeatureWrapper, embed_dim_for, load_backbone
from freuid.config import Config


class GlobalHead(nn.Module):
    """LayerNorm + two-layer MLP → fraud logit.

    LayerNorm normalises the CLS token before the MLP, which stabilises
    training when the backbone is completely frozen (no batch statistics shift).
    Zero-init on the final linear guarantees logit ≈ 0 at init → BCE ≈ ln(2).
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        hidden = max(embed_dim // 4, 64)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))   # (B, D) → (B, 1)


class ConsistencyNet(nn.Module):
    """Frozen DINO backbone + GlobalHead.

    forward(imgs) → (B, 1) logits — identical shape to the baseline CNN, so
    run_epoch / probe / TTA / integrity machinery is reused without modification.

    The backbone attribute is an nn.Module but all its parameters have
    requires_grad=False, so AdamW / SGD only updates the head.
    """

    def __init__(self, backbone: _FeatureWrapper) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = GlobalHead(backbone.embed_dim)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        cls = self.backbone.forward_features(imgs)["cls"]   # (B, D) float32, no grad
        return self.head(cls)                               # (B, 1)

    # ------------------------------------------------------------------
    # Checkpoint: only head weights (backbone reloads from hub at inference)
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        return self.head.state_dict(*args, **kwargs)

    def load_state_dict(self, sd, strict: bool = True):
        return self.head.load_state_dict(sd, strict=strict)


def build_consistency_model(cfg: Config) -> ConsistencyNet:
    """Build a ConsistencyNet from a Config.

    Reads extra.backbone_name (default "dinov2_vitb14").
    Prints trainable param count so it's easy to verify the backbone is frozen.
    """
    backbone_name: str = cfg.extra.get("backbone_name", "dinov2_vitb14")
    wrapper = load_backbone(backbone_name)
    model = ConsistencyNet(wrapper)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"[consistency] backbone={backbone_name} embed_dim={wrapper.embed_dim} "
        f"patch_size={wrapper.patch_size} | "
        f"trainable={n_trainable:,}  frozen={n_frozen:,}"
    )
    return model
