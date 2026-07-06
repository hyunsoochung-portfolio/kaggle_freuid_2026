"""Custom classification head for forensic/tamper detection backbones.

Plain global-average pooling washes out the small, localized artifacts (splice
seams, moire, recapture halos) that give this task away -- a flat mean can't
tell "one bad patch" from "every patch slightly off". GeM (Generalized Mean)
pooling keeps that distinction: raising local activations to a power p > 1
before averaging weights peaky/localized activations far more than a flat
average, so a small forged region isn't diluted by the rest of a genuine
document. p is learned, so training can fall back to plain average pooling
(p -> 1) if that's what a given backbone actually wants.

Works across both CNN (B, C, H, W) and ViT/token (B, N, C) backbone outputs --
``num_prefix_tokens`` says how many leading tokens (CLS/register) to drop
before pooling the patch tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GeM(nn.Module):
    """Generalized-mean pooling over the last (flattened spatial/token) axis.

    Input: (B, C, N). Output: (B, C).
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.full((1,), p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=-1)
        return x.pow(1.0 / self.p)


class ForensicHead(nn.Module):
    """GAP + GeM pooling (concatenated) followed by a linear classifier.

    Accepts raw (unpooled) backbone features from ``forward_features``:
    CNN layout (B, C, H, W) or ViT/token layout (B, N, C).
    """

    def __init__(self, in_features: int, num_prefix_tokens: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.gem = GeM()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features * 2, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:  # CNN: (B, C, H, W) -> (B, C, N)
            b, c, h, w = features.shape
            tokens = features.reshape(b, c, h * w)
        elif features.ndim == 3:  # ViT: (B, N, C) -> drop prefix tokens -> (B, C, N)
            tokens = features[:, self.num_prefix_tokens:, :].transpose(1, 2)
        else:
            raise ValueError(f"ForensicHead: unsupported feature ndim={features.ndim}")
        pooled = torch.cat([tokens.mean(dim=-1), self.gem(tokens)], dim=-1)
        return self.fc(self.dropout(pooled))


class BackboneWithHead(nn.Module):
    """Wraps a ``num_classes=0`` timm backbone with a custom head on raw features.

    Bypasses the backbone's own ``forward`` (which would pool+classify) by
    calling ``forward_features`` directly, so the head sees the unpooled tensor.
    """

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.forward_features(x))
