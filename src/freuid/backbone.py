"""Frozen DINO backbone wrapper for the consistency path.

Loads DINOv3 ViT-B/16 (or DINOv2 ViT-B/14 as fallback) from torch.hub, freezes
all parameters, and exposes a single forward_features() call that returns CLS token,
patch tokens, and grid shape — the inputs the consistency heads need.

The backbone is always frozen (eval + requires_grad=False).  Only the light heads
on top are trained; see models/consistency.py.
"""

from __future__ import annotations

import contextlib
import warnings

import torch
import torch.nn as nn


_EMBED_DIM = {
    "dinov3_vitb16": 768,
    "dinov2_vitb14": 768,
}

_HUB = {
    "dinov3_vitb16": ("facebookresearch/dinov3", "dinov3_vitb16"),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
}


class _FeatureWrapper(nn.Module):
    """Thin wrapper around a raw DINO model.

    Exposes forward_features() with a stable return signature so callers never
    depend on the underlying hub API directly.  The backbone forward runs under
    torch.no_grad() + autocast (on CUDA) and outputs are cast to float32 before
    returning so downstream heads always receive float32 regardless of autocast
    dtype.
    """

    def __init__(self, raw: nn.Module, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self._raw = raw
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    @torch.no_grad()
    def forward_features(self, imgs: torch.Tensor) -> dict:
        """Run the frozen backbone.

        Returns:
            cls          (B, D) float32 — normalised CLS token
            patch_tokens (B, N, D) float32 — normalised patch tokens
            grid_hw      (H, W) ints — patch grid dimensions (H*W == N)
        """
        device = imgs.device
        amp_ctx: contextlib.AbstractContextManager
        if device.type == "cuda":
            amp_ctx = torch.autocast(device_type="cuda")
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            raw_out = self._raw.forward_features(imgs)

        cls = raw_out["x_norm_clstoken"].float()          # (B, D)
        patch = raw_out["x_norm_patchtokens"].float()     # (B, N, D)
        H = imgs.shape[-2] // self.patch_size
        W = imgs.shape[-1] // self.patch_size
        return {"cls": cls, "patch_tokens": patch, "grid_hw": (H, W)}

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convenience: return CLS token directly (used in smoke tests)."""
        return self.forward_features(imgs)["cls"]


def _load_hub(repo: str, model_name: str) -> nn.Module:
    return torch.hub.load(repo, model_name, pretrained=True, verbose=False)


def load_backbone(backbone_name: str) -> _FeatureWrapper:
    """Load and freeze a DINO backbone by name.

    Supported names:
        "dinov3_vitb16" — DINOv3 ViT-B/16; falls back to dinov2_vitb14 if the
                          hub repo is unavailable (gated / not yet published).
        "dinov2_vitb14" — DINOv2 ViT-B/14 (Apache 2.0).
    """
    if backbone_name not in _HUB:
        raise ValueError(
            f"Unknown backbone_name {backbone_name!r}. "
            f"Supported: {list(_HUB)}"
        )

    repo, name = _HUB[backbone_name]
    raw: nn.Module | None = None

    if backbone_name == "dinov3_vitb16":
        try:
            raw = _load_hub(repo, name)
            print(f"[backbone] loaded {backbone_name} from {repo}")
        except Exception as exc:
            warnings.warn(
                f"[backbone] Could not load {backbone_name} from {repo!r}: {exc}. "
                "Falling back to dinov2_vitb14 (Apache 2.0).",
                stacklevel=2,
            )
            backbone_name = "dinov2_vitb14"

    if raw is None:
        repo, name = _HUB[backbone_name]
        raw = _load_hub(repo, name)
        print(f"[backbone] loaded {backbone_name} from {repo}")

    raw.eval()
    raw.requires_grad_(False)

    # patch_size: DINO models expose .patch_size directly
    patch_size = getattr(raw, "patch_size", None)
    if patch_size is None:
        # fallback: read from the patch embedding layer
        patch_size = raw.patch_embed.patch_size
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]

    embed_dim = _EMBED_DIM.get(backbone_name, getattr(raw, "embed_dim", 768))
    return _FeatureWrapper(raw, int(patch_size), int(embed_dim))


def embed_dim_for(backbone_name: str) -> int:
    """CLS token dimension for a given backbone name."""
    return _EMBED_DIM.get(backbone_name, 768)
