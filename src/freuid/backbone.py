"""Frozen DINO backbone wrapper for the consistency path.

Loads DINOv3 ViT-B/16 (or DINOv2 ViT-B/14) from torch.hub or HuggingFace, freezes
all parameters, and exposes a single forward_features() call that returns CLS token,
patch tokens, and grid shape — the inputs the consistency heads need.

DINOv3 loading order:
  1. torch.hub ('facebookresearch/dinov3') — works if fbaipublicfiles.com is accessible
  2. HuggingFace AutoModel ('facebook/dinov3-vitb16-pretrain-lvd1689m') — requires
     HF_TOKEN env var and accepted license on huggingface.co

The backbone is always frozen (eval + requires_grad=False).  Only the light heads
on top are trained; see models/consistency.py.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


_EMBED_DIM = {
    "dinov3_vitb16": 768,
    "dinov2_vitb14": 768,
}

_PATCH_SIZE = {
    "dinov3_vitb16": 16,
    "dinov2_vitb14": 14,
}

_HUB = {
    "dinov3_vitb16": ("facebookresearch/dinov3", "dinov3_vitb16"),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
}

_HF_REPO = {
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}


class _FeatureWrapper(nn.Module):
    """Thin wrapper around a DINO model (hub or transformers backend).

    Exposes forward_features() with a stable return signature so callers never
    depend on the underlying API directly.  Outputs are cast to float32 before
    returning so downstream heads always receive float32 regardless of autocast.
    """

    def __init__(
        self,
        raw: nn.Module,
        patch_size: int,
        embed_dim: int,
        backend: str = "hub",
    ) -> None:
        super().__init__()
        self._raw = raw
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self._backend = backend  # "hub" | "transformers"

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
        amp_ctx = torch.autocast(device_type="cuda") if device.type == "cuda" else contextlib.nullcontext()

        with amp_ctx:
            if self._backend == "transformers":
                out = self._raw(pixel_values=imgs)
                # last_hidden_state: (B, 1+N, D); position 0 = CLS, 1: = patches
                cls = out.last_hidden_state[:, 0, :].float()
                patch = out.last_hidden_state[:, 1:, :].float()
            else:
                raw_out = self._raw.forward_features(imgs)
                cls = raw_out["x_norm_clstoken"].float()
                patch = raw_out["x_norm_patchtokens"].float()

        H = imgs.shape[-2] // self.patch_size
        W = imgs.shape[-1] // self.patch_size
        return {"cls": cls, "patch_tokens": patch, "grid_hw": (H, W)}

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convenience: return CLS token directly (used in smoke tests)."""
        return self.forward_features(imgs)["cls"]


def _load_hub(repo: str, model_name: str) -> nn.Module:
    return torch.hub.load(repo, model_name, pretrained=True, verbose=False)


def _load_dinov3_from_hf(hf_repo: str) -> tuple[nn.Module, str]:
    """Load DINOv3 from HuggingFace (gated — requires HF_TOKEN + license accept)."""
    import os

    from freuid.utils import load_dotenv

    load_dotenv()  # picks up HF_TOKEN from .env if not already set in the environment
    print(f"[backbone] torch.hub failed; trying HuggingFace ({hf_repo})...")
    try:
        from transformers import AutoModel
    except ImportError as e:
        raise RuntimeError(
            "transformers not installed. Run: pip install transformers"
        ) from e
    try:
        model = AutoModel.from_pretrained(hf_repo, token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        raise RuntimeError(
            f"Could not load dinov3_vitb16 from HuggingFace ({hf_repo}): {e}\n"
            "Ensure HF_TOKEN is set (export HF_TOKEN=...) and you have accepted "
            "the license at https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m"
        ) from e
    print(f"[backbone] loaded dinov3_vitb16 from HuggingFace (transformers)")
    return model, "transformers"


def load_backbone(backbone_name: str) -> _FeatureWrapper:
    """Load and freeze a DINO backbone by name.

    Supported names:
        "dinov3_vitb16" — DINOv3 ViT-B/16 (gated; HF_TOKEN required).
                          Tries torch.hub first, then HuggingFace AutoModel.
                          No fallback to DINOv2 — fails loud if unavailable.
        "dinov2_vitb14" — DINOv2 ViT-B/14 (Apache 2.0, torch.hub only).
    """
    if backbone_name not in _HUB:
        raise ValueError(
            f"Unknown backbone_name {backbone_name!r}. "
            f"Supported: {list(_HUB)}"
        )

    backend = "hub"
    raw: nn.Module | None = None

    if backbone_name == "dinov3_vitb16":
        repo, name = _HUB[backbone_name]
        try:
            raw = _load_hub(repo, name)
            print(f"[backbone] loaded {backbone_name} from {repo}")
        except Exception as hub_exc:
            print(f"[backbone] torch.hub error: {hub_exc}")
            hf_repo = _HF_REPO[backbone_name]
            raw, backend = _load_dinov3_from_hf(hf_repo)
    else:
        repo, name = _HUB[backbone_name]
        raw = _load_hub(repo, name)
        print(f"[backbone] loaded {backbone_name} from {repo}")

    raw.eval()
    raw.requires_grad_(False)

    # Prefer the known patch_size; fall back to model attributes
    patch_size = _PATCH_SIZE.get(backbone_name)
    if patch_size is None:
        patch_size = getattr(raw, "patch_size", None)
        if patch_size is None:
            ps = getattr(getattr(raw, "patch_embed", None), "patch_size", 16)
            patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps

    embed_dim = _EMBED_DIM.get(backbone_name, getattr(raw, "embed_dim", 768))
    return _FeatureWrapper(raw, int(patch_size), int(embed_dim), backend=backend)


def embed_dim_for(backbone_name: str) -> int:
    """CLS token dimension for a given backbone name."""
    return _EMBED_DIM.get(backbone_name, 768)
