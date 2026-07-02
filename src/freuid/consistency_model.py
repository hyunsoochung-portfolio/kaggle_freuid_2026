"""Consistency model components.

ConsistencyNet = frozen DINO backbone (external) + a ConsistencyHead that fuses up to
three signals on top of the frozen patch/CLS features:

  - global:  CLS token (LayerNorm only)
  - patch:   PatchConsistencyHead — a learned [outlier] query attends over all patch
             tokens via a small TransformerEncoder, summarising local inconsistency.
  - face:    FaceRegionHead — contrasts patch tokens inside the cached face ROI against
             the rest of the card (requires use_rectify + use_face_region + a cached,
             non-fallback face box).

Each head is independently toggleable via config so they can be ablated. FusionMLP
concatenates whichever embeddings are active and produces the single fraud logit.

The backbone is always passed in from outside so it can be shared / substituted without
touching this file.

Checkpoint discipline: state_dict / load_state_dict expose only head weights (small).
The backbone is always reloaded from torch.hub/HuggingFace at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from freuid.backbone import _FeatureWrapper, load_backbone
from freuid.config import Config

# face_meta layout: [x1_frac, y1_frac, x2_frac, y2_frac, valid] in canonical card-space
# fractions (0..1); valid=0 means "no usable face box" (cache miss, use_face_region off,
# or a fallback/center-square box with SCRFD score==0).
FACE_META_DIM = 5


class PatchConsistencyHead(nn.Module):
    """Learned [outlier] query attends over all patch tokens via a small TransformerEncoder.

    The query is prepended to the patch-token sequence; self-attention lets it summarise
    whichever region(s) look least consistent with the rest of the card. Its output
    embedding (post-LayerNorm) is fed into the fusion MLP.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """patch_tokens: (B, N, D) -> (B, D) outlier-query embedding."""
        b = patch_tokens.size(0)
        query = self.query.expand(b, -1, -1)          # (B, 1, D)
        seq = torch.cat([query, patch_tokens], dim=1)  # (B, 1+N, D)
        out = self.encoder(seq)
        return self.norm(out[:, 0, :])                 # (B, D)


class FaceRegionHead(nn.Module):
    """Contrasts patch tokens inside the cached face ROI against the rest of the card.

    Builds a per-sample boolean patch mask from ``face_meta`` (box fractions in canonical
    card space) and the backbone's patch grid shape, pools inside vs. outside, and feeds
    [diff, cosine similarity] through a small MLP. Samples with ``valid == 0`` (no cache,
    disabled, or a fallback/center-square box) get a zeroed-out embedding: no signal
    instead of a wrong one.
    """

    def __init__(self, embed_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim + 1, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.out_norm = nn.LayerNorm(hidden)
        self.out_dim = hidden

    def forward(
        self,
        patch_tokens: torch.Tensor,
        grid_hw: tuple[int, int],
        face_meta: torch.Tensor | None,
    ) -> torch.Tensor:
        """patch_tokens: (B, N, D); face_meta: (B, FACE_META_DIM) or None -> (B, hidden)."""
        if face_meta is None:
            return patch_tokens.new_zeros(patch_tokens.size(0), self.out_dim)

        h, w = grid_hw
        valid = face_meta[:, 4:5]                       # (B, 1)
        x1, y1, x2, y2 = face_meta[:, 0:1], face_meta[:, 1:2], face_meta[:, 2:3], face_meta[:, 3:4]

        device = patch_tokens.device
        ys = (torch.arange(h, device=device, dtype=patch_tokens.dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=patch_tokens.dtype) + 0.5) / w
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W)
        grid_y = grid_y.reshape(1, -1)  # (1, N)
        grid_x = grid_x.reshape(1, -1)

        inside = (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)  # (B, N)
        inside_f = inside.to(patch_tokens.dtype).unsqueeze(-1)   # (B, N, 1)
        outside_f = 1.0 - inside_f

        inside_mean = (patch_tokens * inside_f).sum(dim=1) / inside_f.sum(dim=1).clamp_min(1.0)
        outside_mean = (patch_tokens * outside_f).sum(dim=1) / outside_f.sum(dim=1).clamp_min(1.0)

        cos = F.cosine_similarity(inside_mean, outside_mean, dim=-1, eps=1e-6).unsqueeze(-1)
        diff = self.norm(inside_mean - outside_mean)
        feat = torch.cat([diff, cos], dim=-1)             # (B, D+1)
        emb = self.fc2(self.act(self.fc1(feat)))          # (B, hidden)
        emb = self.out_norm(emb)                           # match scale of global/patch branches
        return emb * valid                                 # zero out invalid/fallback samples


class FusionMLP(nn.Module):
    """concat(embeddings) -> Linear -> ReLU -> Dropout -> 1 logit.

    Zero-init on the final linear guarantees logit ~= 0 at init (BCE ~= ln(2)) regardless
    of how many heads are fused in.
    """

    def __init__(self, in_dim: int, hidden: int | None = None, dropout: float = 0.3) -> None:
        super().__init__()
        hidden = hidden or max(in_dim // 4, 64)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))  # (B, in_dim) -> (B, 1)


class ConsistencyHead(nn.Module):
    """Bundles global/patch/face embeddings + FusionMLP. This is the entire trainable
    surface of ConsistencyNet (everything that isn't the frozen backbone)."""

    def __init__(
        self,
        embed_dim: int,
        use_patch_consistency: bool = False,
        use_face_region: bool = False,
        patch_layers: int = 2,
        patch_heads: int = 8,
        patch_dropout: float = 0.1,
        face_hidden: int = 128,
        fusion_hidden: int | None = None,
        fusion_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.global_norm = nn.LayerNorm(embed_dim)
        fusion_in = embed_dim

        self.patch_head: PatchConsistencyHead | None = None
        if use_patch_consistency:
            self.patch_head = PatchConsistencyHead(
                embed_dim, num_layers=patch_layers, num_heads=patch_heads, dropout=patch_dropout
            )
            fusion_in += self.patch_head.out_dim

        self.face_head: FaceRegionHead | None = None
        if use_face_region:
            self.face_head = FaceRegionHead(embed_dim, hidden=face_hidden)
            fusion_in += self.face_head.out_dim

        # LayerScale-style gates on the new, unproven branches only (not global): initialized
        # near-zero so the model starts close to global-only behavior and "opens" each pathway
        # only once it earns its keep, instead of contaminating the shared fusion layer's
        # gradients from step 1. Small nonzero init (not exact 0) so the branch still receives
        # gradient through the gate at init -- see docs/problem.md finding #1.
        self.patch_gate: nn.Parameter | None = None
        if self.patch_head is not None:
            self.patch_gate = nn.Parameter(torch.full((self.patch_head.out_dim,), 1e-3))

        self.face_gate: nn.Parameter | None = None
        if self.face_head is not None:
            self.face_gate = nn.Parameter(torch.full((self.face_head.out_dim,), 1e-3))

        self.fusion = FusionMLP(fusion_in, hidden=fusion_hidden, dropout=fusion_dropout)

    def forward(
        self,
        cls: torch.Tensor,
        patch_tokens: torch.Tensor,
        grid_hw: tuple[int, int],
        face_meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [self.global_norm(cls)]
        if self.patch_head is not None:
            parts.append(self.patch_head(patch_tokens) * self.patch_gate)
        if self.face_head is not None:
            parts.append(self.face_head(patch_tokens, grid_hw, face_meta) * self.face_gate)
        fused = torch.cat(parts, dim=-1)
        return self.fusion(fused)


class ConsistencyNet(nn.Module):
    """Frozen DINO backbone + ConsistencyHead (global [+ patch] [+ face] -> fusion).

    forward(imgs, face_meta=None) -> (B, 1) logits -- identical shape to the baseline
    CNN, so run_epoch / probe / TTA / integrity machinery is reused without modification.
    ``face_meta`` is optional and ignored unless the face-region head is enabled.

    The backbone attribute is an nn.Module but all its parameters have
    requires_grad=False, so AdamW / SGD only updates the head.
    """

    def __init__(self, backbone: _FeatureWrapper, **head_kwargs) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = ConsistencyHead(backbone.embed_dim, **head_kwargs)

    def forward(self, imgs: torch.Tensor, face_meta: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.backbone.forward_features(imgs)
        return self.head(feats["cls"], feats["patch_tokens"], feats["grid_hw"], face_meta)

    # ------------------------------------------------------------------
    # Checkpoint: only head weights (backbone reloads from hub at inference)
    # ------------------------------------------------------------------

    def state_dict(self, *args, **kwargs):
        return self.head.state_dict(*args, **kwargs)

    def load_state_dict(self, sd, strict: bool = True):
        return self.head.load_state_dict(sd, strict=strict)


def build_consistency_model(cfg: Config) -> ConsistencyNet:
    """Build a ConsistencyNet from a Config.

    Reads extra.backbone_name (default "dinov2_vitb14") and the per-head toggles
    (extra.use_patch_consistency, extra.use_face_region, default both False -> matches
    the S2 GlobalHead-only architecture). Prints trainable param count so it's easy to
    verify the backbone is frozen.
    """
    backbone_name: str = cfg.extra.get("backbone_name", "dinov2_vitb14")
    wrapper = load_backbone(backbone_name)
    use_patch = bool(cfg.extra.get("use_patch_consistency", False))
    use_face = bool(cfg.extra.get("use_face_region", False))
    model = ConsistencyNet(
        wrapper,
        use_patch_consistency=use_patch,
        use_face_region=use_face,
        patch_layers=int(cfg.extra.get("patch_consistency_layers", 2)),
        patch_heads=int(cfg.extra.get("patch_consistency_heads", 8)),
        patch_dropout=float(cfg.extra.get("patch_consistency_dropout", 0.1)),
        face_hidden=int(cfg.extra.get("face_region_hidden", 128)),
        fusion_dropout=float(cfg.extra.get("fusion_dropout", 0.3)),
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"[consistency] backbone={backbone_name} embed_dim={wrapper.embed_dim} "
        f"patch_size={wrapper.patch_size} | heads: patch={use_patch} face={use_face} | "
        f"trainable={n_trainable:,}  frozen={n_frozen:,}"
    )
    return model
