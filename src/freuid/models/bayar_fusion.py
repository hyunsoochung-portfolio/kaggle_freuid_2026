"""BayarFusionNet: fully fine-tuned DINOv2 CLS embedding, gated-fused with a BayarConv2d
forensic-noise + RGB face-crop stream (lifted from a teammate's feat/overlay-detector branch),
combined via a zero-init MLP head.

Motivating experiment (see CLAUDE.md, "overlay_colab" section): three score-level ensembles of
finetune_v0 + a pretrained overlay_colab model all regressed the public LB, despite
overlay_colab being decisive and directionally right on finetune_v0's most uncertain
predictions. The failure was architectural (a fixed combination rule can't tell overlay's
trustworthy calls from its untrustworthy ones), not conceptual. This model tests whether a
*jointly fine-tuned, gated* fusion head can learn that distinction instead.

Unlike the parked model_type=consistency path (frozen backbone -- see consistency_model.py),
DINOv2 here is FULLY TRAINABLE: `.blocks` is still exposed with num_classes=0, so
freuid.optim.build_llrd_param_groups applies exactly as it does on the baseline finetune path.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from freuid.config import Config
from freuid.consistency_model import FusionMLP

FACE_META_DIM = 5  # matches freuid.data.FACE_META_DIM / freuid.consistency_model.FACE_META_DIM

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# BayarConv2d / NoiseStream -- lifted from feat/overlay-detector's
# src/freuid/models/overlay.py (TwoStreamOverlayNet), unchanged, so this branch's forensic
# feature extractor matches the one already informally validated on this dataset.
# ---------------------------------------------------------------------------

class BayarConv2d(nn.Module):
    """Learnable constrained high-pass filter: center weight fixed to -1, the rest
    normalized to sum to 1 (then negated by the -1 center), so the layer always computes a
    residual regardless of what it learns -- forces the branch to look at noise, not content.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.center = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = kernel_size // 2

    def _constrained_weights(self) -> torch.Tensor:
        w = self.weight.clone()
        w[:, :, self.center, self.center] = 0
        s = w.sum(dim=(2, 3), keepdim=True)
        s = s + (s == 0).float() * 1e-8
        w = w / s
        w[:, :, self.center, self.center] = -1
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self._constrained_weights(), self.bias, padding=self.padding)


class NoiseStream(nn.Module):
    """BayarConv2d frontend -> small CNN body -> global-avg-pooled feature vector."""

    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.frontend = BayarConv2d(3, 16, kernel_size=5)
        self.body = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.frontend(x))


class OverlayStream(nn.Module):
    """Forensic-noise + RGB features on a face crop, concatenated -- no classifier head of
    its own (that's the point: it feeds the gated fusion head below instead).

    Expects an already-cropped, [0,1]-scaled face image; applies ImageNet normalization for
    the RGB branch internally, matching feat/overlay-detector's TwoStreamOverlayNet.
    """

    def __init__(
        self, rgb_backbone: str = "resnet34", noise_feat_dim: int = 128, rgb_pretrained: bool = True
    ) -> None:
        super().__init__()
        self.noise_stream = NoiseStream(feat_dim=noise_feat_dim)
        self.rgb_backbone = timm.create_model(rgb_backbone, pretrained=rgb_pretrained, num_classes=0)
        self.register_buffer("rgb_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))
        self.out_dim = noise_feat_dim + self.rgb_backbone.num_features

    def forward(self, face_crop: torch.Tensor) -> torch.Tensor:
        noise_feat = self.noise_stream(face_crop)
        rgb_feat = self.rgb_backbone((face_crop - self.rgb_mean) / self.rgb_std)
        return torch.cat([noise_feat, rgb_feat], dim=1)


class BayarFusionNet(nn.Module):
    """Fully fine-tuned DINOv2 CLS embedding, gated-fused with a BayarConv2d+RGB face-crop
    branch. forward(imgs, face_crop, face_meta=None) -> (B, 1) logits -- same shape/contract
    as the baseline and consistency paths, so run_epoch/probe/TTA/integrity machinery is
    reused without modification.

    ``face_meta`` (see freuid.data.face_meta_tensor) supplies the validity flag at index
    FACE_META_DIM-1: when 0 (no cached face box for this sample), the overlay branch's
    contribution is zeroed regardless of what pixels were in face_crop, matching
    FaceRegionHead's "no signal instead of a wrong one" convention.

    ``overlay_gate`` is a LayerScale-style per-channel parameter initialized near-zero
    (default 1e-3, same as ConsistencyHead's patch_gate/face_gate) -- the model starts
    close to DINOv2-only behavior and "opens" the overlay pathway only as it earns its
    keep, per the S3 postmortem in CLAUDE.md.
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained: bool = True,
        rgb_backbone: str = "resnet34",
        noise_feat_dim: int = 128,
        rgb_pretrained: bool = True,
        fusion_hidden: int | None = None,
        fusion_dropout: float = 0.3,
        gate_init: float = 1e-3,
    ) -> None:
        super().__init__()
        try:
            self.dino = timm.create_model(
                backbone, pretrained=pretrained, num_classes=0, dynamic_img_size=True,
            )
        except TypeError:
            self.dino = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.overlay = OverlayStream(rgb_backbone, noise_feat_dim, rgb_pretrained)
        self.overlay_gate = nn.Parameter(torch.full((self.overlay.out_dim,), gate_init))
        self.fusion = FusionMLP(
            self.dino.num_features + self.overlay.out_dim,
            hidden=fusion_hidden, dropout=fusion_dropout,
        )

    def forward(
        self,
        imgs: torch.Tensor,
        face_crop: torch.Tensor,
        face_meta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dino_feat = self.dino(imgs)
        overlay_feat = self.overlay(face_crop) * self.overlay_gate
        if face_meta is not None:
            valid = face_meta[:, FACE_META_DIM - 1 : FACE_META_DIM]
            overlay_feat = overlay_feat * valid
        fused = torch.cat([dino_feat, overlay_feat], dim=1)
        return self.fusion(fused)


def build_bayar_fusion_param_groups(
    model: BayarFusionNet, base_lr: float, weight_decay: float, decay: float = 0.7,
) -> list[dict]:
    """LLRD param groups for BayarFusionNet: per-block decay on the ``dino`` sub-module
    (via ``freuid.optim.build_llrd_param_groups``, which needs `.blocks` -- present on
    `model.dino`, not on the fused model itself), plus one additional base-LR group for
    everything else (overlay stream, gate, fusion head): new parameters with no
    pretrained depth structure, so they don't need decayed rates.
    """
    from freuid.optim import build_llrd_param_groups

    dino_groups = build_llrd_param_groups(model.dino, base_lr, weight_decay, decay=decay)
    other_params = [(n, p) for n, p in model.named_parameters() if not n.startswith("dino.")]
    no_decay = [p for n, p in other_params if p.ndim <= 1 or n.endswith(".bias")]
    decay_params = [p for n, p in other_params if p.ndim > 1 and not n.endswith(".bias")]
    other_groups = [
        {"params": decay_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
    ]
    print(f"[bayar_fusion] LLRD: {len(dino_groups)} dino depth groups + 2 base-lr groups "
          f"for overlay/gate/fusion ({sum(p.numel() for _, p in other_params):,} params)")
    return dino_groups + other_groups


def build_bayar_fusion_model(cfg: Config) -> BayarFusionNet:
    """Build a BayarFusionNet from a Config. Reads extra.overlay.* (defaults match
    feat/overlay-detector's own config)."""
    overlay_cfg = cfg.extra.get("overlay", {})
    model = BayarFusionNet(
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        rgb_backbone=overlay_cfg.get("rgb_backbone", "resnet34"),
        noise_feat_dim=int(overlay_cfg.get("noise_feat_dim", 128)),
        rgb_pretrained=bool(overlay_cfg.get("rgb_pretrained", True)),
        fusion_dropout=float(overlay_cfg.get("fusion_dropout", 0.3)),
        gate_init=float(overlay_cfg.get("fusion_gate_init", 1e-3)),
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[bayar_fusion] backbone={cfg.backbone} dino_dim={model.dino.num_features} "
        f"overlay_dim={model.overlay.out_dim} | trainable={n_trainable:,} (fully fine-tuned)"
    )
    return model
