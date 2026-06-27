import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# BayarConv — learnable constrained high-pass filter
# ---------------------------------------------------------------------------

class BayarConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.center = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.padding = kernel_size // 2

    def _constrained_weights(self):
        w = self.weight.clone() 
        # zero the center, normalize the rest to sum to 1, then set center to -1
        w[:, :, self.center, self.center] = 0
        # sum over spacial dims per (out, in) pair
        s = w.sum(dim=(2, 3), keepdim=True)
        # avoid division by zero
        s = s + (s == 0).float() * 1e-8
        w = w / s
        w[:, :, self.center, self.center] = -1
        return w
    
    def forward(self, x):
        return F.conv2d(x, self._constrained_weights(), self.bias, padding=self.padding)

# ---------------------------------------------------------------------------
# SRM fixed-filter bank — 3 standard forensic kernels, not learnable
# ---------------------------------------------------------------------------

class SRMConv2d(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # 3 SRM Kernels (5x5)
        srm1 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=np.float32)

        srm2 = np.array([
            [ 0,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  1, -4,  1,  0],
            [ 0,  0,  1,  0,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=np.float32)

        srm3 = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=np.float32) / 12.0

        kernels = np.stack([srm1, srm2, srm3]) # (3, 5, 5)
        # tile across input channels: (3*in_channels, in_channels, 5, 5)
        weight = np.zeros((3*in_channels, in_channels, 5, 5), dtype=np.float32)
        for i, k in enumerate(kernels):
            for c in range(in_channels):
                weight[i * in_channels + c, c] = k
        
        self.out_channels = 3 * in_channels
        self.register_buffer("weight", torch.from_numpy(weight))
        self.register_buffer("bias", torch.zeros(self.out_channels))
    
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, padding=2)

# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

class NoiseStream(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        frontend_type = cfg["model"].get("noise_frontend", "bayar")
        if frontend_type == "srm":
            self.frontend = SRMConv2d(in_channels=3)
            fe_out = self.frontend.out_channels
        else:
            self.frontend = BayarConv2d(3, 16, kernel_size=5)
            fe_out = 16
        
        feat_dim = cfg["model"].get("noise_feat_dim", 128)
        self.body = nn.Sequential(
            nn.Conv2d(fe_out, 32, 3, padding=1), nn.BatchNorm2d(32),
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

    def forward(self, x):
        return self.body(self.frontend(x))

# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

class TwoStreamOverlayNet(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.noise_stream = NoiseStream(cfg)
        total_feat = self.noise_stream.feat_dim

        self.use_rgb = cfg["model"].get("use_rgb_stream", False)
        if self.use_rgb:
            self.rgb_backbone = timm.create_model(
                cfg["model"]["rgb_backbone"],
                pretrained=cfg["model"].get("rgb_pretrained", True),
                num_classes=0,
            )
            self.register_buffer(
                "rgb_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "rgb_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
            )
            total_feat += self.rgb_backbone.num_features

        fusion_dim = cfg["model"].get("fusion_dim", 128)
        self.head = nn.Sequential(
            nn.Linear(total_feat, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, x):
        noise_feat = self.noise_stream(x)
        if self.use_rgb:
            x_norm = (x - self.rgb_mean) / self.rgb_std
            rgb_feat = self.rgb_backbone(x_norm)
            fused = torch.cat([noise_feat, rgb_feat], dim=1)
        else:
            fused = noise_feat
        return self.head(fused)


def build_overlay_model(cfg) -> nn.Module:
    return TwoStreamOverlayNet(cfg.extra)