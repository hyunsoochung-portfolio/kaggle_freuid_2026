"""DCT-branch dual-stream model for document fraud detection.

Architecture (FFDN 논문 방식 간소화):
    RGB stream:  timm backbone (ViT / EfficientNet 등) -> feature vector
    DCT stream:  이미지를 8x8 블록 DCT 변환 -> 경량 CNN -> feature vector
    Fusion:      concat -> Linear -> 1 logit

DCT stream이 JPEG 압축 아티팩트 (GenAI edits, Print-and-Capture)를
RGB stream이 잡기 어려운 주파수 도메인에서 직접 포착한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# DCT 변환 유틸
# ---------------------------------------------------------------------------

def _dct_basis(n: int = 8) -> torch.Tensor:
    """n×n DCT-II basis matrix (orthonormal). shape: [n, n]"""
    k = torch.arange(n, dtype=torch.float32)
    i = torch.arange(n, dtype=torch.float32).unsqueeze(1)  # [n, 1]
    basis = torch.cos((2 * i + 1) * k * torch.pi / (2 * n))  # [n, n]
    basis[0] *= (1.0 / n) ** 0.5
    basis[1:] *= (2.0 / n) ** 0.5
    return basis  # [n, n]


class BlockDCT(nn.Module):
    """이미지를 8x8 블록으로 나눠 DCT 계수를 추출한다.

    입력:  [B, 3, H, W]  (RGB, 정규화된 값)
    출력:  [B, 3*64, H//8, W//8]  (각 블록의 64개 DCT 계수)
    """

    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
        basis = _dct_basis(block_size)  # [8, 8]
        weight = torch.einsum("ij,kl->ikjl", basis, basis)  # [8, 8, 8, 8]
        weight = weight.reshape(block_size * block_size, 1, block_size, block_size)
        self.register_buffer("weight", weight)  # 학습 안 함 (고정 DCT basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        bs = self.block_size
        x = x.reshape(B * C, 1, H, W)
        dct = F.conv2d(x, self.weight, stride=bs)  # [B*C, 64, H//8, W//8]
        _, _, h, w = dct.shape
        dct = dct.reshape(B, C * 64, h, w)  # [B, 192, H//8, W//8]
        return dct


# ---------------------------------------------------------------------------
# DCT stream CNN (경량)
# ---------------------------------------------------------------------------

class DCTStream(nn.Module):
    """DCT 계수를 받아 feature vector를 출력하는 경량 CNN.

    입력:  [B, 192, H//8, W//8]
    출력:  [B, out_dim]
    """

    def __init__(self, in_channels: int = 192, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # 블록1
            nn.Conv2d(in_channels, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),         nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),
            # 블록2
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.MaxPool2d(2),
            # 블록3
            nn.Conv2d(256, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)  # [B, out_dim]


# ---------------------------------------------------------------------------
# 듀얼 스트림 모델
# ---------------------------------------------------------------------------

# ImageNet 정규화 파라미터 (transforms.py와 동일)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class DualStreamFraudDetector(nn.Module):
    """RGB stream + DCT stream -> fusion -> fraud logit.

    Args:
        backbone:    timm 모델 이름 (RGB stream)
        pretrained:  ImageNet pretrained 여부
        dct_dim:     DCT stream feature 차원
        dropout:     fusion head dropout
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s.in21k",
        pretrained: bool = True,
        dct_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        # --- RGB stream ---
        self.rgb_backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0  # feature만 추출
        )
        rgb_dim = self.rgb_backbone.num_features  # ViT-B: 768, EfficientNet-S: 1280

        # --- DCT stream ---
        self.block_dct = BlockDCT(block_size=8)
        self.dct_stream = DCTStream(in_channels=192, out_dim=dct_dim)

        # --- Fusion head ---
        fusion_dim = rgb_dim + dct_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(fusion_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RGB stream: 정규화된 값 그대로 (ViT가 원래 이렇게 씨)
        rgb_feat = self.rgb_backbone(x)           # [B, rgb_dim]

        # DCT stream: 정규화 역변환 후 0~255로 복원
        # DCT는 픽셀값 0~255 기준으로 설계된 변환이라
        # 정규화된 -2~+2 값을 그대로 넣으면 계수가 의미없는 값이 됨
        mean = _IMAGENET_MEAN.to(x.device)
        std  = _IMAGENET_STD.to(x.device)
        x_pixel = (x * std + mean) * 255.0          # [-2~+2] -> [0~255]
        x_pixel = x_pixel.clamp(0.0, 255.0)         # 범위 보증
        dct_coef = self.block_dct(x_pixel)           # [B, 192, H//8, W//8]
        dct_feat = self.dct_stream(dct_coef)         # [B, dct_dim]

        fused = torch.cat([rgb_feat, dct_feat], dim=1)  # [B, rgb_dim+dct_dim]
        return self.head(fused)                          # [B, 1]


# ---------------------------------------------------------------------------
# 빌드 함수
# ---------------------------------------------------------------------------

def build_dct_model(
    backbone: str = "tf_efficientnetv2_s.in21k",
    pretrained: bool = True,
    dct_dim: int = 256,
    dropout: float = 0.3,
) -> nn.Module:
    return DualStreamFraudDetector(backbone, pretrained, dct_dim, dropout)
