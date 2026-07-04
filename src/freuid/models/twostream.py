"""Two-stream forgery detector: full image + face crop + ELA, fused.

freeze_aux=True freezes face/ELA backbones as fixed feature extractors -- an
unfrozen first attempt overfit badly (AuDET 0.14 vs baseline 0.0000 on the same
MAURITIUS/ID domain holdout); freezing recovered to AuDET=0.0002.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


class TwoStreamModel(nn.Module):
    def __init__(self, full_backbone="tf_efficientnetv2_s.in21k", face_backbone="resnet18",
                 ela_backbone="resnet18", pretrained=True, hidden_dim=256, dropout=0.3,
                 freeze_aux=False):
        super().__init__()
        self.full_net = timm.create_model(full_backbone, pretrained=pretrained, num_classes=0)
        self.face_net = timm.create_model(face_backbone, pretrained=pretrained, num_classes=0)
        self.ela_net = timm.create_model(ela_backbone, pretrained=pretrained, num_classes=0)
        if freeze_aux:
            for net in (self.face_net, self.ela_net):
                for p in net.parameters():
                    p.requires_grad = False
                net.eval()
        self.freeze_aux = freeze_aux
        feat_dim = self.full_net.num_features + self.face_net.num_features + self.ela_net.num_features
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_aux:
            self.face_net.eval()
            self.ela_net.eval()
        return self

    def forward(self, full, face, ela):
        f_full = self.full_net(full)
        ctx = torch.no_grad() if self.freeze_aux else torch.enable_grad()
        with ctx:
            f_face = self.face_net(face)
            f_ela = self.ela_net(ela)
        feat = torch.cat([f_full, f_face, f_ela], dim=1)
        return self.head(feat)


def build_twostream_model(full_backbone="tf_efficientnetv2_s.in21k", face_backbone="resnet18",
                           ela_backbone="resnet18", pretrained=True, freeze_aux=False):
    return TwoStreamModel(full_backbone, face_backbone, ela_backbone,
                          pretrained=pretrained, freeze_aux=freeze_aux)
