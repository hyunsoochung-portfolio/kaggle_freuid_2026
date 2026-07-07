"""Model definitions."""

from typing import TYPE_CHECKING

from freuid.models.baseline import build_model
from freuid.models.bayar_fusion import (
    BayarFusionNet,
    build_bayar_fusion_model,
    build_bayar_fusion_param_groups,
)
from freuid.models.consistency import build_consistency_model

if TYPE_CHECKING:
    import torch.nn as nn

    from freuid.config import Config

__all__ = [
    "build_model",
    "build_consistency_model",
    "BayarFusionNet",
    "build_bayar_fusion_model",
    "build_bayar_fusion_param_groups",
    "build_model_for_config",
]


def build_model_for_config(cfg: "Config") -> "nn.Module":
    """Single source of truth for model_type dispatch (consistency / bayar_fusion / baseline).

    Mirrors the inline dispatch in train.py's build_loaders/model-build section and
    infer.py's main() -- both duplicate this same three-way branch. Callers that only need
    an untrained-but-correctly-shaped module to immediately load_state_dict into (inference,
    analysis/probe scripts) should call this instead of re-deriving the branch a third time;
    train.py keeps its own copy since grad_checkpointing/train_last_k_blocks/LLRD setup is
    entangled with the branch there and isn't worth disturbing for this.
    """
    model_type = cfg.extra.get("model_type", "baseline")
    if model_type == "consistency":
        return build_consistency_model(cfg)
    if model_type == "bayar_fusion":
        return build_bayar_fusion_model(cfg)
    return build_model(cfg.backbone, pretrained=False)
