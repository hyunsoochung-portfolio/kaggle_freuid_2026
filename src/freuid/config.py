"""Config loading. One YAML per experiment keeps runs reproducible and reviewable."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    name: str = "baseline"
    seed: int = 42

    # data
    data_dir: str = "data"
    image_size: int | None = None  # None → use the backbone's native input resolution
    val_fraction: float = 0.1
    # If set, hold out these whole document types as validation (cross-domain split) instead
    # of the random stratified split. e.g. ["MAURITIUS/ID"]. val_fraction is then ignored.
    val_types: list[str] | None = None

    # model
    backbone: str = "tf_efficientnetv2_s.in21k"
    pretrained: bool = True
    head_dropout: float = 0.0  # dropout before the final linear head (0 = timm default)
    pool: str | None = None  # global_pool override ("avg"/"token"/...); None = backbone default

    # train
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    limit: int | None = None  # cap train/val sizes for quick dev runs; None = full data

    # fine-tuning recipe (all default to OFF so existing configs behave exactly as before)
    amp: bool = False  # mixed precision (only takes effect on CUDA); big speed/memory win for ViT
    grad_checkpointing: bool = False  # recompute activations in backward; ~half memory (for 518)
    warmup_epochs: float = 0.0  # >0 → linear LR warmup then cosine decay; 0 → constant LR
    lr_min: float = 0.0  # cosine floor (absolute LR at the end of training)
    llrd_decay: float | None = None  # layer-wise LR decay factor (e.g. 0.75); None = uniform LR

    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    extra = {k: v for k, v in raw.items() if k not in known}
    base = {k: v for k, v in raw.items() if k in known}
    return Config(**base, extra=extra)
