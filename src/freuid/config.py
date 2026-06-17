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
    image_size: int = 384
    val_fraction: float = 0.1

    # model
    backbone: str = "tf_efficientnetv2_s.in21k"
    pretrained: bool = True

    # train
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 8

    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f.name for f in Config.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    extra = {k: v for k, v in raw.items() if k not in known}
    base = {k: v for k, v in raw.items() if k in known}
    return Config(**base, extra=extra)
