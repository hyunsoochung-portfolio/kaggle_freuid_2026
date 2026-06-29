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
    # If set, hold out this whole document `type` for validation (Leave-One-Domain-Out,
    # e.g. "MAURITIUS/ID"); when None, use the random stratified split + val_fraction.
    val_doc_type: str | None = None

    # model
    backbone: str = "tf_efficientnetv2_s.in21k"
    pretrained: bool = True

    # train
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 8
    limit: int | None = None  # cap train/val sizes for quick dev runs; None = full data

    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    # `extra` is handled separately so it isn't passed twice when the YAML has an
    # explicit `extra:` block (new style) AND unknown top-level keys (legacy style).
    known = {f.name for f in Config.__dataclass_fields__.values()} - {"extra"}  # type: ignore[attr-defined]
    explicit_extra: dict = raw.pop("extra", {}) or {}
    legacy_extra = {k: v for k, v in raw.items() if k not in known}
    base = {k: v for k, v in raw.items() if k in known}
    return Config(**base, extra={**explicit_extra, **legacy_extra})
