"""Shared, read-only utilities for the finetune_v0 diagnostic analysis scripts.

Every script in this directory loads the finetune_v0 checkpoint, reproduces its exact
training-time validation split, and scores/inspects it -- nothing here trains, fine-tunes,
or modifies the checkpoint or any file under src/freuid/. All outputs go under
reports/analysis_v0/.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from freuid.config import Config
from freuid.data import load_labels, lodo_split, stratified_split
from freuid.models import build_model, build_model_for_config
from freuid.transforms import build_transforms, resolve_data_config
from freuid.utils import pick_device

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "finetune_v0.pt"
REPORT_DIR = REPO_ROOT / "reports" / "analysis_v0"


def load_checkpoint(checkpoint_path: Path | str = DEFAULT_CHECKPOINT) -> tuple[Config, dict]:
    """Load the checkpoint's stored config + raw state dict (read-only, CPU)."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_cfg = state.get("config", {})
    if not ckpt_cfg:
        raise SystemExit(f"{checkpoint_path} has no stored config -- cannot rebuild the split")
    known = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in ckpt_cfg.items() if k in known})
    return cfg, state


def get_split_ids(cfg: Config) -> tuple[set[str], set[str]]:
    """Reproduce the exact (train_ids, val_ids) split used during finetune_v0 training."""
    if cfg.val_doc_type:
        return lodo_split(cfg.data_dir, cfg.val_doc_type)
    return stratified_split(cfg.data_dir, cfg.val_fraction, cfg.seed)


def split_dataframe(cfg: Config, ids: set[str]) -> pd.DataFrame:
    """Metadata rows (id, path, label, is_digital, type) restricted to `ids`, sorted by id."""
    df = load_labels(cfg.data_dir, "train")
    df = df[df["id"].isin(ids)].sort_values("id").reset_index(drop=True)
    return df


def build_finetuned_model(cfg: Config, state: dict, device) -> torch.nn.Module:
    """The checkpoint's own model (dispatched on cfg.extra["model_type"]: baseline / consistency
    / bayar_fusion), with its fine-tuned weights loaded."""
    model = build_model_for_config(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def build_pretrained_model(cfg: Config, device) -> torch.nn.Module:
    """A fresh, untouched ImageNet/DINOv2-pretrained model of the same architecture.

    Used only as a comparison point for representation-drift analysis -- never trained,
    never saved, never substituted for the real checkpoint anywhere else.
    """
    model = build_model(cfg.backbone, pretrained=True).to(device)
    model.eval()
    return model


def eval_transform(cfg: Config):
    """The exact deterministic (non-augmented) transform finetune_v0's own val_loader used."""
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    tf = build_transforms(data_cfg["image_size"], train=False, mean=data_cfg["mean"], std=data_cfg["std"])
    return tf, data_cfg


@torch.no_grad()
def score_images(model, images, transform, device, batch_size: int = 32) -> np.ndarray:
    """Score a list of already-open PIL images -> P(fraud) in [0,1]. No TTA (single pass)."""
    scores = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        imgs = torch.stack([transform(im) for im in batch]).to(device)
        logits = model(imgs)
        scores.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
    return np.concatenate(scores) if scores else np.array([])


@torch.no_grad()
def score_paths(model, paths, transform, device, batch_size: int = 32) -> np.ndarray:
    """Score a list of file paths directly (opens + converts to RGB internally)."""
    scores = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch]).to(device)
        logits = model(imgs)
        scores.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
    return np.concatenate(scores) if scores else np.array([])


def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.6f}") -> str:
    """Minimal DataFrame -> GitHub-flavored markdown table (no `tabulate` dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def ensure_report_dir(subdir: str | None = None) -> Path:
    d = REPORT_DIR / subdir if subdir else REPORT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def device_and_seed(cfg: Config):
    import random
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    return pick_device()
