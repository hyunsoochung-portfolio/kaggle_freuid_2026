"""Dataset + dataloaders, wired to the real FREUID layout.

The Kaggle archive double-nests each split's images and the label CSV's
``image_path`` column is off by that extra level, so we build paths from the id
instead of trusting ``image_path``:

    data/train/train/<id>.jpeg          labels: train_labels.csv
    data/train_sample/train_sample/...  labels: train_sample_labels.csv
    data/public_test/public_test/...    ids:    sample_submission.csv (label is a placeholder)

Label convention matches metrics.py: 1 = fraud, 0 = bona-fide, -1 = unknown (test).
``train_labels.csv`` columns: id, image_path, label, is_digital, type ("COUNTRY/DOCTYPE").
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageChops
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# face_meta tensor layout: [x1_frac, y1_frac, x2_frac, y2_frac, valid] -- must match
# freuid.consistency_model.FACE_META_DIM. Kept as a plain constant here (rather than an
# import) so the data layer doesn't depend on the model layer.
FACE_META_DIM = 5

# split -> (image subdir relative to data root, labels/ids csv)
SPLITS: dict[str, tuple[str, str]] = {
    "train": ("train/train", "train_labels.csv"),
    "train_sample": ("train_sample/train_sample", "train_sample_labels.csv"),
    "public_test": ("public_test/public_test", "sample_submission.csv"),
}


@dataclass
class Sample:
    id: str
    path: Path
    label: int
    is_digital: bool | None = None
    type: str | None = None  # "COUNTRY/DOCTYPE", e.g. "EGYPT/DL"
    card_path: Path | None = None  # rectified card PNG from regions cache (use_rectify)
    face_box: dict | None = field(default=None, repr=False)  # bbox from regions cache (use_face_region)


def load_labels(root: str | Path, split: str = "train") -> pd.DataFrame:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
    root = Path(root)
    img_dir, csv_name = SPLITS[split]
    df = pd.read_csv(root / csv_name, dtype={"id": str})
    df["path"] = df["id"].map(lambda i: root / img_dir / f"{i}.jpeg")
    if split == "public_test":
        df["label"] = -1
        for col in ("is_digital", "type"):
            df[col] = df.get(col)
    return df


def unpack_batch(batch):
    """(imgs, labels) or (imgs, labels, face_meta) -> (imgs, labels, face_meta_or_None).

    Lets a single loop body (``run_epoch``, sanity checks, inference) handle loaders
    built with or without ``return_face_meta`` without branching on config everywhere.
    """
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    imgs, labels = batch
    return imgs, labels, None


def face_meta_tensor(sample: "Sample", img_size: tuple[int, int]) -> torch.Tensor:
    """Face-box fractions + validity flag for the FaceRegionHead: [x1,y1,x2,y2,valid].

    ``img_size`` is the (W, H) of the image actually opened for this sample (the
    rectified card when ``card_path`` is set) -- the space the cached face box is in.
    Returns an all-zero (invalid) tensor when there's no cached box, or when the box
    is the center-square fallback (SCRFD ``score`` == 0, i.e. no real detection).
    """
    if sample.card_path is None or sample.face_box is None:
        return torch.zeros(FACE_META_DIM, dtype=torch.float32)
    fb = sample.face_box
    if float(fb.get("score", 0.0)) <= 0.0:
        return torch.zeros(FACE_META_DIM, dtype=torch.float32)
    w, h = img_size
    return torch.tensor(
        [fb["x1"] / w, fb["y1"] / h, fb["x2"] / w, fb["y2"] / h, 1.0],
        dtype=torch.float32,
    )


class FreuidDataset(Dataset):
    """Image dataset yielding (transformed_image, label), or (transformed_image, label,
    face_meta) when ``return_face_meta=True`` (see ``face_meta_tensor``).

    Optionally restrict to a subset of ids (for train/val splits) via ``ids``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        ids: set[str] | None = None,
        regions_dir: Path | None = None,
        return_face_meta: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self._regions_dir = regions_dir
        self._return_face_meta = return_face_meta
        df = load_labels(self.root, split)
        if ids is not None:
            df = df[df["id"].isin(ids)]
        self.samples: list[Sample] = []
        for r in df.itertuples(index=False):
            card_path: Path | None = None
            face_box: dict | None = None
            if regions_dir is not None:
                rdir = regions_dir / str(r.id)
                _cp = rdir / "card.png"
                _fp = rdir / "face.json"
                if _cp.exists():
                    card_path = _cp
                if _fp.exists():
                    try:
                        face_box = json.loads(_fp.read_text())
                    except Exception:
                        pass
            self.samples.append(Sample(
                id=r.id,
                path=Path(r.path),
                label=int(r.label),
                is_digital=bool(r.is_digital) if pd.notna(r.is_digital) else None,
                type=r.type if pd.notna(r.type) else None,
                card_path=card_path,
                face_box=face_box,
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        src = s.card_path if s.card_path is not None else s.path
        img = Image.open(src).convert("RGB")
        img_size = img.size  # (W, H) before transform -- the space face_box coords are in
        if self.transform is not None:
            img = self.transform(img)
        if self._return_face_meta:
            return img, s.label, face_meta_tensor(s, img_size)
        return img, s.label


def stratified_split(root, val_fraction=0.1, seed=42, stratify_on=("label", "type")):
    """NOTE: mixes every domain into train+val -- not a cross-domain test.
    Use domain_holdout_split() / lodo_split() for that."""
    df = load_labels(root, "train")
    rng = np.random.default_rng(seed)
    val_ids: set[str] = set()
    for _, group in df.groupby(list(stratify_on)):
        ids = group["id"].to_numpy()
        n_val = min(len(ids), max(1, round(len(ids) * val_fraction)))
        val_ids.update(rng.choice(ids, size=n_val, replace=False).tolist())
    all_ids = set(df["id"])
    return all_ids - val_ids, val_ids


def lodo_split(root: str | Path, val_doc_type: str) -> tuple[set[str], set[str]]:
    """Leave-One-Domain-Out: hold out one whole document ``type`` for validation.

    All ids whose ``type`` equals ``val_doc_type`` become validation, everything else
    is train. Train and val therefore share NO document domain, so val AuDET measures
    cross-domain transfer (a more honest proxy for the unseen-domain private test than
    the in-domain stratified split). Validates both classes are present in the
    held-out domain (AuDET/ROC-AUC is undefined on a single-class validation set).
    """
    df = load_labels(root, "train")
    types = set(df["type"].dropna())
    if val_doc_type not in types:
        raise ValueError(f"val_doc_type {val_doc_type!r} not found; available: {sorted(types)}")
    val_mask = df["type"] == val_doc_type
    val_labels = set(df.loc[val_mask, "label"])
    if val_labels != {0, 1}:
        raise ValueError(
            f"held-out domain {val_doc_type!r} has labels {val_labels}; need both 0 and 1"
        )
    return set(df.loc[~val_mask, "id"]), set(df.loc[val_mask, "id"])


def domain_holdout_split(root, holdout_types):
    """Leave-one-(or more)-domain-out: val = ids whose type is in holdout_types,
    a domain the model never sees while training. Honest cross-domain check.

    Equivalent in spirit to lodo_split() but accepts multiple held-out types at once
    and doesn't enforce both-classes-present -- kept separate since train.py's
    twostream path already depends on this exact signature."""
    if isinstance(holdout_types, str):
        holdout_types = [holdout_types]
    holdout_set = set(holdout_types)
    df = load_labels(root, "train")
    known_types = set(df["type"].unique())
    unknown = holdout_set - known_types
    if unknown:
        raise ValueError(f"holdout_types not found in data: {unknown} (known: {known_types})")
    val_mask = df["type"].isin(holdout_set)
    val_ids = set(df.loc[val_mask, "id"])
    train_ids = set(df.loc[~val_mask, "id"])
    if not val_ids or not train_ids:
        raise ValueError("holdout split produced an empty train or val set")
    return train_ids, val_ids


def load_face_boxes(root):
    path = Path(root) / "face_boxes.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"id": str})
    return {r.id: (bool(r.detected), int(r.x), int(r.y), int(r.w), int(r.h))
            for r in df.itertuples(index=False)}


def crop_face(img, box):
    detected, x, y, w, h = box
    W, H = img.size
    if not detected or w <= 0 or h <= 0:
        side = min(W, H) // 3
        cx, cy = W // 4, H // 2
        return img.crop((max(0, cx - side), max(0, cy - side), cx + side, cy + side))
    pad_w, pad_h = int(w * 0.3), int(h * 0.3)
    left, top = max(0, x - pad_w), max(0, y - pad_h)
    right, bottom = min(W, x + w + pad_w), min(H, y + h + pad_h)
    return img.crop((left, top, right, bottom))


def compute_ela(img, quality=90):
    img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img, resaved)
    extrema = diff.getextrema()
    max_diff = max(e[1] for e in extrema) or 1
    return Image.eval(diff, lambda px: min(255, int(px * 255.0 / max_diff)))


class TwoStreamDataset(Dataset):
    def __init__(self, root, split="train", ids=None,
                 full_transform=None, face_transform=None, ela_transform=None):
        self.root = Path(root)
        self.split = split
        self.full_transform = full_transform
        self.face_transform = face_transform
        self.ela_transform = ela_transform
        self.face_boxes = load_face_boxes(self.root)
        df = load_labels(self.root, split)
        if ids is not None:
            df = df[df["id"].isin(ids)]
        self.samples: list[Sample] = [
            Sample(id=r.id, path=Path(r.path), label=int(r.label),
                   is_digital=bool(r.is_digital) if pd.notna(r.is_digital) else None,
                   type=r.type if pd.notna(r.type) else None)
            for r in df.itertuples(index=False)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        box = self.face_boxes.get(s.id, (False, 0, 0, 0, 0))
        face_img = crop_face(img, box)
        ela_img = compute_ela(img)
        full_t = self.full_transform(img) if self.full_transform else img
        face_t = self.face_transform(face_img) if self.face_transform else face_img
        ela_t = self.ela_transform(ela_img) if self.ela_transform else ela_img
        return full_t, face_t, ela_t, s.label
