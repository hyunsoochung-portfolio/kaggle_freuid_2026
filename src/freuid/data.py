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
from dataclasses import dataclass, field
from pathlib import Path

import logging

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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
    label: int  # 1 = fraud, 0 = bona-fide, -1 = unknown (test)
    is_digital: bool | None = None
    type: str | None = None  # "COUNTRY/DOCTYPE", e.g. "EGYPT/DL"
    card_path: Path | None = None   # rectified card PNG from regions cache (use_rectify)
    face_box: dict | None = field(default=None, repr=False)  # bbox from regions cache (use_face_region)


def load_labels(root: str | Path, split: str = "train") -> pd.DataFrame:
    """Labels dataframe for a split, with a resolved absolute ``path`` column.

    Paths are rebuilt from the id to dodge the double-nesting mismatch. For
    ``public_test`` the csv is the sample submission (no real labels) → label = -1.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
    root = Path(root)
    img_dir, csv_name = SPLITS[split]
    df = pd.read_csv(root / csv_name, dtype={"id": str})  # keep hex ids as strings

    df["path"] = df["id"].map(lambda i: root / img_dir / f"{i}.jpeg")
    if split == "public_test":
        df["label"] = -1
        for col in ("is_digital", "type"):
            df[col] = df.get(col)
    return df
#[id, image_path, label, path, is_digital, type] tables are used in 
#train/val/test splits, and the path column is used to load images. 


class FreuidDataset(Dataset):
    """Image dataset yielding (transformed_image, label).

    Optionally restrict to a subset of ids (for train/val splits) via ``ids``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        ids: set[str] | None = None,
        regions_dir: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self._regions_dir = regions_dir
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        src = s.card_path if s.card_path is not None else s.path
        img = Image.open(src).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label

    # __getitem__ / __len__ 은 파이썬의 "정해진 이름"(특수 메서드)이라, 이것만 구현하면
    # 이 객체는 dataset[i] 와 len(dataset) 으로 다룰 수 있다:
    #   - dataset[i]   → 파이썬이 자동으로 __getitem__(i) 호출
    #   - len(dataset) → 자동으로 __len__() 호출
    # DataLoader는 바로 이 약속(dataset[i], len(dataset))에 기대어 동작한다:
    #   for batch in loader:        # DataLoader가
    #       i = sampler가 고른 인덱스  #   순서를 정하고(shuffle이면 섞음)
    #       sample = dataset[i]     #   우리 __getitem__(i) 를 자동 호출해 한 장씩 받아
    #       ...                     #   batch_size개 모아 텐서로 쌓아(collate) 배치로 넘김
    # 즉 우리가 만든 클래스라도 "정해진 메서드 이름"만 채우면 DataLoader가 알아서 호출한다.


def stratified_split(
    root: str | Path,
    val_fraction: float = 0.1,
    seed: int = 42,
    stratify_on: tuple[str, ...] = ("label", "type"),
) -> tuple[set[str], set[str]]:
    """Split train ids into (train_ids, val_ids), stratified by label×type.

    Stratifying on type as well as label keeps every document domain represented in
    validation, which matters because the test set probes cross-domain generalization.
    """
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

    Mirrors ``freuid-challenge``'s ``get_train_val_split``: all ids whose ``type`` equals
    ``val_doc_type`` become validation, everything else is train. Train and val therefore
    share NO document domain, so val AuDET measures cross-domain transfer (a more honest
    proxy for the unseen-domain private test than the in-domain stratified split).

    Returns ``(train_ids, val_ids)`` — the same shape as ``stratified_split`` so the
    loaders are otherwise unchanged.
    """
    df = load_labels(root, "train")
    types = set(df["type"].dropna())
    if val_doc_type not in types:
        raise ValueError(
            f"val_doc_type {val_doc_type!r} not found; available: {sorted(types)}"
        )
    val_mask = df["type"] == val_doc_type
    val_labels = set(df.loc[val_mask, "label"])
    if val_labels != {0, 1}:
        raise ValueError(
            f"held-out domain {val_doc_type!r} has labels {val_labels}; need both 0 and 1 "
            "(AuDET / ROC-AUC is undefined on a single-class validation set)"
        )
    return set(df.loc[~val_mask, "id"]), set(df.loc[val_mask, "id"])