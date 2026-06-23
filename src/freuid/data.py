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

from dataclasses import dataclass
from pathlib import Path

import logging
import os

import cv2
import numpy as np
import pandas as pd
import torch
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
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        df = load_labels(self.root, split)
        if ids is not None:
            df = df[df["id"].isin(ids)]
        self.samples: list[Sample] = [
            Sample(
                id=r.id,
                path=Path(r.path),
                label=int(r.label),
                is_digital=bool(r.is_digital) if pd.notna(r.is_digital) else None,
                type=r.type if pd.notna(r.type) else None,
            )
            for r in df.itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
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


# ---------------------------------------------------------------------------
# Overlay dataset — face-region crop with cache
# ---------------------------------------------------------------------------

_mtcnn_instance = None


def _get_mtcnn(device="cpu"):
    global _mtcnn_instance
    if _mtcnn_instance is None:
        from facenet_pytorch import MTCNN
        _mtcnn_instance = MTCNN(keep_all=False, device=device, post_process=False)
    return _mtcnn_instance


def _crop_face(image: np.ndarray, mtcnn, crop_margin: float) -> np.ndarray:
    h, w = image.shape[:2]
    pil_img = Image.fromarray(image)
    boxes, _ = mtcnn.detect(pil_img)

    if boxes is not None and len(boxes) > 0:
        x1, y1, x2, y2 = boxes[0]
        bw, bh = x2 - x1, y2 - y1
        mx = bw * crop_margin
        my = bh * crop_margin
        x1 = max(0, int(x1 - mx))
        y1 = max(0, int(y1 - my))
        x2 = min(w, int(x2 + mx))
        y2 = min(h, int(y2 + my))
        return image[y1:y2, x1:x2]

    side = int(min(h, w) * 0.6)
    cy, cx = h // 2, w // 2
    y1 = cy - side // 2
    x1 = cx - side // 2
    return image[y1:y1 + side, x1:x1 + side]


def precache_crops(cfg):
    """Pre-cache face crops for all train + test images. Run once before training."""
    from tqdm import tqdm

    overlay = cfg.extra.get("overlay", {})
    cache_dir = overlay.get("crop_cache_dir", "data/processed/overlay_crops")
    os.makedirs(cache_dir, exist_ok=True)
    crop_margin = overlay.get("crop_margin", 0.75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = _get_mtcnn(device)

    all_paths: list[tuple[str, str, str]] = []
    for split in ("train", "public_test"):
        df = load_labels(cfg.data_dir, split)
        for row in df.itertuples(index=False):
            cache_path = os.path.join(cache_dir, f"{row.id}.png")
            if not os.path.exists(cache_path):
                all_paths.append((str(row.path), row.id, cache_path))

    if not all_paths:
        print(f"all crops already cached in {cache_dir}/")
        return

    print(f"caching {len(all_paths)} face crops (device={device})...")
    for img_path, image_id, cache_path in tqdm(all_paths, desc="crop"):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = _crop_face(image, mtcnn, crop_margin)
        cv2.imwrite(cache_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    print(f"done. {len(all_paths)} crops saved to {cache_dir}/")


class OverlayDataset(Dataset):
    """Face-crop dataset returning ``(image, label)`` for run_epoch / predict_scores."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        ids: set[str] | None = None,
        *,
        crop_margin: float = 0.75,
        cache_dir: str = "data/processed/overlay_crops",
    ) -> None:
        self.transform = transform
        self.crop_margin = crop_margin
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        df = load_labels(root, split)
        if ids is not None:
            df = df[df["id"].isin(ids)]
        self.samples: list[tuple[str, Path, int]] = [
            (row.id, Path(row.path), int(row.label))
            for row in df.itertuples(index=False)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _get_crop(self, img_path: Path, image_id: str) -> np.ndarray:
        cache_path = os.path.join(self.cache_dir, f"{image_id}.png")
        if os.path.exists(cache_path):
            img = cv2.imread(cache_path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = _crop_face(image, _get_mtcnn(), self.crop_margin)
        cv2.imwrite(cache_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        return crop

    def __getitem__(self, idx: int):
        sid, path, label = self.samples[idx]
        image = self._get_crop(path, sid)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label