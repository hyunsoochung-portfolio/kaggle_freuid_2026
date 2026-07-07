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
import torch
from PIL import Image
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


def unpack_batch(batch):
    """(imgs, labels) or (imgs, labels, face_meta) -> (imgs, labels, face_meta_or_None).

    Lets a single loop body (``run_epoch``, sanity checks, inference) handle loaders
    built with or without ``return_face_meta`` without branching on config everywhere.
    """
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    imgs, labels = batch
    return imgs, labels, None


def unpack_and_move(batch, device):
    """unpack_batch(batch) -> (imgs, labels, face_meta_or_None, face_crop_or_None), all
    moved to ``device``.

    ``unpack_batch``'s 3rd element is either a plain face_meta tensor (consistency path,
    unchanged) or a ``{"face_meta":..., "face_crop":...}`` dict (bayar_fusion path) --
    this is the one place both shapes get normalised so callers (train.py, infer.py)
    don't branch on it themselves.
    """
    imgs, labels, extra = unpack_batch(batch)
    imgs = imgs.to(device)
    if isinstance(extra, dict):
        return imgs, labels, extra["face_meta"].to(device), extra["face_crop"].to(device)
    face_meta_dev = extra.to(device) if extra is not None else None
    return imgs, labels, face_meta_dev, None


def forward_with_extras(model, imgs, face_meta=None, face_crop=None):
    """Dispatch a model call by which extra tensors are present: (imgs, face_crop,
    face_meta) for bayar_fusion, (imgs, face_meta) for consistency, (imgs) for baseline."""
    if face_crop is not None:
        return model(imgs, face_crop, face_meta)
    if face_meta is not None:
        return model(imgs, face_meta)
    return model(imgs)


def face_meta_tensor(sample: "Sample", img_size: tuple[int, int]) -> torch.Tensor:
    """Face-box fractions + validity flag for the FaceRegionHead: [x1,y1,x2,y2,valid].

    ``img_size`` is the (W, H) of the ORIGINAL image for this sample -- SCRFD runs on the
    original image, not the rectified card (see freuid.preprocess's module docstring: for
    the mostly-already-full-frame photos in this dataset, rectify_card's "largest quad"
    heuristic frequently warps onto a decorative sub-element -- a flag watermark, a barcode
    -- instead of the card, so face detection no longer depends on its output).
    Returns an all-zero (invalid) tensor when there's no cached box, or when the box
    is the center-square fallback (SCRFD ``score`` == 0, i.e. no real detection).
    """
    if sample.face_box is None:
        return torch.zeros(FACE_META_DIM, dtype=torch.float32)
    fb = sample.face_box
    if float(fb.get("score", 0.0)) <= 0.0:
        return torch.zeros(FACE_META_DIM, dtype=torch.float32)
    w, h = img_size
    return torch.tensor(
        [fb["x1"] / w, fb["y1"] / h, fb["x2"] / w, fb["y2"] / h, 1.0],
        dtype=torch.float32,
    )


def face_crop_image(sample: "Sample", crop_size: int, margin: float = 0.75) -> Image.Image:
    """Crop the cached face region (+ margin) from the ORIGINAL image, resized to a square
    ``crop_size`` x ``crop_size`` -- the input the bayar_fusion overlay branch expects.

    No new face detector is used: this crops from the SCRFD box already cached by
    ``freuid.preprocess.precache_regions`` (``face_box`` on ``Sample``, detected on the
    original image -- see freuid.preprocess's module docstring), the same source
    ``face_meta_tensor`` reads. Margin matches feat/overlay-detector's own ``crop_margin``
    convention (fraction of box width/height added on each side).

    Returns a black ``crop_size``x``crop_size`` image when there's no cached box for this
    sample -- paired with ``face_meta_tensor``'s ``valid=0`` in that same case, the fusion
    gate zeroes this branch's contribution regardless of the placeholder pixels, matching
    ``FaceRegionHead``'s "no signal instead of a wrong one" convention.
    """
    if sample.face_box is None:
        return Image.new("RGB", (crop_size, crop_size))
    img = Image.open(sample.path).convert("RGB")
    w, h = img.size
    fb = sample.face_box
    x1, y1, x2, y2 = fb["x1"], fb["y1"], fb["x2"], fb["y2"]
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (crop_size, crop_size))
    return img.crop((x1, y1, x2, y2)).resize((crop_size, crop_size), Image.BILINEAR)


class FreuidDataset(Dataset):
    """Image dataset yielding (transformed_image, label), or (transformed_image, label,
    face_meta) when ``return_face_meta=True`` (see ``face_meta_tensor``), or
    (transformed_image, label, {"face_meta":..., "face_crop":...}) when
    ``return_face_crop=True`` (see ``face_crop_image`` -- the bayar_fusion path).

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
        return_face_crop: bool = False,
        face_crop_size: int = 224,
        face_crop_margin: float = 0.75,
        face_crop_transform=None,
        use_rectified_as_main: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self._regions_dir = regions_dir
        self._return_face_meta = return_face_meta
        self._return_face_crop = return_face_crop
        self._face_crop_size = face_crop_size
        self._face_crop_margin = face_crop_margin
        self._face_crop_transform = face_crop_transform
        # Default True preserves every existing caller's behavior exactly (card_path
        # preferred as the main image whenever regions_dir is set, e.g. use_rectify=True
        # on the consistency path). Set False when regions_dir is only needed for face
        # crops (bayar_fusion) so the main image stays the raw original -- otherwise
        # enabling the regions cache would silently swap DINOv2's main input too.
        self._use_rectified_as_main = use_rectified_as_main
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
        src = s.path
        if self._use_rectified_as_main and s.card_path is not None:
            src = s.card_path
        img = Image.open(src).convert("RGB")

        # face_box's coordinates are always in the ORIGINAL image's space (SCRFD runs on
        # the raw image, not card.png -- see preprocess.py's module docstring) regardless
        # of which image is "main" here. When the main image is the rectified card instead
        # (use_rectified_as_main=True), re-derive the raw image's own size for the fractions.
        if src == s.path:
            face_img_size = img.size
        else:
            with Image.open(s.path) as _raw:
                face_img_size = _raw.size

        face_crop = None
        if self._return_face_crop:
            face_crop = face_crop_image(s, self._face_crop_size, self._face_crop_margin)
            if self._face_crop_transform is not None:
                face_crop = self._face_crop_transform(face_crop)

        if self.transform is not None:
            img = self.transform(img)

        if self._return_face_crop:
            return img, s.label, {"face_meta": face_meta_tensor(s, face_img_size), "face_crop": face_crop}
        if self._return_face_meta:
            return img, s.label, face_meta_tensor(s, face_img_size)
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