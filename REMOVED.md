# REMOVED — Overlay detector (TwoStreamOverlayNet / MTCNN)

Deleted in branch `rebuild/consistency-v1`. The consistency detector will replace this on the
same infrastructure; see the KEEP section for what is preserved verbatim.

---

## Deleted symbols and files

### `src/freuid/data.py`

| Symbol | Why overlay-specific |
|---|---|
| `_mtcnn_instance` | Global MTCNN singleton for face detection |
| `_get_mtcnn()` | Lazy-init the MTCNN singleton with configurable `min_face_size` |
| `_crop_face()` | Face-region crop (MTCNN detect → margin expand → fallback centre crop) |
| `resolve_cache_dir()` | Resolves `cfg.extra["overlay"]["crop_cache_dir"]` |
| `precache_crops()` | Bulk MTCNN detect + disk-cache for train/test splits |
| `OverlayDataset` | Dataset that serves cached face crops; returns albumentations-style `dict["image"]` |

Removed imports: `cv2`, `torch` (only used in `_get_mtcnn` for `torch.cuda.is_available()`),
`os` (only used in `OverlayDataset` and `precache_crops`).

### `src/freuid/models/overlay.py` — entire file deleted

| Symbol | Notes |
|---|---|
| `BayarConv2d` | Learnable constrained high-pass forensic filter |
| `SRMConv2d` | Fixed 3-kernel SRM filter bank |
| `NoiseStream` | Bayar or SRM frontend → small CNN → 128-d noise embedding |
| `TwoStreamOverlayNet` | Noise stream + optional RGB backbone + fusion head |
| `build_overlay_model()` | Factory called when `model_type == "overlay"` |

### `src/freuid/transforms.py`

| Symbol | Notes |
|---|---|
| `get_overlay_train_transforms()` | Albumentations pipeline for overlay train (no ImageNet norm) |
| `get_overlay_val_transforms()` | Albumentations pipeline for overlay val/inference |

Removed imports: `albumentations as A`, `from albumentations.pytorch import ToTensorV2`
(both were only used by the two overlay transform functions).

### `src/freuid/train.py`

| Item | Notes |
|---|---|
| `build_overlay_loaders()` | Standalone function constructing `OverlayDataset` loaders with `num_workers=0` |
| `is_overlay` flag in `main()` | Branched model construction and loader setup |
| Overlay branch in `build_loaders()` | `if model_type == "overlay": return build_overlay_loaders(cfg)` |
| `num_workers=0` hard-force | Required only for MTCNN fork-safety |
| Imports: `OverlayDataset`, `resolve_cache_dir`, `build_overlay_model`, `get_overlay_train_transforms`, `get_overlay_val_transforms` | |

### `src/freuid/infer.py`

| Item | Notes |
|---|---|
| Overlay branch in `main()` | `is_overlay` flag, `precache_crops` call, `OverlayDataset` loader, `num_workers=0` |
| Imports: `OverlayDataset`, `precache_crops`, `resolve_cache_dir`, `build_overlay_model`, `get_overlay_val_transforms` | |

### `src/freuid/models/__init__.py`

| Item | Notes |
|---|---|
| `build_overlay_model` export | Removed from `__all__` and import |

### Config files deleted

- `configs/overlay_detector.yaml`
- `configs/overlay_detector_colab.yaml`

### `pyproject.toml`

- Removed dependency: `facenet-pytorch>=2.5.3`

---

## Deliberately kept (unchanged)

| Item | Location | Notes |
|---|---|---|
| `FreuidDataset` | `data.py` | Core image dataset, PIL-based |
| `load_labels()` | `data.py` | CSV loader + path resolution |
| `stratified_split()` | `data.py` | Label×type stratified train/val split |
| `lodo_split()` | `data.py` | Leave-One-Domain-Out split |
| `Sample` dataclass | `data.py` | |
| `SPLITS` registry | `data.py` | `train / train_sample / public_test` → path mapping |
| `build_transforms()` | `transforms.py` | Baseline torchvision pipeline |
| `resolve_data_config()` | `transforms.py` | Per-backbone mean/std/size from timm |
| `build_model()` | `models/baseline.py` | timm backbone + 1 fraud logit |
| `run_epoch()` | `train.py` | Train/eval pass, loss, scores |
| `_split_ids()` | `train.py` | LODO vs stratified dispatch |
| `build_loaders()` | `train.py` | Baseline loader builder; `model_type` dispatch scaffold kept for future types |
| `train.py main()` | `train.py` | Epoch loop, checkpointing, cosine scheduler |
| `evaluate()` | `metrics.py` | AuDET + APCER@1%BPCER |
| `audet()` | `metrics.py` | 1 − ROC AUC proxy |
| `apcer_at_bpcer()` | `metrics.py` | |
| `infer.py` | `infer.py` | Full inference → submission CSV pipeline |
| `Config` dataclass + `load_config()` | `config.py` | |
| `cfg.extra` passthrough | `config.py` | Unknown YAML keys land here for new model types |
| `configs/baseline.yaml` | configs/ | EfficientNetV2-S 384px production baseline |
| `configs/smoke.yaml` | configs/ | 2-epoch, limit=100 smoke run |
| Checkpoint format | `train.py` / `infer.py` | `{model, config, epoch, metrics}` dict |
| Data layout handling | `data.py` | Double-nested Kaggle path reconstruction |
