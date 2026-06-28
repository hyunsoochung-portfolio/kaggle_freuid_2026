# Pipeline Context — FREUID 2026

Quick reference for understanding this codebase. Everything here reflects current code state.

---

## Competition in one paragraph

Binary fraud detection on identity-document images. **Label 1 = fraud/attack, 0 = bona-fide.** Model must output a continuous fraud score P(fraud) ∈ [0,1] — never a hard label. Primary metric: **AuDET = area under the DET curve = 1 − ROC AUC. Lower is better.** Secondary: **APCER @ 1% BPCER** (attack pass-rate when genuine-rejection budget is 1%). Both are threshold-sweeping rank metrics; only the ordering of scores matters, not their absolute calibration. Submission format: `id,label` CSV where `label` is the fraud score.

---

## Data layout

```
data/raw/
  train_labels.csv                     # id, image_path, label, is_digital, type
  train/train/<id>.jpeg                # 69,352 training images
  public_test/public_test/<id>.jpeg    # 7,821 local test images (142,818 total ids in sample_submission)
  sample_submission.csv                # id,label — full test id list; label is a placeholder
```

Key facts:
- **5 document types in train**: `EGYPT/DL`, `GUINEA/DL`, `BENIN/DL`, `MOZAMBIQUE/DL`, `MAURITIUS/ID` (~13–16k images each)
- Class balance: **0 → 40,005 / 1 → 29,347** (≈58/42, mildly imbalanced)
- **`is_digital`**: 69,332 True / 20 False — training is **99.97% digital**; print-and-capture is essentially absent, which is the main generalization risk
- `image_path` column in the CSV is unreliable due to double-nesting; paths are rebuilt from `id` as `{split}/{split}/{id}.jpeg`
- `sample_submission.csv` has 142,818 ids; only 7,821 images exist locally — the rest are private/hidden and scored on Kaggle's infrastructure

---

## Two model paths

The config field `model_type: overlay` (under `extra`) selects the overlay path; omitting it uses the baseline.

### Path 1 — Baseline (full-image backbone)

```
FreuidDataset  →  build_transforms(image_size, train)  →  build_model(backbone)
```

- **Dataset**: `FreuidDataset` (`src/freuid/data.py`) — opens each JPEG with PIL, applies torchvision transforms, returns `(image_tensor, label)`
- **Transforms** (`build_transforms`): `Resize → [ColorJitter + RandomRotation(5)] → ToTensor → Normalize(backbone mean/std)`. **No horizontal flip** — document text/orientation is semantically meaningful; flipping produces invalid inputs
- **Normalization**: `resolve_data_config(backbone)` reads mean/std from timm's pretrained config so swapping backbones keeps preprocessing correct
- **Model**: `timm.create_model(backbone, num_classes=1)` — single fraud logit head; any timm name works in config
- **Config example**: `backbone: tf_efficientnetv2_s.in21k`, `image_size: 384`, `epochs: 20`, `batch_size: 32`

### Path 2 — Overlay detector (face-crop + two-stream)

```
OverlayDataset (MTCNN face crop → cache)  →  get_overlay_{train,val}_transforms  →  TwoStreamOverlayNet
```

- **Dataset**: `OverlayDataset` — locates the face region with MTCNN, caches the crop as PNG, returns `(crop_tensor, label)`. Cache is at `data/raw/processed/overlay_crops/` by default
- **Face detection**: `_crop_face()` downscales to `detect_long_side=1024px` before MTCNN runs (fast on large IDs), maps bbox back to full resolution for the actual crop, adds `crop_margin=0.75` padding. Falls back to a centered 60%-of-min-side square if no face is detected
- **MTCNN**: singleton `_mtcnn_instance` — not fork-safe; `num_workers` must be **0** for this path
- **Pre-caching**: `precache_crops(cfg, splits=("train",))` runs MTCNN over all images once before training. Called at the top of `train.py` for the overlay path
- **Transforms** (`get_overlay_{train,val}_transforms`): `Resize → [HorizontalFlip + ColorJitter (train only)] → ToFloat(max_value=255) → ToTensorV2`. **No ImageNet normalization** — the noise stream needs raw [0,1] pixels; the RGB sub-stream normalizes internally in forward()
- **⚠️ Known issue**: `HorizontalFlip(p=0.5)` is still present in overlay train transforms despite the no-flip policy for documents. Should be removed
- **Model** (`TwoStreamOverlayNet`):
  - **Noise stream**: `BayarConv2d` (learnable constrained high-pass) or `SRMConv2d` (fixed forensic filter bank) → 3-block CNN → `AdaptiveAvgPool` → 128-d feature
  - **RGB stream** (optional, `use_rgb_stream: true`): timm backbone (e.g. ResNet-34), ImageNet-normalized inside `forward()`
  - **Head**: `Linear(total_feat, fusion_dim) → ReLU → Dropout(0.3) → Linear(1)`
- **Config example**: `model_type: overlay`, `image_size: 224`, `num_workers: 0`, `noise_frontend: bayar`, `use_rgb_stream: true`, `rgb_backbone: resnet34`

---

## Validation split strategies

Both return `(train_ids, val_ids)` as sets of id strings. Selected by `_split_ids(cfg)` in `train.py`.

### Stratified split (default when `val_doc_type` is not set)

`stratified_split(root, val_fraction=0.1, seed=42)`

Groups by `(label, type)`, samples `val_fraction` from each stratum (minimum 1). All 5 document types and both classes appear in both train and val. Good for measuring in-domain performance but optimistic about generalization.

### Leave-One-Domain-Out / LODO (when `val_doc_type` is set in config)

`lodo_split(root, val_doc_type)`

Holds out **one entire document type** for validation; train sees the remaining types only. Train and val share no document domain → val AuDET measures cross-domain transfer, which is a more honest proxy for the private test. Validates that the held-out type has both classes before proceeding (prevents `roc_auc_score` crash).

**Default held-out type**: `MAURITIUS/ID` (the only `/ID` type; the others are `/DL`).

**Observed issue**: When `val_doc_type: MAURITIUS/ID`, the overlay model achieves near-zero AuDET (~0.0006) from epoch 1 and never improves meaningfully. This means MAURITIUS/ID may be too easy a target for the overlay detector. The near-zero val AuDET did **not** predict the public leaderboard score — significant divergence was observed, confirming the generalization gap.

---

## Training loop (`src/freuid/train.py`)

```python
run_epoch(model, loader, device, criterion, optimizer=None)
```

- With `optimizer` → trains (grad enabled, returns `mean_loss, None, None`)
- Without `optimizer` → evaluates (no grad, collects scores/labels, returns `mean_loss, scores, labels`)
- Loss: `BCEWithLogitsLoss` (no `pos_weight` despite mild imbalance)
- Optimizer: `AdamW(lr, weight_decay)`
- Scheduler: `CosineAnnealingLR(T_max=epochs)` — decays to ~0 by final epoch
- Checkpoint: saved on best val AuDET to `checkpoints/<cfg.name>.pt` as `{"model": state_dict, "config": vars(cfg), "epoch": epoch, "metrics": m}`
- Logs per epoch: `lr`, `train_loss`, `val_loss`, `AuDET`, `APCER@1%BPCER`

**No sanity checks** (init-loss check and single-batch overfit are not currently wired in).

---

## Inference (`src/freuid/infer.py`)

- Reads test ids from `load_labels(data_dir, "public_test")` — starts from the full id list in `sample_submission.csv`
- Checks which images exist locally with `Path(p).exists()`
- Scores present images; defaults missing ids to `0.5` (on Kaggle's run all images are present)
- `backbone` and `image_size` always come from the **checkpoint's stored config**, not the inference config, so preprocessing always matches training weights
- Writes `id,label` CSV (correct column name for Kaggle)

---

## Metrics (`src/freuid/metrics.py`)

```python
evaluate(scores, labels)  # → {"audet": float, "apcer_at_1pct_bpcer": float}
audet(scores, labels)     # = 1 - roc_auc_score(labels, scores)
apcer_at_bpcer(scores, labels, bpcer_target=0.01)
```

`audet()` uses `1 − ROC AUC` as a **linear-axis proxy** — not guaranteed byte-identical to the official Kaggle DET scorer (which may use a probit/normal-deviate axis). Use for relative ranking; reconcile with the official scorer before trusting absolute values.

---

## Config dataclass (`src/freuid/config.py`)

Known fields (all optional with defaults):

| Field | Default | Notes |
|---|---|---|
| `name` | `"baseline"` | Sets the checkpoint filename |
| `seed` | `42` | |
| `data_dir` | `"data"` | Root of the data directory |
| `image_size` | `None` | `None` → backbone's native resolution |
| `val_fraction` | `0.1` | Used by stratified split only |
| `val_doc_type` | `None` | Set to enable LODO (e.g. `"MAURITIUS/ID"`) |
| `backbone` | `"tf_efficientnetv2_s.in21k"` | Any timm name |
| `pretrained` | `True` | |
| `epochs` | `20` | |
| `batch_size` | `32` | |
| `lr` | `3e-4` | |
| `weight_decay` | `1e-4` | |
| `num_workers` | `8` | Must be `0` for overlay path |
| `limit` | `None` | Cap train/val sizes for smoke runs |

Any unknown key lands in `cfg.extra` — used for overlay-specific knobs (`model_type`, `model`, `overlay` sub-dicts).

---

## Key invariants

- **Label 1 = fraud, 0 = bona-fide** everywhere (dataset, loss target, metric)
- **No horizontal flip** for the baseline — overlay transforms still have it (known issue)
- **`num_workers: 0`** required for overlay (MTCNN singleton is not fork-safe)
- **`pin_memory`** is gated on `torch.cuda.is_available()` — disabled on MPS/CPU
- Checkpoint stores the full config so `infer.py` can recover backbone/image_size without a separate config file

---

## Known issues / open risks

| Issue | Impact |
|---|---|
| HorizontalFlip in overlay transforms | Feeds semantically invalid (mirrored text) inputs to the noise stream |
| Val AuDET near-zero on MAURITIUS/ID from epoch 1 | Makes checkpoint selection by AuDET effectively random; val_loss is more informative in this regime |
| Public LB diverges significantly from local val AuDET | Model generalizes poorly to unseen document types; LODO on 5 known types is an insufficient generalization signal |
| 99.97% digital training data | Print-and-capture attacks in the test set are essentially OOD |
| No AMP | Training is slower than necessary; ~1.5–2× throughput available with autocast + GradScaler |
| No sanity checks | Init-loss and batch-overfit checks are not wired into the current training loop |
