# CLAUDE.md — FREUID Challenge 2026

Kaggle team workspace for **[The FREUID Challenge 2026](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai/overview)** (IJCAI-ECAI, Bremen, Aug 2026).
Binary fraud detection on identity documents. **Label 1 = fraud, 0 = bona-fide. Output = continuous fraud score P(fraud). Metric = AuDET, lower is better.**

Full competition brief: `docs/competition.md`. End-to-end workflow: `docs/workflow.md`.

---

## Repo map

```
configs/                   # One YAML per experiment — committed, never hardcode params in code
  baseline.yaml            # EfficientNetV2-S 384px, 20 epochs (production baseline)
  smoke.yaml               # 2 epochs, limit=100, data_smoke/ — quick wiring check
  overlay_detector.yaml    # Two-stream (Bayar noise + ResNet34 RGB) face-region model

src/freuid/
  config.py                # Config dataclass + load_config(yaml); unknown keys → cfg.extra
  data.py                  # FreuidDataset, OverlayDataset, load_labels, stratified_split
                           # Resolves the double-nested Kaggle layout (train/train/<id>.jpeg)
  transforms.py            # build_transforms, get_overlay_{train,val}_transforms, resolve_data_config
  models/
    baseline.py            # build_model(backbone, pretrained) — timm + 1 fraud logit
    overlay.py             # TwoStreamOverlayNet: BayarConv2d / SRMConv2d noise stream + RGB backbone
    __init__.py            # exports build_model, build_overlay_model
  metrics.py               # audet(), apcer_at_bpcer(), evaluate() — local offline ranking
  train.py                 # Entrypoint: python -m freuid.train --config <yaml>
  infer.py                 # Entrypoint: python -m freuid.infer --checkpoint <pt> --out <csv>
  utils.py                 # seed_everything(), pick_device()

data/                      # GITIGNORED — proprietary FREUID data (never commit)
  train_labels.csv         # columns: id, image_path, label, is_digital, type
  train/train/<id>.jpeg
  public_test/public_test/<id>.jpeg
  sample_submission.csv    # all test ids; your label column = fraud score [0,1]
  train_sample/            # tiny labelled sample for smoke tests

data/processed/
  overlay_crops/           # face-region crops cached by OverlayDataset / precache_crops()

checkpoints/               # GITIGNORED — model weights *.pt (state_dict + config + metrics)
submissions/               # GITIGNORED — generated id,label CSVs

notebooks/                 # EDA / exploration — clear outputs before committing
report/                    # Mandatory technical report + experiments.md run log
scripts/
  download_data.sh         # Kaggle CLI download → data/
  pull_submissions.sh      # Helper to pull submission files

tests/
  test_metrics.py          # Sanity checks for audet / apcer_at_bpcer / evaluate
```

---

## Commands

```bash
# Setup
uv sync                    # runtime env
uv sync --extra dev        # + ruff + pytest
uv sync --extra track      # + wandb (optional)

# Sanity checks
uv run python -m freuid.metrics    # prints perfect / random metric values
uv run pytest -q                   # metric unit tests

# Data
bash scripts/download_data.sh      # download competition archive into data/

# Train (standard timm backbone)
uv run python -m freuid.train --config configs/baseline.yaml

# Train (overlay / two-stream model)
uv run python -m freuid.train --config configs/overlay_detector.yaml

# Inference → submission csv
uv run python -m freuid.infer \
    --checkpoint checkpoints/baseline.pt \
    --out submissions/baseline.csv
# backbone/image_size always come from the checkpoint; --config is optional

# Submit to Kaggle
uv run kaggle competitions submit \
    -c the-freuid-challenge-2026-ijcai-ecai \
    -f submissions/baseline.csv \
    -m "baseline effnetv2_s, val AuDET=0.xx"

# Lint
uv run ruff check src tests
```

---

## Key design decisions

- **One config drives one run.** `configs/<exp>.yaml` → `train` → `checkpoints/<name>.pt` → `infer` → `submissions/<name>.csv`. Commit the config; checkpoint and submission are gitignored.
- **backbone/image_size come from the checkpoint at inference**, not the config, so weights always match their preprocessing.
- **`model_type: overlay` in `cfg.extra`** switches train.py and infer.py to the two-stream path (`OverlayDataset`, `build_overlay_model`). The standard path uses `FreuidDataset` and `build_model`.
- **No horizontal flip** in augmentation — documents have text and orientation, and flipping destroys document semantics.
- **MTCNN is a singleton** (`_mtcnn_instance` in data.py) and is not fork-safe — `num_workers: 0` is required for the overlay model.
- **AuDET proxy**: `metrics.py` uses `1 − ROC AUC` as a linear-axis proxy, not byte-identical to the official Kaggle DET scorer. Use it for relative ranking; reconcile with the official scorer before trusting absolute values.
- **Stratified split** in `stratified_split()` stratifies on `(label, type)` to keep all 7 document domains represented in validation. This is the default (in-domain) signal.
- **Leave-One-Domain-Out (opt-in)**: set `val_doc_type:` in a config (e.g. `MAURITIUS/ID`) to hold out one whole document `type` for validation via `lodo_split()`. Train and val then share no domain, so val AuDET measures cross-domain transfer — a more honest proxy for the unseen-domain private test. Leave `val_doc_type` unset to fall back to the stratified split.

## Adding experiments

**New config only (90% of cases):** copy `configs/baseline.yaml`, rename, change `name:` (sets the checkpoint filename), tweak hyperparams, commit. Any [timm](https://github.com/huggingface/pytorch-image-models) model name works in `backbone:`.

**New model:** add `src/freuid/models/<your_model>.py` returning an `nn.Module` with a single fraud logit, then extend `build_model` dispatch in `models/__init__.py`.

**Log every run worth keeping** in `report/experiments.md` with local val AuDET + APCER@1%BPCER.

## Branch / PR conventions

`main` stays runnable. Branch `feat/<name>` or `fix/<name>` → PR → review → merge. Keep PRs scoped to one experiment's config + any code it needs.

**Submission deadline: 2026-07-14** (verify exact time on the Kaggle page).
