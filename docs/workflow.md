# Working in this repo — end-to-end guide

How to go from a clone to a scored Kaggle submission, what belongs in each directory, and how
to add experiments without breaking reproducibility. Tailored to **FREUID 2026** (binary
identity-document fraud detection; **label 1 = fraud, 0 = bona-fide**; output a continuous
**fraud score = P(fraud)**; metric **AuDET**, lower is better). For the competition brief see
[competition.md](competition.md); for setup + data download see the [README](../README.md).

The repo currently ships one experiment, **`finetune_v0`**: a DINOv2 ViT-B/14 backbone,
**fully fine-tuned** (not frozen) at 518px, trained through the `model_type: baseline` code
path. Everything below uses it as the running example — copy `configs/finetune_v0.yaml` as
your starting point for a new run.

---

## 0. The mental model

One config drives one run. The pipeline is a straight line:

```
configs/<exp>.yaml ──▶ freuid.train ──▶ checkpoints/<exp>.pt ──▶ freuid.infer ──▶ submissions/<exp>.csv ──▶ Kaggle
                          │                                          │
                          └─ validates each epoch with the          └─ writes id,label (label = fraud score)
                             local AuDET / APCER@1%BPCER
```

The golden rule: **gate every idea on the local metric before spending a Kaggle submission.**
The public leaderboard is a small validation subset; the prize is decided on a private held-out
test set, so trust your local validation split and don't overfit the public LB.
`scripts/analysis/` holds the validation instruments that go beyond the per-epoch metric (e.g.
`probe_v2.py`, `nondigital_probe.py`, `hesitant_test_images.py`) — reach for these when the
per-epoch AuDET saturates and stops discriminating between candidates.

---

## 1. One-time setup

```bash
uv sync                 # runtime env (.venv) — see README "Setup"
uv sync --extra dev     # + ruff & pytest (for linting/tests)
uv sync --extra track   # + wandb (optional experiment tracking)
```

Then download data per the README's **"Getting the data"** section (Kaggle rules acceptance +
`kaggle auth login` + `bash scripts/download_data.sh`). Data lands in `data/` (gitignored).

Sanity-check the install:

```bash
uv run python -m freuid.metrics    # prints perfect/random metric values
uv run pytest -q                   # metric unit tests should pass
```

---

## 2. The core loop (commands)

### a) Smoke test first (seconds, not hours)

`limit` caps the train/val set so you can prove the wiring end-to-end before a real run:

```bash
# add `limit: 256` to a throwaway copy of configs/finetune_v0.yaml
uv run python -m freuid.train --config configs/finetune_v0.yaml
```

Watch for: it finds train/val counts, runs an epoch, prints `AuDET=…`, and writes
`checkpoints/<name>.pt`. If that works, the full run is just more data + more epochs.

### b) Train

```bash
uv run python -m freuid.train --config configs/finetune_v0.yaml
```

- Trains a binary classifier (`BCEWithLogitsLoss` + a pairwise soft-AUC term), validates **every
  epoch** with `evaluate()` → `AuDET` + `APCER@1%BPCER`, plus a recapture-probe `AuDET` when
  `extra.use_recapture_probe: true`.
- Checkpoints on the metric named by `extra.checkpoint_metric` (finetune_v0 uses
  `probe_audet` — the recapture probe, not raw val AuDET) to `checkpoints/<cfg.name>.pt` (state
  dict + full config + metrics).
- `extra.amp: true` enables mixed precision — required for a full ViT fine-tune to finish in a
  reasonable time; `extra.llrd` configures layer-wise LR decay + warmup for the backbone.
- Device auto-picks `cuda > mps > cpu`.

### c) Inference → submission csv

```bash
uv run python -m freuid.infer \
    --config configs/finetune_v0.yaml \
    --checkpoint checkpoints/finetune_v0.pt \
    --out submissions/finetune_v0.csv
```

- Scores every `public_test` id whose image is present locally; ids with no local image default
  to `extra.missing_id_score` (0.5 — rank-neutral, never 0.0). On Kaggle's grading run all
  images are present, so every id gets a real score.
- TTA runs at every scale in `extra.tta` (must be multiples of the backbone's patch size, e.g.
  `[476, 518, 560]` for a patch-14 ViT) and combines via **rank-averaging**, never raw-score
  averaging.
- Output columns: **`id,label`** where `label` is the fraud score in `[0,1]` (not a hard 0/1).
- Always run `check_submission()` (in `freuid.infer`) afterward — row count, unique-score
  count, exact-zero count, min/max.

### d) Submit to Kaggle

```bash
uv run kaggle competitions submit \
    -c the-freuid-challenge-2026-ijcai-ecai \
    -f submissions/finetune_v0.csv \
    -m "finetune_v0: DINOv2 ViT-B/14 fine-tuned @518px, val AuDET=0.xx"
```

Then log the result (next section) and compare the public LB to your local AuDET.

---

## 3. What goes where

| Directory | Commit? | What belongs here |
|-----------|---------|-------------------|
| `configs/` | ✅ yes | **One YAML per experiment.** This is the source of truth for a run — never hardcode hyperparams in code. Copy `finetune_v0.yaml`, rename, tweak, commit. |
| `src/freuid/` | ✅ yes | All reusable code (data, models, transforms, metrics, train/infer entrypoints). Changes here are reviewed via PR. |
| `submissions/` | ❌ gitignored | Generated `*.csv` (the `id,label` files). Name them after the config: `submissions/<exp>.csv`. Keep, but don't commit. |
| `checkpoints/` | ❌ gitignored | Model weights `*.pt`. Regenerable from config + data + seed, so not committed. |
| `data/` | ❌ gitignored | Proprietary FREUID data. **Never commit** (license + size). |
| `reports/` | ❌ gitignored | Internal analysis artifacts (`scripts/analysis/*.py` output) — regenerable, often derived from proprietary dataset images, so kept local-only. |
| `report/` | ❌ gitignored | The mandatory technical report — drafted locally, finalized/attached before the final submission; not committed during dev. |
| `notebooks/` | ✅ (cleared) | EDA / exploration. **Clear outputs before committing.** |
| `scripts/` | ✅ yes | Helper + analysis scripts (e.g. `download_data.sh`, `scripts/analysis/`). |
| `tests/` | ✅ yes | Unit tests (currently metric + split-logic sanity checks). Add tests when you add logic that can silently break. |

---

## 4. `src/freuid/` — what each module is for, and how to extend it

```
src/freuid/
├── config.py      # Config dataclass + load_config(yaml)
├── data.py        # FreuidDataset, load_labels, stratified_split / lodo_split (label×type)
├── transforms.py  # build_transforms(image_size, train), resolve_data_config
├── augment.py     # recapture_transforms (print-and-capture sim), synth_tamper
├── models/        # build_model(backbone, pretrained)  ← add new architectures here
├── metrics.py     # audet, apcer_at_bpcer, evaluate
├── train.py       # training entrypoint (python -m freuid.train)
├── infer.py       # inference → submission (python -m freuid.infer)
└── utils.py       # seed_everything, pick_device
```

### Add a new experiment (the 90% case — no code change)
1. `cp configs/finetune_v0.yaml configs/<name>.yaml`
2. Edit fields (`name`, `backbone`, `image_size`, `lr`, `epochs`, `extra.*`, …). **Set `name`
   uniquely** — it determines the checkpoint filename.
3. `uv run python -m freuid.train --config configs/<name>.yaml`
4. Verify the diff only touched what you intended:
   `uv run python scripts/config_diff.py configs/finetune_v0.yaml configs/<name>.yaml`
5. Commit the config. The config + `uv.lock` + seed make the run reproducible.

Base config knobs (defaults in [config.py](../src/freuid/config.py)): `seed`, `image_size`,
`val_fraction`, `val_doc_type` (Leave-One-Domain-Out holdout), `backbone`, `pretrained`,
`epochs`, `batch_size`, `lr`, `weight_decay`, `num_workers`, `limit` (cap data for smoke runs;
`null` = full). Unknown keys land in `extra`, so you can stash experiment-specific params there
without touching the dataclass — finetune_v0's are a good reference: `model_type` (dispatches
`train.py`/`infer.py`; defaults to `"baseline"`), `llrd.{enabled,decay,warmup_epochs}`, `amp`,
`train_last_k_blocks`, `grad_checkpointing`, `augment`, `synth_tamper_prob`,
`auc_loss_weight`, `use_recapture_probe`/`recapture_probe_seed`/`checkpoint_metric`, `tta`
(explicit scale list), `missing_id_score`, `val_doc_types`.

### Swap the backbone (usually no code change)
Any [timm](https://github.com/huggingface/pytorch-image-models) model name works in
`backbone:` — `build_model` calls `timm.create_model(backbone, num_classes=1,
dynamic_img_size=True)` (falls back without that kwarg for CNNs). **If it's a patch-based ViT**:
`image_size` and every `extra.tta` scale must be a multiple of the backbone's patch size (14 for
`vit_base_patch14_reg4_dinov2.lvd142m`). Always confirm with
`scripts/config_diff.py configs/finetune_v0.yaml configs/<name>.yaml` that only the keys you meant
to change actually changed.

### Add a genuinely new model
1. Create `src/freuid/models/<your_model>.py` exposing a builder that returns an `nn.Module`
   with a **single fraud logit** (so `BCEWithLogitsLoss` + `sigmoid` still apply).
2. Wire it into model selection (extend the `model_type` dispatch in `train.py` / `infer.py` /
   `models/__init__.py`). Keep the `(backbone, pretrained)` contract or generalize it consciously.

### Add an augmentation
Edit `build_transforms` in [transforms.py](../src/freuid/transforms.py) or
`recapture_transforms` / `synth_tamper` in [augment.py](../src/freuid/augment.py).
**FREUID-specific caution baked in:** no horizontal flip (documents have text/orientation);
augmentation stays mild and should model real **print-and-capture / physical** distortions
(JPEG recompression, slight rotation, color/lighting shifts) — not arbitrary noise that destroys
document semantics. Recapture augmentation and synthetic tampering must stay
**label-independent** — never let "looks recaptured" correlate with the fraud label.

### Touch the metric? Read the caveat first
`audet()` is a **linear-axis proxy (`1 − ROC AUC`)**, not necessarily byte-identical to the
official Kaggle scorer. Use it for **relative ranking** of candidates. If/when the official
scorer's exact definition (e.g. probit-axis DET integration) is confirmed, update
[metrics.py](../src/freuid/metrics.py) and re-rank. Keep `tests/test_metrics.py` green.

---

## 5. Submissions — format & rules (FREUID-specific)

- **File:** CSV with header `id,label`. `id` = the test image id (hex string). `label` = your
  **fraud score in `[0,1]`** (continuous — the DET metrics need a score, *not* a thresholded
  0/1). Higher = more likely fraud.
- **Coverage:** `sample_submission.csv` lists the **full** test set. Your csv must cover all of
  it. `freuid.infer` handles this: it starts from the full id list and fills any id whose image
  isn't present locally with `extra.missing_id_score` (0.5, never 0.0).
- **Scoring:** ranked by **AuDET (lower better)**, with **APCER@1%BPCER** as the operating-point
  metric. Public LB = validation subset; final = private test.
- **Naming:** save as `submissions/<config-name>.csv` so a submission traces back to one config.

---

## 6. Logging results

After each run worth remembering, add a row locally to `report/experiments.md` (gitignored —
this is your own running log, not a committed file) or to the results table in `CLAUDE.md`:

```
| 2026-06-20 | finetune_v0 | hyunsoo | DINOv2 ViT-B/14, fully fine-tuned, 518px | 0.000002 | 0.00744 | +LLRD, AMP, recapture aug |
```

Track **local val AuDET / probe AuDET** (and APCER@1%BPCER), not just the public LB — the
private LB is what pays.

---

## 7. Reproducibility checklist (required for prize eligibility)

The competition requires organizers be able to re-run your ranked result. Before a result
"counts" for the team, make sure:

- [ ] The run is driven by a **committed `configs/*.yaml`** (no uncommitted hyperparams).
- [ ] **`uv.lock` is committed** and unchanged (pinned env). Regenerate + commit if deps change.
- [ ] **Seed is set** (`seed_everything` runs in train/infer; config carries the seed).
- [ ] The exact **train + infer commands** are recorded (in your results log / `CLAUDE.md`).
- [ ] Code stays under the repo's **Apache-2.0** license; no proprietary/non-public data or
      models pulled in (external data/models must be public + license-compatible + cited).
- [ ] The **technical report** (finalized before the submission deadline) reflects the
      submitted result.

---

## 8. Git / team workflow

- `main` stays runnable. Branch per experiment/fix (e.g. `exp/<name>`, `fix/<name>`) → PR →
  review → merge.
- Keep PRs scoped: one experiment's config + any code it needs.
- Don't commit data, checkpoints, submissions, or secrets (`.gitignore` covers these).
- Lint before pushing: `uv run ruff check src tests` (and `uv run pytest -q`).
- Merge Kaggle teams before the **team-merge deadline** (submission quota becomes shared).
  Submission deadline per Kaggle: **2026-07-14** — verify the exact time/zone on the comp page.
