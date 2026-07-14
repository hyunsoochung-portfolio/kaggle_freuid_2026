# Reproducing our FREUID Challenge 2026 submission

This document lets the organizers **reasonably reproduce** our ranked result end-to-end:
environment → data → training → inference → the exact submission CSV. A fully offline
Docker path is provided for the no-network sandbox.

- **Model:** DINOv2 ViT-B/14, fully fine-tuned, attention-pooling head, single fraud logit.
- **Key idea:** `synth_tamper` — a data-grounded synthetic-fraud augmentation that turns
  a fraction of bona-fide cards into realistic forgeries each epoch. See
  [the technical report](report/technical_report.md) for the method and ablations.
- **Canonical config:** [`configs/synth_tamper_v1.yaml`](configs/synth_tamper_v1.yaml)
  (backbone, image size, LR, schedule, TTA, and the augmentation are all pinned here).
- **License:** Apache-2.0 ([`LICENSE`](LICENSE)).

---

## 0. Metric & output convention

- Output a **continuous fraud score** `P(fraud) ∈ [0,1]` per id (**never a hard label**);
  `1 = fraud`, `0 = bona-fide`.
- Primary metric **AuDET** (area under the DET curve; repo proxy `1 − ROC_AUC`), **lower is better**.
- Submission CSV columns: **`id,label`** where `label` is the fraud score. Every id in
  `sample_submission.csv` (~142.8k) must be present; ids without a local image default to
  `0.5` (rank-neutral) — on the grading server all images are present.

## 1. Environment

Python **3.10+**. Install the package and its pinned dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # torch>=2.2, timm>=1.0, numpy<2, pandas, pillow, scikit-learn, albumentations, pyyaml, kaggle
```

A single **NVIDIA A100 (80 GB)** was used for training (AMP). Inference runs on GPU or CPU
(`freuid.utils.pick_device` auto-selects; the checkpoint loads with `map_location="cpu"`).

## 2. Data

Download the competition data from Kaggle and lay it out as (paths are rebuilt from `id`):

```
data/
  train/train/<id>.jpeg                 # 69,352 training images
  train_labels.csv                      # id,label
  public_test/public_test/<id>.jpeg     # 7,821 local test images
  sample_submission.csv                 # full test-id list (~142,818 ids)
```

```bash
kaggle competitions download -c the-freuid-challenge-2026-ijcai-ecai -p data && \
  (cd data && unzip -q '*.zip')
```

## 3. Train (reproduce the checkpoint)

```bash
python -m freuid.train --config configs/synth_tamper_v1.yaml
```

Produces `checkpoints/synth_tamper_v1_last.pt` (last epoch = 25) plus per-epoch snapshots
`checkpoints/synth_tamper_v1_ep{NN}.pt`. Recipe (all in the config): DINOv2 ViT-B/14 @518,
`lr=5e-5`, cosine anneal over **25 epochs with early-stop OFF** (the full anneal is what
makes the last epoch the best), attention pooling (`pool=map`), pairwise soft-AUC loss
(0.1), LLRD 0.7 + 2-epoch warmup, AMP, and `synth_tamper` (prob 0.3). `seed: 42`.

> **Reproducibility note.** GPU + AMP are not bit-deterministic, and the `synth_tamper`
> augmentation draws fresh random forgeries per run, so a re-run lands in the **same score
> band** rather than bit-identical numbers. The **ranked checkpoint we submitted is the
> shipped `synth_tamper_v1_last.pt`** — use it (Docker below) to reproduce the exact CSV.

## 4. Inference (reproduce the submission CSV)

```bash
python -m freuid.infer \
    --checkpoint checkpoints/synth_tamper_v1_last.pt \
    --out submissions/synth_tamper_v1_last.csv
```

Backbone/image-size/TTA/normalization are read from the checkpoint's stored config, so the
model and preprocessing are rebuilt exactly. TTA = multi-scale `[476, 518, 560]`,
rank-averaged (no horizontal flip — documents carry orientation). The run prints an
integrity report: `rows`, `unique_scores`, `exact_zeros` (must be 0), `min`/`max`.

## 5. Submit to Kaggle

```bash
kaggle competitions submit \
    -c the-freuid-challenge-2026-ijcai-ecai \
    -f submissions/synth_tamper_v1_last.csv \
    -m "synth_tamper_v1 last (full-anneal, TTA)"
```

## 6. Offline Docker (no-network sandbox)

Builds a self-contained image; inference needs **no network** (checkpoint holds all weights,
backbone built with `pretrained=False`).

```bash
# one-time: make the trained checkpoint visible to the build
cp checkpoints/synth_tamper_v1_last.pt docker/model/synth_tamper_v1_last.pt
docker build -t freuid-submission .

# run offline; mount the Kaggle test tree (read-only) + an output dir
docker run --rm --network none \
    -v /path/to/data:/data:ro -v /path/to/out:/out \
    freuid-submission
# -> /out/submission.csv    (add `--gpus all` to use a GPU)
```

See [`Dockerfile`](Dockerfile) and [`docker/run_inference.sh`](docker/run_inference.sh).

## 7. Repo map

| Path | What |
|---|---|
| `configs/synth_tamper_v1.yaml` | the canonical winning config (recipe + augmentation) |
| `src/freuid/train.py` | training loop, loaders, LLRD, AUC loss, checkpointing |
| `src/freuid/augment.py` | `synth_tamper` synthetic-fraud generator + wrapper |
| `src/freuid/infer.py` | multi-scale-TTA inference → submission CSV (+ integrity check) |
| `src/freuid/models/` | backbone + attention-pool head (`build_model`) |
| `report/technical_report.md` | methodology, ablations, results |
| `Dockerfile`, `docker/` | offline reproduction image |
