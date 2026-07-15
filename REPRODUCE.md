# Reproducing our FREUID Challenge 2026 submission

End-to-end reproduction of our ranked result: environment → data → training → inference →
the exact submission. A fully offline Docker path (the organizer sandbox contract) is provided.

- **Ranked model:** `synth_recapture_v1` **epoch 11** — DINOv2 ViT-B/14, fully fine-tuned,
  attention-pooling head, trained with `synth_tamper` synthetic-fraud + analog-recapture
  augmentation. Public **AuDET ≈ 0.026**.
- **Canonical config:** [`configs/synth_recapture_v1.yaml`](configs/synth_recapture_v1.yaml).
- **Method / ablations:** [`report/technical_report.pdf`](report/technical_report.pdf).
- **License:** Apache-2.0 ([`LICENSE`](LICENSE)).

---

## 0. Metric & output convention

- Output a continuous **fraud score** `P(fraud) ∈ [0,1]` per id (**never a hard label**);
  `1 = fraud`, higher = more fraudulent. Primary metric **AuDET** (`1 − ROC_AUC`), lower is better.
- Submission columns: **`id,label`** where `label` is the fraud score.

## 1. Environment

Python **3.10+**:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .     # torch>=2.2, timm>=1.0, numpy<2, pandas, pillow, scikit-learn, albumentations, pyyaml, kaggle
```
Training used a single **NVIDIA A100 (80 GB)** with AMP. Inference runs on GPU or CPU
(`freuid.utils.pick_device` auto-selects; checkpoint loads with `map_location="cpu"`).

## 2. Data

```
data/
  train/train/<id>.jpeg                 # 69,352 training images
  train_labels.csv                      # id,label
  public_test/public_test/<id>.jpeg     # local public test images
  sample_submission.csv                 # full test-id list
```
```bash
kaggle competitions download -c the-freuid-challenge-2026-ijcai-ecai -p data && (cd data && unzip -q '*.zip')
```

**External data / models:** none beyond the official FREUID training set and the
**DINOv2 ViT-B/14** self-supervised pretrained weights (`vit_base_patch14_dinov2.lvd142m`,
via `timm`, Apache-2.0), which are fine-tuned end-to-end into the shipped checkpoint. No
external labelled/fraud data, no presentation-attack or forgery-localization networks. See
the technical report §3 for details.

## 3. Train (reproduce the checkpoint)

```bash
python -m freuid.train --config configs/synth_recapture_v1.yaml
# -> checkpoints/synth_recapture_v1_ep11.pt  (+ per-epoch snapshots)
```
Recipe (all in the config): DINOv2 ViT-B/14 @518, `lr=5e-5`, cosine anneal over 20 epochs
(early-stop OFF), attention pooling (`pool=map`), pairwise soft-AUC loss (0.1), LLRD 0.7 +
2-epoch warmup, AMP, `synth_tamper` (prob 0.3) + analog recapture (`recapture_prob 0.5`),
`seed: 42`.

> **Reproducibility note.** GPU + AMP are not bit-deterministic and the augmentation is
> stochastic, so a re-run lands in the same **score band**, not bit-identical numbers. The
> **ranked checkpoint** (`synth_recapture_v1_ep11.pt`, published as a GitHub Release asset —
> see §6) reproduces the exact predictions; use it via the Docker below.

## 4. Inference (Kaggle-format CSV)

```bash
python -m freuid.infer \
    --checkpoint checkpoints/synth_recapture_v1_ep11.pt \
    --out submissions/synth_recapture_v1_ep11.csv
```
Backbone/image-size/TTA/normalization come from the checkpoint's stored config. TTA =
multi-scale `[476, 518, 560]`, rank-averaged (no flip). Prints an integrity report.

## 5. Submit to Kaggle

```bash
kaggle competitions submit -c the-freuid-challenge-2026-ijcai-ecai \
    -f submissions/synth_recapture_v1_ep11.csv -m "synth_recapture_v1 ep11"
```

## 6. Offline Docker (organizer sandbox contract)

Inference needs **no network** (the checkpoint carries all weights; backbone built with
`pretrained=False`).

```bash
# 1) fetch the ranked checkpoint (published as a GitHub Release asset)
curl -L -o docker/model/synth_recapture_v1_ep11.pt \
  https://github.com/hyunsoochung-portfolio/kaggle_freuid_2026/releases/download/freuid-submission/synth_recapture_v1_ep11.pt

# 2) build
docker build -t freuid-repro:local .

# 3) run offline; /data = flat dir of test images ({id}.jpeg), output -> /submissions/submission.csv
docker run --network none \
    -v /path/to/flat/test/images:/data:ro \
    -v "$(pwd)/out:/submissions" \
    freuid-repro:local
```
Output: exactly one `id,label` row per input image (id = filename stem), fraud score
higher = more fraud. Add `--gpus all` to use a GPU. See [`Dockerfile`](Dockerfile) and
[`docker/prepare_submission.py`](docker/prepare_submission.py).

## 7. Repo map

| Path | What |
|---|---|
| `configs/synth_recapture_v1.yaml` | the ranked-model config (recipe + augmentation) |
| `src/freuid/train.py` | training loop, loaders, LLRD, AUC loss, checkpointing |
| `src/freuid/augment.py` | `synth_tamper` synthetic-fraud + analog-recapture generators |
| `src/freuid/infer.py` | multi-scale-TTA inference → Kaggle CSV (+ integrity check) |
| `docker/prepare_submission.py` | offline sandbox entrypoint (flat dir → submission.csv) |
| `report/technical_report.pdf` | methodology, ablations, results |
| `Dockerfile`, `docker/` | offline reproduction image |
