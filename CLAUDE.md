# CLAUDE.md

Context for building this project. Read before writing code.

## Project

FREUID Challenge 2026 — binary fraud detection on identity-document images. Output a continuous
fraud score `P(fraud) ∈ [0,1]` (never a hard label); `1 = fraud`, `0 = bona-fide`. Primary metric
**AuDET** (area under the DET curve; repo proxy `1 - roc_auc_score`, **lower is better**);
secondary **APCER @ 1% BPCER**. Both are rank metrics — only score _ordering_ matters, not
calibration.

The real test is hard on purpose: **print-and-capture ("analog hole") attacks** and **document
types not seen in training**, while the training data is ~99.97% digital. A previous
forensic-noise model scored ~0.0006 locally but ~0.377 public — it leaned on digital noise that
reprinting erases, and was validated on an easy in-domain split. We are rebuilding to be
analog-robust and to generalize across document types.

## The plan

Build incrementally — one change at a time, each version trustworthy and submittable before the
next — toward a frozen-foundation, consistency-based detector.

Start with the simplest thing that works: a single pretrained CNN on the full image producing a
fraud score, with the validation, augmentation, and submission machinery correct from the start.
Then strengthen it — a stronger backbone, a ranking-aware loss, synthetic analog tampering that
manufactures print-and-capture fraud positives from clean images, and test-time augmentation.

Then change the feature extractor: swap the CNN for a **frozen DINOv3 backbone** with a light head
on top, holding everything else fixed so the comparison is clean. Then add the analog-robust core
— a **patch self-consistency** head and a **face-region consistency** head on the frozen patch
features (which brings in document rectification and face detection as cached preprocessing),
fused into a single score.

Finally, push the rank metric: a second frozen backbone rank-averaged in, multi-seed/fold
averaging, and optional light fine-tuning of the backbone's last blocks. Semantic/OCR features
come afterward, only once the above is validated.

## Invariants (true throughout)

- `1 = fraud`, `0 = bona-fide` everywhere; **lower AuDET is better** — checkpoint on the lowest
  recapture-probe AuDET.
- **No horizontal flip** — documents carry orientation; augmentation must never flip.
- Rebuild image paths from `id` as `{split}/{split}/{id}.jpeg`; the CSV `image_path` column is
  unreliable.
- Inference defaults any genuinely-missing test id to **0.5** (rank-neutral), never 0.0; run a
  submission integrity check (unique-score count, exact-zeros, min/max).
- The DINOv3 backbone is **frozen** (`requires_grad=False`, `eval()`, autocast) until the final
  fine-tune step; only heads/fusion train before that.
- Detectors and the backbone run **once** and are cached; precache before training, single-process
  (do not fork the detectors).
- Checkpoints store the **full config** so inference rebuilds the exact model and preprocessing.
- Changes are **additive and gated on config flags**; never silently alter existing behavior.

## Models

Frozen backbone: **DINOv3 ViT-B/16** (fallback **DINOv2 ViT-B/14**, Apache-licensed, if DINOv3's
gated license is a problem). Preprocessing: **FastSAM** (card rectification) and **SCRFD**
(portrait box). Optional ensemble member: **DINOv3 ConvNeXt-Base**.

Restriction: **do not use any model built for ID-fraud / presentation-attack detection, or general
forgery-localization nets** (TruFor, CAT-Net, MVSS-Net, PSCC-Net, Noiseprint). Everything above is
a general vision / segmentation / face model.

## Validation (the part that was broken before)

Do not trust single-domain LODO on the easiest type — it saturates and predicts nothing. Per epoch
the **compass** is a **recapture probe**: apply the analog/print-and-capture augmentation to a
held-out clean split and measure AuDET on it; checkpoint on the lowest probe AuDET. Periodically
run **multi-fold LODO** (hold out each document type in turn, average) as the cross-domain check.
Sanity checks: init BCE ≈ 0.693 on a balanced batch; a single batch must overfit to ~0.

## Compute & environment

Single **A100**. Strategy: freeze the big model, train small heads, so the GPU mostly does one-time
cached forward passes. Use **AMP** (autocast + GradScaler). Don't cache full patch grids for the
whole dataset if storage is tight — recompute the frozen forward on the fly, or cache pooled
features keyed by `id`. On the workspace, work under `/root` (it persists across restarts): keep
the environment, dataset, feature cache, and checkpoints there.

### VESSL workspace (`freuid-hy`)

- **SSH alias:** `freuid-hy` — configured in `~/.ssh/config`, key at `~/.ssh/freuid.pem`
- **Repo:** `/root/repo` (cloned, on branch `rebuild/consistency-v1`)
- **Python:** `/opt/conda/bin/python3` (conda, not system Python)
- **Data:** `/root/repo/data/` — 69,352 train + 7,821 test images, gitignored, never touched by git
- **Checkpoints / submissions:** `/root/repo/checkpoints/`, `/root/repo/submissions/`
- **Training logs:** `/tmp/train_<config-name>.log`

Always start training with `nohup` so it persists after SSH disconnects:
```bash
ssh freuid-hy "cd /root/repo && nohup /opt/conda/bin/python3 -m freuid.train \
    --config configs/<name>.yaml > /tmp/train_<name>.log 2>&1 &"
```

Check progress (strips tqdm noise):
```bash
ssh freuid-hy "grep -av 'it/s\|it]' /tmp/train_<name>.log"
```

Pull code changes to workspace (git never touches `data/`):
```bash
ssh freuid-hy "cd /root/repo && git checkout -- notebooks/ && git pull origin <branch>"
```

Use `/vessl` skill for common workspace tasks.

## Conventions

Config via the existing `Config` dataclass; new knobs go through `cfg.extra` (`model_type`,
`backbone_name`, `image_size`, feature/head flags). Seed from `cfg.seed`. Loss:
`BCEWithLogitsLoss`, optionally a pairwise soft-AUC term. Log per epoch:
`lr, train_loss, val_loss, AuDET, APCER@1%BPCER, probe_AuDET`. Keep both the baseline CNN path and
the frozen-backbone path (`model_type: consistency`) working.
