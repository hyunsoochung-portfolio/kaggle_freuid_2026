# CLAUDE.md

Context for building this project. Read before writing code.

## Project

FREUID Challenge 2026 — binary fraud detection on identity-document images. Output a continuous
fraud score `P(fraud) ∈ [0,1]` (never a hard label); `1 = fraud`, `0 = bona-fide`. Primary metric
**AuDET** (area under the DET curve; repo proxy `1 - roc_auc_score`, **lower is better**);
secondary **APCER @ 1% BPCER**. Both are rank metrics — only score _ordering_ matters, not
calibration.

The real test is hard on purpose: **print-and-capture ("analog hole") attacks**, **GenAI
multimodal edits**, and **document types not seen in training**, while the training data is
~99.97% digital. A previous forensic-noise model scored ~0.0006 locally but ~0.377 public — it
leaned on digital noise that reprinting erases, and was validated on an easy in-domain split.
Everything here is built to avoid repeating that failure.

## Baseline: finetune_v0 (read this before proposing architecture)

The model this repo is built around is **`finetune_v0`**: a **DINOv2 ViT-B/14** backbone
(`vit_base_patch14_reg4_dinov2.lvd142m` via timm, 4 register tokens, Apache-2.0, no HF gating),
**fully fine-tuned end-to-end** (not frozen) at 518px, through the `model_type: baseline` code
path in `train.py` / `infer.py`. Recipe: LLRD (layer-wise LR decay, `decay: 0.7`, 2-epoch
warmup), AMP, `BCEWithLogitsLoss` + a pairwise soft-AUC term, print-and-capture recapture
augmentation + synthetic tamper injection (30% of bona-fide per batch), checkpointed on the
lowest recapture-probe AuDET. Result: **probe_AuDET 0.000002, public LB 0.00744** (rank 33/212
at submission time) — the strongest result so far, and the config every later experiment is
compared against.

Why fine-tune the whole backbone instead of freezing it and training a light head on top:
freezing throws away every layer's ability to adapt to what actually survives print-and-capture
degradation. This shows up directly in `finetune_v0`'s own training curve — epoch 1 (backbone
still ~unmoved during warmup) sits far above the final probe_AuDET; the value collapses only
once backbone weights start adapting. Freezing is the thing to avoid here, not the backbone
choice itself.

**Follow-up experiments off finetune_v0 — several tried since, all negative. Read before
repeating any of these:**

| attempt | change from finetune_v0 | probe_AuDET | public LB | verdict |
| --- | --- | --- | --- | --- |
| `finetune_v1` | DINOv3 ViT-B/16 backbone swap, otherwise identical recipe | — | 0.02695 | ~3.6x worse |
| `finetune_v2` | 784px instead of 518px, otherwise byte-identical config | 0.000000 (ep18) | 0.00750 | wash (noise-level) |
| `bayar_dinov2_v0` | gated fusion with a BayarConv2d forensic-noise + face-crop branch, fine-tuned jointly | 0.0 (exact) | 0.02146 | ~2.9x worse |
| overlay ensemble — hard gate | override finetune_v0 with a standalone forensic-noise model's raw score on ~305 uncertain ids | n/a | 0.01101 | ~1.5x worse |
| overlay ensemble — weighted blend, hesitant subset | weighted rank-blend restricted to the ~300 most uncertain ids (had a local/global rank-scale bug) | n/a | 0.01659 | ~2.2x worse |
| overlay ensemble — weighted blend, all ids | weighted rank-blend (0.8/0.2) across all present ids | n/a | 0.05319 | ~7.1x worse |

Details worth keeping:

- **`finetune_v1` (DINOv3 swap)**: the backbone swap alone made things ~3.6x worse.
  Deprioritize DINOv3 for the primary path. The parked consistency path's DINOv3 loader
  (`src/freuid/backbone.py`, `transformers`-based, unused by `finetune_v0`) has an
  **unverified register-token slicing** risk that this regression now corroborates in
  practice, not just in theory. Don't retry a DINOv3 backbone swap without a specific new
  hypothesis for why it would help.
- **`finetune_v2` (784px)**: the resolution increase was a wash, not a wasted-effort upsampling
  artifact — the native data ceiling is min-side ≈1000px, so 784px is genuine signal, and the
  training augmentation's resize/downscale steps are relative rather than an absolute
  bottleneck (regenerate the underlying census/audit via
  `scripts/analysis/resolution_census.py` and `aug_resolution_audit.py` to re-confirm).
  **Conclusion: resolution is not the lever that matters here** — capacity, ensembling, and
  loss-shaping toward the rank metric are more promising than further resolution tuning. Don't
  retry 1036px: 0% of images natively reach that size (pure upsampling) and the compute cost is
  much worse.
- **`bayar_dinov2_v0`** (`src/freuid/models/bayar_fusion.py`'s `BayarFusionNet`): DINOv2's CLS
  embedding gated-fused (LayerScale-style, near-zero-init gate) with a BayarConv2d-noise +
  RGB-ResNet34 branch on a cached-SCRFD face crop, fully fine-tuned end-to-end with recapture
  augmentation applied to both streams. The local probe hit an exact 0.0, and the checkpoint
  genuinely learned to rely on the branch (100% of the gate's 640 elements moved >10x from
  init) — that reliance is *why* it overfit, not evidence the branch stayed inert.
  Most-to-least confident root causes: (1) the extra branch is pure added capacity to fit the
  closed train/val/probe loop without that fit needing to generalize; (2) the recapture
  augmentation applied to the face-crop stream (JPEG recompress/blur/noise/downscale, built to
  simulate print-and-capture) is exactly the kind of degradation that erases the fine noise
  residue BayarConv2d depends on; (3) SCRFD only finds a valid face on ~1 in 5 documents, and
  the rate differs by split (train 21.5% vs public_test 16.4%) — a real, measured, if
  secondary, train/test mismatch. **Do not retry this fusion family** (BayarConv2d/SRM-style
  noise-residual branches jointly trained under recapture augmentation) without a specific new
  idea for resolving the augmentation-vs-noise-signal conflict.
- **Overlay ensembles**: a separate forensic-noise + face model (BayarConv2d stream, ResNet34
  RGB stream, MLP fusion, 224px MTCNN face crop, **zero recapture/reprint augmentation** in its
  own training pipeline) scores decisively on finetune_v0's most-uncertain present-test
  predictions (91-95% confident calls, holding flat through rank 300) — mechanistically
  sensible, since its forensic-noise signal was never trained to survive a reprint but still
  fires on genuinely-digital (not-yet-reprinted) manipulations. But every fixed-weight or
  fixed-override combination tried made the public LB worse: rank metrics punish a
  confidently-wrong call (pushed to 0.99+) far more than they punish an ambiguous 0.5, so
  *decisiveness* on an uncertain zone doesn't imply *correctness*, and none of these
  combinations could tell which of the second model's calls to trust.

Two different second-branch/second-model ideas (the frozen consistency heads described in
Models below, and this fusion/ensemble line) have now each looked promising in some narrow
sense and failed to transfer to the actual metric, for different reasons each time. Treat this
as a structural mismatch between forensic-noise/consistency features and this project's
reprint-robustness training regime, not a bug to patch.

## Invariants (true throughout)

- `1 = fraud`, `0 = bona-fide` everywhere; **lower AuDET is better** — checkpoint on the lowest
  recapture-probe AuDET (ties broken by probe APCER@1%BPCER — implemented in `train.py`).
- **No horizontal flip** — documents carry orientation; augmentation must never flip.
- Recapture augmentation and synthetic tampering are **label-independent by construction** —
  never let "looks recaptured" correlate with the fraud label (test-set bona-fide are themselves
  printed-and-captured cards).
- Rebuild image paths from `id` as `{split}/{split}/{id}.jpeg`; the CSV `image_path` column is
  unreliable.
- Inference defaults any genuinely-missing test id to **0.5** (rank-neutral), never 0.0; run
  `check_submission()` after every inference (row count, unique-score count, exact-zeros,
  min/max). Write scores at full float precision — truncation creates ties that hurt a rank metric.
- ViT inputs and every TTA scale must be **multiples of the patch size** (14 for DINOv2;
  finetune_v0 uses 518 with TTA `[476, 518, 560]`). `build_model` passes
  `dynamic_img_size=True` to timm (with a `TypeError` fallback for CNNs).
- TTA and ensembling combine by **rank-averaging**, never raw-score averaging.
- Checkpoints store the **full config** so inference rebuilds the exact model and preprocessing.
- Changes are **additive and gated on config flags** (`extra.*`); never silently alter existing
  behavior. Verify recipe changes with `scripts/config_diff.py` (compares fully-resolved configs).
- The old `--sanity` harness (hardcoded SGD lr=0.1) **diverges on full ViTs** — it is only valid
  for the CNN path. For ViTs, sanity = init BCE ≈ 0.693 on a balanced batch + single-batch
  overfit to ~0 under **AdamW** (lr≈1e-3).

## Models

Primary: **`vit_base_patch14_reg4_dinov2.lvd142m` via timm** (DINOv2 ViT-B/14, 4 register
tokens, Apache-2.0, no HF gating), **fully fine-tuned** through the `model_type: baseline`
path — this is `finetune_v0`, described above. Fine-tune recipe (all gated in `cfg.extra`):
LLRD `{enabled, decay: 0.7, warmup_epochs: 2}` (head LR 1e-4, per-block decay, LayerNorm/bias
excluded from weight decay — `src/freuid/optim.py`), `amp: true`, `train_last_k_blocks` /
`grad_checkpointing` as memory fallbacks (unused so far).

Scale-up candidate: `vit_large_patch14_reg4_dinov2.lvd142m`, same recipe. Feasibility confirmed
on the A100: L@518 fits at bs=32 with no checkpointing, ~7.7h/20ep; L@784 needs bs=16 (OOMs at
32), ~21.6h/20ep — both fit a ≤40h budget, though resolution is deprioritized (see follow-up
experiments above), so L@518 is the version worth trying first. Not yet trained/submitted.

Diversity member: ConvNeXt (train at 518px + warmup before ensembling — an untrained ConvNeXt
checkpoint is not ensemble-grade next to finetune_v0's 0.0074). Cheapest diversity available is
multi-seed rank-averaging of finetune_v0's own recipe (no new architecture needed) — also not
yet tried.

**DINOv3: deprioritized.** See `finetune_v1` above — a real regression, not a theoretical risk.
Don't retry a DINOv3 backbone swap without a specific hypothesis for why it would help this
time. If you do: prefer timm tags over `src/freuid/backbone.py`'s `transformers`-based loader,
which has **unverified register-token slicing** (audit with a patch-token PCA visualization
before trusting spatial features) and pulls in DINOv3 weights under Meta's custom license —
check compatibility with an Apache-2.0 release before using in a ranked submission.

A parked `model_type: consistency` path also exists in the codebase (`src/freuid/models/
consistency.py`, `consistency_model.py`): frozen backbone + patch/face self-consistency heads.
It underperformed when frozen (the confound of frozen-vs-fine-tuned was never cleanly resolved
for this specific head design) and was never re-tested on a fine-tuned backbone. Keep it
importable/runnable, but no new work goes there without a specific decision to revisit it.

A `model_type: bayar_fusion` path also exists (`src/freuid/models/bayar_fusion.py`) — this is
`bayar_dinov2_v0` from above. Keep it importable/runnable; **do not retrain it** without a new
idea for the augmentation-vs-noise-signal conflict described above.

Preprocessing (consistency path only, currently parked): FastSAM (card rectification), SCRFD
(portrait box) — cached once, single-process, never inside a forked DataLoader.

Restriction: **do not use any model built for ID-fraud / presentation-attack detection, or
general forgery-localization nets** (TruFor, CAT-Net, MVSS-Net, PSCC-Net, Noiseprint).
Everything above is a general vision / segmentation / face model.

## Validation (the part that keeps biting us)

Per-epoch compass is the **recapture probe** (analog augmentation applied to a held-out clean
split; checkpoint on lowest probe AuDET). **Known limits, respect them**: the probe applies the
_training_ augmentation, so it is partially circular, and it is **saturated** — finetune_v0 hit
2e-6, finetune_v2's best epoch hit exact `0.0`, and `bayar_dinov2_v0` also hit exact `0.0` right
before regressing ~2.9x on the public LB. That last case matters: the probe isn't just
low-resolution between two good candidates (its known limit) — it can be **completely blind to
a real regression**. It has agreed with the LB in *direction* on every submission so far, so
keep it as a regression gate, but don't trust it to rank two good candidates against each other.
Don't trust single-domain LODO on the easiest type either — it saturates the same way. Periodic
multi-fold LODO (hold out each document type in turn) is the cross-domain check. Sanity checks
per the Invariants section.

**Closing this gap is the highest-priority validation task.** Three instruments exist so far,
none sufficient alone:

- `scripts/analysis/nondigital_probe.py` scores checkpoints against the only 20 *real*
  non-digital (print-and-capture) training images. Caveat: 18/20 sit inside every checkpoint's
  training split (same seed=42 split across configs) — only 2 are genuine holdouts, both
  fraud, both correctly and confidently flagged (0.978–1.000) by every checkpoint tried so far.
  Encouraging, but n=2 is not a real signal, and there are zero held-out non-digital bona-fide
  examples to check false-positive behavior at all.
- `scripts/analysis/probe_v2.py` (+ `src/freuid/probe_v2_augment.py`) degrades the held-out val
  split through mechanisms `recapture_transforms` does **not** model at all — halftone/moiré,
  vignetting, specular glare, chromatic aberration, barrel distortion, shot+dust noise,
  posterization, a harsher single-pass JPEG — specifically to catch what the saturated,
  circular recapture probe cannot: a checkpoint that generalizes only to the *trained*
  perturbation family rather than to print-and-capture degradation in general. Built directly
  in response to the `bayar_dinov2_v0` blind spot above. Status: smoke-tested on CPU (n=12,
  `finetune_v2`) — the degradation is visually non-trivial/legible and AuDET reads non-zero
  (unlike the saturated recapture probe); a full val-split run needs VESSL GPU (or a slow CPU
  pass via `--max-images`) and hasn't happened yet.
- `scripts/analysis/hesitant_test_images.py` + `notebooks/hesitant_test_images.ipynb` pull the
  present-test ids whose rank-averaged score sits closest to 0.5 for manual inspection. Useful
  for spotting *systematic* error patterns (a document type, a tamper style) worth targeted
  augmentation — not a numeric ranking instrument by itself. Note: submission scores are
  rank-averaged, so their marginal distribution is close to uniform on (0,1) by construction —
  "score near 0.5" means *median relative rank*, not necessarily *raw sigmoid ≈ 50/50*.

## Leaderboard & next-step priorities

Team `hyunsooochung` (`dylancho80, hyunsooochung, hyunyoungjeong, yuriii2`), rank **33/212**
(top ~16%) as of finetune_v0's submission — competition is `the-freuid-challenge-2026-ijcai-ecai`
on Kaggle, deadline 2026-07-16, submit via `kaggle competitions submit -c
the-freuid-challenge-2026-ijcai-ecai -f submissions/<name>.csv -m "<message>"` (credentials
already on the VESSL box at `~/.kaggle/kaggle.json`). Scores cluster tightly around our rank
(rank 32 = 0.00721, rank 34 = 0.00800) — small gains move several ranks; the big gap is to the
top ~10-15 teams (best is 0.00039), not to us specifically.

Since AuDET/APCER@1%BPCER are pure rank/threshold-sweep metrics (score *distribution shape*
truly doesn't matter, only pairwise ordering — see the FREUID formula in the Project section),
the next-step priorities in rough order:

1. **Run `probe_v2` at scale.** It exists and is smoke-tested but hasn't run a full val split
   against the trained checkpoints yet — do that on VESSL next. Urgency comes from
   `bayar_dinov2_v0`: the local probe hit an exact, saturated 0.0 and the public LB still came
   back ~2.9x *worse* than finetune_v0 — the local instrument wasn't just low-resolution
   between good candidates, it was blind to a real regression. Every idea below faces the same
   blind spot until this instrument is running at scale.
2. **Loss-level: target APCER@1%BPCER specifically.** The pairwise soft-AUC term
   (`auc_loss_weight=0.1`) optimizes the *full* curve; APCER@1%BPCER is a partial-AUC metric that
   only cares about the strict tail near the 99th-percentile bona-fide threshold. A full-curve
   loss doesn't specifically push on that tail — consider a higher `auc_loss_weight`, or a
   partial-AUC / hard-negative-mining term focused on pairs near that boundary.
3. **Multi-seed rank-averaging** of finetune_v0's exact recipe (cheapest diversity — no new
   architecture) before reaching for cross-architecture ensembling.
4. **ViT-L or a retrained ConvNeXt** as genuine cross-architecture ensemble members (feasibility
   already confirmed for ViT-L, see Models section).
5. Inspect the hesitant-images notebook output for *systematic* (not random) error patterns to
   target with augmentation/curriculum changes. This is how the `bayar_dinov2_v0`/overlay line
   of investigation started (see above) — don't repeat that specific fusion family, but a
   different systematic pattern found this way is still worth a fresh look.

## Compute & environment

Single **A100**. Full fine-tune of ViT-B @518 is fast enough with AMP: ~0.27 s/it, ~12 min/epoch,
~4 h for 20 epochs (FP32 was 1.69 s/it — never run full FT without `extra.amp: true`). The old
"freeze the big model, cache features" strategy is obsolete for the primary path; gradient
checkpointing and `train_last_k_blocks` exist if ViT-L doesn't fit. On the workspace, work under
`/root` (persists across restarts): environment, dataset, feature cache, checkpoints all there.

### VESSL workspace (`freuid-hy`)

- **SSH alias:** `freuid-hy` — configured in `~/.ssh/config`, key at `~/.ssh/freuid.pem`
- **Repo:** `/root/repo` (cloned; check the current branch before pushing)
- **Python:** `/opt/conda/bin/python3` (conda, not system Python)
- **Data:** `/root/repo/data/` — 69,352 train + 7,821 test images, gitignored, never touched by
  git. **Never commit dataset files anywhere** (competition license). A separate, full local
  copy also exists on the dev machine at `data/raw/` (one directory level deeper than the VESSL
  layout — `data/raw/train/train/...` vs `/root/repo/data/train/train/...`); read-only CPU
  analysis can run locally against it (see `scripts/analysis/resolution_census.py`'s
  `resolve_data_root` for the layout-detection pattern), but training/inference/anything GPU
  still runs on VESSL only.
- **Checkpoints / submissions:** `/root/repo/checkpoints/`, `/root/repo/submissions/`
- **Training logs:** `/tmp/train_<config-name>.log`
- **Env resets often, mid-session, without warning**: `/opt/conda`'s installed packages
  (including the editable `freuid` install itself) have been observed to disappear more than
  once within a single work session, not just across restarts. If any command fails with
  `ModuleNotFoundError`, don't assume the code broke — first run
  `cd /root/repo && /opt/conda/bin/pip install -e .` and retry.

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
`backbone_name`, `image_size`, `tta` as an explicit scale list, `llrd.*`, `amp`,
`train_last_k_blocks`, `grad_checkpointing`, feature/head flags). Seed from `cfg.seed`. Loss:
`BCEWithLogitsLoss` + pairwise soft-AUC term (`auc_loss_weight`, see `finetune_v0.yaml`). Log
per epoch: `lr_head, train_loss, val_loss, AuDET, APCER@1%BPCER, probe_AuDET`. Log dataset size
and per-class counts at train start. Keep every `model_type` path importable and runnable:
`baseline` (CNN or ViT backbone), the parked `consistency`, and `bayar_fusion`.

## Key docs

- `docs/competition.md` — competition brief (metric, rules, licensing, fraud-type taxonomy)
- `docs/workflow.md` — end-to-end guide: train → infer → submit, what goes in each directory,
  how to add experiments, reproducibility checklist
- `scripts/analysis/nondigital_probe.py`, `probe_v2.py`, `hesitant_test_images.py` +
  `notebooks/hesitant_test_images.ipynb` — the validation instruments described above
- `scripts/analysis/{resolution_census,aug_resolution_audit,vit_res_dryrun}.py` — regenerate the
  resolution/compute analysis behind the `finetune_v2` (784px) conclusion above
- `scripts/config_diff.py` — resolved-config diff tool; verify a new experiment config changes
  only what it claims to
- `scripts/analysis/{degradation_curves,per_slice_metrics,representation_drift,
  score_distribution_audit,shortcut_probe,tamper_bbox,visualize}.py` — additional finetune_v0
  diagnostics (robustness curves, per-slice metrics, representation drift, score distribution,
  shortcut-feature probing, tamper localization, attention/attribution visualization)
