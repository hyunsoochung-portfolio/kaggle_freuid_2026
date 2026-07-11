# FREUID Challenge 2026 — Technical Report

From the DINOv2 baseline (`finetune_v0`) through the current photo-substitution data-augmentation
attempt (`photosub_v0`). Covers what was tried, why, what happened, and what's still open.

**Erratum**: every "public LB" number below is the organizers' combined FREUID score
(DET-F1 harmonic mean of AuDET and APCER@1%BPCER), not raw AuDET, despite this report's own
"Primary metric AuDET" framing below — see `scripts/analysis/official_score_reconciliation_out/
official_score_reconciliation_report.md` and `CLAUDE.md`'s Project section.

## Problem recap

Binary fraud detection on identity-document images. Output a continuous fraud score
`P(fraud) ∈ [0,1]`; `1 = fraud`, `0 = bona-fide`. Primary metric **AuDET** (area under the DET
curve, repo proxy `1 - roc_auc_score`, **lower is better**); secondary **APCER @ 1% BPCER**.
Both are pure rank metrics — only score *ordering* matters, not calibration.

The real (public/private) test set is deliberately hard: **print-and-capture ("analog hole")
attacks**, **GenAI multimodal edits**, and **document types not seen in training** — while the
training data is ~99.97% digital, clean scans. This asymmetry is the throughline of every
result below. A pre-project forensic-noise model scored ~0.0006 locally but ~0.377 public — it
leaned on digital noise that reprinting erases, and was validated on an easy in-domain split.
Everything since has been built to avoid repeating that specific failure.

## Stage 0-3: the retired consistency-based plan

The original plan was a frozen-foundation, consistency-based detector: freeze a pretrained
DINO backbone, train only light heads on top (patch/face self-consistency). This plan was
**retired — the data killed it**:

| Stage | Config | Model | probe_AuDET | public LB |
|---|---|---|---|---|
| S0 | `baseline_v0` | EfficientNetV2-S, fine-tuned | 0.000000 | 0.27106 |
| S1 | `baseline_v1` | ConvNeXt-Small, fine-tuned, 384px | 0.000039 | 0.18129 |
| S2 | `consistency_v0` | DINOv2 ViT-B/14 **frozen** + head | 0.061290 | 0.30743 |
| S3 | `consistency_v1` | DINOv3 ViT-B/16 **frozen** + patch/face heads | 0.115286 | 0.47469 |

The decisive follow-up experiment (below) used the same DINOv2 backbone as S2, fully
fine-tuned with S1's recipe, and beat S1 by 24x and S2 by ~40x on the public LB. **Freezing was
the problem, not the backbone.** Corroborating detail: the fine-tuned model's epoch-1 val_AuDET
(backbone still ~unmoved during LLRD warmup) reproduced S2's frozen-model score almost exactly;
the collapse to near-zero happened the moment backbone weights started adapting. The
frozen-vs-fine-tuned confound was never cleanly separated from "is DINO the right backbone" —
that question stays open but low-priority; `model_type: consistency` is kept importable but
parked.

## finetune_v0: the breakthrough, and current best result

**`vit_base_patch14_reg4_dinov2.lvd142m`** (DINOv2 ViT-B/14, 4 register tokens, Apache-2.0),
**fully fine-tuned end-to-end** (not frozen) at 518px.

Recipe (`configs/finetune_v0.yaml`):
- LLRD (layer-wise LR decay, `decay: 0.7`, 2-epoch warmup), head LR 1e-4, per-block decay,
  LayerNorm/bias excluded from weight decay
- AMP (autocast + GradScaler)
- `BCEWithLogitsLoss` + a pairwise soft-AUC term (`auc_loss_weight: 0.1`)
- Print-and-capture **recapture augmentation** (`recapture_transforms`: resize → JPEG →
  downscale → JPEG again → blur/noise → brightness/contrast/HSV → mild perspective + rotation,
  **no flip**, ever — document orientation must be preserved) simulating the analog-hole attack
  on 99.97%-digital training data
- Synthetic tamper injection: 30% of bona-fide samples per batch get a synthetic copy-move /
  field-smudge / local-splice edit and become synthetic positives (`synth_tamper_prob: 0.3`)
- Checkpointed on lowest **recapture-probe AuDET** (a per-epoch in-loop validation instrument
  applying the same `recapture_transforms` to a held-out clean split), ties broken by probe
  APCER@1%BPCER
- 3-scale TTA `[476, 518, 560]` at inference, all multiples of the ViT-B/14 patch size,
  rank-averaged (never raw-score-averaged)

**Result: probe_AuDET 0.000002, public LB 0.00744** (rank 33/212 at submission time). The
strongest result to date, and the config every later experiment is compared against.

## Follow-up experiments off finetune_v0 (both negative)

| attempt | change | probe_AuDET | public LB | verdict |
|---|---|---|---|---|
| `finetune_v1` | DINOv3 ViT-B/16 backbone swap, otherwise identical recipe | — | 0.02695 | ~3.6x worse |
| `finetune_v2` | 784px instead of 518px, byte-identical config otherwise | 0.000000 (ep18) | 0.00750 | wash (noise-level) |

- **`finetune_v1`**: the backbone swap alone made things ~3.6x worse. DINOv3 is deprioritized
  for the primary path; a parked DINOv3 loader in `src/freuid/backbone.py`
  (`transformers`-based, unused by the active path) has an unverified register-token-slicing
  risk this regression corroborates in practice.
- **`finetune_v2`**: resolution increase was a wash, not a wasted-effort upsampling artifact —
  the native data ceiling is min-side ≈1000px, so 784px is genuine signal, not interpolated
  noise, and training augmentation's resize/downscale steps are relative rather than an
  absolute bottleneck. **Conclusion: resolution isn't the lever that matters here.** Capacity,
  ensembling, and loss-shaping toward the rank metric look more promising than further
  resolution tuning; 1036px was ruled out (0% of images natively reach that size, pure
  upsampling, and the compute cost is worse).

## bayar_dinov2_v0: the fusion-architecture attempt

**Motivation**: a separate, pretrained `overlay_colab` model (BayarConv2d forensic-noise + face
crop) was decisive and directionally right on `finetune_v0`'s most uncertain predictions, but
three *score-level* ensembles of finetune_v0 + overlay_colab all regressed the public LB
(0.01101, 0.01659, 0.05319 vs. 0.00744) despite that signal. The diagnosis: the failure was
architectural (a fixed combination rule can't tell overlay's trustworthy calls from its
untrustworthy ones), not conceptual. `bayar_dinov2_v0` tested whether a **jointly fine-tuned,
gated** fusion could learn that distinction instead.

**Architecture** (`src/freuid/models/bayar_fusion.py`, `BayarFusionNet`):
- Main stream: the same fully fine-tuned DINOv2 ViT-B/14 as `finetune_v0`, seeing the raw
  (unrectified) image — deliberately kept identical to the baseline path so this experiment
  isolates "add an overlay branch," not "also change what DINOv2 sees"
- Overlay stream (`OverlayStream`): a `BayarConv2d`-fronted CNN (`NoiseStream` — a learnable
  constrained high-pass filter, forced to compute a noise residual regardless of what it
  learns) concatenated with a ResNet34 RGB branch, both operating on a 224px face crop
- Fusion: `dino_feat` and `overlay_feat` concatenated, `overlay_feat` scaled by a
  LayerScale-style `overlay_gate` (near-zero init, `1e-3`) before a small `FusionMLP` head —
  the model starts close to DINOv2-only behavior and "opens" the overlay pathway only as it
  earns its keep
- Face crop source (at the time): SCRFD-detected box on `rectify_card()`'s FastSAM-rectified
  512x512 output, cached once via `precache_regions()`
- Otherwise byte-identical recipe to `finetune_v0` (confirmed via `scripts/config_diff.py`):
  same LLRD, AMP, recapture augmentation (applied to **both** streams independently), 30%
  synthetic tamper, AUC loss term, TTA

**Result: probe_AuDET exact 0.0, public LB 0.02146 — ~2.9x worse than finetune_v0.** The
checkpoint genuinely learned to rely on the overlay branch (100% of the `overlay_gate`'s 640
elements moved >10x from init) — that reliance is *why* it overfit, not evidence the branch
stayed inert.

**Postmortem, most-to-least confident root cause:**
1. The extra branch is pure added capacity to fit the closed train/val/probe loop without that
   fit needing to generalize.
2. Recapture augmentation applied to the face-crop stream (JPEG/blur/noise/downscale, built to
   simulate print-and-capture) is exactly the kind of degradation that erases the fine noise
   residue BayarConv2d depends on.
3. SCRFD only found a valid face on ~1 in 5 documents (train 21.5%, public_test 16.4%), a real
   but secondary train/test mismatch — measured, but not yet understood at the time.

Conclusion at the time: don't retry this fusion family (BayarConv2d/SRM-style noise-residual
branches jointly trained under recapture augmentation) without a specific new idea for
resolving the augmentation-vs-noise-signal conflict.

## This session: diagnosing and fixing the face-crop pipeline

Two prerequisite tooling fixes, then two substantive bugs, found via direct investigation
(visual inspection of cached crops, FastSAM's own candidate masks, and live SCRFD tests), not
inference from the postmortem alone.

### Tooling: making `probe_v2` actually usable

`probe_v2` (`scripts/analysis/probe_v2.py` + `src/freuid/probe_v2_augment.py`) is a second,
independently-designed degradation probe — deliberately disjoint from `recapture_transforms`
(halftone/moiré, vignette, specular glare, chromatic aberration, barrel distortion,
Poisson+salt-and-pepper noise, posterization, a harsher single-pass JPEG at a non-overlapping
quality range) — built specifically to catch the blind spot that let `bayar_dinov2_v0`'s
saturated, exact-0.0 recapture probe miss its real regression. As written, it could only build
the plain baseline ViT architecture: it never checked `cfg.extra["model_type"]`, so scoring a
`bayar_fusion` checkpoint failed `load_state_dict` on the first mismatched key, and even past
that, its scoring loop only ever called `model(imgs)` — one positional argument short of
`BayarFusionNet.forward`'s required `face_crop`. It could never score the one checkpoint it was
built to retroactively validate against.

Fixed by adding one shared `build_model_for_config()` dispatcher (mirroring the
`model_type` branch already duplicated in `train.py`/`infer.py`) and rewriting `probe_v2.py` to
build a real `FreuidDataset` and score via the same generic `unpack_and_move`/
`forward_with_extras` plumbing `infer.py` already uses — model-type-agnostic by construction,
so any current or future `model_type` works without the script special-casing it.

### Bug 1: SCRFD was detecting faces on the wrong image

Investigating a hunch that the hesitant-predictions review (most-hesitant public-test scores
skewed toward fraud, mostly face-region tampering) pointed at a face-crop-quality issue led to
building `scripts/analysis/scrfd_coverage.py`, an audit of the cached `face.json` detection
`score` field. It reproduced the known 21.5%/16.4% real-detection rate — but broken down by
document type, the picture was sharply bimodal, not a uniform ~80% miss rate:

| type | detection rate |
|---|---|
| EGYPT/DL | 67.1% |
| MAURITIUS/ID | 31.4% |
| GUINEA/DL | 0.3% |
| BENIN/DL | 0.0% |
| MOZAMBIQUE/DL | 0.0% |

Visual inspection of the cached crops for the near-zero types showed the SCRFD "miss" wasn't a
detector failure at all — `rectify_card()`'s output for those samples didn't contain a face:
FastSAM's rectification had warped onto the document's flag watermark or a barcode strip
instead of the card. Every sampled miss for a given document type landed on the *same*
graphic, deterministically. Inspecting FastSAM's own raw candidate masks confirmed why: even
the "working" Egypt/DL case picked a candidate quad covering only ~2.5% of the frame — the same
order of magnitude as the failing cases (~1.4-1.7%). FastSAM never actually found "the card
boundary" in any of these; this dataset's raw images are already full-frame card photos (no
distinct card-vs-background edge for a "largest quad" heuristic to find), so it always locks
onto whichever internal graphic forms the cleanest rectangle. Egypt/DL "worked" by coincidence
(that graphic happened to be the photo region), not because rectification was doing its job.

**Fix**: run SCRFD directly on the original image instead of `rectify_card()`'s output
(`src/freuid/preprocess.py`). Verified live before committing to a full cache regeneration:
SCRFD found faces at 0.85-0.91 confidence on all four previously-~0%-detection raw images.
`rectify_card()`/`card.png` generation is untouched (still used by the parked `use_rectify`
consistency path for an unrelated reason); only face detection's input changed.

Regenerated the entire regions cache under the fix (GPU-accelerated via `onnxruntime-gpu`,
~30 minutes for 77,173 images): **train detection 21.5% → 100.0%, public_test 16.4% → 99.9%**,
confirmed per-document-type (every type now ≥99%) and score-distribution-healthy (median
confidence 0.86-0.87, up from 0.69-0.76; oversized/false-positive-box rate 0.03-0.5%).

`face_meta_tensor`/`face_crop_image`/`FreuidDataset`/`SynthTamperWrapper` updated to crop from
and normalize against the original image's coordinate space (previously the rectified card's),
since the box is no longer in that space.

### Bug 2: the face-crop stream was degraded during training and double-normalized

Two related issues, found while addressing the postmortem's cause #2 directly:

- `train.py` unconditionally ran the 224px face crop through `recapture_transforms` during
  training — the exact mechanism suspected of erasing BayarConv2d's noise residual before the
  branch could learn anything from it. Removed: the face crop now gets a bare `ToTensor()`
  only, no degradation, at train time.
- Independently: `OverlayStream.forward()` explicitly documents that it expects an
  already-cropped, **[0,1]-scaled** face image and normalizes internally for its own RGB
  branch — but every caller (`train.py`'s `recapture_transforms`, `infer.py`'s
  `build_transforms(...,mean,std)`, `probe_v2`'s `Probe2Transform`) ended with
  `A.Normalize`/`transforms.Normalize` before `ToTensor`, so the crop was being
  ImageNet-normalized once by the dataset-side transform and *again* inside `OverlayStream` —
  feeding it a garbage-scale input regardless of the degradation question. Fixed at all three
  call sites; `probe_v2_augment.py` gained a `normalize=False` path (scale-to-[0,1] via
  `A.ToFloat` instead of `A.Normalize`) for this, since `probe_v2` deliberately *keeps*
  degrading the face crop (a real analog-hole test image's face region is degraded regardless
  of training policy — that's exactly the gap worth testing) but shouldn't double-normalize it.

Spot-checking real val samples confirmed the normalization fix changes individual raw logits
meaningfully (shifts of 0.2-1.4 in logit space) — but re-running the full-scale `probe_v2`
check against the *existing* (pre-fix-trained) `bayar_dinov2_v0` checkpoint produced numerically
identical `AuDET`/`APCER` before and after the normalization fix. This is explainable, not a
broken test: the shift was consistently monotonic across sampled examples, and since AuDET/
APCER are pure rank metrics, a rank-preserving shift leaves them unchanged even though every
individual prediction's raw value moved. It's also a reminder that a fixed, already-trained
checkpoint evaluated under a corrected pipeline is an out-of-distribution test for that
specific checkpoint's overlay branch — informative about robustness, not about what a model
*retrained* under the fix would do.

## bayar_dinov2_v1: retrain under the fixed pipeline

`configs/bayar_dinov2_v1.yaml` — confirmed via `scripts/config_diff.py` to differ from
`bayar_dinov2_v0.yaml` in exactly two resolved keys:

- `name` (`bayar_dinov2_v0` → `bayar_dinov2_v1`)
- `extra.synth_tamper_prob`: `0.3` → `0.0`, a deliberate, separate choice made alongside the
  pipeline fix (not an oversight): real fraud examples already make up 42% of the training set
  (29,347 of 69,352 rows), so synthetic tampering was never needed for class-volume reasons,
  and its uniform-random tamper placement doesn't reflect the face-region tampering pattern
  observed in the hesitant-predictions review.

Everything else — architecture, LLRD schedule, TTA scales, AUC loss weight — stayed identical
to v0; the code-level face-crop pipeline fixes above apply unconditionally (not gated by a
config flag, since they're bug fixes to what `model_type: bayar_fusion` does, not new
experimental knobs).

**Training** (VESSL, A100, 20 epochs, ~15 min/epoch including val+probe passes): sanity check
passed (`init BCE=0.6931 ≈ ln2`), 108,462,305 trainable params, converged smoothly
(`train_loss` 0.14 → 0.027 over 20 epochs). The recapture probe saturated by epoch 9
(`probe_AuDET` hit exact `0.0` three times: epochs 9, 13, 14) — expected and, per the
`bayar_dinov2_v0` precedent, not to be over-trusted. Best checkpoint saved at **epoch 13**.

**Full-scale `probe_v2` comparison** (n=6,935, full held-out val split):

| | `bayar_dinov2_v0` | `bayar_dinov2_v1` | change |
|---|---|---|---|
| `probe_v2_AuDET` | 0.047149 | **0.036830** | ↓ ~22% (better) |
| `probe_v2_APCER@1%BPCER` | 0.236797 | **0.321295** | ↑ ~36% (worse) |

A mixed result, not a clean win: overall ranking improved on an instrument that isn't
saturated, but the strict-tail metric got meaningfully worse. Since this run bundled the
pipeline fix with the `synth_tamper_prob` change, this single data point can't attribute either
effect to a specific cause.

**`overlay_gate` check** (same diagnostic CLAUDE.md used for v0): 558/640 elements (87.2%)
moved more than 10x from init, vs. v0's reported 100%. Still substantial reliance on the
overlay branch — marginally less saturated than v0, but not qualitatively different. This
suggests the postmortem's most-confident root cause (added capacity letting the branch fit the
closed train/val/probe loop without generalizing) likely still applies; nothing done this
session addressed it.

**Status**: submission generated and integrity-checked (142,818 rows, 0 exact-zero scores among
7,821 present ids) but not yet submitted to the public LB — blocked by the competition's shared
daily submission cap, which teammates' parallel experiments (EVA-01, SigLIP2, ConvNeXtV2,
several "data_attention" variants — none of which appear elsewhere in this repo's history, so
apparently tracked outside it) had already exhausted for the day.

## photosub_v0: photo-substitution training augmentation

**Motivation**: manual review of `finetune_v0`'s lowest-ranked-yet-genuinely-fraudulent public-
test predictions (`scripts/analysis/deep_miss_dossiers.py`) found a coherent taxonomy of 9
"deep miss" ids (confirmed fraud, scored as if bona-fide) plus 59 further "boundary" ids at a
~74% fraud rate: **MODE_A** (within-frame physical paste — style-mismatched print, paper rim,
shadow), **MODE_B** (full-cover physical paste, overhangs the frame), **MODE_C** (frame-aligned
digital swap, zero local statistical anomaly, only cross-region semantic evidence), and
**MODE_D** (ghost/secondary-portrait mismatch, only for the 2 of 5 templates that carry one).
Training data is ~99.97% digital and never modeled this evidence family at all — the hypothesis
was that `finetune_v0` misses these not because they're hard, but because it was never shown
anything resembling them.

**What changed vs. `finetune_v0`** (`configs/photosub_v0.yaml`, confirmed via
`scripts/config_diff.py` to differ in exactly one block, `extra.photosub.*` — backbone, LLRD,
AMP, recapture augmentation, AUC loss, TTA all byte-identical to the baseline):

- **Offline photo-substitution generators** (`src/freuid/photosub/generators.py`): deterministic
  functions of `(image, frame/ghost box, donor crop, rng)` implementing MODE_A/B/C/D. MODE_C/D
  are color- *and* degradation-matched (`degradation_match.py` — blur/moiré/JPEG-grain search
  against the target region's own local stats) so they aren't locally distinguishable by cheap
  forensic stats, only by cross-region semantics. MODE_D ships with a **darkened-ghost-variant**
  slice (`ghost_darken_prob=0.3`, factor range 0.15–0.45) to reproduce the real illegible-ghost
  case (id `40dd1055fd`'s ghost is unreadably dark even at 4x zoom) rather than only ever
  generating crisp ghosts.
- **Donor discipline**: donors are bona-fide TRAIN faces only, excluded from any id
  suspiciously close (ArcFace cosine similarity) to a frozen diagnostic probe id, ~40% sampled
  as "hard" (softmax-weighted toward the target's own most-similar match) vs. broad/random.
- **Twin-pair batching** (`src/freuid/photosub/mixing.py`): each generated row's `source_id`
  (the bona-fide TRAIN image it was derived from) is recorded; `TwinPairBatchSampler` tries,
  with probability `twin_pair_prob=0.5`, to pull that same source into the batch alongside its
  tampered derivative, so the pair differs *only* in the substitution evidence.
- **Pair-hinge loss** (`freuid.loss.pair_hinge_loss`, weight `0.5`, margin `1.0`): for every
  twin pair landed together in a batch, penalizes `relu(margin - (tampered_logit -
  clean_logit))` directly — on top of the unchanged BCE + pairwise-soft-AUC terms.
- **Mode-weighted, split-disciplined mixing**: photosub rows are added at `share=0.15` of total
  positives, drawn per confirmed-prevalence mode weights (A=0.44, B=0.33, C=0.22 from the 9
  deep-miss ids' A=4/9, B=3/9, C=2/9; D=0.15 is a modest, *hypothesis-driven* weight, not backed
  by confirmed prevalence — 0/9, but only 2 of 5 templates carry a ghost at all, so n=9 saying
  nothing about it isn't evidence against it). `select_mixed_rows` filters generated rows to
  `source_id ∈ train_ids` for whichever split a given run actually uses, so a row never leaks
  validation-set appearance into training.
- **Per-epoch probe hooks** (`src/freuid/photosub/probes.py`): the 9 deep-miss ids scored
  individually every epoch (logit + rank percentile vs. that epoch's val distribution), plus
  aggregate mean-logit tracking for the 59 boundary ids, 295 clean-floor ids (regression guard:
  must not rise), and a newly-added 200-id ceiling-retention probe (`data/probes/
  ceiling_frauds_sample_ids.csv`, sampled from `logit_census_raw.csv`'s saturated-ceiling block —
  a self-consistency check, since public_test has no ground truth: the fix must not trade away
  `finetune_v0`'s existing high-confidence detections).

**Mass generation** (`scripts/generate_photosub_dataset.py`, full un-restricted bona-fide TRAIN
pool, 9,000 sources, seed 42): **9,000/9,000 rows written, 0 skipped.** Mode counts: A=3,845,
B=2,834, C=1,867, D_main=229, D_ghost=225. A random-sample spot review over the actual generated
corpus (not the curated demo sheets — `scripts/analysis/photosub_spot_review.py`, 80 rows +
full-corpus stats) found: 0/9,000 tiny/degenerate tamper regions, 0/9,000 unexpected template
types, and — the donor-pool-exhaustion concern raised before generating — **not a real problem**:
3,454 distinct donors used across 9,000 rows, the most-reused donor appearing only 9 times
(0.10%), 918/3,454 used exactly once. Darkened-ghost variants were manually verified across the
intended severity spectrum, from fully legible down to a near-illegible worst case (factor
0.157) closely matching the real `40dd1055fd` reference. One infrastructure issue surfaced and
was fixed during this run: the generator originally only flushed its output CSV once, at the
very end of the loop; a mid-run stall (cause unclear — resolved itself on retry, possibly
transient host contention) killed the process and orphaned ~2,800 already-written image/mask
files with no CSV rows to recover them. Fixed to flush incrementally every 50 rows before
re-running the full generation.

One shortfall: **MODE_D fell short of its target weight.** The full (un-restricted) training
run's `share=0.15` target wanted 608 D rows; only 402 were eligible after filtering to that run's
actual `train_ids` (454 were generated in total, but ~52 fell on the val side of the split).
`select_mixed_rows` took all 402 available and printed a warning rather than failing — not
fatal, but D is under-represented relative to its intended (already modest) weight.

**Training** (VESSL A100, 20 epochs, full real split — `train=66,872`, `val=6,935`, no
`--limit`, ~13.3–13.75 min/epoch, ~4.5h total): sanity passed (`init BCE≈ln2`), losses finite
throughout, twin pairs landed every epoch (~1,499/epoch, consistent with `twin_pair_prob=0.5`
over ~4,455 mixed-in rows). The standard val/probe metrics **saturated far faster than
`finetune_v0`'s own 20-epoch run** — `val_loss` 0.517→0.004 and `AuDET`→0.0000 by **epoch 2**,
vs. `finetune_v0` needing the full run to reach its final `probe_AuDET=0.000002`. Best
checkpoint (by `probe_audet`) saved at **epoch 6** (`0.000031`); no later epoch ever beat it, so
the checkpoint actually used for inference/submission is the epoch-6 snapshot, not the final
epoch-20 weights. Floor/ceiling guards held for the whole run: `clean_floor_mean_logit` only
ever dropped (never rose, -0.40 → -8.38), `ceiling_mean_logit` re-saturated and kept climbing
(0.58 → 15.09 by epoch 20) — the fix did not trade away `finetune_v0`'s existing high-confidence
detections.

**Deep-9 diagnostic — the core, and mixed, result.** Raw logit per id (mode in parens; C-tent. =
the one id assigned MODE_C tentatively, by elimination, not positive ghost confirmation):

| id | mode | epoch 1 | epoch 6 (checkpoint) | epoch 20 (final) |
|---|---|---|---|---|
| `c6651aee` | A | -0.11 | -5.53 | -7.79 |
| `b5eebda1` | A | -0.14 | 0.23 | 3.53 |
| `40dd1055` | A | 0.39 | 0.54 | -8.06 |
| `5542f45f` | A | 0.36 | 10.25 | 0.48 |
| `7b409d3b` | B | -0.29 | 9.39 | 12.55 |
| `cd7ad569` | B | -0.30 | 10.25 | 12.21 |
| `2d4ad17d` | B | 0.33 | 10.31 | 15.72 |
| `cceb6a4f` | C (tent.) | -0.37 | 12.39 | 15.43 |
| `a2a3fe5b` | C (conf.) | 0.86 | 12.00 | 15.20 |

Every MODE_B and MODE_C id (**including the tentative one**, which was predicted to be the
likely straggler) converged fast and stayed stably, strongly positive. **MODE_A did the
opposite of the hypothesis**: the mode with the most visually blatant evidence (rims, tears,
tape, style mismatch) ended up the weakest — by epoch 20, 2 of 4 MODE_A ids (`c6651aee`,
`40dd1055`) were stably, strongly negative (confidently scored *bona-fide*, worse than
`finetune_v0`'s own baseline), one (`b5eebda1`) only mildly positive, and one (`5542f45f`) had
round-tripped from a strong positive at epoch 6 back down near zero. Even at the actually-used
epoch-6 checkpoint — healthier than epoch 20, but still not matching the predicted shape — MODE_A
sits at 0.23–10.25 against MODE_B/C's 9.39–12.39. Individual ids oscillated substantially epoch
to epoch before this pattern stabilized (~epoch 9–11 onward), consistent with a model whose
overall loss barely moves once it saturates the easy synthetic distribution but whose weights
keep drifting in ways that swing these specific real, hard examples around.

**Analysis — most-to-least confident**:
1. **Shape-realism gap, previously documented and now corroborated in practice.**
   `generators.py`'s own docstring already flagged this: MODE_A/B paste a (possibly
   small-angle-rotated) **rectangle**, not the fully irregular, hand-cut/arch-shaped silhouette
   several real deep-miss ids actually show (`deep_miss_dossiers.py`'s `CHECKLIST_RESULTS` — e.g.
   `b5eebda1`'s arch-shaped cutout overlapping the crest logo, `40dd1055`'s irregular silhouette
   bulging past the hairline). If the model learned "fraud paste = rectangular color/style-
   mismatched inset + rim + drop shadow," a real *irregular*-boundary paste never triggers that
   rule — and could plausibly read as *more* clean by contrast (no rectangular anomaly to flag),
   a concrete mechanism for the negative reversal, not just a failure to help.
2. **MODE_C/D's signal is narrower and more consistent than MODE_A/B's.** C/D always exactly
   fill the frame/ghost box with a color-and-degradation-matched patch — one tight, repeatable
   recipe. A/B randomize style, rim width, rotation, offset, shadow edges/blur/strength, and an
   optional curl highlight per example — much more heterogeneous. At only 15% of positives split
   four ways, A/B's wider variety may simply be harder to consolidate into one stable,
   generalizable rule than C/D's narrow one, independent of the shape-realism gap above.
3. **Twin-pair supervision never touches these 9 ids directly.** The pair-hinge term only
   applies to synthetic (tampered, own bona-fide source) pairs; none of the 9 real deep-miss ids
   have a synthetic twin. Their behavior is entirely mediated by generalization from whatever
   feature the backbone converged on for "photo substitution" — which this data suggests ended
   up much closer to C/D's clean-swap signature than A/B's physical-paste one.
4. **Not a volume problem.** MODE_A got the *largest* photosub allocation (1,801 rows, weight
   0.4444) and still underperformed B/C — arguing against "just needed more A-mode rows" as the
   primary explanation, in favor of the qualitative mismatch in point 1.
5. **This is a failed gate on its own pre-registered terms.** `configs/photosub_v0.yaml`'s
   docstring pre-registered "majority of the 9 ids' rank percentiles rise... into the ceiling
   region, with no id collapsing further." 5 of 9 (B/C) did rise into the ceiling; MODE_A — the
   modality expected to lead — is the one that collapsed further for 2 of its 4 ids.

**Submission status**: inference run (3-scale TTA `[476, 518, 560]`, rank-averaged) completed
cleanly — 142,818 rows, 0 exact-zero scores, min 0.000981/max 0.996630. Kaggle submission
attempted and rejected (`400 Bad Request`) — file and message both well-formed; almost certainly
the team's shared daily submission cap, already exhausted by teammates' parallel runs earlier
the same day (5 same-day submissions visible in the competition's own history). Checkpoint
(346MB) and submission CSV pulled to the local repo pending a retry after the daily reset
(00:00 UTC). **No public LB number yet** — everything above is inferred from the deep-9 probe
and the standard val/probe/floor/ceiling instruments, not the real held-out test distribution.

## Open questions and recommended next steps

**Historical note**: the `bayar_dinov2_v1` priorities that used to live in this section
(getting probe_v2 at scale, disentangling its bundled changes, the overlay-branch capacity
concern, BayarConv2d's structural noise-residual risk) were never resolved — the project moved
to the photo-substitution line of work above instead of pursuing them. They're restated as item
5 below, not dropped.

**Current priorities, post-`photosub_v0`:**

1. **Get the actual public LB number once the daily submission cap resets.** Everything about
   `photosub_v0` above is inferred from local instruments; the real test distribution (print-
   and-capture, GenAI edits, unseen document types) is the only thing that actually matters and
   hasn't weighed in yet.
2. **Fix the shape-realism gap before iterating on weights or loss terms further.** Add an
   irregular/hand-cut silhouette variant to MODE_A/B (not just a rotated rectangle) — this is
   the most concrete, already-diagnosed candidate for why MODE_A underperformed B/C so
   specifically, per the analysis above. Re-render the curated demo sheets for human review
   before regenerating the mass corpus, per the same gate this experiment already used once.
3. **Run the pre-registered ablations** (`configs/photosub_v0_noablate_nopairs.yaml`,
   `configs/photosub_v0_noablate_nohinge.yaml` — drafted, not yet run) to attribute how much of
   the B/C gain (or the A regression) traces to the photosub data itself vs. twin-pairing vs.
   the hinge term specifically.
4. **Top up MODE_D** (402/608 of its already-modest target) if the ghost-mismatch signal is
   worth pursuing further — a small, targeted supplemental generation run restricted to
   ghost-bearing template types would close this without a full mass-regeneration pass.
5. **The `bayar_dinov2_v1` questions above remain genuinely unanswered**, not resolved — if this
   photo-substitution line of work stalls, they're still there to come back to.

## Appendix: this session's commits (`exp/dinov2+@`)

```
fe57152 feat(configs): add bayar_dinov2_v1, a retrain under the fixed face-crop pipeline
cf13ebb fix(bayar_fusion): stop degrading and double-normalizing the face-crop stream
83949b8 fix(probe_v2): make model construction and scoring model-type agnostic
ce8e82a fix(notebook): filter the full score distribution to present public-test ids
cbe7fe2 fix(preprocess): detect faces on the original image, not the rectified card
```

The SCRFD and face-crop-degradation fixes were also cherry-picked onto `exp/bayar+dinov2`
(the branch VESSL trains from, which retains `configs/bayar_dinov2_v0.yaml` and other
retired-experiment configs this branch's "prune for public display" commit removed) as
`b16f79c` and `c127145`.
