# FREUID Challenge 2026 — Technical Report

From the DINOv2 baseline (`finetune_v0`) through the current fusion-model attempt
(`bayar_dinov2_v1`). Covers what was tried, why, what happened, and what's still open.

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

## Open questions and recommended next steps

1. **Get the actual public LB number.** Everything above about `bayar_dinov2_v1` is inferred
   from `probe_v2`, which is a better instrument than the saturated recapture probe but still
   evaluates a held-out split of the *training* distribution — not real out-of-domain document
   types, GenAI edits, or physically-recaptured photos. It's a harder degradation, not a harder
   source distribution. This is the cheapest, highest-information next step and should happen
   before further architecture changes.
2. **Disentangle the bundled `v1` changes.** A clean ablation restoring `synth_tamper_prob:
   0.3` (isolating just the face-crop pipeline fix) would resolve whether the APCER regression
   traces to the synth-tamper removal or the pipeline fix itself.
3. **The most-confident root cause from the `v0` postmortem is still unaddressed.** The
   `overlay_gate` check above suggests the capacity/overfitting concern is still live. If
   pursued further, the next fix should target that directly — stronger regularization on the
   overlay branch (higher `fusion_dropout`, weight decay on overlay params, or shrinking
   `noise_feat_dim`/the RGB backbone) — rather than more data-pipeline correctness work.
4. **A structural concern worth weighing before investing further**: `BayarConv2d` is
   fundamentally a digital-noise-residual signal, the same category of signal this project's
   founding failure (the pre-project forensic-noise model) leaned on and lost on real,
   reprinted images. Fixing face-crop reliability doesn't resolve whether that signal *type*
   survives the analog hole at all — it only ensures the branch learns a more coherent version
   of it from clean training data. Continued investment here should be weighed against the
   other priorities already identified as structurally safer: multi-seed rank-averaging of
   `finetune_v0`'s own recipe (cheapest diversity, no new architecture), a `ViT-L` scale-up
   (feasibility already confirmed on the A100), and shaping the loss toward the partial-AUC
   `APCER@1%BPCER` tail specifically rather than the full curve.

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
