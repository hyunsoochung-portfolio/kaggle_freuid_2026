# Pipeline Context — finetune_v0 (current best model)

Quick reference for understanding this codebase, in the same format as `docs/pipeline.md`
(which documents an older, now-superseded model path). This doc covers the pipeline as it
exists **today**, centered on `finetune_v0` — a fully fine-tuned Vision Transformer that is
currently the best submission in the project (public LB FREUID 0.00744; see
`docs/finetune.md` for the full experiment writeup and `ROADMAP.md` for the complete
results history). Written so a reader with basic ML knowledge but no transformer/ViT
background can follow every design decision, not just its name.

Everything here reflects current code state. For the other, currently-losing model path
(a **frozen** backbone with hand-designed "consistency" heads), see `docs/consistency.md`.

---

## Competition in one paragraph

Binary fraud detection on identity-document images. **Label 1 = fraud/attack, 0 =
bona-fide.** The model outputs a continuous fraud score `P(fraud) ∈ [0,1]` — never a hard
label. Primary metric: **AuDET = 1 − ROC AUC.** If you're not familiar with ROC AUC: show
the model one fraud image and one genuine image, and ROC AUC is the probability it scores
the fraud one higher. A perfect model gets ROC AUC = 1, so AuDET = 0; a coin-flip model gets
ROC AUC = 0.5, so AuDET = 0.5. **Lower AuDET is better.** Secondary metric: **APCER @ 1%
BPCER** — pick the score threshold that would wrongly flag only 1% of genuine documents as
fraud (BPCER = bona-fide presentation classification error rate = 1%), then ask what
fraction of *real* fraud still sneaks under that threshold (APCER = attack presentation
classification error rate). Both metrics only care about the **ordering** of scores, not
their calibration — a model that outputs `0.9, 0.1` for `(fraud, genuine)` scores exactly
the same as one outputting `0.51, 0.49`.

---

## What kind of model is `finetune_v0`, in plain terms?

`finetune_v0` is a **Vision Transformer (ViT)** — specifically `vit_base_patch14_reg4_dinov2`
from the `timm` library, originally pretrained by Meta as **DINOv2 ViT-B/14**. If you've
only worked with CNNs before, here's the shape of the idea:

1. **Patchify.** The 518×518 input image is cut into a grid of 14×14-pixel patches — that's
   `518 / 14 = 37`, so a **37×37 = 1,369-patch grid**. Each patch is flattened and linearly
   projected into a 768-number vector — a **token**, the same kind of representation an LLM
   uses for a word.
2. **Prefix tokens.** Two kinds of extra tokens are prepended to the sequence, neither of
   which corresponds to a real image patch:
   - **1 CLS token** — a learned "summary" token that, through self-attention, ends up
     aggregating information from the whole image. Its final representation is what the
     classifier head reads to make the fraud/not-fraud decision.
   - **4 register tokens** — a DINOv2-specific addition. Researchers found that ViTs tend
     to "dump" high-norm, hard-to-interpret activity into a few patch tokens as a side
     effect of training; register tokens give the model dedicated scratch space for that,
     keeping the real patch tokens' attention maps interpretable. **Total sequence length:
     1 + 4 + 1,369 = 1,374 tokens.**
3. **Self-attention blocks.** 12 stacked Transformer blocks, each letting every token look
   at every other token (`num_heads=12` attention heads per block, `embed_dim=768`) and mix
   information accordingly, followed by a small per-token MLP. This is what lets a
   tampered-and-repasted region "notice" that it looks inconsistent with the rest of the
   document, even though it's spatially far away from, say, the signature field.
4. **Head.** After the final block, the CLS token's vector goes through one `Linear(768,
   1)` layer to produce a single fraud logit. `~86.6M` parameters total.

**Backbone vs. head, and frozen vs. fine-tuned** — the two ideas that actually decided this
project's outcome: the "backbone" is the big pretrained feature extractor (all 12 blocks +
patch embedding); the "head" is the small layer added on top for our specific task. Earlier
project stages (`consistency_v0`/`v1`, see `docs/consistency.md`) **froze** the backbone —
locked its weights so only the tiny head could learn — which is cheap (fast training, few
parameters to update) but limits how well the features can adapt to *this* task's specific
signal. `finetune_v0` **unfreezes everything** and updates all 86.6M parameters during
training. That one change took probe_AuDET from 0.061 (frozen) to 0.000002 (fine-tuned) and
public LB from 0.307 to 0.007 — freezing, not the ViT architecture itself, turned out to be
the real bottleneck (see `docs/finetune.md`'s "decisive experiment").

---

## Data layout

```
data/
  train_labels.csv                     # id, image_path, label, is_digital, type
  train/train/<id>.jpeg                # 69,352 training images
  public_test/public_test/<id>.jpeg    # 7,821 local test images (142,818 total ids in sample_submission)
  sample_submission.csv                # id,label — full test id list; label is a placeholder
```

Key facts:
- **5 document types in train**: `EGYPT/DL`, `GUINEA/DL`, `BENIN/DL`, `MOZAMBIQUE/DL`,
  `MAURITIUS/ID` (~13k–16k images each; `EGYPT/DL` is the largest and has a notably
  different ~50/50 fraud rate vs. the other four's ~40/60).
- **`is_digital`**: 69,332 True / **only 20 False** — training is 99.97% digital.
  Print-and-capture (photographing a printed document) is the attack the whole project is
  trying to generalize to, and it's almost entirely absent from training data.
- `image_path` in the CSV is unreliable (double-nesting); paths are always rebuilt from
  `id` as `{split}/{split}/{id}.jpeg`.
- This is a **code competition**: `sample_submission.csv` lists all 142,818 test ids, but
  only 7,821 images are shipped locally for development. The rest score on Kaggle's
  infrastructure at grading time.

---

## The model path (`model_type: baseline`, the default — no config flag needed)

```
FreuidDataset → build_transforms(image_size, train, augment="recapture") → build_model(backbone)
```

- **Dataset**: `FreuidDataset` (`src/freuid/data.py`) opens each JPEG with PIL, applies the
  transform pipeline below, and returns `(image_tensor, label)`.
- **Model construction** (`src/freuid/models/baseline.py`): `timm.create_model(backbone,
  num_classes=1)`. For `finetune_v0` specifically, `build_model` first tries
  `dynamic_img_size=True` — a timm flag that lets a ViT accept input resolutions other than
  the one it was pretrained at (needed so the same model can run inference at 476px, 518px,
  and 560px for test-time augmentation, see below) — and falls back to the plain call for
  backbones that don't accept that argument (CNNs like the earlier ConvNeXt baseline).
- **Zero-init head**: the classifier's weight and bias are set to exactly zero at
  construction. Since `sigmoid(0) = 0.5`, this guarantees the model outputs "50% fraud,
  50% genuine" for *any* input before any training happens — a clean, verifiable starting
  point (see "Sanity checks" below) rather than an arbitrary random guess.
- **Normalization**: `resolve_data_config(backbone, image_size)` reads the mean/std this
  specific backbone expects straight from `timm`'s own pretrained config (for
  `finetune_v0`: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)` — the standard
  ImageNet statistics, confirmed identical to what this repo already used elsewhere, so no
  transform changes were needed when swapping to this backbone).
- **Transforms** (`build_transforms` / `recapture_transforms`, see next section):
  **no horizontal flip anywhere** — document text and layout carry meaning, so mirroring
  would create an invalid, nonsensical input. This is a hard invariant of the whole project.

---

## Training-time augmentation: simulating the "analog hole"

The real test set includes **print-and-capture attacks**: someone prints a (possibly
forged) digital document and re-photographs it. That process destroys the pixel-level
digital compression noise a naive detector might rely on, and — per the point above —
training data barely contains any real examples of it. `recapture_transforms`
(`src/freuid/augment.py`) manufactures that degradation synthetically, applied to *every*
training image, in an order matching the real physical chain:

`resize → JPEG-compress → downscale → JPEG-compress again → blur (focus or motion) → sensor
noise → brightness/contrast/hue shift → mild perspective warp + small rotation`

(again: **never a flip**). This is also the basis of the "recapture probe" used for
checkpoint selection, described below.

**Synthetic tamper positives** (`SynthTamperWrapper`, gated by `extra.synth_tamper_prob:
0.3`): real fraud examples are relatively scarce and 99.97%-digital, so 30% of bona-fide
training images per epoch are converted into synthetic fraud examples by one of three edits
— `copy_move` (clone-paste a patch within the same image), `field_smudge` (blur/recolor/
white-out a document-field-shaped region), or `local_splice` (paste a patch from a donor
bona-fide image). The tampered result is always relabeled fraud and always routed through
`recapture_transforms`, so the model only ever sees a synthetic tamper *through* the analog
hole — matching how a real reprint-and-recapture attack would actually look.

### What this actually looks like

Generated by `scripts/analysis/pipeline_examples.py` (real dataset images — output goes to
`reports/pipeline_examples/`, gitignored; open the PNGs locally to view them, they aren't
committed — see the licensing note at the end of this doc).

**`recapture_examples.png`** — the same source image, original vs. three independent draws
of `recapture_transforms`. It's stochastic (random blur strength, JPEG quality, noise
level, etc. are re-sampled every call), so no two draws look identical — this is
deliberate: the model should learn to be robust to *a distribution* of degradation, not one
fixed recipe. Notice the barcode gets visibly blockier/noisier and the overall image loses
crispness, while the document's actual content and layout stay intact.

**`tamper_examples.png`** — one bona-fide source image, original vs. each of the three
synthetic tamper types, with the edited region boxed in green: `copy_move` (a patch cloned
from elsewhere in the *same* image — notice it blends in almost seamlessly, since the
pasted pixels share the same noise/lighting/texture statistics as the rest of the photo,
which is exactly why real copy-move forgeries are notoriously hard to detect), `field_smudge`
(a document field blurred/recolored/blanked-and-noised — usually much more visually
obvious than copy_move), and `local_splice` (a patch pasted in from a completely different
donor image — the clearest edit of the three, since it can carry a different lighting/color
cast).

**`tamper_then_recapture.png`** — the same three tampered images, additionally passed
through the full recapture chain. This is what the model *actually* receives as a training
input for synthetic positives — the raw tamper alone (previous figure) is never fed to the
model directly. Notice how much harder the edits are to spot once the recapture
degradation and (in these particular draws) a perspective warp are layered on top — a
faithful stand-in for what a real print-and-recapture attack would do to a forged document.

---

## Fine-tuning machinery specific to `finetune_v0`

Unfreezing an 86.6M-parameter pretrained transformer needs more care than training a
randomly-initialized head — naively applying one learning rate to every layer risks
destroying the pretrained features before the model learns anything useful. Three
techniques (`src/freuid/optim.py`) address this, all gated behind `extra.llrd` so no other
config is affected:

- **Layer-wise learning-rate decay (LLRD)**: instead of one learning rate for the whole
  model, each of the 12 transformer blocks gets its own, shrinking geometrically the deeper
  (closer to the input) you go: `lr(block_i) = base_lr × decay^(depth - i)`, with
  `decay=0.7` and `base_lr=1e-4` at the head. The intuition: early layers already encode
  generic, broadly-useful visual features (edges, textures) that don't need to change much;
  later layers are closer to task-specific semantics and should adapt faster. The
  patch-embedding/positional-embedding group gets the smallest rate of all (`decay^13 ≈
  9.7e-7`, roughly a thousandth of the head's rate).
- **Linear warmup, then cosine decay**: the learning rate ramps up linearly from 1% of
  `base_lr` to 100% over the first 2 epochs, then decays smoothly toward 0 following a
  cosine curve for the rest of training. Warmup exists because, right at the start,
  Adam-family optimizers' internal gradient-variance estimates are still unreliable — a
  full-strength learning rate applied immediately can take a bad first step that's hard to
  recover from. (Plain `baseline_v1`, which only trains a small CNN head, doesn't need this
  and doesn't use it — warmup only applies inside this gated path.)
- **Mixed-precision training (AMP)**: most of the arithmetic runs in 16-bit floating point
  instead of the usual 32-bit, roughly halving memory use and (on modern GPUs) enabling much
  faster matrix multiplies, while a "loss scaler" keeps gradients from underflowing to zero
  in the lower precision. This was added specifically because a full-precision fine-tune of
  this model measured ~1.7 seconds per training step — enabling AMP cut that to ~0.27
  seconds/step (a ~6.3x speedup), without which a 20-epoch run would have taken the better
  part of a day instead of ~4 hours.

Two more knobs exist in the config but weren't needed for this run (batch size 32 fit
comfortably in memory): `train_last_k_blocks` (freeze every block except the last *k*, a
cheaper fallback if a full fine-tune doesn't fit) and `grad_checkpointing` (trade extra
compute for lower memory by recomputing activations during the backward pass instead of
storing them).

---

## Validation split strategy

`build_loaders` calls `stratified_split(root, val_fraction=0.1, seed=42)` to produce
`(train_ids, val_ids)` as sets of id strings.

### Stratified split (the only split — the one `finetune_v0` uses)

`stratified_split(root, val_fraction=0.1, seed=42)` groups by `(label, type)` and samples
10% from each group. All 5 document types and both classes appear in both train and val.
Stratifying by `type` matters because the hidden test set probes generalization to
document types that differ from training — keeping every type represented in both train and
val keeps the val metric a fair (if in-domain) read on all of them at once.

**Important gap, found via post-hoc diagnostic analysis**: this split does *not* stratify
on `is_digital`. Since only 20 of 69,352 images are non-digital in the first place, one
run's 10% val draw contained just **2** non-digital images (both happened to be fraud —
too few, and only one class, to even compute an AuDET on that slice). Practically, this
means the near-perfect val AuDET this model achieves is a measurement of **in-domain
digital-fraud detection**, and says essentially nothing about generalization to genuinely
non-digital, print-and-recaptured images — see `reports/analysis_v0/README.md` for the full
diagnostic writeup.

---

## The "compass": the recapture probe

Per epoch, alongside plain validation, a **recapture probe** measures the real target
metric: apply the same `recapture_transforms` degradation chain (JPEG/downscale/blur/noise/
perspective/rotation) to a held-out clean split, and compute AuDET *through* that
degradation. **The checkpoint is saved on the lowest probe AuDET, not the lowest plain val
AuDET** — the whole point is to select the model that's robust to the analog hole, not just
the one that memorizes the easy in-domain case. (`finetune_v0`'s best checkpoint was epoch
13, with probe_AuDET = 0.000002.)

---

## Training loop (`src/freuid/train.py`)

```python
run_epoch(model, loader, device, criterion, optimizer=None, auc_weight=0.0, scaler=None)
```

- With `optimizer` → trains (gradients enabled). Without → evaluates (no gradients,
  collects scores/labels for metric computation).
- **Loss**: `BCEWithLogitsLoss` (standard binary cross-entropy on the raw logit), optionally
  plus a **pairwise soft-AUC term** (`extra.auc_loss_weight: 0.1`): for every (fraud,
  bona-fide) pair in a batch, a smooth penalty pushes the fraud score above the bona-fide
  score — directly optimizing *ranking*, which is what AuDET actually measures, rather than
  only optimizing calibrated probability.
- **Optimizer**: `AdamW`, either one flat learning rate (older configs) or the
  LLRD-decayed per-block groups described above (`finetune_v0`).
- **Scheduler**: linear warmup then `CosineAnnealingLR` (LLRD path) or plain
  `CosineAnnealingLR` from step 1 (older configs, unaffected by the LLRD change).
- **Checkpoint**: saved on the lowest `probe_audet` (recapture probe) to
  `checkpoints/<cfg.name>.pt`, storing `{"model": state_dict, "config": vars(cfg), "epoch":
  epoch, "metrics": m}`. Ties on the primary metric are broken by the matching
  APCER@1%BPCER, so two epochs with an identical (to float precision) AuDET don't
  checkpoint arbitrarily.
- **Sanity checks** (now wired in, unlike the older overlay-path doc): (1) **init-loss
  check** — assert BCE loss on the very first batch is within tolerance of `ln(2) ≈
  0.693`, which is exactly what a zero-init head *must* produce (sigmoid(0)=0.5 for every
  sample) — a quick way to catch label-scale bugs or a mis-wired loss before wasting compute
  on a real run. (2) **single-batch overfit** — train a disposable copy of the model on one
  batch for a few hundred steps and confirm the loss collapses toward 0, proving the
  forward/backward pass isn't silently broken and the model has enough capacity. (Note: the
  built-in check hardcodes plain SGD, which turns out to diverge on a full ViT fine-tune for
  unrelated optimizer-stability reasons — see `docs/finetune.md`'s ambiguity #3 for how this
  was diagnosed and worked around.)

---

## Inference & test-time augmentation (`src/freuid/infer.py`)

- `backbone`, `image_size`, and every `extra.*` flag are read from the **checkpoint's stored
  config**, never a separate inference config — preprocessing always matches the exact
  weights that were trained, by construction.
- **Multi-scale TTA, rank-averaged**: the same image is scored at three resolutions (for
  `finetune_v0`: 476px, 518px, 560px — all multiples of the ViT's 14px patch size). Instead
  of averaging the three raw scores, each scale's scores are first converted to **fractional
  ranks** (this image's position among all images, from 0 to 1) and *those* are averaged.
  Why: AuDET only cares about ordering, and different scales can have different implicit
  score "calibration" — averaging ranks is immune to that, while averaging raw scores isn't.
- Any test id with no local image (this is a code competition — most test images aren't
  shipped locally) defaults to `missing_id_score: 0.5` — deliberately rank-neutral, never
  0.0 (a real fraud sample silently scored 0.0 would badly damage AuDET).
- `check_submission()` prints an integrity report after every run: row count, number of
  unique scores, count of exact-zero scores (should be 0), min/max.

---

## Metrics (`src/freuid/metrics.py`)

```python
evaluate(scores, labels)  # → {"audet": float, "apcer_at_1pct_bpcer": float}
audet(scores, labels)     # = 1 - roc_auc_score(labels, scores)
apcer_at_bpcer(scores, labels, bpcer_target=0.01)
```

- **AuDET** = `1 - ROC AUC`. A perfect ranker scores 0; a random one scores ~0.5. This repo
  uses `1 - roc_auc_score` as a **linear-axis proxy** for the true area-under-DET-curve
  metric — not guaranteed byte-identical to Kaggle's official scorer (which may integrate
  on a different axis), but a reliable stand-in for relative ranking between models.
- **APCER@1%BPCER**: fix the decision threshold so only 1% of genuine documents would be
  wrongly rejected (BPCER=1%), then measure what fraction of real fraud still passes as
  "clean" at that threshold (APCER). This is the metric closest to a real deployment
  decision: "if we accept a 1-in-100 false-alarm rate on real customers, how much fraud
  still gets through?"

---

## Config dataclass (`src/freuid/config.py`) + `extra` knobs used by `finetune_v0`

| Field | `finetune_v0` value | Notes |
|---|---|---|
| `name` | `finetune_v0` | Sets the checkpoint filename |
| `backbone` | `vit_base_patch14_reg4_dinov2.lvd142m` | DINOv2 ViT-B/14, timm tag, Apache-licensed, no gating |
| `image_size` | `518` | Must be a multiple of the patch size (14) |
| `epochs` | `20` | |
| `batch_size` | `32` | Fits comfortably with AMP; no memory fallback needed |
| `lr` | `1.0e-4` | The LLRD **head** rate — deeper layers get less, see above |
| `weight_decay` | `5.0e-2` | |
| `seed` | `42` | Same seed as the CNN baseline it's compared against |

`extra` (unknown top-level keys land here; used for anything not in the core dataclass):

| Key | Value | Meaning |
|---|---|---|
| `augment` | `recapture` | Enables the analog-hole augmentation chain |
| `synth_tamper_prob` | `0.3` | Fraction of bona-fide images converted to synthetic fraud per epoch |
| `auc_loss_weight` | `0.1` | Weight of the pairwise soft-AUC loss term |
| `use_recapture_probe` | `true` | Enables the per-epoch recapture-degraded compass metric |
| `checkpoint_metric` | `probe_audet` | Checkpoint on the probe, not plain val AuDET |
| `tta` | `[476, 518, 560]` | Explicit multi-scale list (patch-size-14 multiples) |
| `llrd.enabled` | `true` | Turns on layer-wise LR decay + warmup (see above) |
| `llrd.decay` | `0.7` | Per-block LR multiplier |
| `llrd.warmup_epochs` | `2` | Linear warmup length |
| `train_last_k_blocks` | `null` | Freeze-fallback, unused this run (full fine-tune) |
| `grad_checkpointing` | `false` | Memory/compute tradeoff, unused this run |
| `amp` | `true` | Mixed-precision training (see above) |
| `missing_id_score` | `0.5` | Inference default for genuinely-absent test images |

---

## Key invariants

- **Label 1 = fraud, 0 = bona-fide** everywhere (dataset, loss target, metric).
- **No horizontal flip**, ever, anywhere in the pipeline — documents carry orientation.
- The backbone is **fully trainable** in `finetune_v0` (unlike the frozen `consistency_*`
  path) — every one of the 86.6M parameters gets a gradient update, just at different
  learning rates per layer.
- AMP is **off by default**; only `finetune_v0` (and future configs that explicitly opt in
  via `extra.amp: true`) use it — a disabled `GradScaler` is a documented no-op, so this
  can't silently change behavior for other configs.
- Checkpoint stores the **full config** so `infer.py` always rebuilds the exact model and
  preprocessing that produced the saved weights.

---

## What is the model actually looking at? (attention/attribution audit)

`finetune_v0` reaches probe_AuDET = 0.000002 — but a model can reach a great *number* for
the wrong reasons (e.g. keying on a background pattern or a file-metadata quirk that
happens to correlate with the label in this dataset, rather than real forgery evidence).
Two follow-up analyses probed this directly, both under `reports/analysis_v0/` (aggregate
plots there are committed to git; anything overlaying a real dataset image is gitignored —
generate them locally with the scripts referenced below to view).

**Does the model need the actual document, or just *an* image?**
(`reports/analysis_v0/README.md`, "degradation robustness curves") The val set was scored
under one input corruption at a time. Cropping away the *background* and keeping only the
document (`card_only_70pct_crop`) left AuDET essentially unchanged (≈0). Masking out the
*document* and keeping only a thin background border (`center_masked_15pct_border`) collapsed
AuDET to 0.403 (near-random) and let 98% of fraud slip through. **The model needs the real
document content to make its decision — it is not keying on background or framing.**

**Does it point at the actual edited pixels?** (`reports/analysis_v0/attn/README.md`) Four
attribution methods were implemented for a given image, all overlaid on it:

1. **CLS-to-patch attention** (last 3 transformer blocks, mean over heads) — what the CLS
   "summary" token attended to most when forming its final representation.
2. **Grad-CAM** — the classic CNN class-activation technique, adapted to a ViT.
3. **Occlusion sensitivity** — model-agnostic and the most trustworthy of the four: slide a
   gray patch over the image and record how much the fraud score drops at each position.
4. **Patch-token PCA-to-RGB** — a DINOv2-style visualization comparing what the pretrained
   vs. fine-tuned backbone "sees" in its raw patch features, side by side.

These were run on 200 synthetic tampers (ground-truth edit location known) plus 20 clean
bona-fide controls, scoring **IoU@best-threshold** (does the map's high-confidence region
overlap the true edit?) and **pointing-game accuracy** (is the map's single highest-scoring
pixel inside the true edit?). Individually-inspected example figures (in
`reports/analysis_v0/attn/figures/`, regenerate via `python scripts/analysis/visualize.py`)
show striking alignment — the ground-truth box sits right on top of the hottest region
across all three methods. But the full 200-sample average is more modest: Grad-CAM's
pointing-game hit rate (8.0%) is barely above the ~5.2% random-chance baseline; occlusion
sensitivity (27.5%) is meaningfully better than chance (~5x) but still misses most of the
time. When either method *does* hit the target, though, it does so with real confidence, not
noise (a much higher "concentration" score on hits than misses) — evidence the localization
signal is real, just inconsistent across samples. The likely (not yet confirmed) explanation
is that the three synthetic tamper types differ hugely in real-world detectability —
`copy_move` clones a patch from the *same* image, sharing its noise/color statistics, a
classically hard-to-detect forgery, while `field_smudge`/`local_splice` tend to be visually
obvious — and all the individually-inspected examples happened to be the easier types.

**Bottom line**: not a background/shortcut story (both analyses independently rule that
out), but not pixel-precise forensic localization either — the model's fraud signal likely
comes from a mix of genuinely-localized evidence (confirmed present, just not universal)
and more distributed, whole-document reasoning.

---

## Known issues / open risks

| Issue | Impact |
|---|---|
| Stratified split doesn't control for `is_digital` | Val AuDET (and the recapture probe, which is *derived* from val) can't actually confirm generalization to real non-digital/print-and-capture images — only 2 such images exist in this run's val split |
| A real, if weak, metadata-only shortcut exists in the dataset | Fitting a model on nothing but width/height/file-size/color-channel statistics (no document content) reaches AuDET≈0.10 — far worse than `finetune_v0`'s 0.000002, but confirms the dataset isn't perfectly "clean" of superficial correlates |
| Attribution maps only partially localize tamper regions, and Grad-CAM underperforms occlusion sharply | See "What is the model actually looking at?" above and `reports/analysis_v0/attn/README.md` for the full breakdown — trust occlusion sensitivity over Grad-CAM for this architecture |
| The built-in single-batch-overfit sanity check hardcodes plain SGD | SGD at a flat high learning rate diverges to NaN on a full ViT fine-tune (a known instability for deep transformers); confirmed via an AdamW-based diagnostic that gradient flow and capacity are actually fine — an optimizer-choice mismatch in the shared check, not a real bug |
| VESSL workspace's Python environment and `/tmp` can reset mid-session | Operational risk for long unattended jobs, not a model/data risk — long-running analysis scripts in `scripts/analysis/` are checkpointed/resumable specifically because of this |

---

## A note on the example images in this doc

The illustrative images referenced above (`reports/pipeline_examples/`,
`reports/analysis_v0/attn/figures/`, `reports/analysis_v0/borderline/`) are all real
competition images, which are under a non-commercial license — this repository is public,
so none of them are committed to git (`.gitignore` excludes each of those directories).
Regenerate them locally with `scripts/analysis/pipeline_examples.py` and
`scripts/analysis/visualize.py` to view.
