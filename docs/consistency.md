# Consistency pipeline — current state (as of 2026-07-03)

Written to bring a fresh Claude session with **zero prior context** up to
speed on this project — the challenge, why we're building what we're
building, the staged plan, the architecture, and where results currently
stand — without needing to re-read the whole codebase or conversation
history. See `ROADMAP.md` for the authoritative, continuously-updated results
table and gate checklists, `docs/problem.md` for the full S3-regression
root-cause writeup this doc summarizes, `docs/competition.md` for the full
competition brief, and `CLAUDE.md` for the canonical project instructions
(read that one first if it's available — this doc is a supplement to it, not
a replacement).

## The challenge, in one paragraph

**FREUID Challenge 2026** (IJCAI-ECAI 2026, organized by Microblink) is a
binary fraud-detection task on identity-document images (driver's licenses,
ID cards). Given a document photo, output a continuous **fraud score**
`P(fraud) ∈ [0,1]` — never a hard label — where **1 = fraud, 0 = bona-fide**.
The primary metric is **AuDET** (area under the Detection Error Trade-off
curve; this repo's proxy is `1 - roc_auc_score`), **lower is better**; the
secondary metric is **APCER @ 1% BPCER** (attack pass-rate when the
bona-fide-rejection budget is fixed at 1% — the production-relevant slice of
the curve). Both are **rank metrics**: only the relative ordering of scores
matters, not their calibration. The training data is drawn from a small set
of known document types and is **~99.97% digital** (clean, unaltered
photos/scans), but fraud in the wild comes in three flavors the test set
actually exercises: **physical tampering** (real documents physically
altered then re-photographed), **GenAI multimodal edits**, and — the one this
whole pipeline is built around — **print-and-capture** ("the analog hole"):
a digital forgery is printed and then re-photographed/re-scanned, which
destroys the pixel-level digital noise that naive forensic-noise detectors
rely on. The test set also includes **document types never seen in
training**, forcing cross-domain generalization rather than memorization of
5 known layouts. See `docs/competition.md` for the full brief (prizes,
timeline, rules, dataset licensing).

**Why "rebuild"?** An earlier forensic-noise model scored ~0.0006 AuDET
locally but only ~0.377 on the public leaderboard — it had learned to key on
digital compression/sensor noise patterns that a simple print-and-recapture
attack erases entirely, and its validation split was an easy in-domain
holdout that never exercised that failure mode. This whole codebase is a
ground-up rebuild whose explicit goal is to be robust to the analog hole and
to generalize across document types, validated in a way that would have
caught the earlier model's blind spot before wasting a submission on it.

## The staged roadmap (S0 → S4)

The build strategy is deliberately incremental: get a simple, fully-correct
pipeline (data loading, validation, augmentation, submission format) working
first, submit it, and only then add sophistication — one change at a time,
each stage trustworthy and submittable before starting the next. This is the
plan as laid out in `CLAUDE.md`:

| Stage | Label | Key change | Status |
|---|---|---|---|
| S0 | `baseline_v0` | Single pretrained CNN (EfficientNetV2-S), full image, recapture augmentation, probe-based checkpointing | done, submitted |
| S1 | `baseline_v1` | Stronger CNN (ConvNeXt-Small) + synthetic analog-tamper positives + ranking-aware (soft-AUC) loss + test-time augmentation | done, submitted — **best result so far** |
| S2 | `consistency_v0` | Swap in a **frozen DINOv3/DINOv2 ViT backbone** with a light trainable head on top (global CLS-token feature only), everything else held fixed for a clean comparison | done, submitted — regressed vs. S1 |
| S3 | `consistency_v1` | Add a **patch self-consistency head** and a **face-region consistency head** on the frozen patch features, fused with the global feature into one score | done, submitted twice (buggy, then a gated-fusion fix) — regressed vs. S2, gate not cleared |
| S4 | `ensemble_v0` | (not started) Second frozen backbone rank-averaged in, multi-seed/fold averaging, optional light fine-tuning of the backbone's last blocks | not started |

**The gate to advance a stage**: a new stage must beat the constant-0.5
baseline (the trivial "no information" submission) **and** beat the previous
stage on both the local **probe_AuDET** (see Validation below) and the
**public LB FREUID score** (a harmonic mean of the two competition metrics —
see `ROADMAP.md` header for the exact formula). If a stage doesn't clear the
gate, we don't build on top of it until we understand why and fix it (or
decide to roll back). **As of this doc, we are stuck at the S3 gate** — see
Results below.

Semantic/OCR features and heavier ensembling are explicitly deferred to
*after* the consistency architecture (S2/S3) is validated — the bet behind
this whole plan is that patch/face-level consistency signals, not bigger
models or more features, are what will generalize to the analog-hole and
unseen-document-type test conditions. That bet has not yet paid off in the
results (see below), which is the central open problem right now.

## Where we are right now

We're mid-way through the staged plan: S0/S1 (CNN baselines) are done and
submitted, with **S1 the best-performing submission to date** (public LB
0.181). S2 (frozen backbone, global feature only) is done and submitted but
**lost to S1** on every signal. **S3 (+ patch self-consistency + face-region
heads) has regressed twice** relative to S2 — once with a bug (public LB
0.618), once with a targeted gated-fusion fix that closed part of the gap but
not all of it (public LB 0.475, submitted 2026-07-03). We have not yet
cleared the S3 gate, meaning the core bet of this project (consistency heads
> bigger/frozen backbones alone) is currently unproven and the CNN baseline
from S1 is still the strongest submission we have.

## Pipeline architecture

### Two model paths coexist (`cfg.extra.model_type`)

1. **`baseline`** (default) — full-image CNN. `timm.create_model(backbone,
   num_classes=1)`, standard resize/normalize transforms. This is S0/S1
   (`baseline_v0`, `baseline_v1`). See `src/freuid/models/baseline.py`.
2. **`consistency`** — frozen DINO backbone + trainable head. This is S2/S3
   (`consistency_v0`, `consistency_v1`). See `src/freuid/consistency_model.py`
   and `src/freuid/backbone.py`.

Both paths share the same `train.py` loop, metrics, probe, TTA, and
submission-integrity machinery — only dataset/model construction branches on
`model_type`.

### Consistency model (`ConsistencyNet`)

```
frozen DINO backbone --forward_features()--> {cls, patch_tokens, grid_hw}
                                                      |
                                              ConsistencyHead
                                                      |
                     concat(global [, patch*gate] [, face*gate]) -> FusionMLP -> 1 logit
```

- **Backbone**: DINOv3 ViT-B/16 (`dinov3_vitb16`, gated on HuggingFace — needs
  `HF_TOKEN`; loaded via `transformers` since `torch.hub` needs `torchmetrics`
  which isn't installed on the workspace). S2 used DINOv2 ViT-B/14 instead
  (Apache-licensed, no gating). **Always frozen**: `requires_grad=False`,
  `eval()`. Checkpoints store only head weights; the backbone is reloaded from
  hub/HF at inference time (`ConsistencyNet.state_dict()` overridden to return
  `self.head.state_dict()` only).
- **Global branch** (S2 and S3): CLS token -> `LayerNorm`. Always on.
- **Patch branch** (`PatchConsistencyHead`, S3 only, `use_patch_consistency`):
  a learned `[outlier]` query prepended to the patch-token sequence, run
  through a 2-layer `TransformerEncoder` (8 heads), take the query's output
  position, `LayerNorm`. Intent: let the query attend to whichever region(s)
  look least consistent with the rest of the card (the analog-robust signal
  we actually want — tamper/recapture artifacts show up as local
  inconsistency, which should survive reprinting better than global forensic
  noise). Active on **100% of samples**, ~10.1M params — by far the largest
  and most-exposed new branch.
- **Face branch** (`FaceRegionHead`, S3 only, `use_face_region`): pools patch
  tokens inside vs. outside a cached face bounding box (fractional coords in
  canonical card space), computes `[LayerNorm(inside_mean - outside_mean),
  cosine_similarity(inside_mean, outside_mean)]`, feeds through a small MLP,
  multiplies by a `valid` flag (zeroed when there's no real SCRFD detection —
  only **19.6%** of samples have one; the rest fall back to a center-square
  box and are correctly zeroed out, not fed a wrong signal).
- **Fusion**: `FusionMLP` = `Linear -> ReLU -> Dropout -> Linear`, final layer
  **zero-initialized** so the model starts at logit≈0 (BCE≈ln2) regardless of
  which heads are active.
- **Per-branch gates** (added in the S3 fix, commit `aa17dd0`): patch and face
  embeddings are each multiplied by a learnable per-channel gate
  (`nn.Parameter`, LayerScale-style) initialized to `1e-3` before entering the
  fusion concat. Intent: start the model mathematically close to
  global-only behavior and let training "open" each new pathway only once it
  earns its keep, instead of a noisy/unconverged branch contaminating
  `FusionMLP.fc1`'s shared gradients from step 1. The global branch is *not*
  gated (it's the already-proven S2 signal).

Each head is independently toggleable via config flags — the architecture
supports ablation (`global_only`, `patch_only`, `face_only`, `fusion_all`).

### Preprocessing pipeline (`src/freuid/preprocess.py`)

Runs **once, single-process, before training** (detectors are not fork-safe;
never run inside a multiprocessing `DataLoader`). Cache layout:
`data/processed/regions/{id}/{card.png, face.json}`.

1. **Card rectification (FastSAM)**: segment the largest quadrilateral in the
   raw photo, `warpPerspective` to a canonical 512×512 `card.png`. Falls back
   to a plain resize if FastSAM is unavailable or finds no quad. This is what
   `use_rectify: true` reads from at train/infer time — both training and
   inference use the rectified card, not the raw photo, when enabled.
2. **Face detection (SCRFD via InsightFace)**: run on the *rectified* card,
   take the highest-confidence portrait box (`x1,y1,x2,y2,score` in canonical
   card pixel coords). Falls back to a center square (`0.6 * min(H,W)`,
   `score=0`) if SCRFD is unavailable or finds nothing — this fallback is what
   `face_meta`'s `valid` flag is designed to suppress downstream.

`face_meta` tensor layout: `[x1_frac, y1_frac, x2_frac, y2_frac, valid]`,
fractions in canonical card space. `valid=0` for cache-miss, disabled heads,
or a fallback/center-square box (SCRFD `score==0`).

### Training-time augmentation (`src/freuid/augment.py`)

Two independent, composable pieces, both gated by config (`cfg.extra.augment`,
`cfg.extra.synth_tamper_prob`) so old configs are unaffected:

1. **`recapture_transforms`** — simulates the print-and-capture ("analog
   hole") chain on *every* training image: resize -> JPEG compress -> downscale
   -> re-JPEG-compress -> optical blur (Gaussian/motion) -> sensor noise ->
   brightness/contrast/hue shift -> mild perspective warp + small rotation
   (**never a flip**) -> normalize. This is both the training augmentation
   and the basis of the **recapture probe** (see Validation below).
2. **`synth_tamper` / `SynthTamperWrapper`** — manufactures synthetic fraud
   positives from clean bona-fide images (since real fraud in training is
   scarce and 99.97% digital). Per bona-fide sample, with probability
   `synth_tamper_prob` (0.3 in `baseline_v1`/`consistency_v1`), applies one of:
   - `copy_move` — clone-paste a rectangular patch within the same image
   - `field_smudge` — blur / recolor / white-out-and-noise a document-field-shaped region
   - `local_splice` — paste a patch from a donor bona-fide image (donor pool:
     200 pre-loaded images, sampled once at dataset init)
   The tampered image is always relabeled `1` and always routed through
   `recapture_transforms` (the tamper is only ever seen "through the analog
   hole", matching how a real reprint-and-recapture attack would look).
   Untampered samples go through the clean transform.

### Validation & the recapture probe

Per `CLAUDE.md`: standard single-domain LODO saturates and is not trusted
alone. The actual per-epoch compass is the **recapture probe**
(`use_recapture_probe: true`, `recapture_probe_seed: 1234`): a held-out clean
split has `recapture_transforms` applied and AuDET is measured on it every
epoch; **checkpointing selects on lowest `probe_AuDET`**, not lowest val
AuDET. `checkpoint_metric: probe_audet` in config controls this.

Sanity checks (both currently wired in for the consistency path):
- init BCE ≈ ln(2) ≈ 0.6931 on a balanced batch (`FusionMLP`'s zero-init final
  layer guarantees this regardless of how many heads are active)
- single-batch overfit -> ~0 (confirmed 0.0037 for `consistency_v1`)

### Inference (`src/freuid/infer.py`)

- `backbone` / `image_size` / `model_type` / all `extra.*` flags are read from
  the **checkpoint's stored config**, never from a separate inference config
  — guarantees preprocessing always matches the trained weights.
- TTA: multi-scale (`extra.tta: [448, 512, 528]` for `consistency_v1` — must
  be multiples of DINOv3's `patch_size=16`), **rank-averaged** (not
  score-averaged, since AuDET is a rank metric and rank-averaging is immune to
  per-scale calibration drift), no flip.
- Any test id with no local image is scored `missing_id_score` (default 0.5,
  never 0.0) — real fraud silently defaulting to a 0.0 score would tank AuDET.
- `check_submission()` prints an integrity report (row count, unique-score
  count, exact-zero count, min/max) after every inference run.

## Results so far (see `ROADMAP.md` for the live table)

All AuDET-family numbers are **lower = better** (0 = perfect ranking, 0.5 =
random/no-information). `probe_AuDET` is measured on a held-out clean split
with the print-and-capture augmentation applied (the analog-robustness
compass — see Validation below); `val_AuDET` is the plain in-domain
validation AuDET; `public LB FREUID` is Kaggle's public-leaderboard score, a
harmonic mean of AuDET and APCER@1%BPCER that penalizes whichever metric is
weaker (formula in `ROADMAP.md`'s header) — a constant-0.5 submission scores
FREUID ≈ 1.0, so lower is strictly better here too.

| Stage | Config | Backbone | probe_AuDET (best) | val_AuDET | public LB FREUID |
|---|---|---|---|---|---|
| S0 | baseline_v0 | EfficientNetV2-S | 0.000000 | — | 0.27106 |
| S1 | baseline_v1 | ConvNeXt-Small | 0.000039 | 0.0000 | 0.18129 |
| S2 | consistency_v0 | DINOv2 ViT-B/14 (frozen) + GlobalHead | 0.061290 | 0.0280 | 0.30743 |
| S3 (buggy) | consistency_v1 | DINOv3 ViT-B/16 (frozen) + Global+Patch+Face | 0.117463 | 0.1136 | 0.61753 |
| S3 (gated fix) | consistency_v1 (rerun) | same, + LayerScale gates + FaceRegionHead LayerNorm | 0.115286 | 0.1020 | 0.47469 |

Two things stand out and are worth internalizing:

1. **S2 lost to S1 on both signals** — probe_AuDET (0.061 vs 0.00004) *and*
   public LB (0.307 vs 0.181). The frozen-backbone global-only feature is not
   yet pulling its weight relative to the CNN baseline. The plan's bet is
   that S3's patch/face consistency heads are what should unlock the frozen
   backbone's advantage (analog robustness) — that bet has not paid off yet.
2. **S3 made things worse, twice, but the gated fix partially recovered it.**
   The original run (naive concat fusion, no gating, missing LayerNorm in
   `FaceRegionHead`) regressed hard on every local and LB signal vs. S2. The
   fix (per-branch LayerScale-style gates + the missing LayerNorm) was
   applied and retrained for the full 20 epochs. Locally the move was small —
   probe_AuDET only went from 0.1175 to 0.1153 (~2% relative) — but **public
   LB improved substantially: 0.61753 → 0.47469 (~23% relative)**. The public
   LB response being much larger than the local-probe response is itself a
   signal worth noting: probe_AuDET (recapture-degraded val split) may not be
   fully capturing whatever public LB is picking up on (unseen document
   types? the analog-hole distribution differs from the recapture aug?).
   Regardless, **S3 still hasn't cleared the gate** — 0.475 is worse than both
   S2 (0.307) and S1 (0.181) on public LB. See Open questions for next steps.

## Diagnosis carried over from `docs/problem.md`

Full writeup with code excerpts is in `docs/problem.md`; summary of the
ranked findings:

1. **(primary, partially addressed)** Naive concat fusion has no
   built-in "stay a no-op until useful" guarantee — `FusionMLP.fc1` mixes all
   branches together from gradient step 1, so a noisy/unconverged
   `PatchConsistencyHead` (100% sample coverage, ~65x S2's entire head in
   param count) can drag the good global signal down through the shared
   layer. **Fix applied**: per-branch gates initialized near-zero
   (LayerScale-style). Result: partial recovery — public LB improved 0.618 →
   0.475 (~23% relative) but local probe_AuDET barely moved (0.1175 → 0.1153,
   ~2%). Gap to S2 (0.061 probe / 0.307 LB) still open on both signals. The
   local/public mismatch means either the gates aren't opening (branches stay
   suppressed, contribute little) or they open but the *recapture probe*
   specifically doesn't reward it the way public LB does — gate values have
   not yet been inspected to tell which.
2. **(confirmed bug, fixed)** `FaceRegionHead`'s output wasn't
   scale-normalized like the other two branches (no final `LayerNorm`) — now
   fixed (`self.out_norm = nn.LayerNorm(hidden)` added in `aa17dd0`).
3. **(structural limit, not a bug)** Only 19.6% of samples get a real SCRFD
   face detection; the rest are correctly zeroed via the `valid` flag. Caps
   the face branch's practical value regardless of correctness — worth
   revisiting whether SCRFD is well-suited to the small, stylized ID-card
   portraits in this dataset.
4. **(not yet applied)** `weight_decay=5e-2` is applied uniformly to all
   trainable params including `LayerNorm` weights and biases — standard
   practice excludes these. Low-impact for S2's tiny head, more relevant now
   that `PatchConsistencyHead`'s `TransformerEncoder` adds many more
   LayerNorm/bias params. Proposed but not implemented.

## Open questions / likely next steps

Not decided yet — flagging for whoever picks this up:

- **The gate fix is confirmed insufficient**, now that public LB is in
  (0.475, still worse than S2's 0.307). The interesting sub-question is *why*
  the public LB gain (23% relative) was so much larger than the local probe
  gain (2% relative) — that gap suggests probe_AuDET (recapture-degraded val
  split) isn't a fully faithful stand-in for whatever public LB is testing.
  Worth checking whether probe_AuDET and public LB agree in *direction* more
  often than they agree in *magnitude* across all submitted runs so far.
- **Inspect the learned gate values** after training (`patch_gate`,
  `face_gate` in `ConsistencyHead`) to see whether they opened meaningfully
  above their `1e-3` init or stayed suppressed — this would distinguish
  "gating worked but the branches genuinely aren't useful yet" from "gating
  itself isn't functioning as intended."
- **Ablation re-run** (`global_only` vs `patch_only` vs `face_only` vs
  `fusion_all`) with the gated architecture, to see if the S3 gate condition
  (`fusion_all probe_AuDET <= best single head`) is closer to passing now.
- **Weight-decay exclusion for LayerNorm/bias params** (finding #4 above) —
  still not implemented, cheap to try.
- Consider whether S2 itself (DINOv2 global-only, LB 0.307) needs to be
  revisited/re-baselined with DINOv3 before concluding S3's heads are the
  problem — S2 and S3 aren't on the same backbone (DINOv2 vs DINOv3), so part
  of the S2→S3 LB regression (0.307 → 0.618, original buggy run) is
  confounded by the backbone swap, not just the new heads.

## Where to look in code

| Concern | File |
|---|---|
| Consistency model architecture | `src/freuid/consistency_model.py` |
| Backbone loading (DINOv2/v3, frozen wrapper) | `src/freuid/backbone.py` |
| Baseline CNN model | `src/freuid/models/baseline.py` |
| Card rectification + face detection + caching | `src/freuid/preprocess.py` |
| Recapture aug + synthetic tamper | `src/freuid/augment.py` |
| Training loop, probe, checkpointing | `src/freuid/train.py` |
| Metrics (AuDET, APCER@1%BPCER) | `src/freuid/metrics.py` |
| Inference + TTA + submission integrity | `src/freuid/infer.py` |
| Config dataclass | `src/freuid/config.py` |
| Per-stage YAML configs | `configs/*.yaml` |
| S3 regression root-cause (detailed) | `docs/problem.md` |
| Results table, gate checklists | `ROADMAP.md` |
| Stage plan, invariants, restrictions | `CLAUDE.md` |
