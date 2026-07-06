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

## What happened (read this before proposing architecture)

The original plan was a **frozen-foundation, consistency-based detector**: frozen DINO backbone,
light trainable heads, patch/face self-consistency. That plan is **retired — the data killed it**:

| Stage                | Config           | Model                                          | probe_AuDET     | public LB   |
| -------------------- | ---------------- | ----------------------------------------------- | --------------- | ----------- |
| S0                   | `baseline_v0`    | EfficientNetV2-S, fine-tuned                   | 0.000000        | 0.27106     |
| S1                   | `baseline_v1`    | ConvNeXt-Small, fine-tuned, 384px              | 0.000039        | 0.18129     |
| S2                   | `consistency_v0` | DINOv2 ViT-B/14 **frozen** + head              | 0.061290        | 0.30743     |
| S3                   | `consistency_v1` | DINOv3 ViT-B/16 **frozen** + patch/face heads  | 0.115286        | 0.47469     |
| **current best ✓**  | `finetune_v0`    | **DINOv2 ViT-B/14, fully fine-tuned, 518px**   | **0.000002**    | **0.00744** |
| attempted, worse     | `finetune_v1`    | DINOv3 ViT-B/16, fully fine-tuned, 512px       | —               | 0.02695     |
| attempted, no change | `finetune_v2`    | DINOv2 ViT-B/14, fully fine-tuned, **784px**   | 0.000000 (ep18) | 0.00750     |

The decisive experiment (`docs/finetune.md`): same DINOv2 backbone as S2, fully fine-tuned with
S1's recipe (LLRD + warmup + AMP) → 24× better LB than S1, ~40× better than S2. **Freezing was
the problem, not the backbone.** Corroborating detail: finetune_v0's epoch 1 (backbone still
~unmoved during warmup) reproduced frozen S2's val_AuDET (0.028); the leap to 0.0000 happened the
moment backbone weights adapted. The S2/S3 consistency-heads bet was never fairly tested — the
frozen-vs-fine-tuned confound dominated. Consistency heads on a fine-tuned backbone remain an
open, low-priority question. The `model_type: consistency` path still exists and must keep
importing/running, but no new work goes there without a decision.

**Two follow-up experiments off finetune_v0, both negative — read before repeating them:**

- **`finetune_v1` (DINOv3 ViT-B/16 swap, otherwise identical recipe): the backbone swap alone
  made things ~3.6× worse** (0.00744 → 0.02695). Deprioritize DINOv3 for the primary path; the
  "unverified register-token slicing" caution below is now corroborated by a real regression,
  not just a theoretical risk.
- **`finetune_v2` (784px instead of 518px, otherwise byte-identical config, verified via
  `scripts/config_diff.py`): resolution increase was a wash** (0.00744 → 0.00750, within noise).
  This was **not** a wasted-effort upsampling test — three separate analyses confirmed 784px is
  a genuine, additive resolution increase, not interpolated noise: `resolution_census.md` found
  the native data ceiling is min-side ≈1000px (so 784 is real signal); `aug_audit.md` confirmed
  `recapture_transforms`'s resize/downscale steps are RELATIVE, not an absolute bottleneck;
  `vit_dryrun.md` confirmed it fits comfortably (bs=32, 39.9GB/80GB, ~7.3h/20ep). All three
  reports live under `reports/res_precheck/` (gitignored, regenerate via
  `scripts/analysis/resolution_census.py`, `aug_resolution_audit.py`, `vit_res_dryrun.py`).
  **Conclusion: resolution is not the lever that matters here** — capacity, ensembling, and
  loss-shaping toward the rank metric are more promising than further resolution tuning. Do not
  retry 1036px: the same analyses show it's pure upsampling (0% of images natively reach that
  size) *and* the compute cost is much worse (ViT-L@1036 doesn't even fit a 40h budget).

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
tokens, Apache-2.0, no HF gating), **fully fine-tuned** through the `model_type: baseline` path.
Fine-tune recipe (all gated in `cfg.extra`): LLRD `{enabled, decay: 0.7, warmup_epochs: 2}`
(head LR 1e-4, per-block decay, LayerNorm/bias excluded from weight decay — `src/freuid/optim.py`),
`amp: true`, `train_last_k_blocks` / `grad_checkpointing` as memory fallbacks (unused so far).

Scale-up candidate: `vit_large_patch14_reg4_dinov2.lvd142m`, same recipe. Feasibility confirmed
on the A100 (`reports/res_precheck/vit_dryrun.md`, regenerate via
`scripts/analysis/vit_res_dryrun.py`): L@518 fits at bs=32 with no checkpointing, ~7.7h/20ep;
L@784 needs bs=16 (OOMs at 32), ~21.6h/20ep — both fit a ≤40h budget. Not yet trained/submitted.
Diversity member: ConvNeXt (retrain at 518px + warmup before ensembling; S1's 0.181 checkpoint is
not ensemble-grade next to 0.0074). Cheapest diversity available is multi-seed rank-averaging of
finetune_v0's own recipe (no new architecture needed) — also not yet tried.

**DINOv3: deprioritized, not just "cautioned."** `finetune_v1` swapped in
`vit_base_patch16_dinov3.lvd1689m` (a timm tag, so the register-token-slicing risk below didn't
even apply) with an otherwise identical recipe and public LB regressed ~3.6× (0.00744 → 0.02695)
— a real result, not a theoretical risk. Do not retry a DINOv3 backbone swap without a specific
hypothesis for why it would help this time. The remaining caution below (about
`src/freuid/backbone.py`'s `transformers` loader) still applies to the *other*, unused DINOv3
loading path: it has **unverified register-token slicing** (audit with a patch-token PCA
visualization before trusting spatial features), and DINOv3 weights carry Meta's custom license
— check compatibility with our Apache-2.0 release before using in a ranked submission. Prefer
timm tags if using DINOv3 at all.

Preprocessing (consistency path only, currently parked): FastSAM (card rectification), SCRFD
(portrait box) — cached once, single-process, never inside a forked DataLoader.

Restriction: **do not use any model built for ID-fraud / presentation-attack detection, or
general forgery-localization nets** (TruFor, CAT-Net, MVSS-Net, PSCC-Net, Noiseprint).
Everything above is a general vision / segmentation / face model.

## Validation (the part that keeps biting us)

Per-epoch compass is the **recapture probe** (analog augmentation applied to a held-out clean
split; checkpoint on lowest probe AuDET). **Known limits, respect them**: the probe applies the
_training_ augmentation, so it is partially circular and is now **saturated** (finetune_v0:
2e-6; finetune_v2's best epoch hit exact `0.0`) — it can gate regressions but cannot rank strong
candidates. It has agreed with the LB in *direction* on every submission so far, so keep it, but
decisions between good models need the item-2 instruments (probe_v2 / unseen-type probe / real
printed set). Do not trust single-domain LODO on the easiest type — it saturates and predicts
nothing. Periodic multi-fold LODO (hold out each document type in turn) is the cross-domain
check. Sanity checks per the Invariants section.

**This gap is still open and is the real bottleneck on iterating.** With the probe saturated,
the *only* instrument sensitive enough to rank two good checkpoints against each other right now
is a Kaggle submission — slow and rate-limited. Two small instruments were built to start closing
this but neither is sufficient on its own:

- `scripts/analysis/nondigital_probe.py` scores the checkpoints against the only 20 *real*
  non-digital (print-and-capture) training images. Caveat that matters: 18/20 are in every
  checkpoint's *training* split (same seed=42 stratified split across all `finetune_*` configs)
  — only 2 are genuine holdouts, both fraud, both correctly and confidently flagged by all three
  checkpoints (0.978–1.000). Encouraging but n=2 is not a real validation signal; there are zero
  held-out non-digital *bona-fide* examples to check false-positive behavior at all.
- `scripts/analysis/hesitant_test_images.py` + `notebooks/hesitant_test_images.ipynb` pull the
  present-test ids whose rank-averaged score sits closest to 0.5 for manual inspection. Useful
  for spotting *systematic* error patterns (a document type, a tamper style) worth targeted
  augmentation — not a numeric ranking instrument by itself. Note: submission scores are
  rank-averaged, so their marginal distribution is close to uniform on (0,1) by construction
  (verified empirically: mean 0.50, std 0.28, only 2/7821 <0.01 and 31/7821 >0.99) — "score near
  0.5" means *median relative rank*, not necessarily *raw sigmoid ≈ 50/50*.

Building a real unseen-domain probe (item-2 instrument) remains the highest-priority validation
task before spending more compute on speculative architecture/resolution changes.

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

1. **Build the unseen-domain validation probe** (above) — everything below is hard to evaluate
   without it. Made more urgent, not just still-true, by `exp/bayar+dinov2`: that experiment's
   local probe hit an exact, saturated **0.0** and the public LB still came back ~2.9x *worse*
   than baseline. The local instrument didn't just lack resolution between two good candidates
   (its known limit) — it was completely blind to a real regression. Every remaining idea below
   faces the same blind spot until this exists.
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
   target with augmentation/curriculum changes.

### overlay_colab: a real blind spot found, three ensembles ruled out, feature-fusion is next

Manually inspecting the hesitant-images notebook's output (item 5 above) found a pattern: most
of finetune_v0's ~100 most uncertain present-test predictions visually look like **digitally
edited faces**. A teammate's earlier `feat/overlay-detector` branch (`TwoStreamOverlayNet`:
`BayarConv2d` forensic-noise stream + ResNet34 RGB stream, fused via MLP, operating on a
224px MTCNN face crop — awful standalone public LB, 0.377) was checked against exactly this
zone: **91-95% of finetune_v0's hesitant cases get a decisive, mostly-confident-fraud call from
overlay_colab, holding flat through rank 300** (not a top-100 fluke). Mechanistically this makes
sense — `feat/overlay-detector`'s data pipeline has **zero recapture/reprint augmentation**
(confirmed by grep), so its BayarConv2d noise signal was never trained to survive a reprint, but
still fires correctly on genuinely-digital (not-yet-reprinted) face manipulations.

**Three score-level combinations were tried and all made things worse** — do not repeat these:

| approach | public LB | vs. finetune_v0 (0.00744) |
|---|---|---|
| hard gate: override with overlay's raw score on ~305 hesitant ids | 0.01101 | ~1.5x worse |
| weighted rank-blend restricted to the hesitant 300 (had a local/global rank-scale bug) | 0.01659 | ~2.2x worse |
| weighted rank-blend (0.8/0.2) across all 7821 present ids | 0.05319 | ~7.1x worse |

Lesson: overlay_colab's *decisiveness* on the hesitant zone doesn't mean *correctness* — rank
metrics punish a confidently-wrong call (pushed to 0.99+) far more than they punish an
ambiguous 0.5, so any fixed-weight or fixed-override combination that can't tell which of its
calls to trust is a net loss.

**Follow-up (branch `exp/bayar+dinov2`), tried and also negative — read before repeating.**
Built `BayarFusionNet` (`src/freuid/models/bayar_fusion.py`): DINOv2's CLS embedding gated-fused
(LayerScale-style, near-zero-init `overlay_gate`) with a BayarConv2d-noise + RGB-ResNet34 branch
on a cached-SCRFD face crop, **fully fine-tuned end-to-end** with recapture augmentation applied
to both streams — the same experiment shape as S2/S3 but with both of S3's diagnosed root causes
fixed (frozen backbone → fine-tuned; missing gating → LayerScale gate from init). Config
`configs/bayar_dinov2_v0.yaml`, checkpoint best at epoch 18.

- **Local probe: perfect** (`probe_AuDET=0.0`, exact). **Public LB: 0.02146** — ~2.9x worse than
  finetune_v0 (0.00744), worse than every score-level overlay_colab combination too.
- **The gate was not dormant**: inspected the trained checkpoint directly — 100% of
  `overlay_gate`'s 640 elements moved >10x from their `1e-3` init (mean magnitude 0.062). The
  model genuinely learned to rely on the branch; that reliance is *why* it overfit, not evidence
  the branch stayed inert.
- **Root cause (most to least confident)**: (1) the extra branch is pure added capacity to fit
  the closed train/val/probe loop without that fit needing to generalize — contrast with
  finetune_v0, which *also* hits probe≈0 and still generalizes, so "perfect local score" isn't
  the problem, "an extra unconstrained pathway to a perfect local score" is; (2) `recapture_transforms`
  applied to the face-crop stream (JPEG recompress/blur/noise/downscale) is specifically built to
  simulate print-and-recapture degradation — which is also exactly the kind of degradation that
  erases the fine noise residue BayarConv2d depends on, so joint training under this augmentation
  likely undercut the branch's own intended signal while still leaving room to fit something
  locally predictive but non-generalizing; (3) secondary: SCRFD only finds a *valid* (non-fallback)
  face on ~1 in 5 documents overall (both splits — most ID-card portraits are too small/off-angle
  for a general-purpose detector), and the valid-detection rate differs by split (train 21.5% vs
  public_test 16.4%) — this alone is too small to explain a 2.9x regression, but it's a real,
  measured train/test distribution mismatch in how often the branch's shared fusion-layer weights
  see an actually-informative input. Confirmed *not* a scoring bug: TTA ranges were sane at all
  3 scales, submission integrity was clean, and the eval code path is identical to finetune_v0/v2's.
- **Verdict**: this is the second time a forensic/consistency-style second branch has looked
  great locally and failed to transfer (S2/S3, now this) — different failure mechanism each time,
  same outcome. Treat as a structural mismatch between forensic-noise features and this project's
  reprint-robustness training regime, not a bug to patch. **Do not retry this fusion family**
  (BayarConv2d/SRM-style noise-residual branches, jointly trained under recapture augmentation)
  without a specific new idea for resolving the augmentation-vs-noise-signal conflict — swapping
  the face detector (SCRFD→MTCNN) was considered and isn't expected to help, since the coverage
  gap was a minor factor and detector choice doesn't address the bigger two causes.

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
`BCEWithLogitsLoss` + pairwise soft-AUC term (S1 recipe, kept in finetune_v0). Log per epoch:
`lr_head, train_loss, val_loss, AuDET, APCER@1%BPCER, probe_AuDET`. Log dataset size and
per-class counts at train start. Keep all three paths importable and runnable: baseline CNN,
fine-tuned ViT (both `model_type: baseline`), and the parked `model_type: consistency`.

## Key docs

- `docs/finetune.md` — the decisive experiment (recipe, evidence, verdict)
- `docs/pipeline_finetune.md` — beginner-accessible walkthrough + attention/attribution audit
  (what is the model actually looking at — not a background shortcut, not pixel-precise
  forensic localization either; "a mix of genuinely-localized evidence... and more distributed,
  whole-document reasoning")
- `docs/problem.md` — S3 regression root-cause
- `docs/competition.md` — competition brief (metric, rules, licensing, fraud-type taxonomy)
- `docs/data_analysis.md` — train_labels.csv summary (label/is_digital/type breakdowns)
- `ROADMAP.md` — live results table and gate checklists
- `reports/res_precheck/{resolution_census,aug_audit,vit_dryrun}.md` — gitignored, regenerate via
  `scripts/analysis/{resolution_census,aug_resolution_audit,vit_res_dryrun}.py`; the evidence
  behind finetune_v2's 784px decision and the "don't retry 1036px" conclusion above
- `scripts/analysis/nondigital_probe.py`, `hesitant_test_images.py` +
  `notebooks/hesitant_test_images.ipynb` — the two small validation instruments described above
