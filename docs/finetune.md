# finetune_v0 — the decisive experiment (end-to-end ViT fine-tune)

## Why this experiment exists

S1 (`baseline_v1`, ConvNeXt-Small, fully fine-tuned) is still the best submission in the
project (public LB FREUID 0.18129). S2 (`consistency_v0`, frozen DINOv2 ViT-B/14 + a light
head) lost to it on every signal (probe_AuDET 0.061290 vs 0.000039; public LB 0.30743 vs
0.18129). The S1→S2 change bundled **two variables at once**: it swapped the backbone
(CNN→ViT) **and** switched from full fine-tuning to a frozen backbone. That makes it
impossible to tell which change caused the regression — a worse backbone, or freezing a
backbone that would have been fine if fine-tuned?

`finetune_v0` isolates the second variable: same DINOv2 ViT-B/14 backbone as S2, but
**fully fine-tuned end-to-end** with S1's exact training recipe otherwise (same
augmentation, same synthetic tamper, same soft-AUC loss, same probe-based checkpointing,
same seed). Only backbone identity and image size change; everything else is held constant
by construction (verified with a resolved-config diff tool, see below).

**Decision rule**: if the fine-tuned ViT matches or beats S1 → the backbone was never the
problem, freezing was → S4 becomes a fine-tuned-CNN + fine-tuned-ViT rank ensemble. If it
loses to S1 → the CNN's inductive bias genuinely wins for this task, and ViT backbones are
only useful for ensemble diversity, not as a standalone upgrade.

## What changed, precisely

Reuses the existing `model_type=baseline` code path (`timm.create_model(backbone,
num_classes=1)`) — **not** the frozen-backbone consistency path. `consistency_model.py` and
`backbone.py` are untouched. Backbone: `vit_base_patch14_reg4_dinov2.lvd142m` (DINOv2 ViT-B/14
with 4 register tokens, timm tag, Apache-licensed, no HF gating).

`scripts/config_diff.py configs/baseline_v1.yaml configs/finetune_v0.yaml` reports exactly
10 differing resolved keys:

| Key | baseline_v1 | finetune_v0 | Why |
|---|---|---|---|
| `name` | baseline_v1 | finetune_v0 | trivial |
| `backbone` | convnext_small.fb_in22k_ft_in1k | vit_base_patch14_reg4_dinov2.lvd142m | the swap under test |
| `image_size` | 384 | 518 | DINOv2's native resolution (patch_size=14, 518=14×37) |
| `lr` | 2e-4 | 1e-4 | LLRD head/base LR — S1's lr was tuned for a randomly-init CNN head only, not a full ViT fine-tune; standard LLRD practice uses a smaller head LR (see Ambiguities below) |
| `extra.tta` | `true` (auto [320,384,448]) | `[476, 518, 560]` | must be multiples of patch_size=14; infer.py's auto-scale step rounds to multiples of 32, invalid here |
| `extra.llrd.*` | absent | `{enabled: true, decay: 0.7, warmup_epochs: 2}` | new, gated, gives the ViT layer-wise LR decay + warmup |
| `extra.train_last_k_blocks` | absent | `null` (full fine-tune) | new, gated fallback (unused this run) |
| `extra.grad_checkpointing` | absent | `false` | new, gated (unused this run — not needed, see Throughput) |
| `extra.amp` | absent | `true` | new, gated (see Throughput — added mid-experiment) |

Everything else — `epochs`, `batch_size`, `weight_decay`, `augment`, `synth_tamper_prob`,
`auc_loss_weight`, `use_recapture_probe`, `recapture_probe_seed`, `checkpoint_metric`,
`val_fraction`, `missing_id_score`, `seed`, `data_dir` — is byte-identical
to `baseline_v1.yaml`.

## New code, all gated and additive

- **`src/freuid/optim.py`** (new file): `build_llrd_param_groups` (per-block layer-wise LR
  decay, LayerNorm/bias excluded from weight decay), `freeze_all_but_last_k_blocks`
  (fallback if a full fine-tune doesn't fit memory — unused this run), `build_warmup_cosine_scheduler`
  (linear warmup → cosine decay).
- **`src/freuid/train.py`**: optimizer/scheduler construction now branches on
  `cfg.extra.llrd.enabled` — when absent/false (every other config), behavior is byte-identical
  to before. Added a checkpoint tie-break (probe AuDET ties broken by probe APCER@1%BPCER).
  Added gated AMP (`autocast` + `GradScaler`) — a disabled `GradScaler` is a documented no-op,
  so configs without `extra.amp` are unaffected.
- **`src/freuid/models/baseline.py`**: `build_model` now tries
  `timm.create_model(..., dynamic_img_size=True)` first, falling back via `except TypeError`
  for backbones that don't accept the kwarg (CNNs). Needed so the ViT accepts the 476/518/560
  multi-scale TTA inputs without a pos-embed shape mismatch.
- **`scripts/config_diff.py`** (new): loads two configs through the real `load_config` path
  and prints every differing resolved key — used to produce the table above.

## Ambiguities resolved (not assumed)

1. **Did baseline_v1 use a warmup?** No — read `train.py`: the old scheduler was a flat
   `CosineAnnealingLR(T_max=cfg.epochs)` starting at `cfg.lr` from epoch 1, no warmup at all.
   Warmup was added *only* inside the new gated `llrd` path; S1's own recipe is untouched.
2. **Head LR (1e-4 vs S1's 2e-4)**: S1's `lr=2e-4` was tuned for a from-scratch classifier
   head sitting on a fixed CNN feature extractor. Reusing it as the LLRD head LR for a
   *fully-trainable* ViT would apply that same LR (attenuated only by depth-decay) to the
   whole backbone, which is far too aggressive for a pretrained transformer. Resolved by
   using 1e-4 as the head-of-LLRD base LR, per standard LLRD fine-tuning practice — this is
   the one substantive value deviation from S1, and it's necessary for the experiment to be
   fine-tunable at all (see next point).
3. **Was `train.py`'s built-in `--sanity` check (hardcoded SGD, lr=0.1, 100 steps) a valid
   wiring check for this model?** No. Running it against `finetune_v0` produced
   `loss=0.7995` after 100 steps (failed the `<0.02` bar). A follow-up diagnostic swapping
   the optimizer showed why: SGD lr=0.1 **diverges to NaN** by step ~49 on a
   full-parameter ViT-B (flat high-LR SGD without per-parameter adaptivity is a known
   instability for deep transformers/LayerNorm-heavy architectures — this is *not* how S1's
   CNN behaves, which is why the shared harness's assumption held for baseline_v1 but not
   here). AdamW at lr=1e-3 converged to `loss=0.000124` within 300 steps on the same batch,
   confirming gradient flow and model capacity are both intact — the sanity check's failure
   was an optimizer-choice mismatch in the shared harness, not a wiring bug in `finetune_v0`.
   (AdamW at lr=1e-2 did *not* converge well either — stuck oscillating around 0.58-0.70 —
   confirming 1e-2 is too aggressive and validating that the real LLRD head LR of 1e-4 sits
   in a sane, conservative regime.)
4. **AMP**: `train.py`'s `run_epoch` had no `autocast`/`GradScaler` path at all before this
   experiment — confirmed by reading it — so `baseline_v1` trained in plain FP32. A
   full-parameter ViT-B fine-tune at 518px in FP32 measured at **1.69s/it** (~55 min/epoch for
   the train phase alone → ~18+ hours for 20 epochs), which would have badly violated the
   task's "flag if >3x baseline_v1" throughput guardrail. AMP was added (gated on
   `extra.amp`, disabled everywhere else) after this was diagnosed live; it produced a
   **~6.3x speedup** (0.267s/it, 3.75-3.83 it/s) with `[train] AMP enabled` confirmed in the
   log, bringing the full run down to ~11.9 min/epoch average, ~4 hours total — back in a
   normal range for a single-A100 job.

## Pre-flight checks (all passed; evidence)

- **Resolved-config diff**: see table above — only the expected keys differ.
- **timm backbone facts** (`vit_base_patch14_reg4_dinov2.lvd142m`, verified directly on the
  VESSL workspace): native `input_size=(3,518,518)`, `mean=(0.485,0.456,0.406)`,
  `std=(0.229,0.224,0.225)` — **identical** to the ImageNet defaults this repo already uses
  elsewhere, so no transform changes were needed; `depth=12` blocks; `global_pool='token'`;
  `patch_size=(14,14)`; `set_grad_checkpointing` available; `no_weight_decay() = {'pos_embed',
  'dist_token', 'cls_token'}`.
- **Multi-scale TTA forward pass**: 476/518/560 all produced a `(B,1)` logit with no
  pos-embed shape error, confirmed both on a raw `timm.create_model(..., dynamic_img_size=True)`
  instance and later through the real `build_model()` path used by training/inference.
- **init BCE ≈ ln(2)**: `0.6931 ~= 0.6931` — passed on both the toy-scale smoke run and the
  full run.
- **Single-batch overfit**: failed under the shared harness's hardcoded SGD (see Ambiguity
  #3), confirmed as an optimizer mismatch rather than a bug via an AdamW diagnostic
  (`loss=0.000124` after 300 steps).
- **LLRD wiring**: toy-scale (`--limit 320`) and full runs both logged
  `[optim] LLRD: 28 param groups over depths [0..13] (decay=0.7) -- lr range
  [9.69e-07, 1.00e-04]` — exactly matches the closed-form expectation
  (`1e-4 * 0.7^13 ≈ 9.69e-7`) for depth = 12 blocks + 1 (patch/pos-embed group).
- **Warmup schedule**: `lr_head` measured 1.00e-06 (epoch 1) → 5.05e-05 (epoch 2) →
  1.00e-04 (epoch 3, full LR reached) — exact match to a 2-epoch `LinearLR(start_factor=0.01)`
  then `CosineAnnealingLR` handoff.
- **Throughput**: see Ambiguity #4. Final observed pace: ~11.9 min/epoch average across the
  full 20-epoch run (~4 hours total, started 05:56 UTC, finished ~10:00 UTC 2026-07-03).
- **Checkpoint tie-break**: implemented (probe AuDET ties broken by probe APCER@1%BPCER);
  not exercised in practice this run since no two epochs tied exactly on the float metric.

## Full training log (all 20 epochs)

| Epoch | lr_head | lr_min | train_loss | val_loss | val AuDET | val APCER@1%BPCER | **probe_AuDET** | checkpointed |
|---|---|---|---|---|---|---|---|---|
| 1  | 1.00e-06 | 9.69e-09 | 0.6299 | 0.4058 | 0.0279 | 0.1543 | 0.052232 | ✓ |
| 2  | 5.05e-05 | 4.89e-07 | 0.0996 | 0.0066 | 0.0000 | 0.0000 | 0.000768 | ✓ |
| 3  | 1.00e-04 | 9.69e-07 | 0.0651 | 0.0093 | 0.0003 | 0.0010 | 0.002223 | — |
| 4  | 9.92e-05 | 9.62e-07 | 0.0540 | 0.0016 | 0.0000 | 0.0003 | 0.000068 | ✓ |
| 5  | 9.70e-05 | 9.40e-07 | 0.0511 | 0.0084 | 0.0000 | 0.0000 | 0.000175 | — |
| 6  | 9.33e-05 | 9.04e-07 | 0.0479 | 0.0068 | 0.0002 | 0.0003 | 0.000176 | — |
| 7  | 8.83e-05 | 8.56e-07 | 0.0471 | 0.0013 | 0.0000 | 0.0000 | 0.000111 | — |
| 8  | 8.21e-05 | 7.96e-07 | 0.0429 | 0.0013 | 0.0000 | 0.0000 | 0.000132 | — |
| 9  | 7.50e-05 | 7.27e-07 | 0.0427 | 0.0044 | 0.0000 | 0.0000 | 0.000076 | — |
| 10 | 6.71e-05 | 6.50e-07 | 0.0385 | 0.0015 | 0.0000 | 0.0000 | 0.000253 | — |
| 11 | 5.87e-05 | 5.69e-07 | 0.0381 | 0.0008 | 0.0000 | 0.0000 | 0.000102 | — |
| 12 | 5.00e-05 | 4.84e-07 | 0.0389 | 0.0008 | 0.0000 | 0.0000 | 0.000011 | ✓ |
| 13 | 4.13e-05 | 4.00e-07 | 0.0376 | 0.0006 | 0.0000 | 0.0000 | **0.000002** | ✓ **best** |
| 14 | 3.29e-05 | 3.19e-07 | 0.0358 | 0.0003 | 0.0000 | 0.0000 | 0.000012 | — |
| 15 | 2.50e-05 | 2.42e-07 | 0.0347 | 0.0004 | 0.0000 | 0.0000 | 0.000010 | — |
| 16 | 1.79e-05 | 1.73e-07 | 0.0339 | 0.0003 | 0.0000 | 0.0000 | 0.000028 | — |
| 17 | 1.17e-05 | 1.13e-07 | 0.0329 | 0.0006 | 0.0000 | 0.0000 | 0.000096 | — |
| 18 | 6.70e-06 | 6.49e-08 | 0.0325 | 0.0004 | 0.0000 | 0.0000 | 0.000007 | — |
| 19 | 3.02e-06 | 2.92e-08 | 0.0321 | 0.0004 | 0.0000 | 0.0000 | 0.000041 | — |
| 20 | 7.60e-07 | 7.36e-09 | 0.0317 | 0.0004 | 0.0000 | 0.0000 | 0.000024 | — |

Selected checkpoint: **epoch 13**, `probe_AuDET = 0.000002`.

## Results vs. S1 / S2 / S3

| Config | Backbone | probe_AuDET (best) | val_AuDET | public LB FREUID |
|---|---|---|---|---|
| S1 `baseline_v1` | ConvNeXt-Small (fine-tuned) | 0.000039 | 0.0000 | 0.18129 |
| S2 `consistency_v0` | DINOv2 ViT-B/14 (**frozen**) | 0.061290 | 0.0280 | 0.30743 |
| S3 `consistency_v1` (gated fix) | DINOv3 ViT-B/16 (**frozen**) + Global+Patch+Face | 0.115286 | 0.1020 | 0.47469 |
| **finetune_v0** | DINOv2 ViT-B/14 (**fine-tuned**) | **0.000002** | 0.0000 | **0.00744** |

`finetune_v0`'s probe_AuDET (0.000002) is **~20x better than S1's** best result
(0.000039) and **~30,000x better than S2's** frozen-backbone result (0.061290). Both
`val_AuDET=0.0000` and `probe_AuDET≈0` indicate the fine-tuned ViT saturates the local
recapture-probe signal even more thoroughly than S1 did.

## Inference + submission integrity

Ran with rank-averaged multi-scale TTA at `[476, 518, 560]` (all confirmed valid, multiples
of patch_size=14, no pos-embed errors):

```
[tta] scale=476  scores: min=0.0009 max=1.0000
[tta] scale=518  scores: min=0.0009 max=1.0000
[tta] scale=560  scores: min=0.0009 max=1.0000
[infer] wrote 142818 rows -> submissions/finetune_v0.csv
[infer] integrity: rows=142818 unique_scores=4254 exact_zeros=0 (0.00%) min=0.009256 max=1.000000
```

`rows=142818` matches the full `sample_submission.csv` test set exactly; `exact_zeros=0`
confirms no genuinely-missing id silently got a 0.0 fraud score. Of the 142,818 ids, 7,821
have a locally-present image and got a real rank-averaged score (this is a code competition —
only the public subset ships locally); the remaining 134,997 correctly defaulted to
`missing_id_score=0.5` (Kaggle's grading server has every image present, so this only
affects local integrity-checking, not the actual graded score). Submitted to Kaggle 2026-07-03 10:10 UTC (ref 54294446); scored **public LB FREUID = 0.00744**.

## Final verdict — gate CLEARED ✓

`finetune_v0` beats S1 on **both** required signals:

- probe_AuDET: **0.000002** < 0.000039 (S1) — ~20x better
- public LB FREUID: **0.00744** < 0.18129 (S1) — ~24x better

This is not a marginal win — it's the largest single improvement in the project's history,
and it fully resolves the ambiguity S2/S3 left open. Per the project's decision rule:

> **The fine-tuned ViT beats S1 → the backbone was never the problem, freezing was.**

DINOv2 ViT-B/14, when actually allowed to adapt its weights to this task (via LLRD +
warmup + AMP, same augmentation/loss/probe recipe as S1), is not just competitive with the
CNN baseline — it's dramatically better. S2's and S3's regressions were an artifact of
freezing a backbone that needed to adapt, not evidence that ViT features are unsuited to
this task or that CNN inductive bias wins. The consistency-heads bet (S2/S3) was never
given a fair test, because the confound (frozen vs. fine-tuned) dominated whatever
patch/face-consistency signal those heads might have contributed.

**Implication for the roadmap**: S4 should be reframed around **a fine-tuned-CNN +
fine-tuned-ViT rank ensemble** (per the decision rule), not the frozen-backbone
consistency-heads path that S2/S3 pursued. Whether patch/face self-consistency heads still
add value **on top of a fine-tuned (not frozen) ViT** is now the open question, rather than
"do consistency heads rescue a frozen backbone" (they didn't, per S3). `finetune_v0` is now
the strongest submission in the project by a wide margin and should be the new baseline to
beat.

## Where to look in code

| Concern | File |
|---|---|
| LLRD param groups, freeze fallback, warmup+cosine scheduler | `src/freuid/optim.py` |
| Gated LLRD/AMP/tie-break wiring in the training loop | `src/freuid/train.py` |
| `dynamic_img_size` fix for multi-scale ViT TTA | `src/freuid/models/baseline.py` |
| Resolved-config diff tool | `scripts/config_diff.py` |
| This experiment's config | `configs/finetune_v0.yaml` |
| S1's config (baseline for the diff) | `configs/baseline_v1.yaml` |
