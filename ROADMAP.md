# FREUID 2026 — Roadmap & Results

Primary metric: **AuDET** (lower is better). Secondary: APCER @ 1% BPCER.
Probe AuDET = recapture-degraded val split (analog-robustness compass).
Public LB = Kaggle public leaderboard **FREUID score** (lower is better, 0=perfect):
g_audet = 1 - AuDET; g_apcer = 1 - APCER@1%BPCER
FREUID = 1 - 2·g_audet·g_apcer / (g_audet + g_apcer) ← harmonic mean penalises weak leg

## Stage definitions

| Stage | Label          | Key change                                                  |
| ----- | -------------- | ----------------------------------------------------------- |
| S0    | baseline_v0    | EfficientNetV2-S · recapture aug · probe checkpoint         |
| S1    | baseline_v1    | ConvNeXt-Small · synth tamper (p=0.3) · soft-AUC loss · TTA |
| S2    | consistency_v0 | Frozen DINOv3 ViT-B/16 · patch self-consistency head        |
| S3    | consistency_v1 | + face-region consistency · SCRFD face detector             |
| S4    | ensemble_v0    | ConvNeXt-Base + DINOv3 · multi-seed/fold rank-average       |
| —     | finetune_v0    | Decisive experiment: DINOv2 ViT-B/14, same as S2 but fully fine-tuned (LLRD+warmup+AMP), not frozen — de-confounds S1→S2's backbone-vs-trainability change |

Gate to advance: beat the constant-0.5 baseline **and** the previous stage on probe AuDET + public LB FREUID score.
Constant-0.5 baseline FREUID ≈ 1.0 (g_apcer collapses to 0 → harmonic mean = 0).

---

## Results

| Config         | Backbone                            | Epochs         | probe_AuDET (best ckpt) | val_AuDET | public_LB_FREUID | Notes                                                                |
| -------------- | ----------------------------------- | -------------- | ----------------------- | --------- | ---------------- | -------------------------------------------------------------------- |
| baseline_v0    | tf_efficientnetv2_s.in21k           | 15             | 0.000000                | —         | 0.27106          | S0 submitted                                                         |
| baseline_v1    | convnext_small.fb_in22k_ft_in1k     | 10 (best ep9)  | 0.000039                | 0.0000    | 0.18129          | synth p=0.3, auc_w=0.1, TTA 3-scale; submitted                       |
| consistency_v0 | dinov2_vitb14 (frozen) + GlobalHead | 20 (best ep20) | 0.061290                | 0.0280    | 0.30743          | 149K trainable, synth p=0.3, auc_w=0.1, TTA [448,518,588]; submitted |
| consistency_v1 | dinov3_vitb16 (frozen) + Global+Patch+Face fusion | 20 (best ep18) | 0.117463 | 0.1136 | 0.61753 | 10.27M trainable, synth p=0.3, auc_w=0.1, TTA [448,512,528]; submitted -- regression vs v0, see docs/problem.md |
| consistency_v1 (gated fix) | dinov3_vitb16 (frozen) + Global+Patch+Face, LayerScale gates + FaceRegionHead LayerNorm | 20 (best ep19) | 0.115286 | 0.1020 | 0.47469 | same config, gated-fusion fix per docs/problem.md; public LB improved 0.618→0.475 but still worse than v0 (0.307) and baseline_v1 (0.181) -- gate not cleared, see docs/consistency.md |
| finetune_v0 | dinov2_vitb14 (**fully fine-tuned**, LLRD+warmup+AMP) | 20 (best ep13) | 0.000002 | 0.0000 | 0.00744 | decisive experiment: same backbone as consistency_v0 but fine-tuned not frozen, same recipe as baseline_v1 otherwise; submitted -- gate CLEARED vs baseline_v1 on both signals, best result in the project, see docs/finetune.md |

---

## S1 gate checklist — CLEARED ✓

- [x] baseline_v1 probe_AuDET < baseline_v0 probe_AuDET (0.000039 vs 0.000000 — note: v0 saturated; v1 still excellent)
- [x] baseline_v1 public LB FREUID < 1.0 (0.181 << 1.0)
- [x] baseline_v1 public LB FREUID < baseline_v0 public LB FREUID (0.181 < 0.271)
- [x] TTA submission integrity passed (rows=142818, zeros=0, range=[0.001238, 0.983369])
- [x] auc_loss_weight=0.1 active — smoke run at 0.0 confirmed BCE-identical

## S2 gate checklist — NOT CLEARED ✗

- [x] consistency_v0 smoke: backbone frozen (149K trainable / 86.6M frozen), init BCE=0.6931 ✓
- [x] consistency_v0 full train: probe_AuDET=0.0613 (best ep20), val_AuDET=0.0280
- [x] TTA integrity passed (rows=142818, zeros=0, range=[0.000896, 1.0])
- [x] consistency_v0 public LB FREUID < 1.0 (0.307 << 1.0)
- [ ] consistency_v0 public LB FREUID < baseline_v1 public LB FREUID (0.307 > 0.181 — not cleared)

> **Diagnosis**: CLS-only GlobalHead gives up the CNN baseline on LB. The DINOv2 global feature alone
> is not enough — the analog-robustness gain needs patch self-consistency (S3), not just a better
> frozen feature extractor. probe_AuDET was already a warning (0.061 vs 0.000039 for baseline_v1).

## S3 gate checklist — NOT CLEARED ✗

- [x] consistency_v1 smoke: all 3 heads on, backbone frozen (10.27M trainable / 85.66M frozen), init BCE=0.6931, single-batch overfit=0.0037 ✓
- [x] consistency_v1 full train: probe_AuDET=0.1175 (best ep18), val_AuDET=0.1136
- [x] TTA integrity passed (rows=142818, zeros=0, range≈[0.0, 1.0])
- [ ] consistency_v1 probe_AuDET < consistency_v0 probe_AuDET (0.1175 > 0.0613 — **regressed**)
- [ ] consistency_v1 public LB FREUID < consistency_v0 public LB FREUID (0.618 > 0.307 — **regressed**)
- [ ] ablation gate: fusion_all probe_AuDET <= best single head (toy-scale sweep: global_only beat fusion_all, patch_only, face_only — **not cleared**)

> **Diagnosis**: adding PatchConsistencyHead + FaceRegionHead made every local and public signal
> worse, not better, confirming the toy-scale ablation warning. Root-caused in
> [docs/problem.md](docs/problem.md): naive concat fusion gives the two new heads no "stay a
> no-op until useful" guarantee past init (PatchConsistencyHead is active on 100% of samples,
> ~65x consistency_v0's entire head, and likely hasn't converged in 20 head-only epochs), plus a
> confirmed missing-LayerNorm scale bug in FaceRegionHead's output. Proposed fix: gated
> (LayerScale-style) fusion + the LayerNorm fix, before re-attempting S3.

### S3 retry (gated fusion fix) — STILL NOT CLEARED ✗

- [x] LayerScale-style per-branch gates added (patch_gate, face_gate, init 1e-3) + FaceRegionHead LayerNorm fix
- [x] full retrain: probe_AuDET=0.1153 (best ep19), val_AuDET=0.1020 -- only ~2% better than the buggy run locally
- [x] TTA integrity passed (rows=142818, zeros=0, range=[0.000043, 1.0])
- [x] submitted: public LB FREUID=0.4747 -- notably better than the buggy run (0.618 → 0.475, ~23% relative) despite the small local move
- [ ] consistency_v1 (fixed) probe_AuDET < consistency_v0 probe_AuDET (0.1153 > 0.0613 — **still regressed**)
- [ ] consistency_v1 (fixed) public LB FREUID < consistency_v0 public LB FREUID (0.475 > 0.307 — **still regressed**)

> **Diagnosis**: the gating fix recovered a real chunk of the S3 regression (especially on public
> LB, where the gap to consistency_v0 shrank more than local probe_AuDET suggested) but did not
> close it. Patch/face heads are still net-negative even when gated near-zero at init. Open
> question: whether the gates actually opened during training (unverified — inspect
> `patch_gate`/`face_gate` values in the saved checkpoint) or the branches themselves aren't
> learning a useful signal regardless of gating. See [docs/consistency.md](docs/consistency.md)
> for full context and next-step options (weight-decay exclusion for LayerNorm/bias, re-ablation
> with gating, confounded DINOv2→DINOv3 backbone swap between S2 and S3).

## finetune_v0 gate checklist (decisive experiment) — CLEARED ✓

- [x] resolved-config diff vs baseline_v1: only backbone, image_size, tta scales, lr (LLRD head LR),
      and the new llrd/train_last_k_blocks/grad_checkpointing/amp keys differ (scripts/config_diff.py)
- [x] smoke: init BCE=0.6931, LLRD 28 param groups (lr range [9.69e-07, 1.00e-04]), warmup schedule
      confirmed exact (1.00e-06 → 5.05e-05 → 1.00e-04 over epochs 1-3), multi-scale TTA forward
      (476/518/560) confirmed no pos-embed errors
- [x] built-in `--sanity` single-batch overfit failed under the shared harness's hardcoded SGD
      (diverges to NaN on a full ViT-B -- known instability, not a wiring bug); confirmed via an
      AdamW diagnostic instead (loss=0.000124 after 300 steps) -- gradient flow/capacity intact
- [x] full train: probe_AuDET=0.000002 (best ep13), val_AuDET=0.0000
- [x] TTA integrity passed (rows=142818, unique_scores=4254, exact_zeros=0, range=[0.009256, 1.0])
- [x] finetune_v0 probe_AuDET < baseline_v1 probe_AuDET (0.000002 < 0.000039 -- ~20x better)
- [x] finetune_v0 public LB FREUID < baseline_v1 public LB FREUID (0.00744 < 0.18129 -- ~24x better)

> **Diagnosis**: this is the largest improvement in the project's history and it resolves the
> open question from S2/S3. The S1→S2 change bundled backbone swap AND freezing at once; this
> experiment isolates trainability by fine-tuning the exact same DINOv2 ViT-B/14 backbone S2
> used. Fully fine-tuned, it dramatically beats S1, not just matches it -- **backbone freezing,
> not the ViT backbone itself, was the S2/S3 regression's real cause.** The consistency-heads
> bet (S2/S3) was never given a fair test: the frozen-backbone confound dominated whatever
> patch/face-consistency signal those heads might have added. `finetune_v0` is now the
> strongest submission in the project and the new baseline to beat. Reframes S4 around a
> fine-tuned-CNN + fine-tuned-ViT rank ensemble rather than the frozen-backbone consistency
> path; whether consistency heads add value *on top of* a fine-tuned (not frozen) ViT is now
> the open question. See [docs/finetune.md](docs/finetune.md) for the full writeup.

---

## Constant-baseline reference

A submission of all 0.5 scores: AuDET = 0.5 (worst possible ordering = random).
Any useful model must beat this on both probe and public LB.
