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

Gate to advance: beat the constant-0.5 baseline **and** the previous stage on probe AuDET + public LB FREUID score.
Constant-0.5 baseline FREUID ≈ 1.0 (g_apcer collapses to 0 → harmonic mean = 0).

---

## Results

| Config         | Backbone                            | Epochs         | probe_AuDET (best ckpt) | val_AuDET | public_LB_FREUID | Notes                                                                |
| -------------- | ----------------------------------- | -------------- | ----------------------- | --------- | ---------------- | -------------------------------------------------------------------- |
| baseline_v0    | tf_efficientnetv2_s.in21k           | 15             | 0.000000                | —         | 0.27106          | S0 submitted                                                         |
| baseline_v1    | convnext_small.fb_in22k_ft_in1k     | 10 (best ep9)  | 0.000039                | 0.0000    | 0.18129          | synth p=0.3, auc_w=0.1, TTA 3-scale; submitted                       |
| consistency_v0 | dinov2_vitb14 (frozen) + GlobalHead | 20 (best ep20) | 0.061290                | 0.0280    | 0.30743          | 149K trainable, synth p=0.3, auc_w=0.1, TTA [448,518,588]; submitted |

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

---

## Constant-baseline reference

A submission of all 0.5 scores: AuDET = 0.5 (worst possible ordering = random).
Any useful model must beat this on both probe and public LB.
