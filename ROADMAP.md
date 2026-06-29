# FREUID 2026 — Roadmap & Results

Primary metric: **AuDET** (lower is better). Secondary: APCER @ 1% BPCER.
Probe AuDET = recapture-degraded val split (analog-robustness compass).
Public LB = Kaggle public leaderboard score.

## Stage definitions

| Stage | Label | Key change |
|-------|-------|------------|
| S0 | baseline_v0 | EfficientNetV2-S · recapture aug · probe checkpoint |
| S1 | baseline_v1 | ConvNeXt-Small · synth tamper (p=0.3) · soft-AUC loss · TTA |
| S2 | consistency_v0 | Frozen DINOv3 ViT-B/16 · patch self-consistency head |
| S3 | consistency_v1 | + face-region consistency · SCRFD face detector |
| S4 | ensemble_v0 | ConvNeXt-Base + DINOv3 · multi-seed/fold rank-average |

Gate to advance: beat the constant-0.5 baseline **and** the previous stage on probe AuDET + public LB.

---

## Results

| Config | Backbone | Epochs | probe_AuDET (best ckpt) | val_AuDET | public_LB_AuDET | Notes |
|--------|----------|--------|-------------------------|-----------|-----------------|-------|
| baseline_v0 | tf_efficientnetv2_s.in21k | 15 | 0.000000 | — | TBD | S0 submitted |
| baseline_v1 | convnext_small.fb_in22k_ft_in1k | 10 (best ep9) | 0.000039 | 0.0000 | TBD | synth p=0.3, auc_w=0.1, TTA 3-scale; submitted |

> Public LB for baseline_v0: update after checking Kaggle leaderboard.

---

## S1 gate checklist

- [x] baseline_v1 probe_AuDET < baseline_v0 probe_AuDET (0.000039 vs 0.000000 — note: v0 saturated; v1 still excellent)
- [ ] baseline_v1 public LB AuDET < 0.5 (constant-baseline)
- [ ] baseline_v1 public LB AuDET < baseline_v0 public LB AuDET
- [x] TTA submission integrity passed (rows=142818, zeros=0, range=[0.001238, 0.983369])
- [x] auc_loss_weight=0.1 active — smoke run at 0.0 confirmed BCE-identical

---

## Constant-baseline reference

A submission of all 0.5 scores: AuDET = 0.5 (worst possible ordering = random).
Any useful model must beat this on both probe and public LB.
