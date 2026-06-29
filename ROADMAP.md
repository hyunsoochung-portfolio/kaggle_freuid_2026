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
| baseline_v0 | tf_efficientnetv2_s.in21k | 15 | 0.000000 | — | — | S0 submitted; LB TBD |
| baseline_v1 | convnext_small.fb_in22k_ft_in1k | 20 | TBD | TBD | TBD | S1: synth + AUC loss + TTA |

> Public LB for baseline_v0: update after checking Kaggle leaderboard.

---

## S1 gate checklist

- [ ] baseline_v1 probe_AuDET < baseline_v0 probe_AuDET
- [ ] baseline_v1 public LB AuDET < 0.5 (constant-baseline)
- [ ] baseline_v1 public LB AuDET < baseline_v0 public LB AuDET
- [ ] TTA on/off both produce valid submissions (integrity check passes)
- [ ] auc_loss_weight=0.0 smoke run matches S0 loss curve

---

## Constant-baseline reference

A submission of all 0.5 scores: AuDET = 0.5 (worst possible ordering = random).
Any useful model must beat this on both probe and public LB.
