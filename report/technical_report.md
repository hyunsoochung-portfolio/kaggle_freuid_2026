# Data-Grounded Synthetic Tampering for Identity-Document Fraud Detection

**FREUID Challenge 2026 (IJCAI-ECAI) — Technical Report**
Team: _<TEAM NAME>_ · Kaggle: `hyunsoooochung` et al. · Code: Apache-2.0

---

## 1. Task and challenge

Binary fraud detection on identity-document images: emit a continuous fraud score
`P(fraud) ∈ [0,1]` (`1 = fraud`). The primary metric is **AuDET** (area under the DET
curve; lower is better), a pure **rank** metric. The difficulty is deliberate: the hidden
test is dominated by **print-and-capture ("analog hole") attacks** and **document types not
seen in training**, whereas the released training set is ~99.97% *digital*. A model that
leans on digital forensic noise scores near-zero on an in-domain split but collapses on the
public test, because reprinting erases that noise.

## 2. Method

### 2.1 Backbone and head
A **DINOv2 ViT-B/14** backbone (`vit_base_patch14_dinov2.lvd142m`), **fully fine-tuned**,
with an **attention-pooling** head over the patch tokens and a single fraud logit
(zero-initialized). Full fine-tuning of a strong self-supervised backbone was, by a wide
margin, the biggest lever (swapping an EfficientNet-V2 backbone for a properly-tuned DINOv2
moved AuDET ~0.21 → ~0.07); a *frozen* backbone with a light head was much worse, so
adaptation of the features — not head complexity — is what matters.

### 2.2 `synth_tamper`: data-grounded synthetic fraud
Because genuine cards vastly outnumber frauds in training, the model never sees enough
"what a forgery looks like." We manufacture forgeries on the fly. Each epoch a fraction
(`prob = 0.3`) of bona-fide cards is converted to a synthetic fraud (label 1), reproducing
tells found in a **100-image manual analysis** of the real frauds:

- **Face-paste (≈80% of tampers).** Nearly every real fraud has a pasted portrait: a sharp
  face with a hard rectangular seam over which the card's guilloche/overlay does not
  continue. We paste a donor face (from a per-document-type pool, with validation ids
  excluded to prevent leakage) with a hard seam. A BENIN-specific *colour-face-on-grayscale-
  body* variant reproduces that type's characteristic tell.
- **Field-carve (≈20%, text).** A bounded value field (e.g. date of birth) is rewritten so
  the digits stay readable but the background micro-texture is destroyed — calibrated
  pixel-by-pixel against real Mozambique/Egypt examples.

The augmentation reproduces the *distribution* of real tells, not specific pixels, so it
generalizes across the specific random forgeries drawn each run.

### 2.3 Training recipe
`BCEWithLogits` + a **pairwise soft-AUC** term (weight 0.1), matched to the rank metric.
AdamW with **layer-wise LR decay 0.7**, 2-epoch warmup, and **cosine annealing over 25
epochs**, base `lr = 5e-5`, weight decay 0.05, mixed precision (AMP). Multi-scale
**test-time augmentation** at `[476, 518, 560]`, rank-averaged; **no horizontal flip**
(documents carry orientation). Seed 42.

### 2.4 Validation and checkpoint selection
The split is stratified by (label × document type) so it spans the type mix. A per-epoch
"synth probe" (clean bona-fide vs. its tampered twin) was intended as a compass but, like
every in-domain proxy we tried, **saturates to ≈0** — the model fits our procedural tells
too well. Consequently the local metric cannot rank epochs, and **the public leaderboard is
the only reliable judge**. We therefore run the full schedule and submit the **last
(fully-annealed) epoch**.

## 3. Key findings

| # | Finding | Evidence (public AuDET, lower=better) |
|---|---|---|
| 1 | **`synth_tamper` is the decisive lever.** Adding it to an otherwise-fixed recipe was the first change that helped. | 0.0394 → **0.0157** |
| 2 | **Full anneal + submit-last beats early-stopping.** The heavy augmentation suppresses the overfitting that would otherwise punish 25 epochs, so letting LR fully anneal and submitting the last epoch is strictly better than stopping at the (saturated) validation optimum. | ep12 0.0157 → ep25 **0.0063** |
| 3 | **Synthetic *analog* augmentation HURTS — a robust negative result.** Simulating print-and-capture at train time (mild or strong, on 50–100% of images) *consistently* degraded the score, even though the real test is analog. The model learns features specific to *our* synthetic recapture that do not transfer to real print-capture. | 0.0157 → 0.026 / 0.032 / 0.13 |
| 4 | **Added heads/complexity hurt.** A patch/face consistency branch fused onto the winning model, and longer training on un-augmented data, both regressed. | 0.039 → 0.10 / 0.15 |

Finding 3 is the least intuitive and, we believe, the most useful: for this task,
robustness to the analog channel is better obtained from a strong, well-adapted backbone
than from *synthetically* reproducing the channel, which merely teaches a synthetic-specific
shortcut. A low, LB-visible **unique-score count** was a reliable early warning of the
resulting over-confidence.

## 4. Results

The winning configuration ([`configs/synth_tamper_v1.yaml`](../configs/synth_tamper_v1.yaml))
— DINOv2 ViT-B/14 full FT + attention pool + soft-AUC + multi-scale TTA + `synth_tamper`,
run to the full 25-epoch anneal and submitted at the last epoch — reaches public
**AuDET ≈ 0.006**, versus 0.0157 for the early-stopped variant and 0.039 for the same recipe
without the augmentation.

## 5. Reproducibility

Full instructions are in [`REPRODUCE.md`](../REPRODUCE.md): environment, data layout, the
single training command, inference, and a **no-network Docker** image for the sandbox
(the fine-tuned checkpoint carries all weights, so the backbone is built with
`pretrained=False` and nothing is downloaded at run time). GPU + AMP are not
bit-deterministic and the augmentation is stochastic, so re-runs reproduce the score *band*;
the exact ranked CSV is reproduced from the shipped `synth_tamper_v1_last.pt` checkpoint.

## 6. Limitations and outlook

The approach targets the dominant real tell (portrait substitution) and the most common text
tell; rarer semantic frauds (name/gender mismatch, ghost-portrait mismatch) are not yet
synthesized and are natural next steps, as is diverse-backbone rank-ensembling — the
canonical next lever for a pure rank metric. We deliberately do **not** rely on
forgery-localization or presentation-attack-specific networks; every component is a general
vision model, consistent with the challenge's spirit of generalizable detection.
