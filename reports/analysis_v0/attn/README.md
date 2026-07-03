# finetune_v0 attention / attribution audit

Where does `finetune_v0` actually look when it flags a document as fraud — real tamper
evidence, or backgrounds/borders/shortcuts? `scripts/analysis/visualize.py` implements four
complementary attribution methods and a quantitative audit against ground-truth tamper
regions. No training-code changes: `scripts/analysis/tamper_bbox.py` replicates (not
modifies) `augment.py`'s `synth_tamper` + `recapture_transforms` logic, adding bbox
tracking through the full degradation chain (including the geometric steps) via
albumentations' `bbox_params`.

**Note on figures**: every figure this script produces overlays a real train/val/test
image, so — same as `reports/analysis_v0/borderline/` — they are **not committed to git**
(`reports/analysis_v0/attn/figures/` is gitignored; same non-commercial licensing concern
as the raw competition data). They exist locally/on the VESSL workspace for direct
viewing; this report references them by filename for that purpose. `quant_eval.csv` (pure
scalar metrics, no image content) is committed.

## The four map types

1. **CLS-to-patch attention**: forward hook on `block.attn.attn_drop`'s *input* (the real
   post-softmax attention weights), for the last 3 transformer blocks. **Required
   `timm.layers.set_fused_attn(False)` at import time** — by default timm's `Attention`
   uses `F.scaled_dot_product_attention`, which never calls `attn_drop` explicitly, so the
   hook would silently capture nothing. Token layout (`vit_base_patch14_reg4_dinov2`:
   `[CLS, 4 register tokens, patch tokens]`, grid 37×37=1369, total 1374 tokens) is
   asserted at runtime everywhere it matters.
2. **Grad-CAM** (`pytorch-grad-cam`), target layer = final block's `norm1`, reshape
   transform drops the 5 prefix tokens and reshapes to 37×37, gradient taken w.r.t. the raw
   fraud logit (`ClassifierOutputTarget(0)`).
3. **Occlusion sensitivity** (model-agnostic — the most trustworthy of the four): a 56px
   mid-gray patch (fixed neutral value in normalized space, not each image's own mean)
   slides with stride 28 (17×17=289 positions, batched), recording the fraud-probability
   drop per position.
4. **Patch-token PCA-to-RGB** (DINOv2-style): PCA(3) of the final block's raw patch-token
   output, pretrained DINOv2 vs. fine-tuned `finetune_v0`, same image, side by side.

## Qualitative spot-checks (4 of 200 synthetic tampers inspected directly)

Four synthetic-tamper examples (`synthfig_00`, `02`, `05`, `08` — chosen at even intervals,
not cherry-picked) were inspected directly. **All four show the ground-truth tamper bbox
(green rectangle) sitting almost exactly on top of the hottest region in CLS attention,
Grad-CAM, *and* occlusion sensitivity simultaneously** — e.g. `synthfig_00`'s field-smudge
edit over a text field, `synthfig_02`'s edited "Signature" field, and `synthfig_05`'s
altered document-type text all produce sharp, correctly-placed hotspots across all three
methods. The fine-tuned PCA panel also visibly differs from the pretrained one near the
tamper region in each case (a distinct color cluster the pretrained PCA doesn't show).

This initially looked like strong, uncomplicated evidence of precise localization — see
§ Reconciling the qualitative/quantitative gap below for why the full 200-sample audit
tells a more nuanced story.

## Quantitative audit (200 synthetic tampers + 20 clean controls)

| Metric | Grad-CAM | Occlusion sensitivity |
|---|---|---|
| Pointing-game hit rate (synth, n=200) | **8.0%** | **27.5%** |
| — random-chance baseline (mean bbox area / image area) | 5.2% | 5.2% |
| IoU@best-threshold, median (synth) | 0.005 | 0.073 |
| IoU@best-threshold, mean (synth) | 0.052 | (see `quant_eval.csv`) |

- **Grad-CAM is barely better than chance** (8.0% vs. a 5.2% random-argmax baseline derived
  from the actual mean tamper-bbox size) — essentially uninformative as a localizer here,
  despite the one qualitative example above looking clean.
- **Occlusion sensitivity is meaningfully better than chance** (~5.3x the baseline rate) —
  a real, above-chance localization signal, though it still misses the exact tamper region
  in nearly 3 out of 4 samples.
- **When either method *does* hit the target, it does so with high confidence, not noise.**
  Splitting synth samples by pointing-game outcome:

  | | Grad-CAM concentration† | Occlusion concentration† |
  |---|---|---|
  | hit (n=16 / n=55) | 0.89 (mean) | 0.51 (mean) |
  | miss (n=184 / n=145) | 0.19 (mean) | 0.38 (mean) |

  († fraction of total map "energy" in the top 10% of pixels — see `visualize.py`'s
  `concentration_score`.) Hits are sharply more concentrated than misses for both methods,
  especially Grad-CAM (0.89 vs. 0.19). This is not consistent with pure noise: when the
  model does localize the tamper, the signal is genuinely peaked, not a lucky diffuse
  coincidence.
- **Clean-control concentration is confounded by a likely metric artifact** and should not
  be over-interpreted: raw clean-control occlusion concentration (mean 0.65) is *bimodal*
  (min/25th percentile = 0, median/75th = ~1.0), not a clean "low and diffuse" signal as
  hoped. Because `concentration_score` clips the map to non-negative values before
  computing the top-10% share, an image where occlusion barely moves the (already very
  low) fraud probability *anywhere* has its output dominated by tiny numerical noise —
  whichever few positions happen to have a small positive residual absorb ~100% of the
  (tiny) remaining "energy," producing a spuriously high concentration score despite no
  real localized effect. This metric isn't well-suited to near-zero-signal cases; a
  magnitude-aware comparison (e.g. raw max probability drop, not clipped-and-normalized
  concentration) would be needed to properly characterize control diffuseness, and wasn't
  captured in this run.

## Reconciling the qualitative/quantitative gap

Four visually excellent spot-checks alongside an aggregate 8-27.5% hit rate initially looks
contradictory. The likely explanation, **not directly confirmed in this run** (tamper type
wasn't logged per-sample — flagged as a follow-up): `synth_tamper` (in both `augment.py` and
its replication here) picks uniformly among three edit types with very different real-world
detectability:

- **`copy_move`** — pastes a patch from *elsewhere in the same image* over another region.
  Classically the hardest forgery type to detect, forensically: the pasted content shares
  the exact same noise/color/texture statistics as its surroundings (that's the entire
  point of a copy-move attack in the literature), so there's no foreign-source signal to
  key on.
- **`field_smudge`** / **`local_splice`** — blur/recolor/fill a field, or splice in content
  from a *different* image. Both tend to be visually obvious (legibility loss, texture
  mismatch, a differently-lit/differently-styled inserted region).

All four spot-checked examples show large, visually obvious edits over text fields —
consistent with `field_smudge`/`local_splice`, not `copy_move`. If `copy_move` (roughly
1/3 to 1/2 of the 200 synthetic samples, depending on donor-pool availability) is
substantially harder to localize than the other two types, that alone would explain both
observations: a handful of "easy" spot-checks looking great, and a much lower aggregate hit
rate once all edit types are pooled together. **This is a well-motivated hypothesis, not a
confirmed finding** — logging tamper type per sample and re-running the audit stratified by
type is the natural next step.

## Verdict

Neither pole of "genuine forgery detection" vs. "shortcut learning" fits cleanly — the
honest read is a nuanced middle ground:

- **Not a background/shortcut story.** This is corroborated from two independent
  directions: (a) the degradation-curve test in `reports/analysis_v0/README.md` §2 already
  showed that hiding the document collapses performance to near-random while hiding the
  background does nothing; (b) here, occlusion sensitivity (model-agnostic, the most
  trustworthy of the four methods) localizes the true tamper region ~5x more often than
  chance, and does so with high confidence when it succeeds. A model exploiting a pure
  background/framing shortcut would not show either pattern.
- **But not precise, reliable, pixel-level forensic localization either.** Most individual
  samples (especially via Grad-CAM) do not have their attribution map's peak sitting inside
  the ground-truth tamper region. The fraud signal for a majority of samples likely comes
  from more distributed/holistic evidence (cross-field consistency, overall document
  structure) rather than a single sharp "here's the edit" localization — or, per the
  hypothesis above, the model may simply be weaker at localizing specific harder-to-detect
  edit types (`copy_move`) even while still classifying them correctly overall.
- **Grad-CAM specifically underperforms occlusion sensitivity by a wide margin** (8% vs.
  27.5% hit rate; near-zero vs. modest IoU) despite targeting the same underlying
  representation. This likely reflects a known limitation of adapting CNN-native
  class-activation methods to Vision Transformers — gradients at a single pre-attention
  `norm1` layer don't localize as cleanly as they do for a CNN's local receptive fields —
  rather than telling us anything new about the model's actual reasoning. Trust the
  occlusion numbers over the Grad-CAM numbers here.

## Files

| File | Contents |
|---|---|
| `quant_eval.csv` | Per-sample IoU/pointing-game/concentration for 200 synth + 20 clean (committed) |
| `figures/` | Per-image 6-panel comparison + attention companion figures, 60 images (**gitignored** — real dataset image content) |
| `../../../scripts/analysis/visualize.py` | Main script: all 4 map types, figure generation, quantitative eval |
| `../../../scripts/analysis/tamper_bbox.py` | Replicated tamper + recapture-with-bbox generation |
