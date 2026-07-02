# S3 regression: patch/face heads hurt instead of help

## Summary

`consistency_v1` (S3: frozen DINOv3 + GlobalHead + PatchConsistencyHead + FaceRegionHead,
fused) trained for the full 20 epochs on the full dataset and came out **worse** than
`consistency_v0` (S2: frozen DINOv3 + GlobalHead only) on every local signal:

| Config | AuDET (val) | probe_AuDET (best ckpt) |
|---|---|---|
| consistency_v0 (S2, global only) | 0.0280 | 0.0613 |
| consistency_v1 (S3, +patch +face) | 0.1136 | 0.1175 |

This is the wrong direction for the S3 gate (`fusion_all >= best_single`, lower is better).
An ablation harness run at toy scale (limit=1024, 5 epochs) showed the same pattern earlier
(`global_only` beat `fusion_all`, `patch_only`, and `face_only`), so this isn't noise from
the full run — it reproduces at both scales.

Training itself is healthy: init BCE ≈ ln(2), single-batch overfit → 0.004, train/val loss
both decrease monotonically over 20 epochs, submission integrity is clean (142,818 rows, 0
exact-zeros). The regression is about what the added heads are doing to the *fused* signal,
not a wiring failure.

## Findings, ranked by likely impact

### 1. Naive concat fusion has no "stay a no-op until useful" guarantee past init (most likely primary cause)

`ConsistencyHead.forward` (`consistency_model.py`) does:

```python
parts = [self.global_norm(cls)]
if self.patch_head is not None:
    parts.append(self.patch_head(patch_tokens))
if self.face_head is not None:
    parts.append(self.face_head(patch_tokens, grid_hw, face_meta))
fused = torch.cat(parts, dim=-1)
return self.fusion(fused)      # FusionMLP: Linear -> ReLU -> Dropout -> Linear(zero-init)
```

`FusionMLP`'s **final** layer is zero-init, which is why `_check_init_loss` passes (logit ≈ 0
regardless of how many heads are on, at step 0). But `FusionMLP`'s **first** layer
(`self.fc1`) is a single shared `Linear` that mixes the concatenated
`[global | patch | face]` vector together from the very first gradient step. There is no
mechanism that lets the network fall back to "global-only" behavior while the newer branches
are still learning — a noisy, not-yet-useful patch/face embedding immediately contaminates
the shared hidden representation that the (already good) global signal also has to pass
through.

`PatchConsistencyHead` is the most exposed to this: it's active on **100%** of samples and
carries **~10.1M params** (a 2-layer TransformerEncoder over 1024 patch tokens + a learned
query) — about **65x** `consistency_v0`'s entire head (149K params) — trained head-only for
the same 20-epoch budget S2 used to converge. It's plausible it simply hasn't learned a
net-useful signal yet, and its gradients are dragging the otherwise-good global pathway down
through the shared fusion layer.

**Standard fix:** a learnable per-branch gate/scale on each embedding before concatenation,
initialized near zero (LayerScale-style: `parts[i] * self.gate_i`, `gate_i` a
`nn.Parameter` initialized to 0 or a small value). This makes the model start
mathematically equivalent to global-only and lets training *open* the patch/face pathways
only once their contribution actually reduces loss, instead of mixing everything in from
step 1.

### 2. `FaceRegionHead`'s output embedding isn't scale-normalized (confirmed bug)

`global_norm(cls)` and `PatchConsistencyHead`'s output (`self.norm(out[:, 0, :])`) both end
in `nn.LayerNorm`, giving both branches comparable, controlled scale before concatenation.
`FaceRegionHead.forward` does not:

```python
emb = self.fc2(self.act(self.fc1(feat)))   # plain Linear -> GELU -> Linear, no norm
return emb * valid
```

Its output magnitude is uncontrolled (depends on random `Linear`/`GELU` init), unlike the
other two branches. Because `FusionMLP.fc1` is a single shared layer, a branch with
mismatched scale can distort that layer's gradients out of proportion to how informative the
branch actually is.

**Fix:** add a final `nn.LayerNorm(hidden)` to `FaceRegionHead`, matching the pattern already
used by the other two heads.

### 3. Face signal is sparse — limits (but doesn't eliminate) bug #2's impact

Sampled 3,000 cached `face.json` entries from `data/processed/regions/`:

```
sampled=3000  real_detection=587  fallback=2413  missing=0
valid_rate=0.196
```

Only **19.6%** of samples get a real SCRFD detection; the other **80.4%** fall back to the
center-square box and are correctly zeroed out (`valid == 0` -> `emb * valid == 0`, by
design — see `face_meta_tensor` in `data.py`). So bug #2 only actively distorts fusion on
about 1 in 5 samples. This makes the face head a plausible secondary contributor but an
unlikely primary cause of a regression this large, given `PatchConsistencyHead` (finding #1)
is active on every sample and carries far more capacity.

This also means the face branch's practical value is capped in this dataset: even a fully
correct FaceRegionHead can only inform ~20% of samples. Worth revisiting whether the
SCRFD portrait detector is well suited to the (small, stylized) photos found on these ID
cards, independent of the fusion issue.

### 4. `weight_decay` is applied uniformly to LayerNorm/bias params (secondary tuning issue)

`train.py`'s `main()`:

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
```

`cfg.weight_decay` for `consistency_v1.yaml` is `5.0e-2`, applied to every trainable
parameter with no exclusions. Standard practice excludes `LayerNorm` weights and all biases
from weight decay, since decaying them toward zero has no regularization benefit and can hurt
convergence. `consistency_v0`'s `GlobalHead` had only one `LayerNorm` + two `Linear` biases,
so this was low-impact before. `PatchConsistencyHead`'s `TransformerEncoder` (2 layers) has
many more such parameters (per-layer LayerNorms, attention biases, MLP biases), so this is a
first-time-meaningful effect introduced by S3, not a pre-existing issue that just resurfaced.

## Proposed next steps (not yet implemented)

1. Add the missing `LayerNorm` to `FaceRegionHead` (finding #2) — small, mechanical fix.
2. Add a learnable per-branch gate to `FusionMLP`/`ConsistencyHead`, initialized near zero
   (finding #1) — the fix most likely to actually close the gap, since it directly addresses
   why adding heads can regress performance instead of only improving or no-op'ing it.
3. Optionally split the AdamW param groups so `weight_decay=0` for `LayerNorm` weights and
   biases (finding #4).
4. Re-run the full 20-epoch `consistency_v1` training and compare probe_AuDET /
   val AuDET against `consistency_v0` (0.0613 / 0.0280) to see how much of the gap closes.
