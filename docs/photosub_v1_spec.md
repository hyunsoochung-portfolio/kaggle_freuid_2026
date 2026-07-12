# photosub_v1 spec

Follow-up to `photosub_v0` (see `docs/technical_report.md`'s photosub_v0 section for the full
postmortem this builds on). Read alongside `scripts/analysis/movement_census_report.md`,
`movement_census_part2_report.md`, `threshold_pricing_out/threshold_pricing_report.md`, and
`scripts/analysis/deep_miss_dossiers.py`'s module docstring / `CHECKLIST_RESULTS` (the MODE_A
taxonomy this spec's shape-realism work is built on).

**Status as of this pass: code + config + tests done. Smoke test BLOCKED on VESSL connectivity
(see "Smoke test" below). Full 20-epoch run and any submission are separate, later go
decisions — nothing here authorizes either.**

## Why: what photosub_v0's postmortem said to fix

Three findings, most-to-least confident (`threshold_pricing_out/threshold_pricing_report.md`'s
strategy memo + `docs/technical_report.md`'s deep-9 diagnostic):

1. **Boundary-59 is already caught by finetune_v0 itself** (59/59 above the full-population
   threshold, 16/59 even above the stricter present-only threshold) — not a gap that needs new
   capability. v1 does not target this population specifically.
2. **Deep-9 is the real, still-open gap**, but the fix that photosub_v0 tried (MODE_A weighted
   at 44% of photosub rows) made it WORSE for MODE_A specifically: 2 of 4 real MODE_A deep-miss
   ids ended up more confidently bona-fide than finetune_v0's own baseline. The diagnosed cause:
   the generator only ever pasted a rotated rectangle, while the real exemplars show an
   arch-shaped cutout (`b5eebda1`), an irregular hand-cut silhouette (`40dd1055`), a diagonal
   tear (`5542f45f`), and (a MODE_B exemplar, but the same visual tell family) a tape strip
   (`cd7ad569`) — **not a clean rectangle**. This is the shape-realism fix below. **MODE_A is
   kept, not dropped** — the fix is what it generates, not whether it runs.
3. **Ceiling-zone confusion is a real, distinct, separately-priced cost**: ~36-37 expected
   bona-fide mass sits inside the ceiling zone, score-indistinguishable from confirmed fraud
   (embedding collapse, `hesitant_report.md`). Both `bayar_dinov2_v0` and `photosub_v0` traded
   away some of finetune_v0's existing ceiling confidence while chasing a different gain — this
   is the APCER guard photosub_v0 needed and didn't have. v1's ceiling-per-template gate exists
   specifically to catch this DURING training, not after a submission.

## Item 1: data/generator changes from v0

### MODE_A — kept, with silhouette variety

Four visual tells from the real exemplars (`deep_miss_dossiers.py`'s `CHECKLIST_RESULTS`): arch
shape, hand-cut polygon wobble, torn edge, tape strips.

| tell | generator | status |
|---|---|---|
| hand-cut polygon wobble | `_irregular_patch_alpha` / `irregular_shape_prob` | **already existed, reviewed, approved this pass** |
| torn edge | `_apply_tear_effect` / `tear_prob` | **already existed, reviewed, approved this pass** |
| arch shape | `_arch_patch_alpha` / `arch_shape_prob` (NEW) | implemented + unit-tested, **NOT rendered/reviewed** |
| tape strips | `_apply_tape_strips` / `tape_prob` (NEW) | implemented + unit-tested, **NOT rendered/reviewed** |

**Rectangle remains one variant** — `irregular_shape_prob`/`arch_shape_prob` are mutually
exclusive slices of one probability line in `generate_mode_a` (a single `rng.random()` draw
picks arch first, then irregular, then falls through to rect), so as long as their sum is < 1,
`rect` keeps a real, nonzero share by construction. `tear_prob`/`tape_prob` are independent
overlays on top of whichever shape was chosen.

All four kwargs default to `0.0` and are byte-identical to the pre-existing rectangle-only
behavior at that default (verified by test —
`test_mode_a_arch_shape_prob_zero_is_byte_identical_to_original` /
`test_mode_a_tape_prob_zero_is_byte_identical_to_original`).

### Render-gate review (this pass)

`docs/photosub_renders/mode_a_shape_variants_sheet.png` (5 templates × the 4 pre-existing
variants: rect/irregular/tear/irregular_tear) was reviewed visually this pass. Findings, stated
plainly rather than rubber-stamped:

- **Tear effect: consistently good.** Legible, localized (corner-anchored, not a full bisecting
  crack), reads as physical paper damage across all 5 templates. No reservations.
- **Irregular shape: mostly good, uneven.** Guinea/DL and Mozambique/DL show a convincing
  wavy/jagged hand-cut boundary. Benin/DL's irregular sample reads as a **geometric
  diamond/kite shape** — too regular, less like a hand-cut paper edge than the others. Egypt/DL
  and Mauritius/ID's irregular columns render in a grayscale/pencil-sketch print style that
  reads as a `print_style` variation, not a shape defect, but worth noting since it makes the
  shape itself harder to judge in those two cells.
- **No catastrophic failures**: no broken compositing, no mask/frame-box misalignment, no
  garish/unrealistic artifacts.

**Decision: approved for real weights.** The Benin geometric-kite instance is a real, noted
weakness (not every irregular draw looks equally hand-cut), but it's one sample out of many
possible draws per template, not a systemic failure — `_irregular_patch_alpha`'s radial jitter
does produce genuine variety, this was one less-convincing draw. Weights set moderately (not
maximal) partly in response to this: `irregular_shape_prob: 0.4`, `tear_prob: 0.25` in
`configs/photosub_v1.yaml` (independent draws: rect ≈45%, irregular-only ≈30%, tear-only ≈15%,
both ≈10%).

`arch`/`tape` (this pass's new additions) were added to `_SHAPE_VARIANTS` in
`scripts/analysis/photosub_render_sheets.py`'s `render_shape_variants` stage, ready to run —
**but this stage needs the regions cache (VESSL-only) and VESSL was unreachable this session**
(see "Smoke test" below). `arch_shape_prob`/`tape_prob` stay at **`0.0`** in
`configs/photosub_v1.yaml` until that sheet is regenerated and reviewed — no exception, same
gate every prior shape/effect addition went through (MODE_D's ghost-darkening decision,
irregular/tear above).

### MODE_B — unchanged

No code or weight changes. `irregular_shape_prob` exists on `generate_mode_b` from an earlier
pass (offered as an optional diversity variant since none of MODE_B's 3 confirmed deep-miss ids
showed the shape-realism gap) but stays at its existing default; not part of this spec's scope.

### MODE_C — evidence-bearing configurations only

**Ghostless-C dropped.** `generate_mode_c` (frame-aligned digital swap, color- and
degradation-matched, zero local statistical anomaly) is now restricted at generation time
(`scripts/generate_photosub_dataset.py`'s `no_ghost_modes`) to ghost-bearing templates only
(EGYPT/DL, MAURITIUS/ID) — the same restriction MODE_D already had. Rationale: on a template
with no ghost/secondary-portrait feature, a "clean" digital swap has genuinely **zero** evidence
any generator in this family can point to (no visible tell, no cross-region signature) — exactly
the "too clean" signature photosub_v0's deep-9 diagnostic suspects the model over-generalized
from (MODE_B/C converged fast and stayed strongly positive in training, but MODE_A — the mode
requiring an actual visible tell — collapsed for 2 of 4 real ids).

On a ghost-bearing template, `generate_mode_c` needs no code change: it never touches the ghost
region, so swapping the main portrait automatically creates a real cross-region mismatch
against the (untouched) ghost. **This makes ghost-restricted C functionally identical to
`generate_mode_d`'s `swap_target="main"` (D_main) branch** — both are "evidence must be real"
gates converging on the same 2-template population. This is a documented, accepted consequence,
not a bug — `C` and `D` stay separately tracked/weighted buckets (for `mode_weights` and the
per-mode twin-pairing control below) even though their generation recipe now overlaps on ghost
templates.

### MODE_D — unchanged in kind, still ghost-only

`ghost_darken_prob=0.3` (reproducing the real illegible-ghost case, `40dd1055fd`) carried over
unchanged from v0.

### Per-template cap

`freuid.photosub.mixing.select_mixed_rows`'s new `per_template_cap` parameter bounds any single
document template's share of EACH mode's own selection (`_select_with_template_cap`) —
independently per mode, applied AFTER the mode-weight allocation. Directly answers
`movement_census_part2_report.md`'s finding: MODE_D's ghost-mismatch generator concentrated
2.81x on MAURITIUS/ID vs 1.82x on EGYPT/DL relative to their real-fraud share. Set to `0.6` in
`configs/photosub_v1.yaml` — with only 2 eligible templates for C/D, this means neither can
exceed 60% of that mode's rows (vs. unconstrained in v0). Falls back to filling over-cap (with a
loud warning) rather than silently under-shooting the mode's target if the cap can't be met —
tested (`test_per_template_cap_fills_over_cap_when_only_one_template_available`).

### Twin pairs per-mode: ON for A/B, OFF for C/D

`freuid.photosub.mixing.is_pairable_mode` / `PhotosubTwinDataset.is_pairable` /
`TwinPairBatchSampler.is_pairable` restrict twin-pair seeking (and therefore
`pair_hinge_loss`'s targets) to A/B rows only. **Rationale — "the near-identical C/D twin is the
standing forgetting suspect"**: MODE_C/D's swap is color- and degradation-matched with zero
local statistical anomaly by design, so its bona-fide twin is near-pixel-identical to the
tampered row except for the exact substitution. photosub_v0's postmortem flagged this as the
most plausible mechanism for encouraging the model to fixate on tiny, non-generalizable pixel
artifacts from the generation process itself rather than a stable, generalizable "twin"
signal — unlike an A/B physical-paste pair, which differs by genuinely different local
statistics (rim, shadow, style mismatch). C/D rows still train normally via BCE + `auc_loss`;
they simply never seek a twin in-batch nor contribute to the hinge term. No config flag needed —
automatic based on each row's own `mode` column.

### Rehearsal weighting — measured, not assumed

**Finding, stated honestly**: a dedicated loader test
(`tests/test_photosub_mixing.py::TestRehearsalFrequency`) measured, rather than assumed, whether
real fraud rows keep finetune_v0-era (1x/epoch) sampling frequency once photosub rows are mixed
in. At photosub_v1-realistic settings (`share=0.15`, `twin_pair_prob=0.5`, on a 1000-row/300-fraud
synthetic population), **measured per-epoch fraud-row coverage came out ~90-93%, not ~100%** — a
real, non-negligible ~7-10% rehearsal-frequency cost from `TwinPairBatchSampler`'s displacement
mechanism (when a photosub item's forced twin needs a batch slot, the displaced item "simply
isn't seen this batch," per that class's own docstring — same tradeoff family as `drop_last`,
but with a materially larger effect size than `drop_last` alone). A control run at
`twin_pair_prob=0.0` confirmed the loss is attributable to the twin-pairing displacement
specifically, not some other bug (coverage loss there matches `drop_last`'s remainder exactly).

This was NOT part of this pass's requested fixes (the task asked to VERIFY the claim via a
loader test, not to redesign the sampler), so **v1 ships with this cost unaddressed** — reported
here so it's a known, quantified tradeoff for whoever reviews the full-run results, not a
silent assumption. A future iteration wanting to close this gap would need
`TwinPairBatchSampler` to either widen its search for a non-forced displacement slot beyond the
current batch, or accept a small increase in `drop_last`-style coverage loss as the cost of twin
pairing at this scale — out of scope here.

## Item 2: training instrumentation

Already built (prior session) and confirmed wired for any `photosub.enabled=True` config,
`photosub_v1.yaml` included — nothing new needed here beyond confirming the wiring:

- **`threshold_watch`** (`freuid.photosub.probes.threshold_watch`): exact per-epoch 1%-BPCER
  operating point from the current epoch's real val labels (no MC, no imputation needed).
- **Per-template ceiling probes**: `run_probe_hooks`'s `probe_ceiling_by_template` — each of the
  200 ceiling self-consistency probe ids classified by template (color-histogram classifier,
  `scripts/analysis/classify_ceiling_probe_types.py`), reporting each template's
  above-threshold count every epoch.
- **Deep-9 per-id table**: `probe_deep_detail`, logged individually every epoch (id, logit,
  score, pct_rank).
- **freuid/apcer in every probe evaluation**: `freuid.metrics.evaluate()` returns `freuid`
  alongside `audet`/`apcer_at_1pct_bpcer` (vendored-scorer-exact) for the main val pass, the
  recapture probe, and `probe_v2.py`.

## Item 3: pre-registered gates + checkpoint selection

Implemented in `freuid.photosub.checkpoint_select` (no prior spec existed for this in the repo —
confirmed by search before writing it). All three gates must pass for an epoch to be
**gate-composite eligible**:

1. **deep-9**: majority (`deep9_majority_frac=0.5`, i.e. ≥5/9) of the deep-9 ids score at or
   above this epoch's `threshold_watch` crossing score, **AND** no individual id's current logit
   falls below its own frozen finetune_v0-era baseline logit (`data/probes/
   missed_frauds_deep_ids.csv`'s own `logit` column) — a per-id regression check, not just an
   aggregate count, so one id quietly regressing can't hide behind the others improving.
2. **ceiling-200 per-template**: no template's above-threshold rate
   (`probe_ceiling_by_template`) drops by more than `ceiling_drop_tolerance` (0.20, relative)
   from this RUN's own epoch-1 baseline for that template. This is the APCER guard photosub_v0
   needed and didn't have — checked PER TEMPLATE, not in aggregate, so one template's regression
   can't be diluted by four others holding steady.
3. **boundary-59**: the above-threshold count does not drop below this run's own epoch-1
   baseline count.

Epoch-1 is used as the in-run baseline for gates 2/3 (not an external finetune_v0 number)
because these specific per-template/threshold-relative metrics are new instrumentation with no
finetune_v0-era measurement to compare against — the same convention the existing clean_floor/
ceiling mean-logit guards already use (tracked within one run).

**diff_gate vs finetune_v0 reference**: a POST-hoc, offline check (`scripts/analysis/
diff_gate.py --candidate submissions/photosub_v1.csv --reference submissions/finetune_v0.csv`)
against a real generated submission — not a per-epoch training-time gate, since it needs a full
TTA inference pass. Required to PASS (candidate not worse than reference in all 3 imputation
schemes) before any submission, per that tool's own gate logic.

### Checkpoint selection

- **Gate-composite eligibility**: epochs where all 3 gates above pass (`CheckpointTracker.
  eligible_epochs`).
- **Latest-stable**: the latest epoch E such that E and the next `stability_window - 1` (=1,
  i.e. 2 consecutive epochs total) are ALL eligible (`CheckpointTracker.latest_stable_epoch`) —
  avoids checkpointing on a single noisy good epoch surrounded by failures. Motivated directly
  by `docs/technical_report.md`'s own finding that photosub_v0's deep-9 per-id logits
  "oscillated substantially epoch to epoch before this pattern stabilized" — a real, observed
  risk in this exact training regime, not a hypothetical one.
- **Last-k averaging** (optional, `last_k_averaging: 3`): SWA-style arithmetic mean of the last
  3 eligible checkpoints' weights (`average_state_dicts`) — offered as an alternative candidate
  checkpoint, not a replacement for latest-stable.

**Additive, not destructive**: `train.py` still writes the existing `probe_audet`-based
`checkpoints/<name>.pt` exactly as before. Gate-composite selection writes
`checkpoints/<name>_gate_selected.pt` and (if enabled) `checkpoints/<name>_lastk_avg.pt`
alongside it — a run that opts in but whose gates never confirm eligible loses nothing; a config
that doesn't opt in (`checkpoint_selection.enabled` absent or `false`) is completely unaffected.

## Item 4: smoke test

**Plan**: `configs/photosub_v1.yaml` with `limit: <small>` (e.g. 256) via `--limit`, generate a
matching small `rows.csv` via `scripts/generate_photosub_dataset.py --n-total <small>
--restrict-ids-file <the --limit split's train ids> --irregular-shape-prob 0.4 --tear-prob 0.25
--enable-mode-d`, run `python -m freuid.train --config configs/photosub_v1.yaml --limit 256` for
~100 steps, confirm every instrumentation line emits (per-id deep-9 table, threshold-watch line,
per-template ceiling exposure line, `[gate]` composite line, all present in the log), then STOP
— no full 20-epoch run, no submission.

**Status: BLOCKED.** VESSL (`freuid-hy`) was unreachable this session — `ssh` timed out
(`Connection timed out` to `betelgeuse.cloud.vessl.ai:31430`) on two separate attempts, workspace
most likely suspended. Generation needs the regions cache and real train images (VESSL-only per
CLAUDE.md); training needs GPU (VESSL-only). Neither is runnable from this local CPU-only
session. Everything that does NOT need VESSL is done and verified locally: all generator/mixing/
checkpoint-selection code, all new unit tests (254 passing), the v1 config
(`scripts/config_diff.py` confirms it differs from `photosub_v0.yaml` in exactly the 8 documented
keys), and the render-sheet code extension (not yet executed).

**Next step, once VESSL is reachable**: (1) regenerate
`docs/photosub_renders/mode_a_shape_variants_sheet.png` with the arch/tape variants and review
before touching their `0.0` defaults, (2) run the smoke command above and confirm instrumentation
emits, (3) only then consider a full run — a separate, later go decision this spec does not make.

## Explicit non-goals (carried over from `threshold_pricing_out/threshold_pricing_report.md`)

**Do not hand-edit individual known ids' scores in a submission file** to game the public
leaderboard. Every diagnostic id referenced in this spec (the deep-9, the boundary-59, the 4
reviewed ceiling-B ids, `4c1ac0279e`) is evidence for what CAPABILITY is missing, never a
checklist of individual predictions to fix by hand. This would be pure public-LB probing that
cannot transfer to the private test set and directly contradicts this project's reprint-
robustness thesis (CLAUDE.md's Project section: a prior model that scored ~0.0006 locally but
~0.377 public by exploiting exactly this kind of local-only signal). If this is proposed later,
flag it as this same anti-pattern.
