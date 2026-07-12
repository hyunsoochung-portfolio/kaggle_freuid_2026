# MODE_A shape-realism fix -- human-review gate

`mode_a_shape_variants_sheet.png`: one row per template (EGYPT/DL, GUINEA/DL, BENIN/DL, MOZAMBIQUE/DL, MAURITIUS/ID), one column per variant (rect, irregular, tear, irregular_tear) -- `rect` is today's unchanged baseline (irregular_shape_prob=0.0, tear_prob=0.0), the other 3 exercise freuid.photosub.generators.generate_mode_a's new kwargs.

Compare against the real deep-miss exemplars this is fixing (deep_miss_dossiers.py's CHECKLIST_RESULTS / deep_miss_dossiers.html): `b5eebda1` (arch-shaped cutout overlapping the crest logo), `40dd1055` (irregular silhouette bulging past the hairline, dark shadow strip), `5542f45f` (literal diagonal tear exposing a lighter backing patch).

**This is the gate before any mass regeneration or weight decision for irregular_shape_prob / tear_prob** -- same convention already used for the MODE_D ghost-darkening decision. Do not proceed to mass generation until this sheet has been reviewed and a weight approved.
