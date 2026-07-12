# Photo-substitution render sheets -- human-review gate

Donor pool: 3866 bona-fide TRAIN faces (post probe-overlap exclusion). Templates: all 5 known document types (EGYPT/DL, GUINEA/DL, BENIN/DL, MOZAMBIQUE/DL, MAURITIUS/ID) -- **only 5 exist in this dataset**, see this script's module docstring; there is no unseen 6th template to add.

## Sheets

- `mode_A_sheet.png` -- 12 examples (6 hard-donor + 6 broad), each a 2x/4x frame-perimeter zoom, pre- vs. post-recapture@518 side by side
- `mode_B_sheet.png` -- 12 examples (6 hard-donor + 6 broad), each a 2x/4x frame-perimeter zoom, pre- vs. post-recapture@518 side by side
- `mode_C_sheet.png` -- 12 examples with the stats-delta + PASS/FAIL verdict in each caption
- `stats_check.csv` / `stats_check_report.md` -- MODE_C/D natural-baseline pass rate

**This is the gate before mass generation -- no full dataset has been written.** Review the sheets above (and the shape-realism / frame-box-accuracy caveats in freuid.photosub.generators and .template_regions' module docstrings) before approving.

