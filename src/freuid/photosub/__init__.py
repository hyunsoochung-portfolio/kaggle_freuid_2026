"""Photo-substitution training-data generators (MODE_A/B/C/D).

See scripts/analysis/deep_miss_dossiers.py's module docstring for the taxonomy these modes
implement (confirmed against 3 real deep-miss frauds, extended-by-eye to 6 more -- see that
script's CHECKLIST_RESULTS). This package turns that taxonomy into deterministic, offline
generators that read bona-fide TRAIN images and a donor face pool and write new (image, mask,
mode, params) rows -- see ``pipeline.py`` for how a row is saved to disk, and
``scripts/analysis/photosub_render_sheets.py`` for the human-review render sheets.

No training happens here. Generation, once approved via the render-sheet human gate, is a
separate follow-up step (writing the full offline dataset), not part of this package.
"""

from __future__ import annotations
