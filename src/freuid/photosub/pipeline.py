"""Offline dataset-row glue: turns a generators.TamperResult into files on disk plus a
training-CSV-compatible row.

Generation happens OFFLINE, once, ahead of training -- NOT inside a DataLoader ``__getitem__``
like the existing on-the-fly ``freuid.augment.synth_tamper``. The saved image is just another
JPEG on disk with label=1; when (after the render-sheet human gate -- see
scripts/analysis/photosub_render_sheets.py) these rows are appended to a real training split,
they load through the exact same ``freuid.data.FreuidDataset`` / ``freuid.augment
.recapture_transforms`` pipeline as every other row, at the SAME augmentation probability/
parameters -- no per-mode special-casing anywhere in the loading path. This is a deliberate
design constraint (CLAUDE.md's recapture-augmentation invariant is label-independent by
construction; by the same logic it must also be mode-independent), not an oversight: MODE_C/D's
fiction is "digital", but they must still be seen through the analog-hole recapture chain
exactly like everything else, since the real test-set bona-fide cards are themselves
printed-and-captured.

Row schema (extra columns beyond ``train_labels.csv``'s own id/image_path/label/is_digital/type):
    source_id  -- the bona-fide TRAIN id this row was derived from. Split-discipline bookkeeping:
                  a train/val splitter must keep a row and its source_id on the SAME side, or the
                  synthetic row leaks validation-set appearance into training (twin pairing).
    mode       -- "A" / "B" / "C" / "D_main" / "D_ghost".
    mask_path  -- path to the saved binary tamper mask (generators.TamperResult.mask).
    params     -- JSON-encoded generator params dict, for provenance/debugging.
    donor_id   -- the donor pool id whose face was pasted/swapped in (None if not passed by the
                  caller -- e.g. earlier smoke-scale rows generated before this field existed).
                  Exists so a mass-generation pass can be audited for hard-case donor-pool
                  reuse/exhaustion (scripts/analysis/photosub_spot_review.py) -- not used by
                  training itself.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from freuid.photosub.generators import TamperResult


def photosub_generated_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "processed" / "photosub_generated"


def save_tamper_result(
    result: TamperResult,
    source_id: str,
    doc_type: str | None,
    data_dir: str | Path,
    row_index: int = 0,
    donor_id: str | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Writes the tampered JPEG + mask PNG to disk and returns a dataset row dict.

    ``row_index`` disambiguates multiple generations from the same source id (e.g. a hard-case
    and a broad-case donor variant of the same bona-fide row). ``donor_id`` is optional
    provenance (see module docstring's row-schema note) -- omit it and the row just carries an
    empty donor_id, same as rows generated before this field existed.

    ``out_dir`` overrides the default ``photosub_generated_dir(data_dir)`` location for the
    written images/masks. Needed because ids are deterministic
    (``{source_id}_photosub_{mode}_{row_index}``): a second generation run sharing the same
    default directory would silently overwrite an earlier run's files in place for any row with a
    matching (source_id, mode, row_index) -- exactly what happened when a v1 shape-realism
    regeneration pass shared a data_dir with the already-spot-reviewed v0 corpus, before this
    param existed. ``None`` (default) reproduces the original single-directory behavior exactly.
    """
    out_dir = Path(out_dir) if out_dir is not None else photosub_generated_dir(data_dir)
    img_dir, mask_dir = out_dir / "images", out_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    new_id = f"{source_id}_photosub_{result.mode}_{row_index:03d}"
    image_path = img_dir / f"{new_id}.jpeg"
    mask_path = mask_dir / f"{new_id}.png"

    result.image.convert("RGB").save(image_path, quality=95)
    Image.fromarray(result.mask).save(mask_path)

    return {
        "id": new_id,
        "image_path": str(image_path),
        "label": 1,
        "is_digital": True,  # a generated composite, not a real analog recapture (yet)
        "type": doc_type,
        "source_id": source_id,
        "mode": result.mode,
        "mask_path": str(mask_path),
        "params": json.dumps(result.params),
        "donor_id": donor_id,
    }


_ROW_FIELDS = [
    "id", "image_path", "label", "is_digital", "type", "source_id", "mode", "mask_path",
    "params", "donor_id",
]


def append_rows_csv(rows: list[dict], csv_path: str | Path) -> None:
    """Append generated rows to a photosub dataset-rows CSV, writing the header if new."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_ROW_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in _ROW_FIELDS})
