"""Per-id photo-substitution template metadata: photo-frame box + ghost/secondary-portrait box.

New cache file under ``{data_dir}/processed/photosub_template/{id}/template.json`` -- does NOT
mutate the existing regions cache (``card.png`` / ``face.json`` under
``{data_dir}/processed/regions/{id}/``, see ``freuid.preprocess``).

**Photo-frame box** is NOT a manually-surveyed per-template rectangle (no such survey exists for
these document types, unlike the ghost regions below) -- it is derived from the id's own cached
SCRFD face box, expanded by ``FRAME_MARGIN_FRAC`` on every side. This is the exact convention
``scripts/analysis/deep_miss_dossiers.py`` established for its "photo-frame perimeter" zoom
panels (``FRAME_MARGIN_FRAC = 0.6`` there too). It is an approximation of the true rectangular
photo slot printed on the card, not a ground-truth template annotation -- MODE_C in particular
assumes a precise frame edge ("seam precisely on the frame edge"), so how well this approximation
holds is exactly the kind of thing the render-sheet human gate (``docs/photosub_renders/``)
should scrutinize before any mass generation.

**Ghost/secondary-portrait box** is a manually-verified per-TYPE fractional bbox, duplicated here
from ``scripts/analysis/deep_miss_dossiers.py``'s ``GHOST_TEMPLATES`` survey (identical values --
see that module's ``survey_templates`` stage for the visual evidence). Duplicated rather than
imported because ``src/freuid`` must not depend on ``scripts/analysis`` (the reverse is the
existing layering throughout this repo). Only EGYPT/DL and MAURITIUS/ID carry one; every other
known type -- and any unseen type -- has no ghost box, and MODE_D generation is skipped for them.
"""

from __future__ import annotations

import json
from pathlib import Path

FRAME_MARGIN_FRAC = 0.6  # matches scripts/analysis/deep_miss_dossiers.py's FRAME_MARGIN_FRAC

# Fractions of the full (original, un-rectified) image's (width, height) -- same convention as
# scripts/analysis/deep_miss_dossiers.py's GHOST_TEMPLATES and ghost_resolvability.csv.
GHOST_TEMPLATES: dict[str, tuple[float, float, float, float] | None] = {
    "EGYPT/DL": (0.855, 0.28, 0.985, 0.55),
    "MAURITIUS/ID": (0.716, 0.625, 0.834, 0.793),
    "GUINEA/DL": None,
    "BENIN/DL": None,
    "MOZAMBIQUE/DL": None,
}


def template_regions_dir(data_dir: str | Path) -> Path:
    """Canonical path to the NEW photosub template-metadata cache under a data root."""
    return Path(data_dir) / "processed" / "photosub_template"


def frame_box_from_face(
    face_box: dict, img_w: int, img_h: int, margin_frac: float = FRAME_MARGIN_FRAC,
) -> dict:
    """Photo-frame box for a card image: the SCRFD face box expanded by ``margin_frac`` on
    each side, clamped to the image bounds. See module docstring for the approximation caveat."""
    x1, y1, x2, y2 = face_box["x1"], face_box["y1"], face_box["x2"], face_box["y2"]
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin_frac, bh * margin_frac
    fx1 = max(0, int(round(x1 - mx)))
    fy1 = max(0, int(round(y1 - my)))
    fx2 = min(img_w, int(round(x2 + mx)))
    fy2 = min(img_h, int(round(y2 + my)))
    return {"x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2}


def ghost_box_px(doc_type: str | None, img_w: int, img_h: int) -> dict | None:
    """Ghost-region box in pixel coords for ``doc_type``, or None if that type has no ghost
    (includes unseen/unknown types, since GHOST_TEMPLATES.get(None) is also None)."""
    frac = GHOST_TEMPLATES.get(doc_type)
    if frac is None:
        return None
    x1, y1, x2, y2 = frac
    return {"x1": int(x1 * img_w), "y1": int(y1 * img_h), "x2": int(x2 * img_w), "y2": int(y2 * img_h)}


def build_template_metadata(id_: str, doc_type: str | None, face_box: dict, img_w: int, img_h: int) -> dict:
    """The full template-metadata record for one id -- what generators.py consumes."""
    return {
        "id": id_,
        "type": doc_type,
        "frame_box": frame_box_from_face(face_box, img_w, img_h),
        "frame_box_source": "face_box_expanded",  # see module docstring's approximation caveat
        "frame_margin_frac": FRAME_MARGIN_FRAC,
        "ghost_box": ghost_box_px(doc_type, img_w, img_h),
    }


def precache_template_metadata(
    data_dir: str | Path,
    regions_dir_path: Path,
    rows: list,  # iterable of objects/rows with .id, .type, .path (e.g. load_labels() itertuples)
    limit: int | None = None,
) -> None:
    """Write ``template.json`` for each row whose ``regions_dir_path/{id}/face.json`` exists and
    has a real (non-fallback) SCRFD detection (``score > 0``). Idempotent -- skips ids whose
    ``template.json`` already exists, mirroring ``freuid.preprocess.precache_regions``'s
    convention. Only meaningful for bona-fide TRAIN rows (the generators' paste targets); rows
    with an unknown ``type`` still get a ``frame_box`` (face-derived) but ``ghost_box=None``.
    """
    out_root = template_regions_dir(data_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    items = list(rows)
    if limit is not None:
        items = items[:limit]

    n_done = n_skip = 0
    for row in items:
        out_path = out_root / str(row.id) / "template.json"
        if out_path.exists():
            n_skip += 1
            continue
        face_path = Path(regions_dir_path) / str(row.id) / "face.json"
        if not face_path.exists():
            n_skip += 1
            continue
        try:
            face_box = json.loads(face_path.read_text())
        except Exception:
            n_skip += 1
            continue
        if float(face_box.get("score", 0.0)) <= 0.0:
            n_skip += 1
            continue
        if not Path(str(row.path)).exists():
            n_skip += 1
            continue

        from PIL import Image

        with Image.open(row.path) as img:
            img_w, img_h = img.size
        doc_type = getattr(row, "type", None)
        meta = build_template_metadata(str(row.id), doc_type, face_box, img_w, img_h)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(meta), encoding="utf-8")
        n_done += 1

    print(f"[photosub.template_regions] wrote {n_done} template.json, skipped {n_skip}")


def load_template_metadata(data_dir: str | Path, id_: str) -> dict | None:
    p = template_regions_dir(data_dir) / str(id_) / "template.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None
