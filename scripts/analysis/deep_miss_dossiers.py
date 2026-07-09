"""Per-id evidence dossiers for finetune_v0's confirmed low-ranked frauds, plus a
photo-substitution-taxonomy check across the 9 deep-miss ids and the 59 TRANS_5 boundary ids.

Background (see review_package_deep_ingest_report.md / CLAUDE.md): the deep-review CSV
(`review_package_deep_done.csv`) confirmed 9 ids in TRANS_1/3/4 are genuinely fraud sitting at
low/mid rank ("deep misses"), plus 59 TRANS_5 ids right at the ceiling boundary. A close visual
read of 3 of the 9 deep-miss ids (done outside this script, by eye) found a coherent
photo-substitution taxonomy:

    MODE_A -- within-frame physical paste: style-mismatched printed portrait (grayscale/
              halftone vs color card), paper rim, physical shadow, severed background print at
              the frame; then recaptured.
    MODE_B -- full-cover physical paste: photo covers or overhangs the entire photo region;
              paste boundary coincides with the card's structural edge; rim + corner shadows +
              style mismatch; recaptured.
    MODE_C -- frame-aligned digital swap: whole portrait replaced inside the photo frame, so the
              splice seam hides in a legitimate structural boundary; zero local statistical
              anomaly; evidence is cross-region inconsistency (ghost image shows a different
              person; gender field mismatch; lighting/rendering style mismatch).

    Assigned (by eye, prior to this script): c6651aee9e494aaa89747630524f1ab7 (Mozambique/DL,
    TRANS_1) = MODE_A; 7b409d3bd41844e5b62ae6a936c9b0e8 (Benin/DL, TRANS_3) = MODE_B;
    a2a3fe5bcd3c4a808b188e355ed052b7 (Mauritius/ID, TRANS_4) = MODE_C. Benin/DL being caught
    when digitally edited (elsewhere in the training data) but missed here when physically
    pasted is the controlled contrast that motivates checking whether this is a MODALITY gap
    (digital-edit detection vs physical-recapture-of-a-paste detection), not a document-type gap.

This script does NOT propose a fix -- evidence gathering only:

  --stage freeze_probes      Freeze the 3 diagnostic probe id lists (deep-miss/boundary/clean-
                              floor) + a provenance README under data/probes/. Pure CSV, local.
  --stage survey_templates   Render several bona-fide TRAIN images per known document type at
                              full card size, to visually check which templates carry a ghost/
                              secondary-portrait security feature and roughly where. Local only
                              (no regions cache needed -- full-card renders, no face box).
  --stage dossiers           Per-id evidence dossier for the 9 deep-miss ids: full card, SCRFD
                              box, 2x/4x zooms of the face region and the photo-frame perimeter,
                              a ghost-region zoom where GHOST_TEMPLATES defines one for that id's
                              doc_type_proxy, plus metadata (raw logit, per-TTA-scale logit,
                              pct_rank, degradation stats). Needs the regions cache -> VESSL.
  --stage boundary_grids      Same evidence checklist, batched as 5x5 grids, for the 59 TRANS_5
                              boundary F ids. Needs the regions cache -> VESSL.
  --stage ghost_resolvability Measures each ghost region's pixel size at native resolution and
                              after resize to each TTA scale, for every (template, reviewed-id)
                              pair where a ghost region is defined. Pure CPU/PIL, local.
  --stage assemble           Combine everything into deep_miss_dossiers.html. Local, pure
                              string/image assembly -- no GPU.

No training, no submissions, no modification of existing source files or the regions cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.data import load_labels  # noqa: E402
from freuid.preprocess import regions_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import df_to_md  # noqa: E402
from hesitant_clusters import KNOWN_TYPES  # noqa: E402
from occlusion_test import read_face_box  # noqa: E402
from review_package import _load_font, render_cell, render_sheet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEEP_DONE_CSV = Path(__file__).resolve().parent / "review_package_deep_done.csv"
DEFAULT_LOGIT_CENSUS_CSV = Path(__file__).resolve().parent / "logit_census_raw.csv"
DEFAULT_PROBES_DIR = REPO_ROOT / "data" / "probes"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "deep_miss_dossiers_out"
DEFAULT_HTML_OUT = Path(__file__).resolve().parent / "deep_miss_dossiers.html"

TTA_SCALES = (476, 518, 560)  # CLAUDE.md: finetune_v0's TTA scale list, all multiples of 14

# --- Photo-substitution taxonomy (assigned by eye, outside this script -- see module docstring) ---
MODE_ASSIGNMENTS: dict[str, str] = {
    "c6651aee9e494aaa89747630524f1ab7": "A",  # Mozambique/DL, TRANS_1
    "7b409d3bd41844e5b62ae6a936c9b0e8": "B",  # Benin/DL, TRANS_3
    "a2a3fe5bcd3c4a808b188e355ed052b7": "C",  # Mauritius/ID, TRANS_4
}
MODE_DESCRIPTIONS: dict[str, str] = {
    "A": "within-frame physical paste (style-mismatched print, paper rim, physical shadow, "
         "severed background print at the frame; recaptured)",
    "B": "full-cover physical paste (photo covers/overhangs the whole photo region, paste "
         "boundary coincides with the card's structural edge, rim + corner shadows + style "
         "mismatch; recaptured)",
    "C": "frame-aligned digital swap (whole portrait replaced inside the photo frame, seam "
         "hides in a legitimate structural boundary, zero local statistical anomaly -- evidence "
         "is cross-region: ghost shows a different person, gender/name mismatch, lighting/"
         "rendering style mismatch)",
}

# --- Ghost/secondary-portrait template survey. Filled in via --stage survey_templates + a
# direct visual check of 2 bona-fide TRAIN samples per known type (both samples agreed in every
# case -- see survey_templates_report.md for the rendered evidence): EGYPT/DL and MAURITIUS/ID
# both carry a small secondary/duplicate portrait as a legitimate security feature (grayscale,
# top-right near the "ET" oval for EGYPT/DL; a tinted rounded-square near the "SC" mark for
# MAURITIUS/ID); GUINEA/DL, BENIN/DL, MOZAMBIQUE/DL do not. None = confirmed no ghost feature.
# Bounding boxes are FRACTIONS of the card image's (width, height), since templates are
# standardized layouts reproduced at different source resolutions. ---
GHOST_TEMPLATES: dict[str, tuple[float, float, float, float] | None] = {
    "EGYPT/DL": (0.855, 0.28, 0.985, 0.55),
    "MAURITIUS/ID": (0.716, 0.625, 0.834, 0.793),
    "GUINEA/DL": None,
    "BENIN/DL": None,
    "MOZAMBIQUE/DL": None,
}

# --- doc_type_proxy is a 5-NN vote in a fine-tuned embedding space already documented as
# collapsed/unreliable for near-median-rank populations (hesitant_report.md). Direct visual
# inspection of all 9 deep-miss ids' full-resolution originals (notebooks/trans_1_4_flagged_out/)
# found the proxy WRONG for 7/9 of them -- so ghost-template lookup and reporting for these 9
# specific ids uses this manually-confirmed override instead of the proxy. ---
VISUAL_TYPE_OVERRIDES: dict[str, str] = {
    "c6651aee9e494aaa89747630524f1ab7": "MOZAMBIQUE/DL",  # proxy said MAURITIUS/ID
    "b5eebda1d20743d9a364523ee2b6e506": "BENIN/DL",        # proxy said MAURITIUS/ID
    "40dd1055fd7b4fedb5a34d3ad34e8960": "MAURITIUS/ID",    # proxy agreed
    "5542f45f55a74485802f33efdd58d677": "MOZAMBIQUE/DL",  # proxy said EGYPT/DL
    "7b409d3bd41844e5b62ae6a936c9b0e8": "BENIN/DL",        # proxy said EGYPT/DL
    "cd7ad569a66244ff9adb29f7182cc270": "BENIN/DL",        # proxy said EGYPT/DL
    "2d4ad17d2e1149bd8632f503d82e7455": "BENIN/DL",        # proxy said EGYPT/DL
    "cceb6a4f987d4cdaa92979217cf99667": "MAURITIUS/ID",    # proxy said BENIN/DL
    "a2a3fe5bcd3c4a808b188e355ed052b7": "MAURITIUS/ID",    # proxy agreed
}


def visual_type(id_: str, proxy_fallback: str) -> str:
    return VISUAL_TYPE_OVERRIDES.get(id_, proxy_fallback)


# --- Evidence checklist for the 6 uninspected deep-miss ids (TRANS_1/3/4 F-verdict ids not
# among the 3 pre-assigned above). Filled in by direct visual inspection of --stage dossiers'
# rendered panels (dossiers_out/*_full.jpeg, *_face_*x.jpeg, *_frame_*x.jpeg, *_ghost_*x.jpeg)
# -- a manual/judgment call, same as MODE_ASSIGNMENTS, not something this script automates.
# Each entry: paper_rim / paste_shadow / style_mismatch / boundary_aligned / ghost_present /
# ghost_mismatch / gender_name_mismatch are "yes"/"no"/"n/a" (n/a = no ghost region defined for
# this id's type, so that question doesn't apply); proposed_mode is "A"/"B"/"C"/"other" -- flag
# 'other' honestly rather than forcing a fit. Populated after --stage dossiers has been run on
# VESSL and the renders pulled back; empty dict = not yet inspected. ---
CHECKLIST_RESULTS: dict[str, dict] = {
    "b5eebda1d20743d9a364523ee2b6e506": {
        # TRANS_1, Benin/DL. Arch-shaped hand-cut paper silhouette overlapping the crest logo at
        # top; grayscale/halftone print against the card's color background; the paper's own
        # gray background is visible, severed at an irregular (non-rectangular) boundary -- the
        # classic MODE_A signature. No tape/shadow as pronounced as the confirmed MODE_B example.
        "paper_rim": "yes", "paste_shadow": "no (none clearly visible)", "style_mismatch": "yes (grayscale/halftone vs color card)",
        "boundary_aligned": "no (arch shape overlaps the crest logo, doesn't match the rectangular photo slot)",
        "ghost_present": "n/a (Benin/DL has no ghost template)", "ghost_mismatch": "n/a",
        "gender_name_mismatch": "n/a (no independent field to cross-check)",
        "proposed_mode": "A",
        "notes": "Compared directly against confirmed MODE_B exemplar 7b409d3bd4 (same template): "
                 "that one has a rectangular tape+shadow paste; this one has an arch-shaped cutout "
                 "with severed background -- a materially different, MODE_A-matching signature.",
    },
    "40dd1055fd7b4fedb5a34d3ad34e8960": {
        # TRANS_1, Mauritius/ID (has ghost template). Irregular hand-cut silhouette bulging past
        # the person's own hair/shoulders, grayscale halftone print, visible dark shadow strip
        # beneath the paste, distinct dark purple-gray backing behind the cutout -- very clear
        # physical-paste evidence, arguably clearer than the original c6651aee exemplar.
        "paper_rim": "yes (irregular hand-cut edge, clearly visible)", "paste_shadow": "yes (dark strip beneath the paste)",
        "style_mismatch": "yes (grayscale/halftone vs color card)",
        "boundary_aligned": "no (hand-cut silhouette, doesn't match any regular frame shape)",
        "ghost_present": "yes", "ghost_mismatch": "UNKNOWN -- ghost region is present but the source image is very dark/underexposed; essentially illegible even at 4x zoom (see ghost_resolvability_report.md's contrast-vs-size distinction)",
        "gender_name_mismatch": "n/a (ghost illegible, can't cross-check)",
        "proposed_mode": "A",
        "notes": "Strongest MODE_A physical-paste evidence of the 6 uninspected ids. The ghost "
                 "region exists but is unreadable due to exposure/darkness, not size -- a separate "
                 "legibility axis from the resolution question in ghost_resolvability_report.md.",
    },
    "5542f45f55a74485802f33efdd58d677": {
        # TRANS_3, Mozambique/DL. A literal diagonal TEAR/rip visible through the upper-right of
        # the photo, exposing a lighter patch behind it -- unambiguous physical paper damage/
        # paste evidence. Photo is in COLOR (unlike the grayscale MODE_A exemplars), so the
        # "style mismatch" sub-symptom doesn't apply here even though the core paste evidence does.
        "paper_rim": "yes (via a visible diagonal tear exposing what's behind)", "paste_shadow": "unclear (tear dominates, no separate shadow clearly visible)",
        "style_mismatch": "no (color-on-color, unlike the grayscale MODE_A exemplar)",
        "boundary_aligned": "unclear (photo occupies the normal slot, but the tear itself is the key anomaly, not the outer boundary)",
        "ghost_present": "n/a (Mozambique/DL has no ghost template)", "ghost_mismatch": "n/a",
        "gender_name_mismatch": "n/a",
        "proposed_mode": "A",
        "notes": "Classified as MODE_A on the strength of the physical tear alone (clearest single "
                 "piece of physical-paste evidence among all 9 ids), even though the grayscale/"
                 "halftone sub-symptom doesn't apply -- MODE_A's defining feature is the physical "
                 "paste/severed-paper evidence, not specifically the color style mismatch.",
    },
    "cd7ad569a66244ff9adb29f7182cc270": {
        # TRANS_4, Benin/DL. Visible tan/yellow TAPE strip at the top of the photo; photo is
        # visibly ROTATED/crooked relative to the card's own rectangular frame, with a sliver of
        # the card's background peeking out on the right; soft shadow along the right/bottom
        # edges consistent with a slightly raised paste. Near-identical signature family to the
        # confirmed MODE_B exemplar (7b409d3bd4).
        "paper_rim": "yes", "paste_shadow": "yes (soft shadow along right/bottom edges)",
        "style_mismatch": "no (color-on-color)",
        "boundary_aligned": "yes (paste sized to cover the full designated photo region, tilted but not overhanging into surrounding elements)",
        "ghost_present": "n/a (Benin/DL has no ghost template)", "ghost_mismatch": "n/a",
        "gender_name_mismatch": "n/a",
        "proposed_mode": "B",
        "notes": "Tape + shadow + crooked full-cover placement closely mirrors the confirmed "
                 "MODE_B exemplar on the same template.",
    },
    "2d4ad17d2e1149bd8632f503d82e7455": {
        # TRANS_4, Benin/DL. The pasted photo's own tan/beige background forms a visible
        # rectangle that does NOT contain the full hair silhouette -- his hair overflows past the
        # top of that rectangle onto the card's own white background, revealing the paste
        # boundary. Less dramatic than cd7ad569a6 (no obvious tape/shadow/tilt spotted), but the
        # same underlying template/signature family and the same rectangle-boundary-mismatch tell.
        "paper_rim": "yes (tan paste-background rectangle visibly smaller than the subject's hair)", "paste_shadow": "unclear (subtle, not confident)",
        "style_mismatch": "no (color-on-color)",
        "boundary_aligned": "no (hair overflows the pasted rectangle's own background)",
        "ghost_present": "n/a (Benin/DL has no ghost template)", "ghost_mismatch": "n/a",
        "gender_name_mismatch": "n/a",
        "proposed_mode": "B",
        "notes": "Lower-confidence than cd7ad569a6 -- less dramatic evidence -- but grouped with "
                 "it as the same template/signature family (both Benin/DL, both a full-slot-sized "
                 "paste rather than an irregular hand-cut arch like the MODE_A Benin/Mozambique "
                 "cases).",
    },
    "cceb6a4f987d4cdaa92979217cf99667": {
        # TRANS_4, Mauritius/ID (has ghost template). No visible physical-paste evidence at all
        # (no rim, tape, shadow, tear, or style mismatch) -- consistent with MODE_C's "zero local
        # statistical anomaly". The ghost region IS present but this source image is one of the
        # smaller/lower-resolution ones in the set (858x541 native) and the ghost crop is blurry;
        # it shows a short-haired silhouette that does not obviously contradict the main
        # portrait's short-haired man -- i.e. it does NOT provide the same crisp cross-person
        # mismatch the confirmed MODE_C exemplar (a2a3fe5b) shows.
        "paper_rim": "no", "paste_shadow": "no", "style_mismatch": "no",
        "boundary_aligned": "yes (clean digital-looking insert, no visible seam)",
        "ghost_present": "yes", "ghost_mismatch": "INCONCLUSIVE -- ghost too low-resolution/blurry to confirm or rule out a different person; silhouette doesn't obviously contradict the main portrait",
        "gender_name_mismatch": "no obvious mismatch found",
        "proposed_mode": "C (tentative -- by elimination, not by positive ghost confirmation)",
        "notes": "Genuinely the most uncertain of the 6. Absence of any physical-paste evidence "
                 "points away from A/B, but the positive MODE_C signature (ghost showing a "
                 "DIFFERENT person) that made a2a3fe5b unambiguous is NOT clearly present here -- "
                 "flagging this honestly as tentative rather than forcing a confident C.",
    },
}


# ---------------------------------------------------------------------------
# Shared data loading
# ---------------------------------------------------------------------------

def load_deep_done(csv_path: Path = DEFAULT_DEEP_DONE_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"id": str, "verdict": str, "note": str})
    df["verdict"] = df["verdict"].fillna("").str.strip()
    return df


def deep_miss_ids(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["stratum"].isin(["TRANS_1", "TRANS_2", "TRANS_3", "TRANS_4"])) & (df["verdict"] == "F")]
    return sub.sort_values(["stratum", "logit"]).reset_index(drop=True)


def boundary_ids(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["stratum"] == "TRANS_5") & (df["verdict"] == "F")]
    return sub.sort_values("logit", ascending=False).reset_index(drop=True)


def clean_floor_ids(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df["stratum"].isin(["FLOOR_DEEP_TOP", "FLOOR_DEEP_REST"])) & (df["verdict"] == "B")]
    return sub.sort_values(["stratum", "logit"]).reset_index(drop=True)


def attach_paths_and_scale_logits(df: pd.DataFrame, data_dir: str, logit_census_csv: Path) -> pd.DataFrame:
    test_meta = load_labels(data_dir, "public_test")[["id", "path"]]
    df = df.merge(test_meta, on="id", how="left")
    census = pd.read_csv(logit_census_csv, dtype={"id": str})
    scale_cols = [c for c in census.columns if c.startswith("logit_")]
    df = df.merge(census[["id", *scale_cols]], on="id", how="left")
    return df


# ---------------------------------------------------------------------------
# Stage: freeze_probes
# ---------------------------------------------------------------------------

def freeze_probes(args) -> None:
    df = load_deep_done(Path(args.deep_done_csv))
    deep = deep_miss_ids(df)
    boundary = boundary_ids(df)
    floor = clean_floor_ids(df)

    out_dir = Path(args.probes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["id", "stratum", "logit", "pct_rank", "doc_type_proxy", "verdict"]
    deep_out = deep[cols].copy()
    deep_out["mode"] = deep_out["id"].map(MODE_ASSIGNMENTS).fillna("")
    deep_out.to_csv(out_dir / "missed_frauds_deep_ids.csv", index=False)
    boundary[cols].to_csv(out_dir / "missed_frauds_boundary_ids.csv", index=False)
    floor[cols].to_csv(out_dir / "clean_floor_sample_ids.csv", index=False)

    print(f"[deep_miss] froze {len(deep_out)} deep-miss ids, {len(boundary)} boundary ids, "
          f"{len(floor)} clean-floor ids -> {out_dir}")

    readme = f"""# Diagnostic probe id lists -- provenance

**These are diagnostic probes for manual/automated evidence-gathering, NEVER training data.**
Nothing here is used to fit, validate, or select a checkpoint; they exist only to let follow-up
scripts (e.g. `scripts/analysis/deep_miss_dossiers.py`) re-load exactly the same id sets used in
this round of manual review without re-deriving them.

All three files are filtered rows of `scripts/analysis/review_package_deep_done.csv` (the
manually-reviewed deep-review CSV from `review_package.py --stage build_deep`), which itself
samples finetune_v0's public-test predictions per `scripts/analysis/review_package_deep_report.md`.

## missed_frauds_deep_ids.csv ({len(deep_out)} ids)

Rows: `stratum in (TRANS_1, TRANS_2, TRANS_3, TRANS_4)` AND `verdict == 'F'` -- ids in the
transitional zone (excluding TRANS_5, which sits immediately against the ceiling mode and is
handled separately as `missed_frauds_boundary_ids.csv`) that a manual reviewer confirmed are
genuinely fraud despite the model ranking them as if bona-fide. `mode` column carries the
photo-substitution-taxonomy assignment (A/B/C) for the 3 ids visually inspected before this
script existed; blank for the 6 not yet assigned.

## missed_frauds_boundary_ids.csv ({len(boundary)} ids)

Rows: `stratum == 'TRANS_5'` AND `verdict == 'F'`. TRANS_5 sits immediately adjacent to the
ceiling mode (see `logit_census_report.md`) -- a high F rate there is an expected boundary
effect, not itself surprising. This set exists to check what FRACTION of that boundary mass is
also photo-substitution (same taxonomy) vs. something else, since a fix that recovers the
photo-substitution modes could plausibly recover a meaningful share of this ~59/80 = 73.8%-rate
population too.

## clean_floor_sample_ids.csv ({len(floor)} ids)

Rows: `stratum in (FLOOR_DEEP_TOP, FLOOR_DEEP_REST)` AND `verdict == 'B'`. **Note**: the build
spec called for "300 floor B-verdict ids"; the actual filtered count is **{len(floor)}**, not
300 -- 5 of the 300 originally-sampled floor ids were marked `U` (unfamiliar/unreadable) rather
than `B`, so they're correctly excluded here rather than padded to force a round number. Serves
as a negative-control / clean-example set for any follow-up that needs bona-fide examples known
NOT to carry the photo-substitution evidence being catalogued in the other two files.

## Regenerating

`python scripts/analysis/deep_miss_dossiers.py --stage freeze_probes`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[deep_miss] wrote provenance README -> {out_dir / 'README.md'}")


# ---------------------------------------------------------------------------
# Stage: survey_templates
# ---------------------------------------------------------------------------

SURVEY_SEED = 1
SURVEY_N_PER_TYPE = 2


def survey_templates(args) -> None:
    """Renders (reproducibly, same seed/sample as the manual pass that filled in
    GHOST_TEMPLATES above) the bona-fide TRAIN samples used to determine which known document
    types carry a ghost/secondary-portrait feature. The ghost/no-ghost determination itself is a
    visual call (recorded in GHOST_TEMPLATES's own comment + the report text below) -- this
    stage exists to make that call re-inspectable, not to re-derive it automatically."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_labels(args.data_dir, "train")
    bonafide = train_df[train_df["label"] == 0].copy()
    bonafide = bonafide[bonafide["path"].map(lambda p: Path(p).exists())]

    rows = []
    for t in KNOWN_TYPES:
        sub = bonafide[bonafide["type"] == t]
        sample = sub.sample(n=min(SURVEY_N_PER_TYPE, len(sub)), random_state=SURVEY_SEED)
        for _, row in sample.iterrows():
            dest = out_dir / f"survey_{t.replace('/', '_')}_{row['id'][:10]}.jpeg"
            Image.open(row["path"]).convert("RGB").save(dest)
            rows.append({"type": t, "id": row["id"], "rendered": str(dest)})
            print(f"[deep_miss] survey sample: {t} {row['id'][:10]} -> {dest}")

    lines = ["# Document-template ghost/secondary-portrait survey\n"]
    lines.append(
        f"{SURVEY_N_PER_TYPE} bona-fide TRAIN samples per known type (seed={SURVEY_SEED}), "
        "rendered below for re-inspection. The ghost/no-ghost call itself was made by direct "
        "visual read (both samples agreed in every case) -- this is a manual, evidence-only "
        "determination, same as the mode-assignment checklist elsewhere in this script.\n"
    )
    lines.append("## Findings\n")
    lines.append(
        "| type | has ghost? | location | evidence |\n"
        "| --- | --- | --- | --- |\n"
        "| EGYPT/DL | **YES** | grayscale secondary portrait, top-right near the 'ET' oval mark "
        "(fractional bbox `(0.855, 0.28, 0.985, 0.55)`) | both samples show the ghost matching "
        "the main portrait, as expected for bona-fide |\n"
        "| GUINEA/DL | no | -- (ECOWAS seal graphic occupies that area, no secondary face) | -- |\n"
        "| BENIN/DL | no | -- (Statue-of-Liberty watermark graphic, no secondary face) | -- |\n"
        "| MOZAMBIQUE/DL | no | -- (barcode strip at bottom, no secondary face) | -- |\n"
        "| MAURITIUS/ID | **YES** | tinted rounded-square secondary portrait near the 'SC' mark "
        "and swan/bird graphic (fractional bbox `(0.716, 0.625, 0.834, 0.793)`) | both samples "
        "show the ghost matching the main portrait; tint color varies (teal, purple) between "
        "samples, possibly per-image watermark randomization |\n"
    )
    lines.append("\n## Rendered samples\n")
    for r in rows:
        lines.append(f"- {r['type']} `{r['id'][:10]}`: {_relpath_or_abs(Path(r['rendered']))}")
    (out_dir / "survey_templates_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[deep_miss] wrote survey report -> {out_dir / 'survey_templates_report.md'}")


def _relpath_or_abs(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# Ghost resolvability math (pure, no GPU) -- used by --stage ghost_resolvability
# ---------------------------------------------------------------------------

def ghost_px_at_scale(
    card_size: tuple[int, int], ghost_bbox_frac: tuple[float, float, float, float], scale: int,
) -> tuple[float, float]:
    """Native (width, height) of a fractional bbox on a card of `card_size`, resized ANISOTROPICALLY
    to (scale, scale) (freuid.transforms.build_transforms uses `Resize((image_size, image_size))`
    -- a direct squash/stretch to a square, NOT an aspect-preserving resize+crop, so width and
    height scale by DIFFERENT factors whenever the card isn't already square)."""
    w, h = card_size
    x1, y1, x2, y2 = ghost_bbox_frac
    native_w, native_h = (x2 - x1) * w, (y2 - y1) * h
    scaled_w = native_w * (scale / w)
    scaled_h = native_h * (scale / h)
    return scaled_w, scaled_h


def ghost_native_px(card_size: tuple[int, int], ghost_bbox_frac: tuple[float, float, float, float]) -> tuple[float, float]:
    w, h = card_size
    x1, y1, x2, y2 = ghost_bbox_frac
    return (x2 - x1) * w, (y2 - y1) * h


# ---------------------------------------------------------------------------
# Stage: ghost_resolvability
# ---------------------------------------------------------------------------

def ghost_resolvability(args) -> None:
    """Pure CPU/PIL: measures each ghost region's pixel size at native resolution and after
    resize to each TTA scale, for (a) the 2 template samples/type surveyed in --stage
    survey_templates, and (b) the reviewed ids (9 deep-miss + 59 boundary) whose visual/proxy
    type has a defined ghost region. No GPU, no regions cache needed -- card size alone
    (PIL Image.size) plus the fixed fractional bbox from GHOST_TEMPLATES is enough."""
    rows = []

    # (a) template survey samples -- re-derive the same seeded sample as survey_templates()
    train_df = load_labels(args.data_dir, "train")
    bonafide = train_df[train_df["label"] == 0].copy()
    bonafide = bonafide[bonafide["path"].map(lambda p: Path(p).exists())]
    for t, bbox in GHOST_TEMPLATES.items():
        if bbox is None:
            continue
        sub = bonafide[bonafide["type"] == t]
        sample = sub.sample(n=min(SURVEY_N_PER_TYPE, len(sub)), random_state=SURVEY_SEED)
        for _, row in sample.iterrows():
            img = Image.open(row["path"])
            native_w, native_h = ghost_native_px(img.size, bbox)
            row_out = {
                "source": "train_template_sample", "type": t, "id": row["id"],
                "card_w": img.size[0], "card_h": img.size[1],
                "ghost_native_w_px": native_w, "ghost_native_h_px": native_h,
                "ghost_native_short_side_px": min(native_w, native_h),
            }
            for scale in TTA_SCALES:
                sw, sh = ghost_px_at_scale(img.size, bbox, scale)
                row_out[f"short_side_px_at_{scale}"] = min(sw, sh)
            rows.append(row_out)

    # (b) reviewed test ids (deep-miss + boundary) whose type has a ghost region defined
    df = load_deep_done(Path(args.deep_done_csv))
    reviewed = pd.concat([deep_miss_ids(df), boundary_ids(df)], ignore_index=True)
    reviewed = attach_paths_and_scale_logits(reviewed, args.data_dir, Path(args.logit_census_csv))
    for _, row in reviewed.iterrows():
        t = visual_type(row["id"], row["doc_type_proxy"])
        bbox = GHOST_TEMPLATES.get(t)
        if bbox is None or not Path(str(row["path"])).exists():
            continue
        img = Image.open(row["path"])
        native_w, native_h = ghost_native_px(img.size, bbox)
        row_out = {
            "source": "reviewed_test_id", "type": t, "id": row["id"],
            "card_w": img.size[0], "card_h": img.size[1],
            "ghost_native_w_px": native_w, "ghost_native_h_px": native_h,
            "ghost_native_short_side_px": min(native_w, native_h),
        }
        for scale in TTA_SCALES:
            sw, sh = ghost_px_at_scale(img.size, bbox, scale)
            row_out[f"short_side_px_at_{scale}"] = min(sw, sh)
        rows.append(row_out)

    result_df = pd.DataFrame(rows)
    out_csv = Path(args.out_dir) / "ghost_resolvability.csv"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv, index=False)
    print(f"[deep_miss] wrote {len(result_df)} ghost-resolvability rows -> {out_csv}")

    below_20 = result_df[result_df[[f"short_side_px_at_{s}" for s in TTA_SCALES]].min(axis=1) < 20]
    write_ghost_resolvability_report(Path(args.out_dir) / "ghost_resolvability_report.md", result_df, below_20)


def write_ghost_resolvability_report(report_path: Path, result_df: pd.DataFrame, below_20: pd.DataFrame) -> None:
    lines = ["# Ghost-region resolvability across TTA scales\n"]
    lines.append(
        "Each row's `short_side_px_at_{476,518,560}` is the SHORTER of the ghost bbox's width/"
        "height after `freuid.transforms.build_transforms`'s `Resize((scale, scale))` -- an "
        "ANISOTROPIC squash to a square, not an aspect-preserving resize, so width and height "
        "shrink by different factors whenever the source card isn't already square (that's why "
        "`ghost_native_w_px`/`ghost_native_h_px` can differ from a simple uniform scaling of "
        "`ghost_native_short_side_px`).\n"
    )
    lines.append(df_to_md(result_df.round(1)))
    lines.append("")

    if len(below_20):
        lines.append(
            f"\n**{len(below_20)}/{len(result_df)} rows land below ~20px (short side) at EVERY "
            "inference TTA scale** -- flagged prominently per the task spec. At that size a "
            "ghost-mismatch signal is not plausibly learnable from the standard 476/518/560 "
            "input alone; a fix targeting this evidence would need an added higher-resolution or "
            "cropped view of the ghost region specifically, not just better use of the existing "
            "resized input.\n"
        )
    else:
        lines.append(
            "\nNo row lands below ~20px (short side) at every TTA scale -- the ghost region "
            "stays nominally legible-sized after resize in every case measured here (this says "
            "nothing about CONTRAST/exposure legibility, which the per-id dossier renders speak "
            "to separately).\n"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[deep_miss] wrote ghost-resolvability report -> {report_path}")


# ---------------------------------------------------------------------------
# Stage: dossiers -- per-id evidence panels for the 9 deep-miss ids (needs regions cache -> VESSL)
# ---------------------------------------------------------------------------

FRAME_MARGIN_FRAC = 0.6  # how far beyond the SCRFD face box the "photo-frame perimeter" zoom extends


def render_id_panels(row: pd.Series, rdir: Path, out_dir: Path, prefix: str) -> dict:
    """Renders (and saves to disk) this id's evidence panels: full card w/ SCRFD box, 2x/4x face
    zoom, 2x/4x photo-frame-perimeter zoom (face box expanded by FRAME_MARGIN_FRAC -- shows the
    rim/shadow/frame area a face-only crop would cut off), and 2x/4x ghost-region zoom where
    GHOST_TEMPLATES defines one for this id's (manually-confirmed, not proxy) document type.
    Returns a dict of panel name -> relative path (None where not applicable)."""
    img = Image.open(row["path"]).convert("RGB")
    w, h = img.size
    fb = read_face_box(rdir, row["id"])
    has_face = fb is not None and float(fb.get("score", 0.0)) > 0.0
    panels: dict[str, str | None] = {}

    card = img.copy()
    if has_face:
        draw = ImageDraw.Draw(card)
        box = (int(fb["x1"]), int(fb["y1"]), int(fb["x2"]), int(fb["y2"]))
        draw.rectangle(box, outline=(0, 255, 0), width=max(2, w // 250))
    full_path = out_dir / f"{prefix}_full.jpeg"
    card.save(full_path)
    panels["full"] = full_path.name

    if has_face:
        x1, y1, x2, y2 = int(fb["x1"]), int(fb["y1"]), int(fb["x2"]), int(fb["y2"])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        face_crop = img.crop((x1, y1, x2, y2))
        for zoom in (2, 4):
            z = face_crop.resize((max(1, int(face_crop.width * zoom)), max(1, int(face_crop.height * zoom))))
            p = out_dir / f"{prefix}_face_{zoom}x.jpeg"
            z.save(p)
            panels[f"face_{zoom}x"] = p.name

        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * FRAME_MARGIN_FRAC), int(bh * FRAME_MARGIN_FRAC)
        fx1, fy1, fx2, fy2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
        frame_crop = img.crop((fx1, fy1, fx2, fy2))
        for zoom in (2, 4):
            z = frame_crop.resize((max(1, int(frame_crop.width * zoom)), max(1, int(frame_crop.height * zoom))))
            p = out_dir / f"{prefix}_frame_{zoom}x.jpeg"
            z.save(p)
            panels[f"frame_{zoom}x"] = p.name
    else:
        panels.update({"face_2x": None, "face_4x": None, "frame_2x": None, "frame_4x": None})

    t = visual_type(row["id"], row.get("doc_type_proxy", ""))
    ghost_bbox = GHOST_TEMPLATES.get(t)
    if ghost_bbox:
        gx1, gy1, gx2, gy2 = ghost_bbox
        gbox = (int(gx1 * w), int(gy1 * h), int(gx2 * w), int(gy2 * h))
        ghost_crop = img.crop(gbox)
        for zoom in (2, 4):
            z = ghost_crop.resize((max(1, int(ghost_crop.width * zoom)), max(1, int(ghost_crop.height * zoom))))
            p = out_dir / f"{prefix}_ghost_{zoom}x.jpeg"
            z.save(p)
            panels[f"ghost_{zoom}x"] = p.name
    else:
        panels["ghost_2x"] = panels["ghost_4x"] = None

    return panels


def dossiers(args) -> None:
    df = load_deep_done(Path(args.deep_done_csv))
    deep = deep_miss_ids(df)
    deep = attach_paths_and_scale_logits(deep, args.data_dir, Path(args.logit_census_csv))

    rdir = regions_dir(args.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this needs VESSL")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in deep.iterrows():
        prefix = f"{row['stratum']}_{row['id'][:10]}"
        panels = render_id_panels(row, rdir, out_dir, prefix)
        t = visual_type(row["id"], row["doc_type_proxy"])
        record = {
            "id": row["id"], "stratum": row["stratum"], "prefix": prefix,
            "logit": float(row["logit"]), "pct_rank": float(row["pct_rank"]),
            "logit_476": float(row["logit_476"]), "logit_518": float(row["logit_518"]),
            "logit_560": float(row["logit_560"]),
            "doc_type_proxy": row["doc_type_proxy"], "visual_type": t,
            "face_score": float(row["face_score"]), "blur_laplacian_var": float(row["blur_laplacian_var"]),
            "moire_fft_score": float(row["moire_fft_score"]), "blockiness_score": float(row["blockiness_score"]),
            "min_side_px": float(row["min_side_px"]),
            "mode_preassigned": MODE_ASSIGNMENTS.get(row["id"], ""),
            "panels": panels,
        }
        records.append(record)
        print(f"[deep_miss] rendered dossier panels for {row['id'][:10]} ({row['stratum']}) -> {prefix}_*")

    meta_path = out_dir / "dossiers_meta.json"
    meta_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[deep_miss] wrote {len(records)} dossier records -> {meta_path}")


# ---------------------------------------------------------------------------
# Stage: boundary_grids -- same evidence checklist, batched as grids, for the 59 TRANS_5 ids.
# Reuses review_package.py's render_cell/render_sheet (card + SCRFD box + 2x face zoom) rather
# than duplicating that machinery; ghost zooms (where the type_proxy guess has one) are rendered
# as a separate grid since render_cell has no ghost-panel concept.
# ---------------------------------------------------------------------------

def boundary_grids(args) -> None:
    df = load_deep_done(Path(args.deep_done_csv))
    boundary = boundary_ids(df)
    boundary = attach_paths_and_scale_logits(boundary, args.data_dir, Path(args.logit_census_csv))
    # review_package.render_cell expects `mean_logit`; the deep-review CSV's own `logit` column
    # already IS the mean logit (see review_package.py's build_deep, which writes
    # `"logit": float(row["mean_logit"])`) -- just alias the name.
    boundary = boundary.rename(columns={"logit": "mean_logit"})

    rdir = regions_dir(args.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this needs VESSL")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Main checklist grid: card + SCRFD box + 2x face zoom (review_package.render_cell), ordered
    # by logit descending (matches review_package_deep_report.md's TRANS_5 sheet order).
    ordered = boundary.sort_values("mean_logit", ascending=False).reset_index(drop=True)
    cells = [render_cell(row, rdir, cell_width=340) for _, row in ordered.iterrows()]
    n_sheets = (len(cells) + 24) // 25
    for s in range(n_sheets):
        render_sheet(cells[s * 25:(s + 1) * 25], out_dir / f"boundary_sheet{s + 1:02d}.png")
    print(f"[deep_miss] boundary checklist grid: {len(cells)} ids -> {n_sheets} sheet(s)")

    # Ghost-zoom grid, restricted to ids whose type_proxy guess has a ghost region defined.
    # type_proxy is NOT manually verified for these 59 (only the original 9 deep-miss ids got a
    # full visual override) -- flagged explicitly wherever this subset is reported on.
    ghost_rows = ordered[ordered["doc_type_proxy"].map(lambda t: GHOST_TEMPLATES.get(t) is not None)]
    ghost_cells = []
    for _, row in ghost_rows.iterrows():
        img = Image.open(row["path"]).convert("RGB")
        w, h = img.size
        bbox = GHOST_TEMPLATES[row["doc_type_proxy"]]
        gx1, gy1, gx2, gy2 = bbox
        crop = img.crop((int(gx1 * w), int(gy1 * h), int(gx2 * w), int(gy2 * h)))
        crop = crop.resize((max(1, crop.width * 3), max(1, crop.height * 3)))
        cell = Image.new("RGB", (crop.width, crop.height + 30), (255, 255, 255))
        cell.paste(crop, (0, 0))
        d = ImageDraw.Draw(cell)
        d.text((2, crop.height + 2), f"{row['id'][:10]} pct={row['pct_rank']:.1f}", fill=(0, 0, 0), font=_load_font(12))
        ghost_cells.append(cell)
    if ghost_cells:
        # Pad all cells to the same size (ghost crops can differ slightly by source resolution)
        max_w = max(c.width for c in ghost_cells)
        max_h = max(c.height for c in ghost_cells)
        padded = []
        for c in ghost_cells:
            p = Image.new("RGB", (max_w, max_h), (255, 255, 255))
            p.paste(c, (0, 0))
            padded.append(p)
        n_ghost_sheets = (len(padded) + 24) // 25
        for s in range(n_ghost_sheets):
            render_sheet(padded[s * 25:(s + 1) * 25], out_dir / f"boundary_ghost_sheet{s + 1:02d}.png")
        print(f"[deep_miss] boundary ghost grid: {len(padded)} ids (type_proxy-based, unverified) -> {n_ghost_sheets} sheet(s)")
    else:
        print("[deep_miss] no TRANS_5 ids have a ghost-template type_proxy guess -- skipping ghost grid")

    ordered[[
        "id", "stratum", "mean_logit", "pct_rank", "doc_type_proxy", "face_score",
        "blur_laplacian_var", "moire_fft_score", "blockiness_score", "min_side_px",
    ]].to_csv(out_dir / "boundary_grid_meta.csv", index=False)
    print(f"[deep_miss] wrote boundary grid metadata -> {out_dir / 'boundary_grid_meta.csv'}")


# ---------------------------------------------------------------------------
# Stage: assemble -- combine everything into deep_miss_dossiers.html. Pure local, no GPU.
# Evidence only -- no fix recommendations anywhere in this output.
# ---------------------------------------------------------------------------

def _html_escape(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _checklist_row_html(id_: str, record: dict) -> str:
    mode = record.get("mode_preassigned") or ""
    if mode:
        cl = {"paper_rim": "--", "paste_shadow": "--", "style_mismatch": "--", "boundary_aligned": "--",
              "ghost_present": "--", "ghost_mismatch": "--", "gender_name_mismatch": "--",
              "proposed_mode": mode, "pre_assigned": "yes (before this script existed)"}
    else:
        cl = CHECKLIST_RESULTS.get(id_)
        if cl is None:
            cl = {k: "NOT YET SCORED" for k in (
                "paper_rim", "paste_shadow", "style_mismatch", "boundary_aligned", "ghost_present",
                "ghost_mismatch", "gender_name_mismatch", "proposed_mode",
            )}
            cl["pre_assigned"] = "no"
        else:
            cl = {**cl, "pre_assigned": "no"}
    cells = "".join(f"<td>{_html_escape(cl.get(k, ''))}</td>" for k in (
        "paper_rim", "paste_shadow", "style_mismatch", "boundary_aligned", "ghost_present",
        "ghost_mismatch", "gender_name_mismatch", "proposed_mode", "pre_assigned",
    ))
    return f"<tr><td>{_html_escape(id_[:12])}</td>{cells}</tr>"


def _dossier_section_html(record: dict, out_dir_name: str) -> str:
    mode = record.get("mode_preassigned") or ""
    if mode:
        mode_note = f"<p class='mode-badge'>Pre-assigned MODE_{mode}: {_html_escape(MODE_DESCRIPTIONS.get(mode, ''))}</p>"
    else:
        cl = CHECKLIST_RESULTS.get(record["id"])
        if cl is None:
            mode_note = "<p class='mode-badge unassigned'>Not yet visually scored.</p>"
        else:
            mode_note = (
                f"<p class='mode-badge unassigned'><b>Proposed MODE_{_html_escape(cl['proposed_mode'])}</b> "
                f"(scored from these renders, see checklist table above for the yes/no breakdown)<br>"
                f"<i>{_html_escape(cl.get('notes', ''))}</i></p>"
            )
    panels = record["panels"]

    def img_tag(key: str, label: str) -> str:
        fname = panels.get(key)
        if not fname:
            return f"<div class='panel missing'>{label}<br>(no face detected -- n/a)</div>"
        return f"<div class='panel'><div class='panel-label'>{label}</div><img src='{out_dir_name}/{fname}' loading='lazy'></div>"

    panel_html = "".join([
        img_tag("full", "Full card (SCRFD box)"),
        img_tag("face_2x", "Face 2x"),
        img_tag("face_4x", "Face 4x"),
        img_tag("frame_2x", "Photo-frame perimeter 2x"),
        img_tag("frame_4x", "Photo-frame perimeter 4x"),
        img_tag("ghost_2x", "Ghost region 2x"),
        img_tag("ghost_4x", "Ghost region 4x"),
    ])

    return f"""
    <section class="dossier">
      <h3>{_html_escape(record['id'][:12])} -- {_html_escape(record['stratum'])}</h3>
      {mode_note}
      <table class="meta-table">
        <tr><td>logit (mean)</td><td>{record['logit']:.4f}</td>
            <td>pct_rank</td><td>{record['pct_rank']:.2f}</td></tr>
        <tr><td>logit@476</td><td>{record['logit_476']:.4f}</td>
            <td>logit@518</td><td>{record['logit_518']:.4f}</td></tr>
        <tr><td>logit@560</td><td>{record['logit_560']:.4f}</td>
            <td>doc_type_proxy (unreliable)</td><td>{_html_escape(record['doc_type_proxy'])}</td></tr>
        <tr><td>visual_type (confirmed)</td><td>{_html_escape(record['visual_type'])}</td>
            <td>face_score</td><td>{record['face_score']:.4f}</td></tr>
        <tr><td>blur_laplacian_var</td><td>{record['blur_laplacian_var']:.2f}</td>
            <td>moire_fft_score</td><td>{record['moire_fft_score']:.2f}</td></tr>
        <tr><td>blockiness_score</td><td>{record['blockiness_score']:.4f}</td>
            <td>min_side_px</td><td>{record['min_side_px']:.0f}</td></tr>
      </table>
      <div class="panels">{panel_html}</div>
    </section>
    """


def assemble(args) -> None:
    out_dir = Path(args.out_dir)
    meta_path = out_dir / "dossiers_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found -- run --stage dossiers first")
    records = json.loads(meta_path.read_text(encoding="utf-8"))

    ghost_csv = out_dir / "ghost_resolvability.csv"
    ghost_df = pd.read_csv(ghost_csv) if ghost_csv.exists() else None

    boundary_meta_csv = out_dir / "boundary_grid_meta.csv"
    boundary_df = pd.read_csv(boundary_meta_csv, dtype={"id": str}) if boundary_meta_csv.exists() else None
    boundary_sheets = sorted(out_dir.glob("boundary_sheet*.png"))
    boundary_ghost_sheets = sorted(out_dir.glob("boundary_ghost_sheet*.png"))

    out_dir_name = out_dir.name

    checklist_rows = "".join(_checklist_row_html(r["id"], r) for r in records)
    dossier_sections = "".join(_dossier_section_html(r, out_dir_name) for r in records)

    n_scored = sum(1 for r in records if r.get("mode_preassigned") or r["id"] in CHECKLIST_RESULTS)
    unscored_note = (
        "" if n_scored == len(records) else
        f"<p class='warn'><b>{len(records) - n_scored}/{len(records)} ids not yet visually "
        "scored</b> -- CHECKLIST_RESULTS in deep_miss_dossiers.py is still empty for them. "
        "Re-run --stage assemble after scoring.</p>"
    )

    ghost_table_html = ""
    if ghost_df is not None:
        ghost_table_html = ghost_df.round(1).to_html(index=False, classes="meta-table")
        n_below_20 = int((ghost_df[[f"short_side_px_at_{s}" for s in TTA_SCALES]].min(axis=1) < 20).sum())
        ghost_table_html += (
            f"<p>{n_below_20}/{len(ghost_df)} rows land below ~20px (short side) at every TTA "
            "scale.</p>" if n_below_20 else
            "<p>No row lands below ~20px (short side) at every TTA scale -- size alone doesn't "
            "explain any ghost-legibility failures found; see per-id dossier renders above for "
            "contrast/exposure-driven legibility issues instead (e.g. a very dark source image).</p>"
        )

    boundary_html = ""
    if boundary_df is not None:
        boundary_html = (
            f"<p>{len(boundary_df)} TRANS_5 boundary ids (all confirmed F-verdict). "
            f"{len(boundary_sheets)} main checklist sheet(s), "
            f"{len(boundary_ghost_sheets)} ghost-zoom sheet(s) (type_proxy-based subset, NOT "
            "manually type-verified for these 59 -- see caveat below).</p>"
            "<p><b>Qualitative pass over all 3 main sheets (59 faces)</b>: at this thumbnail "
            "scale, NONE show the dramatic physical-paste tells found in the 9 deep-miss dossiers "
            "(no visible tears, tape strips, or arch-shaped hand-cut silhouettes) -- most look "
            "like plausible, cleanly-composited portraits. This is a genuinely different "
            "population from TRANS_1-4: every one of these 59 ids sits at pct_rank 94.9-100 (i.e. "
            "already ranked at or near the very top of the ENTIRE population by the model) -- so "
            "unlike the TRANS_1-4 deep misses, these are not really 'hidden'; the model already "
            "flags them as confidently fraud, they just fall a hair outside the ceiling-mode "
            "tolerance window. <b>Caveat, stated honestly</b>: this is a thumbnail-scale "
            "impression, not a scored checklist pass -- the deep-miss evidence (tears, tape, "
            "arch-cutouts) only became visible after 2x/4x zoom cropping, so a true per-id "
            "photo-substitution rate for this population would need the same zoom treatment as "
            "the 9 deep-miss ids, which this pass did not do for all 59.</p>"
        )
        for p in boundary_sheets + boundary_ghost_sheets:
            boundary_html += f"<div class='panel'><div class='panel-label'>{p.name}</div><img src='{out_dir_name}/{p.name}' loading='lazy'></div>"

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Deep-miss evidence dossiers</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; color: #222; }}
h1, h2, h3 {{ color: #111; }}
.mode-badge {{ background: #eef6ff; border-left: 4px solid #3576d3; padding: 8px 12px; }}
.mode-badge.unassigned {{ background: #fff8e6; border-left-color: #d3a135; }}
table.meta-table {{ border-collapse: collapse; margin: 10px 0; }}
table.meta-table td, table.meta-table th {{ border: 1px solid #ccc; padding: 4px 8px; font-size: 13px; }}
.panels {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }}
.panel {{ border: 1px solid #ddd; padding: 4px; }}
.panel img {{ max-width: 260px; max-height: 320px; display: block; }}
.panel.missing {{ width: 200px; height: 100px; display: flex; align-items: center; justify-content: center; color: #999; text-align: center; }}
.panel-label {{ font-size: 11px; color: #555; margin-bottom: 2px; }}
.dossier {{ border-top: 2px solid #333; padding-top: 16px; margin-top: 24px; }}
.warn {{ background: #fff0f0; border-left: 4px solid #d33; padding: 8px 12px; }}
</style>
</head>
<body>
<h1>Deep-miss evidence dossiers</h1>
<p><b>Evidence only -- no fix recommendations in this document.</b> Background: 9 TRANS_1/3/4
ids and 59 TRANS_5 ids were manually confirmed as genuinely fraud despite finetune_v0 ranking
them as if bona-fide (see <code>review_package_deep_ingest_report.md</code>). A visual read of
3 of the 9 found a photo-substitution taxonomy (MODE_A/B/C, defined below); this document
gathers the evidence needed to confirm or amend that taxonomy on the remaining 6, and to check
how much of the 59-id TRANS_5 boundary population shares it.</p>

<h2>Mode taxonomy</h2>
<ul>
<li><b>MODE_A</b> -- {_html_escape(MODE_DESCRIPTIONS['A'])}</li>
<li><b>MODE_B</b> -- {_html_escape(MODE_DESCRIPTIONS['B'])}</li>
<li><b>MODE_C</b> -- {_html_escape(MODE_DESCRIPTIONS['C'])}</li>
</ul>

<h2>Mode-assignment / checklist table (9 deep-miss ids)</h2>
{unscored_note}
<table class="meta-table">
<tr><th>id</th><th>paper rim?</th><th>paste shadow?</th><th>style mismatch?</th>
<th>boundary aligned?</th><th>ghost present?</th><th>ghost mismatch?</th>
<th>gender/name mismatch?</th><th>proposed mode</th><th>pre-assigned?</th></tr>
{checklist_rows}
</table>

<h2>Per-id dossiers</h2>
{dossier_sections}

<h2>Ghost resolvability across TTA scales</h2>
{ghost_table_html or "<p>Run --stage ghost_resolvability first.</p>"}

<h2>TRANS_5 boundary batch check (59 ids)</h2>
<p>Same evidence checklist as above, batched as grids rather than per-id pages -- the question
here is what fraction of the boundary misses are also photo-substitution (a fix could recover
much of this ~73.8% F-rate stratum too) vs. something else. Ghost-zoom subset uses
<code>doc_type_proxy</code> (NOT manually visually verified for these 59, unlike the 9 deep-miss
ids) to decide which ids get a ghost panel -- a wrong proxy guess means a missing or spurious
ghost panel for that id, not a wrong verdict.</p>
{boundary_html or "<p>Run --stage boundary_grids first.</p>"}

</body></html>
"""
    Path(args.html_out).write_text(html, encoding="utf-8")
    print(f"[deep_miss] wrote assembled report -> {args.html_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["freeze_probes", "survey_templates", "dossiers", "boundary_grids", "ghost_resolvability", "assemble"],
        required=True,
    )
    parser.add_argument("--deep-done-csv", default=str(DEFAULT_DEEP_DONE_CSV))
    parser.add_argument("--logit-census-csv", default=str(DEFAULT_LOGIT_CENSUS_CSV))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--probes-dir", default=str(DEFAULT_PROBES_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--html-out", default=str(DEFAULT_HTML_OUT))
    args = parser.parse_args()

    if args.stage == "freeze_probes":
        freeze_probes(args)
    elif args.stage == "survey_templates":
        survey_templates(args)
    elif args.stage == "ghost_resolvability":
        ghost_resolvability(args)
    elif args.stage == "dossiers":
        dossiers(args)
    elif args.stage == "boundary_grids":
        boundary_grids(args)
    elif args.stage == "assemble":
        assemble(args)
    else:
        raise SystemExit(f"--stage {args.stage} not implemented yet")


if __name__ == "__main__":
    main()
