"""Offline photo-substitution dataset generator: turns the reviewed generators (freuid.photosub.
generators, gated behind scripts/analysis/photosub_render_sheets.py's render-sheet human review)
into real training rows on disk.

**Scale is caller-controlled via --n-total.** This script itself has no "mass" vs "smoke" mode --
it is the same driver either way; run it with a small --n-total for a training-side smoke run
(see configs/photosub_v0.yaml) and a large one only after the full-dataset generation gate is
explicitly approved (per CLAUDE.md / the render-sheet review). Nothing here decides that gate.

Reuses the donor pool + probe-overlap exclusion already built by photosub_render_sheets.py
--stage build_index (donor_index.pkl) rather than re-deriving that logic -- run that stage
first if it hasn't been run (or its output is stale) on this machine.

For each of --n-total sampled bona-fide TRAIN sources (sampled WITHOUT restricting to any
particular train/val split -- split discipline is enforced at TRAIN TIME instead, by
freuid.photosub.mixing.select_mixed_rows filtering on source_id, since a generated row's
eligibility depends on which split a given training run uses, not on generation time):
  1. assigns a mode (A/B/C/D) via --mode-weights (same default as configs/photosub_v0.yaml --
     D is now ON at a modest weight, but still requires --enable-mode-d at the CLI level too, a
     deliberate second gate; a source whose OWN type has no ghost feature never draws C OR D at
     all -- see the no_ghost_modes renormalization in generate() -- rather than drawing one and
     silently falling back to something else, which would have skewed all the counts. C's
     ghost-only restriction is a photosub_v1 change: a "clean" digital swap on a non-ghost
     template has zero evidence any generator here can point to -- see generators.py's module
     docstring for why this is suspected to matter, not just tidiness)
  2. picks a donor via freuid.photosub.donor_pool.sample_donor (--hard-fraction, default 0.4,
     per CLAUDE.md's hard-case donor discipline) -- donor_id is recorded on the output row so a
     mass run can be audited for donor-pool reuse/exhaustion (scripts/analysis/
     photosub_spot_review.py), a real risk when --n-total is large relative to the donor pool
  3. runs the matching generator (D_ghost rows get --ghost-darken-prob applied, per
     generate_mode_d's ghost-legibility design decision) and writes (image, mask, row) via
     freuid.photosub.pipeline

Needs the regions cache (SCRFD face.json) and the real train images -> VESSL.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from freuid.data import load_labels  # noqa: E402
from freuid.photosub.donor_pool import (  # noqa: E402
    DonorFace,
    build_donor_record,
    face_crop_from_box,
    sample_donor,
)
from freuid.photosub.face_embedding import best_available_embed_fn  # noqa: E402
from freuid.photosub.generators import GENERATORS  # noqa: E402
from freuid.photosub.mixing import load_photosub_rows  # noqa: E402
from freuid.photosub.pipeline import (  # noqa: E402
    append_rows_csv,
    photosub_generated_dir,
    save_tamper_result,
)
from freuid.photosub.template_regions import GHOST_TEMPLATES, build_template_metadata  # noqa: E402
from freuid.preprocess import regions_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = REPO_ROOT / "scripts" / "analysis" / "photosub_render_out" / "donor_index.pkl"

# Same defaults as configs/photosub_v0.yaml's mode_weights: A/B/C from the n=9 deep-miss ids'
# confirmed prevalence (A=4/9, B=3/9, C=2/9 -- see that config's docstring for the provenance
# caveat); D=0.15 is a modest, hypothesis-driven weight (0/9 confirmed -- only 2 of 5 templates
# even carry a ghost feature, so n=9 saying nothing about it isn't evidence it doesn't happen).
DEFAULT_MODE_WEIGHTS = {"A": 0.4444, "B": 0.3333, "C": 0.2222, "D": 0.15}
DEFAULT_HARD_FRACTION = 0.4
DEFAULT_GHOST_DARKEN_PROB = 0.3  # "a slice" of D_ghost rows deliberately dimmed -- see generate_mode_d


def _load_template_metadata_for_row(row, rdir: Path):
    face_path = rdir / row["id"] / "face.json"
    if not face_path.exists():
        return None, None
    try:
        face_box = json.loads(face_path.read_text())
    except Exception:
        return None, None
    if float(face_box.get("score", 0.0)) <= 0.0:
        return None, None
    with Image.open(row["path"]) as img:
        img_w, img_h = img.size
    template = build_template_metadata(row["id"], row["type"], face_box, img_w, img_h)
    return face_box, template


def generate(args) -> None:
    rdir = regions_dir(args.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this needs VESSL")
    index_path = Path(args.index_path)
    if not index_path.exists():
        raise SystemExit(
            f"{index_path} not found -- run scripts/analysis/photosub_render_sheets.py "
            "--stage build_index first"
        )
    with open(index_path, "rb") as f:
        donors: list[DonorFace] = pickle.load(f)
    print(f"[generate_photosub] loaded {len(donors)} donors from {index_path}")

    embed_fn = best_available_embed_fn()  # must match build_index's embedding space

    mode_weights = dict(DEFAULT_MODE_WEIGHTS)
    if args.mode_weights:
        mode_weights.update(json.loads(args.mode_weights))
    if not args.enable_mode_d:
        mode_weights["D"] = 0.0
    active_modes = [m for m, w in mode_weights.items() if w > 0.0]
    if not active_modes:
        raise SystemExit(f"mode_weights has no positive weight: {mode_weights!r}")
    probs = np.array([mode_weights[m] for m in active_modes], dtype=np.float64)
    probs /= probs.sum()
    print(f"[generate_photosub] mode weights: {dict(zip(active_modes, probs.tolist()))}")

    # v1: a source only OFFERS "C" OR "D" as a candidate mode when its own type has a ghost box.
    # D always needed this (the generator itself requires a ghost_box). C is NEW in v1: a
    # "clean" digital swap on a non-ghost template leaves ZERO evidence any generator in this
    # family can point to (no visible tell, no cross-region signature) -- exactly the "too
    # clean" signature photosub_v0's deep-9 diagnostic suspects the model over-generalized from
    # (see generators.py's module docstring, MODE_C entry). Drawing "C"/"D" for a non-ghost type
    # and silently falling back to something else would inflate that something beyond its
    # intended weight; renormalize a separate ghost-only-excluded distribution once, up front,
    # rather than fixing up draws after the fact.
    no_ghost_modes = [m for m in active_modes if m not in ("C", "D")]
    if no_ghost_modes:
        no_ghost_probs = np.array([mode_weights[m] for m in no_ghost_modes], dtype=np.float64)
        no_ghost_probs /= no_ghost_probs.sum()
    else:
        no_ghost_modes, no_ghost_probs = active_modes, probs

    train_df = load_labels(args.data_dir, "train")
    bonafide = train_df[train_df["label"] == 0].copy()
    bonafide = bonafide[bonafide["path"].map(lambda p: Path(p).exists())]
    bonafide = bonafide[bonafide["type"].isin(GHOST_TEMPLATES)]  # only known template types
    if args.restrict_ids_file:
        restrict_ids = set(Path(args.restrict_ids_file).read_text().split())
        bonafide = bonafide[bonafide["id"].isin(restrict_ids)]
        print(f"[generate_photosub] --restrict-ids-file: {len(bonafide)} eligible sources "
              f"after restricting to {len(restrict_ids)} ids from {args.restrict_ids_file}")
    rng = np.random.default_rng(args.seed)
    if args.n_total < len(bonafide):
        sources = bonafide.sample(n=args.n_total, random_state=args.seed).reset_index(drop=True)
    else:
        print(f"[generate_photosub] WARNING: n_total={args.n_total} >= {len(bonafide)} eligible "
              "sources -- using all of them")
        sources = bonafide.reset_index(drop=True)

    rows_csv = Path(args.rows_csv)
    pending_rows = []  # flushed to rows_csv every progress_every rows -- see the flush below
    n_written, n_skipped = 0, 0
    t0 = time.monotonic()
    for i, row in enumerate(sources.itertuples(index=False)):
        row_t0 = time.monotonic()
        row_d = row._asdict()
        face_box, template = _load_template_metadata_for_row(row_d, rdir)
        if face_box is None or template is None:
            n_skipped += 1
            continue

        if template["ghost_box"] is not None:
            mode = str(rng.choice(active_modes, p=probs))
        else:
            mode = str(rng.choice(no_ghost_modes, p=no_ghost_probs))

        image = Image.open(row_d["path"]).convert("RGB")
        target_rec = build_donor_record(row_d["id"], row_d["path"], row_d["type"], face_box, embed_fn=embed_fn)
        gen_rng = np.random.default_rng(args.seed + 1_000_003 * (i + 1))
        donor_rec = sample_donor(
            donors, gen_rng, target_rec or donors[i % len(donors)],
            hard_fraction=args.hard_fraction,
        )
        with Image.open(donor_rec.path) as d_img:
            donor_crop = face_crop_from_box(d_img.convert("RGB"), donor_rec.face_box)

        if mode == "D":
            result = GENERATORS["D"](
                image, template["frame_box"], template["ghost_box"], donor_crop, gen_rng,
                enabled=True, ghost_darken_prob=args.ghost_darken_prob,
            )
        elif mode == "A":
            result = GENERATORS["A"](
                image, template["frame_box"], donor_crop, gen_rng,
                irregular_shape_prob=args.irregular_shape_prob, arch_shape_prob=args.arch_shape_prob,
                tear_prob=args.tear_prob, tape_prob=args.tape_prob,
            )
        elif mode == "B":
            result = GENERATORS["B"](
                image, template["frame_box"], donor_crop, gen_rng,
                irregular_shape_prob=args.irregular_shape_prob_b,
            )
        else:
            result = GENERATORS[mode](image, template["frame_box"], donor_crop, gen_rng)

        out_row = save_tamper_result(
            result, row_d["id"], row_d["type"], args.data_dir, row_index=0, donor_id=donor_rec.id,
            out_dir=rows_csv.parent,
        )
        pending_rows.append(out_row)
        n_written += 1

        row_elapsed = time.monotonic() - row_t0
        if row_elapsed > args.slow_row_warn_secs:
            print(f"[generate_photosub] WARNING: row {row_d['id']} (mode={mode}) took "
                  f"{row_elapsed:.1f}s (> {args.slow_row_warn_secs}s) -- possible stall")

        if n_written % args.progress_every == 0:
            elapsed = time.monotonic() - t0
            print(f"[generate_photosub] {n_written}/{len(sources)} written "
                  f"({n_skipped} skipped, no face box) elapsed={elapsed:.0f}s rate={n_written / max(elapsed, 1e-6):.2f}/s")
            # Flush incrementally: a mass run is long enough that losing everything to a kill
            # partway through (append_rows_csv previously only ran once, at the very end --
            # confirmed the hard way when a real stall mid-run orphaned ~2800 already-written
            # images/masks with no CSV rows to recover them) is real, avoidable data loss.
            append_rows_csv(pending_rows, rows_csv)
            pending_rows = []

    if pending_rows:
        append_rows_csv(pending_rows, rows_csv)

    all_rows = load_photosub_rows(rows_csv)
    mode_counts = all_rows["mode"].value_counts().to_dict() if len(all_rows) else {}
    print(
        f"[generate_photosub] wrote {n_written} rows this run ({n_skipped} sources skipped, no "
        f"usable face box) -> {rows_csv} (appended, flushed incrementally). "
        f"Mode counts (FULL corpus, not just this run): {mode_counts}"
    )
    print(f"[generate_photosub] output images/masks under {photosub_generated_dir(args.data_dir)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    p.add_argument("--rows-csv", default=str(photosub_generated_dir("data") / "rows.csv"))
    p.add_argument("--n-total", type=int, required=True,
                   help="number of bona-fide TRAIN sources to generate photosub rows from "
                        "(1 row per source). Pass a small number for a training-side smoke run; "
                        "the full-dataset scale is a separate, explicitly-approved run.")
    p.add_argument("--mode-weights", type=str, default=None,
                   help='JSON dict overriding DEFAULT_MODE_WEIGHTS, e.g. \'{"A": 0.5}\'')
    p.add_argument("--hard-fraction", type=float, default=DEFAULT_HARD_FRACTION)
    p.add_argument("--enable-mode-d", action="store_true",
                   help="allow MODE_D (ghost mismatch) -- off by default at the CLI level (a "
                        "second, deliberate gate on top of generate_mode_d's own enabled=True "
                        "requirement)")
    p.add_argument("--ghost-darken-prob", type=float, default=DEFAULT_GHOST_DARKEN_PROB,
                   help="fraction of D_ghost rows to additionally dim (see generate_mode_d's "
                        "ghost_darken_prob) -- reproduces the real illegible-ghost case")
    p.add_argument("--irregular-shape-prob", type=float, default=0.0,
                   help="fraction of MODE_A rows to give an irregular hand-cut silhouette instead "
                        "of a rotated rectangle (see generate_mode_a's irregular_shape_prob) -- "
                        "the shape-realism fix for photosub_v0's MODE_A collapse. 0.0 (default) "
                        "reproduces the original rectangle-only corpus exactly; do not set this "
                        "above 0.0 for a real run until scripts/analysis/photosub_render_sheets.py "
                        "--stage render_shape_variants has been reviewed and a weight approved.")
    p.add_argument("--tear-prob", type=float, default=0.0,
                   help="fraction of MODE_A rows to additionally overlay a tear/rip effect (see "
                        "generate_mode_a's tear_prob) -- same shape-realism fix and same "
                        "render-sheet-gate caveat as --irregular-shape-prob.")
    p.add_argument("--arch-shape-prob", type=float, default=0.0,
                   help="fraction of MODE_A rows to give an arch/doorway-shaped silhouette "
                        "instead of a rotated rectangle (see generate_mode_a's arch_shape_prob, "
                        "mutually exclusive with --irregular-shape-prob -- their sum must stay "
                        "<= 1.0 so rect remains a real variant). photosub_v1 addition, NOT yet "
                        "rendered/reviewed as of this file -- keep at 0.0 until "
                        "docs/photosub_renders/mode_a_shape_variants_sheet.png has been "
                        "regenerated with this variant and a weight approved.")
    p.add_argument("--tape-prob", type=float, default=0.0,
                   help="fraction of MODE_A rows to additionally overlay tape-strip marks (see "
                        "generate_mode_a's tape_prob). photosub_v1 addition, same "
                        "not-yet-reviewed caveat as --arch-shape-prob.")
    p.add_argument("--irregular-shape-prob-b", type=float, default=0.0,
                   help="fraction of MODE_B rows to give an irregular hand-cut silhouette (see "
                        "generate_mode_b's irregular_shape_prob). MODE_B's own deep-miss ids "
                        "converged fine in photosub_v0, so this is an optional diversity variant, "
                        "not a required fix -- keep at 0.0 unless deliberately testing it.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--slow-row-warn-secs", type=float, default=8.0,
                   help="print a WARNING if a single row takes longer than this (typical rate "
                        "is ~0.7s/row) -- catches a stall immediately instead of it blending "
                        "into the next --progress-every interval's average")
    p.add_argument("--restrict-ids-file", type=str, default=None,
                   help="optional path to a whitespace/newline-separated file of TRAIN ids -- "
                        "restricts eligible sources to this set (e.g. so a smoke-scale "
                        "generation run's rows are guaranteed to overlap a --limit-truncated "
                        "train split; see freuid.data.stratified_split)")
    args = p.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
