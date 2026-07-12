"""Human-review render sheets for the photo-substitution training-data generators
(freuid.photosub.*) -- the gate before any mass generation.

Per the build request: 12 examples per MODE_A/B/C (MODE_D stays off by default -- see
freuid.photosub.generators.generate_mode_d's docstring on the ghost-legibility caveat; pass
--enable-mode-d to include it anyway), spanning the parameter ranges, across every known
document template (including the ghost-bearing ones), with 6 deliberately-hard (lookalike
donor) examples per mode.

Gate-grade upgrade (second pass, after the first render sheets surfaced a real MODE_C stats
failure -- see freuid.photosub.degradation_match's module docstring): MODE_A/B examples now
render as a per-example 2x/4x zoom of the frame PERIMETER (rim, shadow, overhang), pre- and
post-recapture-augmentation side by side, so rim/shadow survival at 518px is judgeable by eye
instead of guessed from a full-card thumbnail. MODE_C/D examples carry the stats-delta numbers
and a PASS/FAIL verdict (against the natural intra-card baseline -- see
freuid.photosub.baseline_stats) directly in their caption.

**Only 5 real document types exist in this dataset** (confirmed: EGYPT/DL, GUINEA/DL, BENIN/DL,
MOZAMBIQUE/DL, MAURITIUS/ID are the entire `type` universe in train_labels.csv -- there is no
6th template to add). The build request asked for "at least 6 templates" assuming more existed;
rather than forcing a 6th out of nothing, this script renders across all 5 and says so here,
same convention as `deep_miss_dossiers.py`'s freeze_probes reporting 295 clean-floor ids instead
of a requested round 300.

Needs the regions cache (SCRFD face.json) and the real train/public_test images -> VESSL.

Stages:
  --stage build_index    Build the donor pool (bona-fide TRAIN faces w/ a real SCRFD detection),
                          exclude any donor that's suspiciously close to a frozen probe id
                          (data/probes/*.csv), and pickle the result for the render stage. ALSO
                          builds the natural intra-card baseline (freuid.photosub.baseline_stats)
                          over ~200 untouched bona-fide cards and writes its p90 thresholds --
                          the empirical acceptance criterion the render stage checks MODE_C/D
                          against, replacing eyeballing.
  --stage render          12 examples/mode (6 hard + 6 broad donor picks) across all 5 known
                          templates, saved as PNG grids under docs/photosub_renders/. Also (as
                          part of this same stage, not a separate CLI stage) writes
                          stats_check.csv / stats_check_report.md -- the MODE_C/D pass rate
                          against the natural-baseline p90 thresholds.

No training, no submissions, no modification of existing source files or the regions cache.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.data import load_labels  # noqa: E402
from freuid.photosub.baseline_stats import build_natural_baseline, compute_p90_thresholds  # noqa: E402
from freuid.photosub.donor_pool import (  # noqa: E402
    DonorFace, build_donor_index, build_donor_record, exclude_probe_overlap,
    face_crop_from_box, sample_donor,
)
from freuid.photosub.face_embedding import arcface_available, best_available_embed_fn  # noqa: E402
from freuid.photosub.generators import (  # noqa: E402
    generate_mode_a, generate_mode_b, generate_mode_c, generate_mode_d,
)
from freuid.photosub.stats import STAT_COLS, region_stats  # noqa: E402
from freuid.photosub.template_regions import GHOST_TEMPLATES, build_template_metadata  # noqa: E402
from freuid.preprocess import regions_dir  # noqa: E402
from freuid.transforms import build_transforms, resolve_data_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import df_to_md  # noqa: E402
from deep_miss_dossiers import DEFAULT_PROBES_DIR, visual_type  # noqa: E402
from hesitant_clusters import KNOWN_TYPES  # noqa: E402
from review_package import _load_font, render_sheet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "photosub_render_out"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs" / "photosub_renders"
DEFAULT_INDEX_PATH = DEFAULT_OUT_DIR / "donor_index.pkl"
DEFAULT_BASELINE_PATH = DEFAULT_OUT_DIR / "baseline_thresholds.json"

# 5 real document types -- see module docstring; this is the FULL template universe, not a subset.
TEMPLATES = list(KNOWN_TYPES)
assert set(TEMPLATES) == set(GHOST_TEMPLATES), "TEMPLATES must match GHOST_TEMPLATES's key set"

BACKBONE = "vit_base_patch14_reg4_dinov2.lvd142m"  # finetune_v0's backbone (CLAUDE.md)
RECAPTURE_SCALE = 518  # finetune_v0's training resolution
PERIMETER_MARGIN_FRAC = 0.6  # matches deep_miss_dossiers.py's FRAME_MARGIN_FRAC convention

TEMPLATE_SAMPLE_SEED = 2  # distinct from deep_miss_dossiers.py's SURVEY_SEED (=1)
DONOR_INDEX_SEED = 3
N_EXAMPLES_PER_MODE = 12
N_HARD_PER_MODE = 6


# ---------------------------------------------------------------------------
# Stage: build_index
# ---------------------------------------------------------------------------

def _load_probe_donor_faces(data_dir: str, rdir: Path, embed_fn) -> list[DonorFace]:
    """Frozen diagnostic probe ids (data/probes/*.csv) as DonorFace records, for the paranoid
    probe-exclusion guard. Type comes from `visual_type` (manually-confirmed for the 9 deep-miss
    ids, falls back to the unreliable doc_type_proxy for the other 354 -- a conservative choice:
    see donor_pool.exclude_probe_overlap's own docstring on why this guard is "paranoid, cheap",
    not exact). ``embed_fn`` MUST be the same one used to build the donor pool -- comparing
    embeddings from two different spaces would make the exclusion check meaningless."""
    import time

    probes_dir = DEFAULT_PROBES_DIR
    test_meta = load_labels(data_dir, "public_test")[["id", "path"]]
    records: list[DonorFace] = []
    n_seen = 0
    t0 = time.monotonic()
    for csv_name in ("missed_frauds_deep_ids.csv", "missed_frauds_boundary_ids.csv", "clean_floor_sample_ids.csv"):
        csv_path = probes_dir / csv_name
        if not csv_path.exists():
            print(f"[photosub_render] WARNING: {csv_path} not found -- skipping this probe set")
            continue
        df = pd.read_csv(csv_path, dtype={"id": str})
        df = df.merge(test_meta, on="id", how="left")
        for _, row in df.iterrows():
            n_seen += 1
            if n_seen % 100 == 0:
                print(f"[photosub_render] probe faces: {n_seen} seen, {len(records)} embedded "
                      f"(elapsed {time.monotonic() - t0:.0f}s)")
            face_path = rdir / row["id"] / "face.json"
            if not face_path.exists() or not Path(str(row["path"])).exists():
                continue
            try:
                face_box = json.loads(face_path.read_text())
            except Exception:
                continue
            t = visual_type(row["id"], row.get("doc_type_proxy", ""))
            rec = build_donor_record(row["id"], row["path"], t, face_box, embed_fn=embed_fn)
            if rec is not None:
                records.append(rec)
    print(f"[photosub_render] probe faces done: {len(records)}/{n_seen} embedded in "
          f"{time.monotonic() - t0:.0f}s")
    return records


def build_index(args) -> None:
    rdir = regions_dir(args.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this needs VESSL")

    embed_fn = best_available_embed_fn()
    print(f"[photosub_render] donor identity embedding: "
          f"{'ArcFace (insightface buffalo_l/w600k_r50)' if arcface_available() else 'cheap fallback (approximate)'}")

    train_df = load_labels(args.data_dir, "train")
    bonafide = train_df[train_df["label"] == 0].copy()
    bonafide = bonafide[bonafide["path"].map(lambda p: Path(p).exists())]
    if args.max_donors and len(bonafide) > args.max_donors:
        bonafide = bonafide.sample(n=args.max_donors, random_state=DONOR_INDEX_SEED).reset_index(drop=True)
    print(f"[photosub_render] building donor index over {len(bonafide)} bona-fide TRAIN rows "
          f"(max_donors={args.max_donors})...")

    donors = build_donor_index(rdir, bonafide.itertuples(index=False), embed_fn=embed_fn)
    print(f"[photosub_render] {len(donors)}/{len(bonafide)} had a real SCRFD detection AND a "
          "usable embedding -> donor candidate")

    probes = _load_probe_donor_faces(args.data_dir, rdir, embed_fn)
    print(f"[photosub_render] loaded {len(probes)} frozen-probe faces for the exclusion guard")

    kept = exclude_probe_overlap(donors, probes, similarity_threshold=args.exclusion_threshold)
    n_excluded = len(donors) - len(kept)
    exclusion_rate = n_excluded / len(donors) if donors else 0.0
    print(f"[photosub_render] probe-overlap exclusion: {n_excluded}/{len(donors)} donors dropped "
          f"({exclusion_rate * 100:.1f}%, similarity_threshold={args.exclusion_threshold})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.index_path, "wb") as f:
        pickle.dump(kept, f)
    print(f"[photosub_render] wrote {len(kept)} donors -> {args.index_path}")

    print(f"[photosub_render] building natural intra-card baseline over "
          f"{args.n_baseline_cards} cards...")
    baseline_df = build_natural_baseline(
        bonafide.itertuples(index=False), rdir, n_cards=args.n_baseline_cards,
    )
    thresholds = compute_p90_thresholds(baseline_df)
    baseline_csv = out_dir / "baseline_raw.csv"
    baseline_df.to_csv(baseline_csv, index=False)
    Path(args.baseline_path).write_text(json.dumps({
        "n_cards": len(baseline_df), "thresholds_p90": thresholds,
    }, indent=2), encoding="utf-8")
    print(f"[photosub_render] natural baseline over {len(baseline_df)} cards -- p90 thresholds: "
          f"{thresholds} -> {args.baseline_path} (raw: {baseline_csv})")


# ---------------------------------------------------------------------------
# Stage: render
# ---------------------------------------------------------------------------

def _pick_target_rows(data_dir: str, rdir: Path) -> dict[str, dict]:
    """One bona-fide TRAIN row per known template (deterministic), with its face_box + template
    metadata attached. Returns {type: {"row": row, "face_box": ..., "template": ...}}."""
    train_df = load_labels(data_dir, "train")
    bonafide = train_df[train_df["label"] == 0].copy()
    bonafide = bonafide[bonafide["path"].map(lambda p: Path(p).exists())]

    targets: dict[str, dict] = {}
    for t in TEMPLATES:
        sub = bonafide[bonafide["type"] == t].sample(frac=1.0, random_state=TEMPLATE_SAMPLE_SEED)
        for _, row in sub.iterrows():
            face_path = rdir / row["id"] / "face.json"
            if not face_path.exists():
                continue
            try:
                face_box = json.loads(face_path.read_text())
            except Exception:
                continue
            if float(face_box.get("score", 0.0)) <= 0.0:
                continue
            with Image.open(row["path"]) as img:
                img_w, img_h = img.size
            template = build_template_metadata(row["id"], t, face_box, img_w, img_h)
            targets[t] = {"row": row, "face_box": face_box, "template": template}
            break
    missing = set(TEMPLATES) - set(targets)
    if missing:
        print(f"[photosub_render] WARNING: no valid target found for {missing}")
    return targets


def _visual_denormalize(tensor, mean: tuple, std: tuple) -> Image.Image:
    arr = tensor.numpy().transpose(1, 2, 0)
    arr = arr * np.array(std) + np.array(mean)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _perimeter_box(frame_box: dict, img_w: int, img_h: int, margin_frac: float = PERIMETER_MARGIN_FRAC) -> dict:
    x1, y1, x2, y2 = frame_box["x1"], frame_box["y1"], frame_box["x2"], frame_box["y2"]
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin_frac, bh * margin_frac
    return {
        "x1": max(0, int(x1 - mx)), "y1": max(0, int(y1 - my)),
        "x2": min(img_w, int(x2 + mx)), "y2": min(img_h, int(y2 + my)),
    }


def _scale_box(box: dict, orig_size: tuple[int, int], new_size: tuple[int, int]) -> dict:
    """Proportionally rescale a box from ``orig_size`` to ``new_size`` -- exact for
    recapture_transforms's initial anisotropic Resize step. Its LATER probabilistic Perspective
    (p=0.3) / Rotate (p=0.3) steps can shift the box's true post-transform location by a few
    percent when they fire; this is a deliberate simplification for a qualitative "does the rim/
    shadow survive" check, not a claim of pixel-exact alignment."""
    ow, oh = orig_size
    nw, nh = new_size
    sx, sy = nw / ow, nh / oh
    x1 = max(0, min(int(box["x1"] * sx), nw - 1))
    y1 = max(0, min(int(box["y1"] * sy), nh - 1))
    x2 = max(x1 + 1, min(int(box["x2"] * sx), nw))
    y2 = max(y1 + 1, min(int(box["y2"] * sy), nh))
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


_ZOOM_PANEL_MAX_DIM = 320  # caps a single zoom panel's larger side, regardless of source resolution


def _zoom_crop(img: Image.Image, box: dict, zoom: int, max_dim: int = _ZOOM_PANEL_MAX_DIM) -> Image.Image:
    """Crop then zoom by `zoom`x -- capped at `max_dim` on the longer side. Source photos in
    this dataset range from ~500px to ~1600px wide (card_w in ghost_resolvability.csv); a naive
    zoom*native_crop_size blew a 320x400-ish frame-perimeter crop up to multi-thousand-pixel
    panels for the higher-resolution sources, producing a 318-megapixel, 128MB sheet PNG on the
    first render of this fix -- unopenable in practice. The cap means "zoom" is no longer a
    literal pixel-for-pixel multiplier for large sources, only a qualitative "look closer"
    level, which is what this sheet is actually for (a human visual check, not a measurement)."""
    crop = img.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
    w, h = max(1, crop.width * zoom), max(1, crop.height * zoom)
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
    return crop.resize((w, h))


def _ab_dossier_cell(pre_img: Image.Image, post_img: Image.Image, frame_box: dict, caption: str) -> Image.Image:
    """Per-example cell for MODE_A/B: 2x and 4x zooms of the frame PERIMETER (rim, shadow,
    overhang), pre- and post-recapture side by side, so survival at 518px is judgeable by eye."""
    perim_pre = _perimeter_box(frame_box, pre_img.width, pre_img.height)
    perim_post = _scale_box(perim_pre, pre_img.size, post_img.size)

    rows = []
    for zoom in (2, 4):
        pre_z = _zoom_crop(pre_img, perim_pre, zoom)
        post_z = _zoom_crop(post_img, perim_post, zoom)
        h = max(pre_z.height, post_z.height)
        row = Image.new("RGB", (pre_z.width + post_z.width + 6, h), (255, 255, 255))
        row.paste(pre_z, (0, 0))
        row.paste(post_z, (pre_z.width + 6, 0))
        rows.append(row)

    max_w = max(r.width for r in rows)
    caption_h = 40
    canvas = Image.new("RGB", (max_w, sum(r.height for r in rows) + caption_h), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height
    d = ImageDraw.Draw(canvas)
    d.text((2, y + 2), caption, fill=(0, 0, 0), font=_load_font(11))
    d.text((2, y + 20), "left=pre-recapture  right=post-recapture@518 (row1=2x row2=4x)",
           fill=(90, 90, 90), font=_load_font(9))
    return canvas


def _pad_cells_to_common_size(cells: list[Image.Image]) -> list[Image.Image]:
    if not cells:
        return cells
    max_w = max(c.width for c in cells)
    max_h = max(c.height for c in cells)
    padded = []
    for c in cells:
        p = Image.new("RGB", (max_w, max_h), (255, 255, 255))
        p.paste(c, (0, 0))
        padded.append(p)
    return padded


def _cd_dossier_cell(
    img: Image.Image, caption: str, cell_size: tuple[int, int] = (240, 280),
) -> Image.Image:
    cw, ch = cell_size
    thumb_h = ch - 40
    scale = min(cw / img.width, thumb_h / img.height)
    new_w, new_h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
    thumb = img.resize((new_w, new_h))
    cell = Image.new("RGB", (cw, ch), (255, 255, 255))
    cell.paste(thumb, ((cw - new_w) // 2, (thumb_h - new_h) // 2))
    d = ImageDraw.Draw(cell)
    d.text((2, thumb_h + 2), caption, fill=(0, 0, 0), font=_load_font(10))
    return cell


_MODE_FNS = {"A": generate_mode_a, "B": generate_mode_b, "C": generate_mode_c}


def render(args) -> None:
    rdir = regions_dir(args.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this needs VESSL")
    if not Path(args.index_path).exists():
        raise SystemExit(f"{args.index_path} not found -- run --stage build_index first")
    with open(args.index_path, "rb") as f:
        donors: list[DonorFace] = pickle.load(f)
    print(f"[photosub_render] loaded {len(donors)} donors from {args.index_path}")

    # MUST be the same embed_fn build_index used for the donor pool -- sample_donor's cosine
    # similarity against a target built with a different embedding (different dimensionality
    # entirely, cheap=1024 vs ArcFace=512) would crash or, worse, silently compare nonsense.
    embed_fn = best_available_embed_fn()

    if not Path(args.baseline_path).exists():
        raise SystemExit(f"{args.baseline_path} not found -- run --stage build_index first")
    baseline = json.loads(Path(args.baseline_path).read_text())
    thresholds = baseline["thresholds_p90"]
    print(f"[photosub_render] loaded natural-baseline p90 thresholds (n_cards={baseline['n_cards']}): "
          f"{thresholds}")

    targets = _pick_target_rows(args.data_dir, rdir)
    out_dir = Path(args.docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = resolve_data_config(BACKBONE, RECAPTURE_SCALE)
    recapture_tf = build_transforms(
        data_cfg["image_size"], train=True, mean=data_cfg["mean"], std=data_cfg["std"], augment="recapture",
    )

    modes = list(_MODE_FNS)
    if args.enable_mode_d:
        modes.append("D")

    stats_rows = []
    for mode in modes:
        cells = []
        for i in range(N_EXAMPLES_PER_MODE):
            t = TEMPLATES[i % len(TEMPLATES)]
            if t not in targets:
                continue
            info = targets[t]
            row, template = info["row"], info["template"]
            image = Image.open(row["path"]).convert("RGB")
            target_face_box = info["face_box"]
            target_rec = build_donor_record(row["id"], row["path"], t, target_face_box, embed_fn=embed_fn)

            hard = i < N_HARD_PER_MODE
            # Fixed per-mode offset (not hash(mode), which is salted per-process for strings)
            # so re-running this script reproduces byte-identical render sheets.
            mode_offset = {"A": 0, "B": 10_000, "C": 20_000, "D": 30_000}[mode]
            gen_rng = np.random.default_rng(mode_offset + 1000 * (i + 1))
            donor_rec = sample_donor(
                donors, gen_rng, target_rec or donors[i % len(donors)],
                hard_fraction=1.0 if hard else 0.0,
            )
            with Image.open(donor_rec.path) as d_img:
                donor_crop = face_crop_from_box(d_img.convert("RGB"), donor_rec.face_box)

            if mode == "D":
                ghost_box = template["ghost_box"]
                if ghost_box is None:
                    continue
                result = generate_mode_d(
                    image, template["frame_box"], ghost_box, donor_crop, gen_rng, enabled=True,
                )
            else:
                result = _MODE_FNS[mode](image, template["frame_box"], donor_crop, gen_rng)

            caption = f"{t} {'HARD' if hard else 'broad'} donor={donor_rec.id[:8]}"

            if mode in ("A", "B"):
                post_tensor = recapture_tf(result.image)
                post_img = _visual_denormalize(post_tensor, data_cfg["mean"], data_cfg["std"])
                cells.append(_ab_dossier_cell(result.image, post_img, template["frame_box"], caption))

            if mode in ("C", "D"):
                # D_ghost only touches the ghost box (frame_box would show a trivial zero
                # delta, since nothing changed there) -- compare whichever region this
                # particular result actually altered.
                stats_box = template["ghost_box"] if result.mode == "D_ghost" else template["frame_box"]
                fx1, fy1, fx2, fy2 = (stats_box[k] for k in ("x1", "y1", "x2", "y2"))
                s_tampered = region_stats(result.image.crop((fx1, fy1, fx2, fy2)))
                s_orig = region_stats(image.crop((fx1, fy1, fx2, fy2)))
                deltas = {c: abs(s_tampered[c] - s_orig[c]) for c in STAT_COLS}
                passed = all(deltas[c] <= thresholds[c] for c in STAT_COLS)
                stats_rows.append({
                    "mode": result.mode, "type": t, "example": i, "hard": hard, "passed": passed,
                    **{f"delta_{c}": deltas[c] for c in STAT_COLS},
                    **{f"threshold_p90_{c}": thresholds[c] for c in STAT_COLS},
                })
                stats_caption = (
                    f"{t} {'HARD' if hard else 'broad'} "
                    f"blurΔ={deltas['blur_laplacian_var']:.0f} moireΔ={deltas['moire_fft_score']:.0f} "
                    f"blockΔ={deltas['blockiness_score']:.2f} [{'PASS' if passed else 'FAIL'}]"
                )
                cells.append(_cd_dossier_cell(result.image, stats_caption))

        cells = _pad_cells_to_common_size(cells)
        sheet_path = out_dir / f"mode_{mode}_sheet.png"
        render_sheet(cells, sheet_path, ncols=4, nrows=3)
        print(f"[photosub_render] mode {mode}: {len(cells)} examples -> {sheet_path}")

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        stats_csv = out_dir / "stats_check.csv"
        stats_df.to_csv(stats_csv, index=False)
        print(f"[photosub_render] wrote statistical self-check -> {stats_csv}")
        _write_stats_report(out_dir / "stats_check_report.md", stats_df, thresholds, baseline["n_cards"])

    _write_index_report(out_dir, modes, len(donors))


def _write_stats_report(path: Path, stats_df: pd.DataFrame, thresholds: dict, n_baseline_cards: int) -> None:
    lines = ["# MODE_C/D statistical self-check: natural-baseline pass rate\n"]
    lines.append(
        "Design intent (see freuid.photosub.generators' module docstring): MODE_C/D evidence "
        "should be SEMANTIC (a different person, a cross-region mismatch), not a local pixel-"
        "statistics anomaly. Eyeballing render sheets can't confirm that, so this replaces it with "
        "an empirical acceptance criterion (freuid.photosub.baseline_stats): the SAME 3 cheap stats "
        "(freuid.photosub.stats -- blur_laplacian_var, moire_fft_score, blockiness_score) measured "
        f"between two random frame-sized regions on {n_baseline_cards} UNTOUCHED bona-fide cards "
        "give the natural intra-card spread; a MODE_C/D composite PASSES when its tampered-vs-"
        "original delta, for every stat, falls within that natural distribution's 90th percentile.\n"
    )
    lines.append("## p90 thresholds (natural intra-card baseline)\n")
    for c in STAT_COLS:
        lines.append(f"- `{c}`: {thresholds[c]:.3f}")
    lines.append("")

    pass_rate = float(stats_df["passed"].mean())
    lines.append(f"## Overall pass rate: {pass_rate * 100:.1f}% ({int(stats_df['passed'].sum())}/{len(stats_df)})\n")
    if pass_rate < 0.90:
        lines.append(
            f"**Below the ~90% bar -- iterate the degradation matching, don't hand-wave.** "
            f"{len(stats_df) - int(stats_df['passed'].sum())}/{len(stats_df)} examples are locally "
            "distinguishable from their surroundings by these cheap stats despite the semantic-only "
            "design intent.\n"
        )
    else:
        lines.append("At or above the ~90% bar -- composites are not, in aggregate, local-statistics outliers "
                     "relative to how much two random regions on an ordinary untouched card naturally differ.\n")

    lines.append("## Per-stat pass rate\n")
    for c in STAT_COLS:
        stat_pass = float((stats_df[f"delta_{c}"] <= stats_df[f"threshold_p90_{c}"]).mean())
        lines.append(f"- `{c}`: {stat_pass * 100:.1f}% within threshold")
    lines.append("")

    display_cols = ["mode", "type", "hard", "passed"] + [f"delta_{c}" for c in STAT_COLS]
    lines.append("## Per-example detail\n")
    lines.append(df_to_md(stats_df[display_cols].round(3)))
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[photosub_render] wrote stats report -> {path}")


def _write_index_report(out_dir: Path, modes: list[str], n_donors: int) -> None:
    lines = ["# Photo-substitution render sheets -- human-review gate\n"]
    lines.append(
        f"Donor pool: {n_donors} bona-fide TRAIN faces (post probe-overlap exclusion). Templates: "
        f"all {len(TEMPLATES)} known document types ({', '.join(TEMPLATES)}) -- **only 5 exist in "
        "this dataset**, see this script's module docstring; there is no unseen 6th template to add.\n"
    )
    lines.append("## Sheets\n")
    for mode in modes:
        if mode in ("A", "B"):
            lines.append(f"- `mode_{mode}_sheet.png` -- {N_EXAMPLES_PER_MODE} examples "
                         f"({N_HARD_PER_MODE} hard-donor + {N_EXAMPLES_PER_MODE - N_HARD_PER_MODE} broad), "
                         "each a 2x/4x frame-perimeter zoom, pre- vs. post-recapture@518 side by side")
        else:
            lines.append(f"- `mode_{mode}_sheet.png` -- {N_EXAMPLES_PER_MODE} examples with the "
                         "stats-delta + PASS/FAIL verdict in each caption")
    if "C" in modes or "D" in modes:
        lines.append("- `stats_check.csv` / `stats_check_report.md` -- MODE_C/D natural-baseline pass rate")
    lines.append(
        "\n**This is the gate before mass generation -- no full dataset has been written.** "
        "Review the sheets above (and the shape-realism / frame-box-accuracy caveats in "
        "freuid.photosub.generators and .template_regions' module docstrings) before approving.\n"
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[photosub_render] wrote index -> {out_dir / 'README.md'}")


# ---------------------------------------------------------------------------
# Stage: render_shape_variants -- human-review gate for the MODE_A shape-realism fix
# ---------------------------------------------------------------------------

# rect = today's unchanged baseline; the other 6 exercise generate_mode_a's shape/effect kwargs
# (see freuid.photosub.generators' module docstring for the deep-miss exemplars --
# b5eebda1/40dd1055/5542f45f/cd7ad569 -- this is fixing). arch/tape are the photosub_v1
# additions (irregular/tear were already reviewed and approved in an earlier pass -- see
# SHAPE_VARIANTS_README.md's provenance) -- NOT yet reviewed as of this file; do not set
# arch_shape_prob/tape_prob above 0.0 in a real config until this sheet has been regenerated
# (needs the regions cache, VESSL-only -- see docs/photosub_v1_spec.md's render-gate status)
# and reviewed.
_SHAPE_VARIANTS: dict[str, dict] = {
    "rect": {},
    "irregular": {"irregular_shape_prob": 1.0},
    "arch": {"arch_shape_prob": 1.0},
    "tear": {"tear_prob": 1.0},
    "tape": {"tape_prob": 1.0},
    "irregular_tear": {"irregular_shape_prob": 1.0, "tear_prob": 1.0},
    "arch_tape": {"arch_shape_prob": 1.0, "tape_prob": 1.0},
}


def render_shape_variants(args) -> None:
    """Human-review gate for the MODE_A shape-realism fix (irregular hand-cut silhouette + tear
    effect, freuid.photosub.generators): one example per (template, variant) -- rect / irregular
    / tear / irregular_tear -- each the same 2x/4x frame-perimeter zoom, pre- vs. post-recapture,
    that `render`'s MODE_A sheet already uses (`_ab_dossier_cell`). Separate stage and output file
    from `render` -- does not touch or re-run the existing MODE_A/B/C/D sheets, stats check, or
    index report. Per this project's established convention (the same gate MODE_D's ghost-
    darkening decision went through), review this sheet and pick real irregular_shape_prob /
    tear_prob weights BEFORE any mass regeneration."""
    rdir = regions_dir(args.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- this needs VESSL")
    if not Path(args.index_path).exists():
        raise SystemExit(f"{args.index_path} not found -- run --stage build_index first")
    with open(args.index_path, "rb") as f:
        donors: list[DonorFace] = pickle.load(f)
    print(f"[photosub_render] loaded {len(donors)} donors from {args.index_path}")

    embed_fn = best_available_embed_fn()
    targets = _pick_target_rows(args.data_dir, rdir)
    out_dir = Path(args.docs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = resolve_data_config(BACKBONE, RECAPTURE_SCALE)
    recapture_tf = build_transforms(
        data_cfg["image_size"], train=True, mean=data_cfg["mean"], std=data_cfg["std"], augment="recapture",
    )

    cells = []
    for t in TEMPLATES:
        if t not in targets:
            continue
        info = targets[t]
        row, template = info["row"], info["template"]
        image = Image.open(row["path"]).convert("RGB")
        target_rec = build_donor_record(row["id"], row["path"], t, info["face_box"], embed_fn=embed_fn)

        for variant_i, (variant_name, kwargs) in enumerate(_SHAPE_VARIANTS.items()):
            # Fixed (template_index, variant_index) offset -- NOT hash(t)/hash(variant_name),
            # which are salted per-process for strings (see this file's own mode_offset comment
            # in `render` for the same caveat) -- so re-running this stage is byte-reproducible.
            gen_rng = np.random.default_rng(TEMPLATES.index(t) * 1000 + variant_i * 100 + 1)
            donor_rec = sample_donor(donors, gen_rng, target_rec or donors[0], hard_fraction=0.5)
            with Image.open(donor_rec.path) as d_img:
                donor_crop = face_crop_from_box(d_img.convert("RGB"), donor_rec.face_box)
            result = generate_mode_a(image, template["frame_box"], donor_crop, gen_rng, **kwargs)

            post_tensor = recapture_tf(result.image)
            post_img = _visual_denormalize(post_tensor, data_cfg["mean"], data_cfg["std"])
            caption = (
                f"{t} [{variant_name}] donor={donor_rec.id[:8]} "
                f"shape={result.params['shape']} tear={result.params['tear']} "
                f"tape={result.params['tape']}"
            )
            cells.append(_ab_dossier_cell(result.image, post_img, template["frame_box"], caption))

    cells = _pad_cells_to_common_size(cells)
    sheet_path = out_dir / "mode_a_shape_variants_sheet.png"
    n_variants = len(_SHAPE_VARIANTS)
    render_sheet(cells, sheet_path, ncols=n_variants, nrows=len(TEMPLATES))
    print(f"[photosub_render] shape-variant sheet: {len(cells)} examples "
          f"({len(TEMPLATES)} templates x {n_variants} variants) -> {sheet_path}")

    readme = out_dir / "SHAPE_VARIANTS_README.md"
    readme.write_text(
        "# MODE_A shape-realism fix -- human-review gate\n\n"
        f"`{sheet_path.name}`: one row per template ({', '.join(TEMPLATES)}), one column per "
        f"variant ({', '.join(_SHAPE_VARIANTS)}) -- `rect` is today's unchanged baseline "
        "(every prob 0.0), the other 6 exercise "
        "freuid.photosub.generators.generate_mode_a's shape/effect kwargs.\n\n"
        "**photosub_v1 status**: `irregular`/`tear`/`irregular_tear` were reviewed and approved "
        "in an earlier pass (see docs/photosub_v1_spec.md's render-gate section for the review "
        "notes and approved weights). `arch`/`tape`/`arch_tape` are NEW as of photosub_v1 -- "
        "NOT yet reviewed. Regenerate this sheet (needs the regions cache, VESSL-only) and "
        "review before setting arch_shape_prob/tape_prob above 0.0 in any real config.\n\n"
        "Compare against the real deep-miss exemplars this is fixing (deep_miss_dossiers.py's "
        "CHECKLIST_RESULTS / deep_miss_dossiers.html): `b5eebda1` (arch-shaped cutout "
        "overlapping the crest logo), `40dd1055` (irregular silhouette bulging past the "
        "hairline, dark shadow strip), `5542f45f` (literal diagonal tear exposing a lighter "
        "backing patch), `cd7ad569` (visible tan/yellow tape strip, filed under MODE_B but "
        "offered as a MODE_A overlay here too since the tell itself isn't paste-mode-specific)."
        "\n\n"
        "**This is the gate before any mass regeneration or weight decision for "
        "irregular_shape_prob / arch_shape_prob / tear_prob / tape_prob** -- same convention "
        "already used for the MODE_D ghost-darkening decision. Do not proceed to mass "
        "generation until this sheet has been reviewed and weights approved.\n",
        encoding="utf-8",
    )
    print(f"[photosub_render] wrote {readme}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=["build_index", "render", "render_shape_variants"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR))
    p.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH))
    p.add_argument("--baseline-path", default=str(DEFAULT_BASELINE_PATH))
    p.add_argument("--max-donors", type=int, default=4000,
                    help="cap the donor pool for this render-sheet demo run (mass generation "
                         "later should use the full pool -- no cap)")
    p.add_argument("--n-baseline-cards", type=int, default=200)
    p.add_argument(
        "--exclusion-threshold", type=float, default=0.5,
        help="cosine-similarity cutoff for the probe-overlap guard. 0.5 is a starting point "
             "for ArcFace's similarity scale (genuinely different people cluster much lower "
             "than the 0.9 that was tuned for the old cheap pixel embedding) -- re-tune from "
             "the reported exclusion rate, not assumed correct.",
    )
    p.add_argument("--enable-mode-d", action="store_true",
                    help="also render MODE_D examples (see generate_mode_d's ghost-legibility caveat)")
    args = p.parse_args()

    if args.stage == "build_index":
        build_index(args)
    elif args.stage == "render":
        render(args)
    elif args.stage == "render_shape_variants":
        render_shape_variants(args)


if __name__ == "__main__":
    main()
