"""Random-sample spot-review pass over the ACTUAL mass-generated photosub dataset
(freuid.photosub.pipeline's rows.csv) -- distinct from photosub_render_sheets.py's curated,
hand-picked 12-per-mode demo sheets. Those sheets show the generators working as designed on
deliberately chosen templates/donors; this script instead audits what the pipeline actually
produced at scale, checking for edge-case failures a curated sheet wouldn't surface:

  - tiny/degenerate tamper regions (mask area far smaller than a typical frame/ghost box --
    could mean a bad face-box detection produced a tiny frame_box, see template_regions.py's
    frame-box-accuracy caveat)
  - unexpected/odd template types (anything outside the 5 known document types)
  - hard-case donor-pool reuse/exhaustion: freuid.photosub.donor_pool.sample_donor's hard path
    is a SHARP softmax (hard_temperature=0.05, close to argmax) over each target's own top-10
    most-similar candidates -- at mass-generation scale, many different targets could plausibly
    collapse onto the same handful of generically-similar-looking donors from a pool that was
    only ~3866 faces at last count (scripts/analysis/photosub_render_out/donor_index.pkl),
    which would show up as a small number of donor ids accounting for a disproportionate share
    of rows. This is exactly the failure this script's corpus-wide donor_id frequency table is
    built to catch quantitatively, not just by eye.

Two outputs (both read-only over rows.csv / the generated images+masks -- writes nothing back):
  1. photosub_spot_review_report.md -- corpus size, mode/type distribution, the donor-reuse
     frequency table (corpus-wide, not sample-limited), and the tiny-frame / odd-template flag
     lists.
  2. spot_review_sheet_XX.png (paginated) -- N rows sampled UNIFORMLY AT RANDOM (not curated)
     from rows.csv, each rendered with its mask's bounding box drawn on the thumbnail and a
     caption carrying mode/type/donor_id/area_frac/flags.

No training, no submissions, no modification of rows.csv or the generated images/masks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.photosub.mixing import load_photosub_rows  # noqa: E402
from freuid.photosub.pipeline import photosub_generated_dir  # noqa: E402
from freuid.photosub.template_regions import GHOST_TEMPLATES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import df_to_md  # noqa: E402
from review_package import _load_font, render_sheet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "photosub_spot_review_out"
DEFAULT_ROWS_CSV = photosub_generated_dir("data") / "rows.csv"

TINY_AREA_FRAC_THRESHOLD = 0.005  # tamper region < 0.5% of image area flagged as "tiny"
DONOR_REUSE_TOP_N = 20
SAMPLE_SEED = 5  # distinct from every other seed used in this pipeline so far
N_SAMPLE = 80
CELLS_PER_SHEET = 20


def _mask_area_frac(mask_path: str, img_w: int, img_h: int) -> float:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    return float((mask > 127).sum()) / float(img_w * img_h)


def _mask_bbox(mask_path: str) -> tuple[int, int, int, int] | None:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _compute_flags(rows: pd.DataFrame) -> pd.DataFrame:
    """Adds area_frac / odd_type / tiny columns, computed over the FULL corpus (not just the
    random sample) so the corpus-wide counts in the report are exact, not sample-estimated."""
    known_types = set(GHOST_TEMPLATES)
    area_fracs, odd_types = [], []
    for row in rows.itertuples(index=False):
        try:
            with Image.open(row.image_path) as img:
                w, h = img.size
            area_fracs.append(_mask_area_frac(row.mask_path, w, h))
        except Exception:
            area_fracs.append(float("nan"))
        odd_types.append(row.type not in known_types)
    out = rows.copy()
    out["area_frac"] = area_fracs
    out["odd_type"] = odd_types
    out["tiny"] = out["area_frac"] < TINY_AREA_FRAC_THRESHOLD
    return out


def _donor_reuse_table(rows: pd.DataFrame) -> pd.DataFrame:
    have_donor = rows[rows["donor_id"].notna() & (rows["donor_id"] != "")]
    if have_donor.empty:
        return pd.DataFrame(columns=["donor_id", "n_uses", "pct_of_rows_with_donor_id"])
    counts = have_donor["donor_id"].value_counts()
    table = counts.reset_index()
    table.columns = ["donor_id", "n_uses"]
    table["pct_of_rows_with_donor_id"] = 100.0 * table["n_uses"] / len(have_donor)
    return table


def _spot_cell(row, cell_size: tuple[int, int] = (260, 300)) -> Image.Image:
    cw, ch = cell_size
    thumb_h = ch - 56
    img = Image.open(row.image_path).convert("RGB")
    draw_img = img.copy()
    bbox = _mask_bbox(row.mask_path)
    if bbox is not None:
        d = ImageDraw.Draw(draw_img)
        d.rectangle(bbox, outline=(255, 0, 0), width=max(2, img.width // 200))

    scale = min(cw / draw_img.width, thumb_h / draw_img.height)
    new_w, new_h = max(1, int(draw_img.width * scale)), max(1, int(draw_img.height * scale))
    thumb = draw_img.resize((new_w, new_h))
    cell = Image.new("RGB", (cw, ch), (255, 255, 255))
    cell.paste(thumb, ((cw - new_w) // 2, (thumb_h - new_h) // 2))

    flags = []
    if row.tiny:
        flags.append("TINY")
    if row.odd_type:
        flags.append("ODD_TYPE")
    flag_str = f" [{'/'.join(flags)}]" if flags else ""
    donor_str = str(row.donor_id)[:8] if pd.notna(row.donor_id) and row.donor_id else "?"
    caption = (
        f"{row.type} {row.mode}{flag_str}\n"
        f"donor={donor_str} area={row.area_frac * 100:.2f}%"
    )
    d = ImageDraw.Draw(cell)
    d.multiline_text((4, thumb_h + 4), caption, fill=(0, 0, 0), font=_load_font(11))
    return cell


def run(rows_csv: Path, out_dir: Path, n_sample: int, seed: int) -> None:
    rows = load_photosub_rows(rows_csv)
    if rows.empty:
        raise SystemExit(f"{rows_csv} is empty or missing -- nothing to review")
    print(f"[spot_review] loaded {len(rows)} rows from {rows_csv}")

    rows = _compute_flags(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_counts = rows["mode"].value_counts().to_dict()
    type_counts = rows["type"].value_counts().to_dict()
    tiny_rows = rows[rows["tiny"]]
    odd_rows = rows[rows["odd_type"]]
    donor_table = _donor_reuse_table(rows)

    n_sample = min(n_sample, len(rows))
    sample = rows.sample(n=n_sample, random_state=seed).reset_index(drop=True)

    cells = [_spot_cell(r) for r in sample.itertuples(index=False)]
    n_sheets = -(-len(cells) // CELLS_PER_SHEET)
    for s in range(n_sheets):
        chunk = cells[s * CELLS_PER_SHEET : (s + 1) * CELLS_PER_SHEET]
        render_sheet(chunk, out_dir / f"spot_review_sheet_{s:02d}.png", ncols=5, nrows=4)
    print(f"[spot_review] wrote {n_sheets} sheet(s) of {n_sample} randomly-sampled rows -> {out_dir}")

    lines = ["# Photosub mass-generation spot review (random sample, not curated)\n"]
    lines.append(
        f"Corpus: **{len(rows)} rows** in `{rows_csv}`. Sampled **{n_sample}** uniformly at "
        f"random (seed={seed}) for visual review below -- this is NOT the curated 12-per-mode "
        "demo sheet (scripts/analysis/photosub_render_sheets.py); it's what the pipeline "
        "actually produced at scale.\n"
    )
    lines.append("## Mode distribution (full corpus)\n")
    lines.append(df_to_md(pd.DataFrame(list(mode_counts.items()), columns=["mode", "n"])))
    lines.append("")
    lines.append("## Type distribution (full corpus)\n")
    lines.append(df_to_md(pd.DataFrame(list(type_counts.items()), columns=["type", "n"])))
    lines.append("")

    lines.append(f"## Tiny tamper-region flags (area < {TINY_AREA_FRAC_THRESHOLD * 100:.1f}% of image)\n")
    lines.append(f"**{len(tiny_rows)}/{len(rows)}** rows flagged.")
    if len(tiny_rows):
        lines.append("")
        lines.append(df_to_md(tiny_rows[["id", "mode", "type", "area_frac"]].head(30).round(5)))
    lines.append("")

    lines.append("## Odd/unexpected template type flags\n")
    lines.append(f"**{len(odd_rows)}/{len(rows)}** rows flagged (types outside {sorted(GHOST_TEMPLATES)}).")
    if len(odd_rows):
        lines.append("")
        lines.append(df_to_md(odd_rows[["id", "mode", "type"]].head(30)))
    lines.append("")

    lines.append(f"## Donor-pool reuse (corpus-wide, top {DONOR_REUSE_TOP_N})\n")
    have_donor_n = int(donor_table["n_uses"].sum()) if len(donor_table) else 0
    n_missing_donor = len(rows) - have_donor_n
    lines.append(
        f"{len(donor_table)} distinct donor ids used across {have_donor_n} rows that recorded "
        f"one ({n_missing_donor} rows have no donor_id -- generated before that field existed, "
        "or a caller that didn't pass it).\n"
    )
    if len(donor_table):
        top = donor_table.head(DONOR_REUSE_TOP_N).copy()
        top["donor_id"] = top["donor_id"].str[:12]
        lines.append(df_to_md(top.round(2)))
        lines.append("")
        max_uses = int(donor_table["n_uses"].max())
        top1_pct = float(donor_table.iloc[0]["pct_of_rows_with_donor_id"])
        n_singleton = int((donor_table["n_uses"] == 1).sum())
        lines.append(
            f"Most-reused donor: **{max_uses} uses** ({top1_pct:.2f}% of donor-tagged rows). "
            f"{n_singleton}/{len(donor_table)} donors used exactly once. If the top donor(s) "
            "account for a disproportionate share, that's the hard-donor-pool-exhaustion "
            "failure mode this report exists to catch -- consider rebuilding donor_index.pkl "
            "with a larger --max-donors before training.\n"
        )

    lines.append("## Sheets\n")
    for s in range(n_sheets):
        lines.append(f"- `spot_review_sheet_{s:02d}.png`")
    (out_dir / "photosub_spot_review_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[spot_review] wrote report -> {out_dir / 'photosub_spot_review_report.md'}")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows-csv", default=str(DEFAULT_ROWS_CSV))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--n-sample", type=int, default=N_SAMPLE)
    p.add_argument("--seed", type=int, default=SAMPLE_SEED)
    args = p.parse_args()
    run(Path(args.rows_csv), Path(args.out_dir), args.n_sample, args.seed)


if __name__ == "__main__":
    main()
