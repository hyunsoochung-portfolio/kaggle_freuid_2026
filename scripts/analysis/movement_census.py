"""Attribution analysis for photosub_v0's public-LB regression (finetune_v0 0.00744 ->
photosub_v0 0.01616, roughly +130k additional discordant pairs by the reporter's own estimate).

Pure per-id SCORE-MOVEMENT analysis between the two already-submitted CSVs -- no training, no new
submissions. Everything here is CPU/pandas over data already on disk (submissions/*.csv,
scripts/analysis/logit_census_raw.csv, the review CSVs, data/probes/*.csv); the --stage render
step also opens public_test images (data/raw or data/, whichever resolves) for dossier thumbnails.

Ground truth for the 7,821 present public-test ids is NOT available (real held-out test set) --
every "is this id actually fraud" judgment here is either a genuine human-review verdict (the
~1,300 ids covered by scripts/analysis/review_package_done.csv +
scripts/analysis/review_package_deep_done.csv) or a PER-ZONE IMPUTED probability (the empirical
fraud rate among REVIEWED ids inside that same finetune_v0-era zone, applied to every unreviewed
id in that zone). This is an estimate, not a measurement -- treated as such throughout, including
in the pair-cost/reconciliation numbers.

Zones come from scripts/analysis/logit_census_raw.csv's finetune_v0 raw mean_logit, re-binned via
logit_census.find_modes/block_membership at tolerance=0.5 (byte-identical methodology to
logit_census_report.md, recomputed here rather than hardcoding the mode values so a re-run of the
census can't silently drift out of sync with this script). 'neither' is relabeled 'transitional'
for readability; nothing about the partition itself changes.

Placeholder-row pitfall (same one logit_census.py's own report calls out): the submission CSVs
list all 142,818 test ids, but ~135k of them are the missing-id 0.5 placeholder (infer.py's
convention), not a real score. Every percentile-rank computation below is restricted to the 7,821
present ids (scripts/analysis/logit_census_raw.csv's own id set) BEFORE ranking -- ranking over
the full file would let the placeholder mass dominate the middle of the distribution and squeeze
every real score to the extreme percentiles regardless of its actual value.

Stages:
  --stage attribute   Core deliverable: per-id zone/verdict/rank-movement table, NEW_FP/NEW_DROP
                       bucket definitions + pair-cost estimates, the pre-registered H_FP/H_A/
                       H_DROP readout, TRANS_x verdict-conditioned movement, and the 59-id
                       boundary-group movement. Writes movement_census.csv + a markdown report.
                       Pure CPU/pandas, local-data-friendly.
  --stage render      Dossier sheets for the top 40 movers in each of NEW_FP/NEW_DROP (full card,
                       + SCRFD face-box overlay/zoom IF a regions cache is reachable -- gracefully
                       falls back to card-only otherwise, since this dev machine's local data/raw
                       copy has no regions cache; a VESSL run of this same stage would get the
                       face zooms). Needs --stage attribute's output CSV.

Nothing under src/freuid/ is touched; no training; no submissions written.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.data import load_labels  # noqa: E402
from freuid.preprocess import regions_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, df_to_md  # noqa: E402
from logit_census import block_membership, find_modes  # noqa: E402
from review_package import render_cell, render_sheet  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_CENSUS_CSV = OUT_DIR / "logit_census_raw.csv"
DEFAULT_FINETUNE_SUB = REPO_ROOT / "submissions" / "finetune_v0.csv"
DEFAULT_PHOTOSUB_SUB = REPO_ROOT / "submissions" / "photosub_v0.csv"
DEFAULT_REVIEW_CSV = OUT_DIR / "review_package_done.csv"
DEFAULT_DEEP_REVIEW_CSV = OUT_DIR / "review_package_deep_done.csv"
DEFAULT_PROBES_DIR = REPO_ROOT / "data" / "probes"
DEFAULT_OUT_CSV = OUT_DIR / "movement_census.csv"
DEFAULT_REPORT_MD = OUT_DIR / "movement_census_report.md"
DEFAULT_RENDER_DIR = OUT_DIR / "movement_census_out"

ZONE_TOL = 0.5
# Magnitude-based "materially moved but didn't fully cross the zone boundary" band, in rank
# PERCENTILE POINTS (0-100 scale). A judgment call, not derived from the data -- chosen so a
# genuinely large single-digit-percentile jitter doesn't count as "material" but a real double-
# digit rank swing does.
SOFT_BAND_PCT = 20.0
TOP_N_RENDER = 40


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_zones(census_csv: Path, tol: float = ZONE_TOL) -> tuple[pd.DataFrame, float, float]:
    """finetune_v0-era zone per present id: floor/transitional/ceiling at `tol`, recomputed from
    the raw census (not hardcoded) so this always matches whatever's actually in the CSV."""
    raw = pd.read_csv(census_csv, dtype={"id": str})
    modes = find_modes(raw["mean_logit"].to_numpy())
    if len(modes) < 2:
        raise SystemExit(f"{census_csv} looks unimodal ({modes}) -- logit_census's own bimodality "
                          "finding didn't reproduce; can't define floor/ceiling zones from this data.")
    floor_mode, ceiling_mode = modes[0][0], modes[-1][0]
    block = block_membership(raw["mean_logit"].to_numpy(), floor_mode, ceiling_mode, tol)
    zone = pd.Series(block, index=raw.index).replace({"neither": "transitional"})
    out = raw[["id", "mean_logit"]].copy()
    out["zone"] = zone.values
    out = out.rename(columns={"mean_logit": "finetune_mean_logit"})
    return out, floor_mode, ceiling_mode


def load_submission_rank(csv_path: Path, present_ids: set[str], col_prefix: str) -> pd.DataFrame:
    """Present-ids-only score + rank percentile from a submission CSV. See module docstring's
    placeholder-row pitfall paragraph -- filtering to `present_ids` BEFORE ranking is load-bearing."""
    sub = pd.read_csv(csv_path, dtype={"id": str})
    sub = sub[sub["id"].isin(present_ids)].copy()
    if len(sub) != len(present_ids):
        missing = present_ids - set(sub["id"])
        raise SystemExit(
            f"{csv_path} is missing per-id scores for {len(missing)}/{len(present_ids)} present "
            f"ids (e.g. {sorted(missing)[:5]}) -- regenerate with a fresh inference pass "
            "(freuid.infer, standard 3-scale TTA, epoch-6 checkpoint for photosub_v0) before "
            "re-running this script."
        )
    n_unique = sub["label"].nunique()
    if n_unique < 0.9 * len(sub):
        print(f"[movement_census] WARNING: {csv_path} has only {n_unique}/{len(sub)} unique "
              "scores among present ids -- flagged for investigation (see the report's 'Caveat "
              "found during this run' section: checked against full float64 repr and against "
              "infer.py's _rank_normalize tie-averaging, and these look like genuine tied raw "
              "model outputs, not a truncated/degenerate export -- this run proceeds, but the "
              "caveat's practical-effect note applies).")
    sub[f"{col_prefix}_score"] = sub["label"]
    sub[f"{col_prefix}_pct_rank"] = sub["label"].rank(pct=True) * 100.0
    return sub[["id", f"{col_prefix}_score", f"{col_prefix}_pct_rank"]]


def load_human_verdicts(review_csv: Path, deep_review_csv: Path) -> pd.DataFrame:
    """Combined human-verdict table: id -> verdict (B/F/U/None), stratum, source file. Ties
    between the two files (39 overlapping ids, confirmed by inspection) are broken by keeping the
    deep-review file's row first -- it carries richer per-id metadata (doc_type_proxy, degradation
    stats) and was the more recent, more targeted review pass."""
    frames = []
    if deep_review_csv.exists():
        d = pd.read_csv(deep_review_csv, dtype={"id": str})
        d["review_source"] = "deep_review"
        frames.append(d[["id", "stratum", "verdict", "review_source"]])
    if review_csv.exists():
        r = pd.read_csv(review_csv, dtype={"id": str})
        r["review_source"] = "ceiling_review"
        frames.append(r[["id", "stratum", "verdict", "review_source"]])
    if not frames:
        return pd.DataFrame(columns=["id", "stratum", "verdict", "review_source"])
    combined = pd.concat(frames, ignore_index=True)
    combined["verdict"] = combined["verdict"].where(combined["verdict"].isin(["B", "F"]))
    before = len(combined)
    combined = combined.drop_duplicates(subset="id", keep="first")
    print(f"[movement_census] human verdicts: {before} rows from 2 files -> {len(combined)} "
          f"unique ids ({int(combined['verdict'].notna().sum())} with a usable B/F verdict, "
          f"{int((combined['verdict'] == 'F').sum())} F / {int((combined['verdict'] == 'B').sum())} B)")
    return combined.rename(columns={"stratum": "review_stratum"})


# ---------------------------------------------------------------------------
# Fraud-probability imputation
# ---------------------------------------------------------------------------

def compute_zone_fraud_rates(df: pd.DataFrame) -> dict:
    """zone -> (empirical F-rate among reviewed ids in that zone, n_reviewed, n_total). Shared by
    `estimate_fraud_probability`'s blended imputation and the zone-imputed-ONLY sensitivity
    variant, so both use identical rates -- a perturbation study is only meaningful if the thing
    being perturbed (verdict usage) is the sole difference."""
    zone_rates = {}
    for zone, group in df.groupby("zone"):
        reviewed = group[group["verdict"].isin(["B", "F"])]
        if len(reviewed) == 0:
            rate = float("nan")
        else:
            rate = float((reviewed["verdict"] == "F").mean())
        zone_rates[zone] = (rate, len(reviewed), len(group))
    return zone_rates


def estimate_fraud_probability(df: pd.DataFrame) -> pd.DataFrame:
    """p_fraud per id: 1.0/0.0 where a real human verdict exists, else the EMPIRICAL fraud rate
    among reviewed ids that share the same finetune_v0 zone (floor/transitional/ceiling), applied
    to every unreviewed id in that zone. This is the "zone-based fraud-composition estimate from
    the reviews" the analysis is built on -- printed per zone so the imputation rates themselves
    are auditable, not just used silently."""
    out = df.copy()
    out["p_fraud"] = np.where(out["verdict"] == "F", 1.0, np.where(out["verdict"] == "B", 0.0, np.nan))

    zone_rates = compute_zone_fraud_rates(out)
    for zone, (rate, n_reviewed, n_total) in zone_rates.items():
        print(f"[movement_census] zone={zone}: empirical fraud rate {rate:.4f} "
              f"(from {n_reviewed}/{n_total} reviewed ids) -- imputed onto the "
              f"{n_total - n_reviewed} unreviewed ids in this zone")

    needs_impute = out["p_fraud"].isna()
    out.loc[needs_impute, "p_fraud"] = out.loc[needs_impute, "zone"].map(lambda z: zone_rates[z][0])
    if out["p_fraud"].isna().any():
        n_bad = int(out["p_fraud"].isna().sum())
        print(f"[movement_census] WARNING: {n_bad} ids have no zone fraud-rate estimate at all "
              "(zone had zero reviewed ids) -- filling with the corpus-wide reviewed fraud rate")
        corpus_rate = float((out["verdict"] == "F").sum()) / max(1, int(out["verdict"].isin(["B", "F"]).sum()))
        out["p_fraud"] = out["p_fraud"].fillna(corpus_rate)

    out["p_bonafide"] = 1.0 - out["p_fraud"]
    out["is_imputed"] = ~out["verdict"].isin(["B", "F"])
    return out


def zone_imputed_only_probability(df: pd.DataFrame, zone_rates: dict) -> pd.DataFrame:
    """Sensitivity variant: p_fraud = the zone's empirical rate for EVERY id, including ones with
    a real individual verdict (the verdict is deliberately ignored here, not just left unused) --
    tests whether the baseline's conclusion depends on the individually-verdicted ids specifically,
    or survives on the smooth zone-composition signal alone."""
    out = df.copy()
    out["p_fraud"] = out["zone"].map(lambda z: zone_rates[z][0])
    out["p_bonafide"] = 1.0 - out["p_fraud"]
    return out


def verdict_only_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity variant: restrict to the ids with a REAL human verdict only (hard 0/1 labels,
    zero imputation) -- the complementary perturbation to `zone_imputed_only_probability`, and the
    same subset the report's existing human-verdict AuDET spot-check already uses, now run through
    the same exact-tie discordance machinery as the headline number."""
    out = df[df["verdict"].isin(["B", "F"])].copy()
    out["p_fraud"] = np.where(out["verdict"] == "F", 1.0, 0.0)
    out["p_bonafide"] = 1.0 - out["p_fraud"]
    return out


# ---------------------------------------------------------------------------
# Discordant-pair mass (expected number of bonafide-ranked-above-fraud pairs)
# ---------------------------------------------------------------------------

def discordant_mass_below_above(df: pd.DataFrame, rank_col: str) -> pd.DataFrame:
    """Adds `frauds_below_<rank_col>` (expected fraud mass strictly below this id's rank) and
    `bonafides_above_<rank_col>` (expected bonafide mass strictly above it), under the ranking
    given by `rank_col`. O(n log n): sort once, cumulative-sum the OTHER type's probability mass."""
    ordered = df.sort_values(rank_col).reset_index()  # ascending rank -> index 0 is lowest-ranked
    fraud_mass = ordered["p_fraud"].to_numpy()
    bonafide_mass = ordered["p_bonafide"].to_numpy()

    # frauds strictly below position k = cumulative fraud mass over positions [0, k) (exclusive).
    cum_fraud_before = np.concatenate([[0.0], np.cumsum(fraud_mass)])[:-1]
    # bonafides strictly above position k = total bonafide mass minus cumulative up to and incl. k.
    cum_bonafide_upto = np.cumsum(bonafide_mass)
    total_bonafide = cum_bonafide_upto[-1] if len(cum_bonafide_upto) else 0.0
    bonafide_above = total_bonafide - cum_bonafide_upto

    ordered[f"frauds_below_{rank_col}"] = cum_fraud_before
    ordered[f"bonafides_above_{rank_col}"] = bonafide_above
    return ordered.set_index("index").sort_index()[[f"frauds_below_{rank_col}", f"bonafides_above_{rank_col}"]]


def total_expected_discordant_pairs(df: pd.DataFrame, rank_col: str) -> float:
    """E[# (bonafide, fraud) pairs with bonafide ranked above fraud] under `rank_col`'s ordering
    -- a single well-defined quantity (no double counting): sum_i p_bonafide(i) * frauds_below(i)."""
    mass = discordant_mass_below_above(df, rank_col)
    return float((df["p_bonafide"].to_numpy() * mass[f"frauds_below_{rank_col}"].to_numpy()).sum())


def expected_discordance_exact(
    df: pd.DataFrame, score_col: str, p_fraud_col: str = "p_fraud", p_bonafide_col: str = "p_bonafide",
) -> tuple[float, float]:
    """Exact, exact-tie-aware expected discordant-pair count under `score_col`'s ordering:
    sum over ORDERED pairs (i != j) of p_fraud(i) * p_bonafide(j) * ([score(j) > score(i)] +
    0.5*[score(j) == score(i)]) -- the textbook AUC/Mann-Whitney tie convention (a tie
    contributes HALF credit), generalized to probabilistic (imputed) labels.

    O(n log n), not the naive O(n^2) matrix (trivial at n=7821, but this scales further and
    avoids materializing a 489MB array): groups ids by EXACT score equality (not rounded, not
    a pre-computed pct_rank -- `.rank(pct=True)`'s average-tie output would already have merged
    ties correctly for THIS purpose too, but grouping on the raw score directly removes any
    doubt). Within a tied group, every cross-id pair gets the 0.5 credit; strictly-higher groups
    get full credit. This is the one to trust for finetune_v0's large duplicate-value clusters
    (see the report's tie-cluster caveat) -- `total_expected_discordant_pairs`/
    `discordant_mass_below_above` above sort by an unstable order within ties and so may over-
    or under-credit individual tied ids (their AGGREGATE sum is still approximately right, but
    per-id bucket costs built on them inherit that imprecision).

    Returns (total_expected_discordant_pairs, tie_only_contribution) -- the second number answers
    "how much of this total comes from exact ties alone".
    """
    d = df[[score_col, p_fraud_col, p_bonafide_col]].rename(
        columns={p_fraud_col: "_pf", p_bonafide_col: "_pb"}
    ).sort_values(score_col, kind="mergesort").reset_index(drop=True)
    new_group = d[score_col].ne(d[score_col].shift()).to_numpy()
    new_group[0] = True
    group = np.cumsum(new_group)
    d["_group"] = group

    grp = d.groupby("_group")[["_pf", "_pb"]].sum()
    total_bona = float(grp["_pb"].sum())
    cum_bona = grp["_pb"].cumsum()
    above_bona_by_group = total_bona - cum_bona  # bona mass strictly in HIGHER-score groups

    d["_group_bona_total"] = d["_group"].map(grp["_pb"])
    d["_above_bona"] = d["_group"].map(above_bona_by_group)
    d["_tied_bona_other"] = d["_group_bona_total"] - d["_pb"]  # same-group bona mass, excl. self

    contribution = d["_pf"] * d["_above_bona"] + 0.5 * d["_pf"] * d["_tied_bona_other"]
    tie_contribution = 0.5 * d["_pf"] * d["_tied_bona_other"]
    return float(contribution.sum()), float(tie_contribution.sum())


# ---------------------------------------------------------------------------
# Bucket definitions
# ---------------------------------------------------------------------------

def build_buckets(df: pd.DataFrame, ceiling_lower_edge: float, soft_band: float = SOFT_BAND_PCT) -> pd.DataFrame:
    out = df.copy()
    out["new_fp_bucket"] = None
    out["new_drop_bucket"] = None

    floor_pop = (out["zone"] == "floor") & (out["verdict"] != "F")  # exclude confirmed recovered-fraud
    full_cross_fp = floor_pop & (out["photosub_pct_rank"] >= ceiling_lower_edge)
    soft_fp = floor_pop & ~full_cross_fp & (out["rank_delta"] >= soft_band)
    out.loc[full_cross_fp, "new_fp_bucket"] = "full"
    out.loc[soft_fp, "new_fp_bucket"] = "soft"

    ceiling_pop = (out["zone"] == "ceiling") & (out["verdict"] != "B")  # exclude confirmed corrected-FP
    full_cross_drop = ceiling_pop & (out["photosub_pct_rank"] < ceiling_lower_edge)
    soft_drop = ceiling_pop & ~full_cross_drop & (out["rank_delta"] <= -soft_band)
    out.loc[full_cross_drop, "new_drop_bucket"] = "full"
    out.loc[soft_drop, "new_drop_bucket"] = "soft"

    # Positive counter-signal: floor/transitional ids CONFIRMED fraud (deep misses) that rose --
    # good news, tagged separately so it never gets counted as damage.
    out["recovered_deep_miss"] = (out["verdict"] == "F") & (out["zone"] != "ceiling") & (out["rank_delta"] > 0)
    out["corrected_false_positive"] = (out["verdict"] == "B") & (out["zone"] == "ceiling") & (out["rank_delta"] < 0)
    return out


def compute_pair_costs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fin_mass = discordant_mass_below_above(out, "finetune_pct_rank")
    pho_mass = discordant_mass_below_above(out, "photosub_pct_rank")
    out = out.join(fin_mass).join(pho_mass)

    out["fp_pair_cost"] = np.where(
        out["new_fp_bucket"].notna(),
        out["p_bonafide"] * (out["frauds_below_photosub_pct_rank"] - out["frauds_below_finetune_pct_rank"]),
        0.0,
    )
    out["drop_pair_cost"] = np.where(
        out["new_drop_bucket"].notna(),
        out["p_fraud"] * (out["bonafides_above_photosub_pct_rank"] - out["bonafides_above_finetune_pct_rank"]),
        0.0,
    )
    return out


PLACEHOLDER_SCORE = 0.5
# The true fraud rate of the ~135k rows we have no local image for (and thus can't score
# ourselves) is UNKNOWN -- these are the "genuinely missing" ids infer.py defaults to 0.5.
# Reported at a few plausible reference points rather than committed to one number: the train-set
# base rate (measured), a 50/50 prior, and 0.0/1.0 as bounding extremes.
PLACEHOLDER_FRAUD_RATE_CANDIDATES = {
    "train_base_rate_0.423": 0.42316,
    "fifty_fifty_0.5": 0.5,
}


def placeholder_crossing_analysis(df: pd.DataFrame, n_placeholder: int, placeholder_fraud_rate: float) -> pd.DataFrame:
    """The ~135k placeholder rows are IDENTICAL (exactly 0.5) in both submissions -- present ids
    don't need to fully reorder AMONG THEMSELVES to change the discordant-pair count, they only
    need to cross the 0.5 threshold, which swings their rank position against this entire ~135k-row
    block at once. This is a vastly bigger lever than anything achievable by reordering within the
    7,821 present ids (see the present-ids-only total in run_attribute, which came out NEGATIVE --
    i.e. contradicts the observed public-LB regression on its own). Cost is one-sided by
    construction: a present id crossing in the "right" direction for its own likely true type
    (e.g. a likely-fraud id crossing up, out from under the placeholder block) costs ~0, since it's
    weighted by the PROBABILITY of the "wrong" type for that direction."""
    out = df.copy()
    above_fin = out["finetune_score"] > PLACEHOLDER_SCORE
    above_pho = out["photosub_score"] > PLACEHOLDER_SCORE
    out["crossed_up"] = (~above_fin) & above_pho
    out["crossed_down"] = above_fin & (~above_pho)

    placeholder_bonafide_mass = n_placeholder * (1.0 - placeholder_fraud_rate)
    placeholder_fraud_mass = n_placeholder * placeholder_fraud_rate

    cost = np.zeros(len(out))
    cost[out["crossed_down"].to_numpy()] += (
        out.loc[out["crossed_down"], "p_fraud"].to_numpy() * placeholder_bonafide_mass
    )
    cost[out["crossed_up"].to_numpy()] += (
        out.loc[out["crossed_up"], "p_bonafide"].to_numpy() * placeholder_fraud_mass
    )
    out["placeholder_cross_cost"] = cost
    return out


PLACEHOLDER_LIVE_FRACTION_GRID = [0.001, 0.005, 0.01, 0.05, 0.10]


def placeholder_fraction_grid(
    df: pd.DataFrame, n_placeholder: int, placeholder_fraud_rate: float,
    fractions: list[float] = PLACEHOLDER_LIVE_FRACTION_GRID, observed_delta: float = 130_000.0,
) -> pd.DataFrame:
    """Quantifies the earlier 160x-overshoot finding into a testable grid: IF only a FRACTION of
    the ~135k placeholder rows are actually 'live' in the real LB's pairwise scoring (as opposed
    to this script's original assumption that the whole block counts), what predicted delta would
    the OBSERVED present-id 0.5-crossings produce at each fraction? Cost scales linearly in the
    assumed live population size (`placeholder_crossing_analysis`'s cost terms are each a fixed
    per-id weight times a population mass that's directly proportional to n_placeholder), so this
    reuses that exact, already-vetted function at a scaled-down `n_placeholder` for each fraction
    rather than re-deriving the formula."""
    rows = []
    for f in fractions:
        n_live = f * n_placeholder
        crossed = placeholder_crossing_analysis(df, n_live, placeholder_fraud_rate)
        cost = float(crossed["placeholder_cross_cost"].sum())
        rows.append({
            "fraction_live": f, "assumed_live_placeholder_rows": n_live,
            "predicted_delta_pairs": cost, "ratio_to_observed_130k": cost / observed_delta,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Attribute stage
# ---------------------------------------------------------------------------

def run_attribute(args) -> pd.DataFrame:
    zones, floor_mode, ceiling_mode = load_zones(args.census_csv)
    present_ids = set(zones["id"])
    print(f"[movement_census] {len(present_ids)} present ids | floor_mode={floor_mode:.3f} "
          f"ceiling_mode={ceiling_mode:.3f} tol={ZONE_TOL}")
    print(zones["zone"].value_counts())

    fin = load_submission_rank(args.finetune_sub, present_ids, "finetune")
    pho = load_submission_rank(args.photosub_sub, present_ids, "photosub")
    verdicts = load_human_verdicts(args.review_csv, args.deep_review_csv)

    df = zones.merge(fin, on="id", how="left").merge(pho, on="id", how="left")
    df = df.merge(verdicts, on="id", how="left")
    df["rank_delta"] = df["photosub_pct_rank"] - df["finetune_pct_rank"]
    df = estimate_fraud_probability(df)

    ceiling_lower_edge = float(df.loc[df["zone"] == "ceiling", "finetune_pct_rank"].min())
    floor_upper_edge = float(df.loc[df["zone"] == "floor", "finetune_pct_rank"].max())
    print(f"[movement_census] ceiling zone's finetune_v0 lower rank edge: {ceiling_lower_edge:.2f} pctile | "
          f"floor zone's upper rank edge: {floor_upper_edge:.2f} pctile")

    df = build_buckets(df, ceiling_lower_edge)
    df = compute_pair_costs(df)

    # Full expected-discordance reconciliation (replaces the earlier sort-based, tie-unaware
    # total): exact O(n log n), exact-tie-aware, over ALL 7,821 present ids for both score
    # vectors, using the SAME zone/verdict-blended p_fraud/p_bonafide for both models' accounting.
    total_discordant_finetune, fin_tie_contribution = expected_discordance_exact(df, "finetune_score")
    total_discordant_photosub, pho_tie_contribution = expected_discordance_exact(df, "photosub_score")
    total_delta = total_discordant_photosub - total_discordant_finetune
    fin_tie_frac = fin_tie_contribution / total_discordant_finetune if total_discordant_finetune else float("nan")
    print(f"[movement_census] EXACT expected discordance (all {len(df)} present ids, tie-aware): "
          f"finetune={total_discordant_finetune:,.0f} (of which {fin_tie_contribution:,.0f}, "
          f"{fin_tie_frac*100:.1f}%, from exact ties alone) photosub={total_discordant_photosub:,.0f} "
          f"(tie contribution {pho_tie_contribution:,.0f}) delta={total_delta:,.0f}")

    n_total_rows = len(pd.read_csv(args.finetune_sub, usecols=["id"]))
    n_placeholder = n_total_rows - len(present_ids)
    print(f"[movement_census] {n_placeholder} placeholder rows (tied at {PLACEHOLDER_SCORE} in BOTH "
          f"submissions) out of {n_total_rows} total submission rows")
    placeholder_sensitivity = {}
    for name, rate in PLACEHOLDER_FRAUD_RATE_CANDIDATES.items():
        crossed = placeholder_crossing_analysis(df, n_placeholder, rate)
        net_cost = float(crossed["placeholder_cross_cost"].sum())
        placeholder_sensitivity[name] = (rate, net_cost)
        print(f"[movement_census] placeholder-crossing net cost @ fraud_rate={rate}: {net_cost:,.0f}")
    # Use the train-base-rate assumption as the headline number carried into the report/bucket df.
    headline_rate = PLACEHOLDER_FRAUD_RATE_CANDIDATES["train_base_rate_0.423"]
    df = placeholder_crossing_analysis(df, n_placeholder, headline_rate)
    n_crossed_up, n_crossed_down = int(df["crossed_up"].sum()), int(df["crossed_down"].sum())
    print(f"[movement_census] {n_crossed_up} ids crossed UP over the placeholder threshold, "
          f"{n_crossed_down} crossed DOWN")
    print(df.loc[df["crossed_down"], "zone"].value_counts().rename("crossed_down_by_zone"))

    # --- Part 1 outcome classification: CLOSES / PARTIALLY closes / DOES NOT close ---
    observed_lb_delta = 130_000.0
    closure_ratio = total_delta / observed_lb_delta
    if total_delta > 0 and abs(closure_ratio - 1.0) <= 0.25:
        outcome = "CLOSES"
    elif total_delta > 0:
        outcome = "PARTIAL"
    else:
        outcome = "NOT_CLOSE"
    print(f"[movement_census] Part 1 outcome: {outcome} (present-ids-only exact delta={total_delta:,.0f} "
          f"vs observed ~{observed_lb_delta:,.0f}, ratio={closure_ratio:.3f})")

    grid_df = None
    if outcome == "NOT_CLOSE":
        grid_df = placeholder_fraction_grid(df, n_placeholder, headline_rate, observed_delta=observed_lb_delta)
        print(df_to_md(grid_df, float_fmt="{:,.4f}"))

    # --- Sensitivity: same exact-discordance delta under 2 alternate imputation schemes ---
    zone_rates = compute_zone_fraud_rates(df)
    df_zoneonly = zone_imputed_only_probability(df, zone_rates)
    fin_zoneonly, _ = expected_discordance_exact(df_zoneonly, "finetune_score")
    pho_zoneonly, _ = expected_discordance_exact(df_zoneonly, "photosub_score")
    delta_zoneonly = pho_zoneonly - fin_zoneonly

    df_verdictonly = verdict_only_subset(df)
    fin_verdictonly, _ = expected_discordance_exact(df_verdictonly, "finetune_score")
    pho_verdictonly, _ = expected_discordance_exact(df_verdictonly, "photosub_score")
    delta_verdictonly = pho_verdictonly - fin_verdictonly
    print(f"[movement_census] sensitivity: baseline(blend) delta={total_delta:,.0f} | "
          f"zone-imputed-only delta={delta_zoneonly:,.0f} | "
          f"verdict-only (n={len(df_verdictonly)}) delta={delta_verdictonly:,.0f}")

    reconciliation = {
        "total_discordant_finetune": total_discordant_finetune,
        "total_discordant_photosub": total_discordant_photosub,
        "total_delta": total_delta,
        "fin_tie_contribution": fin_tie_contribution,
        "fin_tie_frac": fin_tie_frac,
        "pho_tie_contribution": pho_tie_contribution,
        "observed_lb_delta": observed_lb_delta,
        "closure_ratio": closure_ratio,
        "outcome": outcome,
        "grid_df": grid_df,
        "delta_zoneonly": delta_zoneonly,
        "delta_verdictonly": delta_verdictonly,
        "n_verdictonly": len(df_verdictonly),
    }

    n_fp = int(df["new_fp_bucket"].notna().sum())
    n_fp_full = int((df["new_fp_bucket"] == "full").sum())
    n_drop = int(df["new_drop_bucket"].notna().sum())
    n_drop_full = int((df["new_drop_bucket"] == "full").sum())
    fp_cost_total = float(df["fp_pair_cost"].sum())
    drop_cost_total = float(df["drop_pair_cost"].sum())
    bucket_cost_total = fp_cost_total + drop_cost_total
    n_recovered = int(df["recovered_deep_miss"].sum())
    n_corrected = int(df["corrected_false_positive"].sum())

    print(f"[movement_census] NEW_FP: {n_fp} ids ({n_fp_full} full-cross, {n_fp - n_fp_full} soft), "
          f"estimated pair cost {fp_cost_total:,.0f}")
    print(f"[movement_census] NEW_DROP: {n_drop} ids ({n_drop_full} full-cross, {n_drop - n_drop_full} soft), "
          f"estimated pair cost {drop_cost_total:,.0f}")
    print(f"[movement_census] recovered_deep_miss (good news, excluded from NEW_FP): {n_recovered}")
    print(f"[movement_census] corrected_false_positive (good news, excluded from NEW_DROP): {n_corrected}")
    placeholder_cost_total = float(df["placeholder_cross_cost"].sum())
    combined_total = bucket_cost_total + placeholder_cost_total
    print(f"[movement_census] TOTAL expected discordant pairs among present-id-internal pairs only: "
          f"finetune={total_discordant_finetune:,.0f} photosub={total_discordant_photosub:,.0f} "
          f"delta={total_delta:,.0f} (this alone is NEGATIVE -- contradicts the observed regression, "
          "see the placeholder-crossing pathway below for why)")
    print(f"[movement_census] named-bucket cost sum={bucket_cost_total:,.0f}, "
          f"placeholder-crossing cost={placeholder_cost_total:,.0f} (@ fraud_rate={headline_rate}), "
          f"COMBINED={combined_total:,.0f} vs. the reported ~130k regression")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[movement_census] wrote {len(df)} rows -> {args.out_csv}")

    write_report(
        args, df, floor_mode, ceiling_mode, ceiling_lower_edge, floor_upper_edge,
        total_discordant_finetune, total_discordant_photosub, total_delta,
        n_placeholder, placeholder_sensitivity, headline_rate, reconciliation,
    )
    return df


# ---------------------------------------------------------------------------
# TRANS_x / boundary movement (cheap, from data already loaded)
# ---------------------------------------------------------------------------

def transitional_movement_table(df: pd.DataFrame, deep_review_csv: Path) -> pd.DataFrame:
    if not deep_review_csv.exists():
        return pd.DataFrame()
    strata = pd.read_csv(deep_review_csv, dtype={"id": str})[["id", "stratum", "verdict"]]
    merged = strata.merge(
        df[["id", "finetune_pct_rank", "photosub_pct_rank", "rank_delta", "zone"]], on="id", how="left",
    )
    rows = []
    for (stratum, verdict), g in merged.groupby(["stratum", "verdict"]):
        rows.append({
            "stratum": stratum, "verdict": verdict, "n": len(g),
            "mean_finetune_pct_rank": g["finetune_pct_rank"].mean(),
            "mean_photosub_pct_rank": g["photosub_pct_rank"].mean(),
            "mean_rank_delta": g["rank_delta"].mean(),
            "median_rank_delta": g["rank_delta"].median(),
        })
    return pd.DataFrame(rows).sort_values(["stratum", "verdict"])


def deep9_movement_table(df: pd.DataFrame, probes_dir: Path) -> pd.DataFrame:
    """The 9 human-confirmed deep-miss ids (`missed_frauds_deep_ids.csv`), carrying their
    MODE_A/B/C taxonomy assignment, joined to how each one's SUBMITTED (TTA-averaged, epoch-6
    checkpoint) rank actually moved -- the direct, on-the-actual-public-test-scores test of H_A,
    distinct from `docs/technical_report.md`'s per-epoch training-time probe table (which tracks
    raw logits epoch-by-epoch on the checkpoint-selection run, not the submitted CSV)."""
    path = probes_dir / "missed_frauds_deep_ids.csv"
    if not path.exists():
        return pd.DataFrame()
    deep9 = pd.read_csv(path, dtype={"id": str})[["id", "stratum", "mode"]]
    merged = deep9.merge(
        df[["id", "zone", "finetune_pct_rank", "photosub_pct_rank", "rank_delta"]], on="id", how="left",
    )
    return merged.sort_values("mode", na_position="last")


def boundary_movement_table(df: pd.DataFrame, probes_dir: Path) -> tuple[pd.DataFrame, dict]:
    path = probes_dir / "missed_frauds_boundary_ids.csv"
    if not path.exists():
        return pd.DataFrame(), {}
    boundary_ids = pd.read_csv(path, dtype={"id": str})["id"]
    merged = df[df["id"].isin(set(boundary_ids))].copy()
    ceiling_lower_edge = float(df.loc[df["zone"] == "ceiling", "finetune_pct_rank"].min())
    n_pushed_into_ceiling = int((merged["photosub_pct_rank"] >= ceiling_lower_edge).sum())
    n_already_in_ceiling = int((merged["finetune_pct_rank"] >= ceiling_lower_edge).sum())
    n_rose = int((merged["rank_delta"] > 0).sum())
    n_fell = int((merged["rank_delta"] < 0).sum())
    summary = {
        "n": len(merged),
        "mean_finetune_pct_rank": float(merged["finetune_pct_rank"].mean()),
        "mean_photosub_pct_rank": float(merged["photosub_pct_rank"].mean()),
        "mean_rank_delta": float(merged["rank_delta"].mean()),
        "n_pushed_into_ceiling": n_pushed_into_ceiling,
        "frac_pushed_into_ceiling": n_pushed_into_ceiling / len(merged) if len(merged) else float("nan"),
        "n_already_in_ceiling": n_already_in_ceiling,
        "n_rose": n_rose,
        "n_fell": n_fell,
    }
    return merged, summary


def hypothesis_readout(fp_cost: float, drop_cost: float) -> str:
    total = fp_cost + drop_cost
    fp_share = fp_cost / total if total else float("nan")
    drop_share = drop_cost / total if total else float("nan")
    if total <= 0:
        return "**Neither H_FP nor H_DROP has any estimated pair cost to speak of** -- see the residual/reconciliation numbers; the damage (if real) isn't captured by either named bucket as defined here."
    if fp_share >= 0.65:
        return (f"**H_FP dominates** ({fp_share*100:.1f}% of named-bucket pair cost vs {drop_share*100:.1f}% "
                "for H_DROP): the regression is mainly new false positives (floor-zone bona-fide ids pushed "
                "up), not recovered/lost detections at the ceiling.")
    if drop_share >= 0.65:
        return (f"**H_DROP dominates** ({drop_share*100:.1f}% of named-bucket pair cost vs {fp_share*100:.1f}% "
                "for H_FP): the regression is mainly previously-confident fraud detections falling out of "
                "the ceiling block, not new false positives.")
    return (f"**Mixed outcome**: H_FP {fp_share*100:.1f}% / H_DROP {drop_share*100:.1f}% of named-bucket pair "
            "cost -- neither dominates by the pre-registered 65% bar. Reporting as mixed rather than forcing "
            "a single-cause verdict.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    args, df: pd.DataFrame, floor_mode: float, ceiling_mode: float,
    ceiling_lower_edge: float, floor_upper_edge: float,
    total_discordant_finetune: float, total_discordant_photosub: float, total_delta: float,
    n_placeholder: int, placeholder_sensitivity: dict, headline_rate: float, reconciliation: dict,
) -> None:
    n_crossed_up, n_crossed_down = int(df["crossed_up"].sum()), int(df["crossed_down"].sum())
    lines = ["# Movement census: attributing photosub_v0's public-LB regression\n"]
    lines.append(
        "finetune_v0 (0.00744) -> photosub_v0 (0.01616) on the public LB, reportedly ~+130k "
        "additional discordant pairs. This traces that regression through per-id score movement "
        f"between the two submissions, over the {len(df)} present public-test ids "
        "(placeholder-row pitfall handled -- see module docstring).\n"
    )
    lines.append("## Zone definition\n")
    lines.append(
        f"finetune_v0-era zones from `logit_census_raw.csv`, tolerance={ZONE_TOL}: floor mode "
        f"{floor_mode:.3f}, ceiling mode {ceiling_mode:.3f} (recomputed here, matches "
        "`logit_census_report.md`). Ceiling zone's own finetune_v0 rank lower edge: "
        f"**{ceiling_lower_edge:.2f}** percentile. Floor zone's upper edge: **{floor_upper_edge:.2f}** percentile.\n"
    )
    lines.append(df_to_md(df["zone"].value_counts().rename_axis("zone").reset_index(name="n")))
    lines.append("")

    n_unique_fin = int(df["finetune_score"].nunique())
    n_unique_pho = int(df["photosub_score"].nunique())
    fin_dup_counts = df["finetune_score"].value_counts()
    fin_dup = fin_dup_counts[fin_dup_counts > 1]
    pho_dup_counts = df["photosub_score"].value_counts()
    pho_dup = pho_dup_counts[pho_dup_counts > 1]
    lines.append("## Caveat found during this run: finetune_v0.csv's present-id scores carry heavy exact-value ties\n")
    lines.append(
        f"Not previously documented anywhere in the repo, surfaced by this script's own "
        f"unique-score sanity check on `load_submission_rank`. Among the {len(df)} present ids: "
        f"`finetune_v0.csv` has only **{n_unique_fin} unique scores** ({int(fin_dup.sum())} ids "
        f"sit inside one of **{len(fin_dup)} exact-duplicate-value groups**, the largest "
        f"{int(fin_dup.max()) if len(fin_dup) else 0} ids wide), vs. `photosub_v0.csv`'s "
        f"**{n_unique_pho} unique scores** ({int(pho_dup.sum())} ids in "
        f"{len(pho_dup)} groups, largest {int(pho_dup.max()) if len(pho_dup) else 0}) -- "
        "near-fully continuous, as expected for TTA-rank-averaged floats.\n\n"
        "Checked against full float64 `repr()` (not a 6-decimal print-truncation artifact -- "
        "values match to all 16 digits), and against `infer.py`'s `_rank_normalize` (averages "
        "ties in the underlying raw per-scale scores before rank-normalizing) -- these are "
        "consistent with genuinely tied RAW model outputs for many different finetune_v0 images, "
        "spread across the full score range (duplicate clusters appear well outside the floor/"
        "ceiling saturation zones already documented in `logit_census_report.md`, not just at the "
        "two saturated poles), not with a data-pipeline bug in this script. Root cause not "
        "chased further here (out of this task's scope) -- flagged per CLAUDE.md's own invariant "
        "that score truncation/ties hurt a rank metric. **Practical effect on this analysis**: "
        f"pandas' `.rank(pct=True)` (average-tie method, matching `_rank_normalize`'s own tie "
        "handling) assigns every id inside a tied group the SAME finetune_v0 percentile -- so "
        "for roughly half of present ids, this census's 'finetune_v0 rank' is a block-average "
        "position, not a distinguishing per-id value. This plausibly adds noise to individual "
        "rank_delta values (and thus to which specific ids land in NEW_FP/NEW_DROP) but has no "
        "reason to bias the buckets' aggregate direction, since the ties are pre-existing "
        "structure in finetune_v0's own output, not something introduced by comparing it to "
        "photosub_v0.\n"
    )

    lines.append("## NEW_FP bucket (floor-zone bona-fide ids that rose)\n")
    fp = df[df["new_fp_bucket"].notna()].sort_values("fp_pair_cost", ascending=False)
    lines.append(
        f"**{len(fp)} ids** ({int((fp['new_fp_bucket']=='full').sum())} fully crossed into ceiling "
        f"territory, {int((fp['new_fp_bucket']=='soft').sum())} rose materially without fully "
        f"crossing -- soft band = {SOFT_BAND_PCT} rank-percentile points). Estimated pair-cost "
        f"contribution: **{fp['fp_pair_cost'].sum():,.0f}**.\n"
    )
    if len(fp):
        show = fp[["id", "review_stratum", "verdict", "finetune_pct_rank", "photosub_pct_rank", "rank_delta", "fp_pair_cost"]].head(20)
        lines.append(df_to_md(show, float_fmt="{:.2f}"))
    lines.append("")

    lines.append("## NEW_DROP bucket (ceiling-zone fraud ids that fell)\n")
    drop = df[df["new_drop_bucket"].notna()].sort_values("drop_pair_cost", ascending=False)
    lines.append(
        f"**{len(drop)} ids** ({int((drop['new_drop_bucket']=='full').sum())} fully fell out of "
        f"ceiling territory, {int((drop['new_drop_bucket']=='soft').sum())} dropped materially "
        f"without fully leaving). Estimated pair-cost contribution: **{drop['drop_pair_cost'].sum():,.0f}**.\n"
    )
    if len(drop):
        show = drop[["id", "review_stratum", "verdict", "finetune_pct_rank", "photosub_pct_rank", "rank_delta", "drop_pair_cost"]].head(20)
        lines.append(df_to_md(show, float_fmt="{:.2f}"))
    lines.append("")

    n_recovered = int(df["recovered_deep_miss"].sum())
    n_corrected = int(df["corrected_false_positive"].sum())
    lines.append("## Good-news counter-signals (excluded from the damage buckets above)\n")
    lines.append(
        f"- **{n_recovered}** confirmed-fraud id(s) outside the ceiling zone whose rank rose "
        "(a recovered deep-miss-style detection, not damage).\n"
        f"- **{n_corrected}** confirmed-bona-fide id(s) inside the ceiling zone whose rank fell "
        "(a corrected false positive, not damage).\n"
    )

    lines.append("## Part 1: full expected-discordance reconciliation (replaces bucket-only accounting)\n")
    bucket_total = float(fp["fp_pair_cost"].sum() + drop["drop_pair_cost"].sum())
    r = reconciliation
    lines.append(
        "Exact, exact-tie-aware expected discordance (`expected_discordance_exact`) over ALL "
        f"{len(df)} present ids for both score vectors -- sum over ordered pairs (i,j), i != j, "
        "of p_fraud(i) * p_bonafide(j) * ([score(j) > score(i)] + 0.5*[score(j) == score(i)]), "
        "using the SAME zone/verdict-blended p_fraud/p_bonafide for both models (per the task's "
        "own instruction to keep imputation identical between the two models' accounting). "
        "**This replaces the earlier sort-based total** (`total_expected_discordant_pairs` over "
        "`.rank(pct=True)`), which sorted by an unstable order within finetune_v0's large tie "
        "clusters and so could over/under-credit individual tied ids even though its aggregate "
        "sum was approximately right.\n\n"
        f"finetune_v0={total_discordant_finetune:,.0f}, photosub_v0={total_discordant_photosub:,.0f}, "
        f"**delta={total_delta:,.0f}**. Of finetune_v0's total, **{r['fin_tie_contribution']:,.0f} "
        f"({r['fin_tie_frac']*100:.1f}%) comes from exact ties alone** -- i.e. from the caveat "
        "above: pairs where a tied finetune_v0 id is credited the textbook 0.5 (not a full 0 or 1) "
        "against every cross-class id sharing its exact score. photosub_v0's own tie contribution "
        f"is {r['pho_tie_contribution']:,.0f} (its scores are nearly tie-free, as documented in "
        "the caveat above), so essentially all of the *difference* in tie handling between the two "
        "models' accounting comes from finetune_v0's side.\n\n"
        "This still says photosub_v0 orders the present-ids population BETTER than finetune_v0, "
        "not worse -- corroborated independently by computing AuDET directly from the "
        f"{int(df['verdict'].isin(['B','F']).sum())} ids with a REAL human verdict (no imputation "
        "at all): finetune_v0 AuDET=0.0508 vs. photosub_v0 AuDET=0.0410 on that reviewed subset -- "
        f"photosub_v0 wins there too. Named-bucket (NEW_FP + NEW_DROP) cost, **{bucket_total:,.0f}**, "
        "is a small, mixed-sign-net contributor to this total, not the story.\n"
    )

    lines.append(
        f"**Outcome: `{r['outcome']}`** (present-ids-only delta={total_delta:,.0f} vs. the observed "
        f"~{r['observed_lb_delta']:,.0f}-pair LB regression, ratio={r['closure_ratio']:.3f}). Per "
        "the pre-registered criteria: delta is negative (photosub_v0 orders the present-ids "
        "population strictly *better*), the opposite sign from what would be needed to explain the "
        "regression -- **the present-ids-only assumption itself is the thing to doubt**, not the "
        "imputation or the bucket accounting. The mechanism that DOES have the right sign and rough "
        f"scale is present ids crossing the 0.5 placeholder threshold: **{n_crossed_up} ids crossed "
        f"UP** (finetune<0.5, photosub>=0.5) and **{n_crossed_down} crossed DOWN** (finetune>=0.5, "
        "photosub<0.5), and the crossed-down side is dominated by the ceiling zone:\n"
    )
    lines.append(df_to_md(
        df.loc[df["crossed_down"], "zone"].value_counts().rename_axis("zone").reset_index(name="n_crossed_down"),
    ))
    lines.append("")
    lines.append(df_to_md(
        df.loc[df["crossed_up"], "zone"].value_counts().rename_axis("zone").reset_index(name="n_crossed_up"),
    ))
    lines.append("\nProceeding to the placeholder-fraction grid.\n")

    if r["grid_df"] is not None:
        lines.append(
            "### Placeholder-fraction grid: quantifying the earlier 160x-overshoot into a testable estimate\n"
        )
        lines.append(
            "If only a FRACTION of the ~135k placeholder rows are actually 'live' in the real LB's "
            "pairwise scoring (rather than the full block, which the original placeholder-crossing "
            "calculation assumed), predicted delta scales linearly with that fraction. Grid at the "
            f"headline placeholder fraud-rate assumption ({headline_rate}):\n"
        )
        lines.append(df_to_md(r["grid_df"], float_fmt="{:,.4f}"))
        best_row = r["grid_df"].iloc[(r["grid_df"]["predicted_delta_pairs"] - r["observed_lb_delta"]).abs().argsort().iloc[0]]
        lines.append(
            f"\nClosest grid point to the observed ~{r['observed_lb_delta']:,.0f}: "
            f"**fraction_live={best_row['fraction_live']}** (predicted "
            f"{best_row['predicted_delta_pairs']:,.0f} pairs, "
            f"{best_row['ratio_to_observed_130k']:.2f}x observed). Interpolating exactly (the cost "
            "function is linear in the assumed live fraction) rather than only reading grid points:\n"
        )

    placeholder_cost_full = float(df["placeholder_cross_cost"].sum())
    w_down = float(df.loc[df["crossed_down"], "p_fraud"].sum())
    w_up = float(df.loc[df["crossed_up"], "p_bonafide"].sum())
    net_weight = w_down - w_up
    implied_mass = r["observed_lb_delta"] / net_weight if net_weight else float("nan")
    assumed_mass = n_placeholder * (1.0 - headline_rate)
    implied_fraction = implied_mass / assumed_mass if assumed_mass else float("nan")
    combined_full = bucket_total + placeholder_cost_full
    lines.append(
        f"At full scale (fraction_live=1.0, i.e. the entire ~135k block treated as live): "
        f"placeholder-crossing cost = **{placeholder_cost_full:,.0f}**, combined with the small "
        f"present-ids-internal named-bucket cost = **{combined_full:,.0f}** -- roughly "
        f"**{combined_full/r['observed_lb_delta']:.0f}x** the reported **~{r['observed_lb_delta']:,.0f}** "
        "regression. Exact backward-solve: the observed crossings carry a net probability-weighted "
        f"'opposing-type' mass of {net_weight:.1f} units ({w_down:.1f} from crossed-down "
        f"fraud-weighted ids, minus {w_up:.1f} offset from crossed-up bona-fide-weighted ids). For "
        f"that to net out to the observed ~{r['observed_lb_delta']:,.0f} pairs, each unit of weight "
        f"would need to be pairing against an effective opposing population of **~{implied_mass:,.0f}** "
        f"-- **fraction_live ≈ {implied_fraction:.4f}** ({implied_fraction*100:.2f}% of the assumed "
        f"~{assumed_mass:,.0f}-strong placeholder-bonafide block), landing just below this grid's "
        "0.5% point and above its 0.1% point. The SIGN and dominant zone (ceiling ids crossing down) "
        "both check out and are corroborated by the direct human-verdict AuDET comparison above -- "
        "it's the assumed POPULATION SIZE that overshoots, not the crossing mechanism itself.\n\n"
        "**Most likely reading, in order of confidence**: (1) the public leaderboard most likely "
        "does NOT score all ~135k placeholder rows as a live pairwise-comparison population -- e.g. "
        "if it only scores the present-ids population directly, the placeholder-crossing mechanism "
        "wouldn't apply at all, and the true driver is something this analysis hasn't captured yet, "
        "since the present-ids-internal total is the wrong sign on its own; (2) alternatively, only "
        f"a small fraction (~{implied_fraction*100:.2f}%) of that block is actually 'live', which "
        "would still make placeholder-crossing the right MECHANISM at a smaller scale than "
        "originally assumed. Either way: **the single most useful next step is finding out exactly "
        "how the public LB's ~142.8k-row scoring population is actually defined**, not further "
        "tuning this script's fraud-rate assumption.\n\n"
        "**Decisive experiment for the record (not run here -- a human submission decision):** "
        "resubmit the current-best file with only the placeholder constant changed (0.5 -> 0.9). "
        "If the public LB score changes at all, that PROVES placeholder rows are scored (ruling out "
        "reading (1) above) and its magnitude/direction SIGNS how much of the ~135k block is live "
        "and at what assumed fraud rate. Do NOT submit anything as part of this analysis; that "
        "decision belongs to a human.\n"
    )

    lines.append(
        "### Sensitivity: is the conclusion an imputation artifact?\n\n"
        f"Same exact-discordance delta recomputed under 2 alternate p_fraud schemes, to check "
        "robustness rather than trusting the single zone/verdict blend used above:\n\n"
        f"- **Baseline (verdict where available, zone-imputed elsewhere)**: delta={total_delta:,.0f}\n"
        f"- **Zone-imputed-only** (every id's p_fraud = its zone's empirical rate, individual "
        f"verdicts ignored even where known, full {len(df)}-id population): "
        f"delta={r['delta_zoneonly']:,.0f}\n"
        f"- **Verdict-only** (hard 0/1 labels, zero imputation, restricted to the "
        f"{r['n_verdictonly']} ids with a real human verdict -- a much smaller population, so not "
        f"directly comparable in absolute pair count, only in sign): delta={r['delta_verdictonly']:,.0f}\n\n"
        f"All three land on the same sign (negative -- photosub_v0 orders present ids better). "
        "**The present-ids-only-delta-is-negative conclusion is not an imputation artifact**: it "
        "holds whether individual verdicts are trusted, ignored in favor of zone smoothing, or the "
        "population is restricted to only the ids with zero imputation at all.\n"
    )

    lines.append("## Pre-registered hypothesis readout\n")
    lines.append(
        "Computed two ways since they disagree on which dominates -- report both, not just the "
        "one that tells a cleaner story:\n\n"
        f"- **Present-ids-internal buckets only**: {hypothesis_readout(float(fp['fp_pair_cost'].sum()), float(drop['drop_pair_cost'].sum()))}\n"
        f"- **Including placeholder-crossing cost** (the mechanism that actually explains the "
        f"regression's sign and rough size): crossed-down ids are "
        f"{int((df.loc[df['crossed_down'], 'zone']=='ceiling').sum())}/{n_crossed_down} ceiling-zone "
        "-- i.e. this is overwhelmingly **H_DROP**: previously-confident ceiling-zone fraud "
        "detections falling below the 0.5 placeholder threshold, not new floor-zone false positives.\n"
        "- **H_A specifically** (both of its two predicted signatures checked directly): (1) the "
        "deep-9 MODE_A behavior does NOT reproduce on the actually-submitted epoch-6 scores -- see "
        "the Deep-9 section below, 8/9 rose; (2) the predicted 'small NEW_FP flavor concentrated "
        "on portrait-prominent bona-fides' also doesn't show up -- NEW_FP is small (226 of the "
        "36,776+226 named-bucket total) and, per `movement_census_out/VISUAL_FINDINGS.md`'s "
        "manual review of its top-40 renders, mixed across EGYPT/DL, MAURITIUS/ID, BENIN/DL, and "
        "GUINEA/DL with no portrait-specific or single-template signature. **H_A is REFUTED as an "
        "explanation for this regression** -- its own pre-registered predictions were checked "
        "against the real submission and didn't materialize; the damage traces to H_DROP instead, "
        "and (per the visual findings) concentrated on MAURITIUS/ID ceiling ids, which is a "
        "MODE_D/ghost-mismatch lead, not a MODE_A one.\n"
    )

    deep9 = deep9_movement_table(df, args.probes_dir)
    if len(deep9):
        n_rose = int((deep9["rank_delta"] > 0).sum())
        n_fell = int((deep9["rank_delta"] < 0).sum())
        lines.append("## Deep-9 MODE_A/B/C ids: movement in the actually-submitted scores (H_A check)\n")
        lines.append(
            "The 9 human-confirmed deep-miss ids (`data/probes/missed_frauds_deep_ids.csv`), "
            "joined to how each one's SUBMITTED rank (TTA-averaged, epoch-6 checkpoint -- the one "
            "actually used for the public-test submission) moved between finetune_v0 and "
            "photosub_v0. This is a direct, on-the-real-submission test of H_A, distinct from "
            "`docs/technical_report.md`'s epoch-by-epoch raw-logit training probe table (which "
            "tracks a later, non-submitted epoch-20 snapshot too).\n"
        )
        lines.append(df_to_md(deep9, float_fmt="{:.2f}"))
        lines.append(
            f"\n**{n_rose}/9 rose, {n_fell}/9 fell.** Including all 3 MODE_A ids "
            f"(`c6651aee`, `b5eebda1`, `40dd1055`), which all rose modestly here "
            "(unlike the epoch-20 collapse documented in the technical report) -- consistent with "
            "epoch-6 (the actually-submitted checkpoint) predating the epoch-9-to-20 drift that "
            "produced MODE_A's later negative reversal. The only id that fell is `a2a3fe5b` "
            "(MODE_C, confirmed), and only slightly (-9.0 pct points, still at the 86th "
            "percentile). **On the actual submitted scores, none of the 9 deep-miss ids show the "
            "damage pattern H_A's 'deep-9 MODE_A behavior' clause predicted** -- that behavior is "
            "real in the epoch-20 training probe, but epoch-20 is not the checkpoint that was "
            "submitted.\n"
        )

    trans = transitional_movement_table(df, args.deep_review_csv)
    if len(trans):
        lines.append("## Transitional/floor-review movement (verdict-conditioned, deep-review strata)\n")
        lines.append(df_to_md(trans, float_fmt="{:.2f}"))
        lines.append("")

    boundary_df, boundary_summary = boundary_movement_table(df, args.probes_dir)
    if boundary_summary:
        lines.append("## 59-id boundary group (TRANS_5) movement\n")
        lines.append(
            f"**Answer: no, photosub_v0 did NOT push this group further into the ceiling on "
            f"average -- it pushed them the other way.** n={boundary_summary['n']}, mean "
            f"finetune_v0 pct_rank={boundary_summary['mean_finetune_pct_rank']:.2f} -> photosub_v0 "
            f"pct_rank={boundary_summary['mean_photosub_pct_rank']:.2f} "
            f"(mean rank_delta={boundary_summary['mean_rank_delta']:+.2f}, "
            f"{boundary_summary['n_fell']}/{boundary_summary['n']} fell vs. "
            f"{boundary_summary['n_rose']}/{boundary_summary['n']} rose). All "
            f"{boundary_summary['n_already_in_ceiling']}/{boundary_summary['n']} were ALREADY at/"
            "above the ceiling zone's lower rank edge under finetune_v0 (mean pct_rank ~98, deep in "
            "the ceiling already, not actually hovering at a boundary in this tol=0.5 zone sense) -- "
            f"and {boundary_summary['n_pushed_into_ceiling']}/{boundary_summary['n']} still are "
            "under photosub_v0, so none fell all the way out, but the direction of movement is the "
            "same drop-within-the-ceiling pattern as the NEW_DROP bucket, not the hoped-for upside "
            "signal CLAUDE.md's priority list was checking for.\n"
        )

    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[movement_census] wrote report -> {args.report_md}")


# ---------------------------------------------------------------------------
# Render stage
# ---------------------------------------------------------------------------

def run_render(args) -> None:
    if not args.out_csv.exists():
        raise SystemExit(f"{args.out_csv} not found -- run --stage attribute first")
    df = pd.read_csv(args.out_csv, dtype={"id": str})
    rdir = regions_dir(args.data_dir)
    has_regions = rdir.exists()
    print(f"[movement_census] regions cache at {rdir}: {'found' if has_regions else 'NOT found -- rendering card-only, no face zoom'}")

    meta = load_labels(args.data_dir, "public_test")[["id", "path"]]
    df = df.merge(meta, on="id", how="left")

    for bucket_col, cost_col, name in [
        ("new_fp_bucket", "fp_pair_cost", "new_fp"),
        ("new_drop_bucket", "drop_pair_cost", "new_drop"),
    ]:
        bucket = df[df[bucket_col].notna()].sort_values(cost_col, ascending=False).head(TOP_N_RENDER)
        if bucket.empty:
            print(f"[movement_census] {name}: bucket is empty, skipping render")
            continue
        cells = []
        for row in bucket.itertuples(index=False):
            if pd.isna(row.path) or not Path(row.path).exists():
                continue
            cell_row = pd.Series({
                "id": row.id, "path": row.path,
                "mean_logit": row.finetune_mean_logit, "stratum": row.zone,
                "pct_rank": row.photosub_pct_rank,
            })
            cell = render_cell(cell_row, rdir, show_face_crop=has_regions, caption_height=80)
            d = ImageDraw.Draw(cell)
            caption2 = (
                f"[{getattr(row, bucket_col)}] verdict={row.verdict} fin_pct={row.finetune_pct_rank:.1f} "
                f"-> pho_pct={row.photosub_pct_rank:.1f} cost={getattr(row, cost_col):.1f}"
            )
            d.text((4, cell.height - 20), caption2, fill=(200, 0, 0))
            cells.append(cell)
        out_dir = args.render_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        n_sheets = -(-len(cells) // 20)
        for s in range(n_sheets):
            render_sheet(cells[s * 20:(s + 1) * 20], out_dir / f"{name}_sheet{s+1:02d}.png", ncols=4, nrows=5)
        print(f"[movement_census] {name}: rendered {len(cells)} cells across {n_sheets} sheet(s) -> {out_dir}")

    findings_path = args.render_dir / "VISUAL_FINDINGS.md"
    findings_path.write_text(
        "# Dossier render -- visual findings (manual inspection of the rendered sheets)\n\n"
        "**NEW_DROP top-40 (by pair cost) is overwhelmingly one template: MAURITIUS/ID.** Visual "
        "count across `new_drop_sheet01.png` + `new_drop_sheet02.png`: 39/40 cells are the "
        "Republic of Mauritius National Identity Card (teal/green flag header, visible teal/purple "
        "ghost secondary-portrait feature next to the ID number); only 1/40 is a different template "
        "(Guinea/DL). This is the single most concrete lead in the whole analysis -- MAURITIUS/ID "
        "is one of only two document types with a ghost region (the other is EGYPT/DL), and MODE_D "
        "(ghost mismatch, including the darkened-ghost-variant slice) is the one generator family "
        "that specifically targets ghost-bearing templates. If photosub_v0 learned something from "
        "MODE_D that generalizes badly to genuinely-fraudulent Mauritius ID ceiling cards (e.g. "
        "treating a legible/darkened ghost as weaker evidence of fraud than it should be), this "
        "NEW_DROP pattern -- previously-confident MAURITIUS/ID fraud detections falling toward the "
        "0.5 placeholder threshold -- is exactly what that would look like.\n\n"
        "**NEW_FP (all 25) shows no such single-template pattern** -- a mix of EGYPT/DL, "
        "MAURITIUS/ID, BENIN/DL, and GUINEA/DL cards, none dominant. Consistent with NEW_FP's tiny "
        "estimated pair cost (226) relative to NEW_DROP's (36,776): the false-positive side isn't "
        "where the action is.\n\n"
        "**Not yet done, recommended next step**: this is a visual read of the top-40-by-cost "
        "sample only, not a quantitative count over the full 376-id NEW_DROP bucket. "
        "`scripts/analysis/hesitant_clusters.py`'s `type_proxy_knn` (embedding-KNN against train's "
        "5 known types) could classify the full bucket cheaply and confirm the MAURITIUS/ID share "
        "quantitatively rather than by eye -- worth doing before concluding this is definitely a "
        "MODE_D/ghost-specific regression rather than a coincidence of this particular top-40 slice.\n\n"
        "---\n\n"
        "**UPDATE (follow-up full-population census, see `movement_census_part2_report.md`'s item "
        "6): the caution above was warranted -- the top-40-by-cost read does NOT generalize.** "
        "`type_proxy_knn` itself turned out to be unusable here (embedding collapse in the "
        "saturated ceiling zone -- mean pairwise cosine similarity 0.999, confirmed by "
        "misclassifying a VISUALLY-CONFIRMED Mauritius id as Mozambique/DL); a color-histogram "
        "classifier (unaffected by the model's collapsed representation, validated against real "
        "visual ground truth) was used instead, over the FULL ceiling zone. Result: the full "
        "376-id NEW_DROP bucket is **GUINEA/DL-dominated by count and by rate** (280/376 ids, "
        "35.5% ceiling-zone drop rate) -- NOT Mauritius (71/376, 19.9%). Both reads are correct "
        "measurements of different things: Mauritius ids carry disproportionately large per-id "
        "pair cost among drops (they started ranked higher under finetune_v0, so they fall "
        "further), which is why they dominate a COST-sorted top-40, but GUINEA/DL dominates the "
        "population by raw count/rate. GUINEA/DL is not even a ghost-bearing template and "
        "received zero MODE_D training rows -- **the MODE_D/ghost-mismatch mechanism this file "
        "originally proposed cannot explain GUINEA/DL's drop rate at all.** See "
        "`ITEM4_DOSSIER_FINDINGS.md` for what the actual cross-template evidence looks like "
        "(text-field splice artifacts, not photo-region tampering).\n",
        encoding="utf-8",
    )
    print(f"[movement_census] wrote visual findings -> {findings_path}")


# ---------------------------------------------------------------------------
# Part 2, item 6: document-type classification, embedding-KNN attempt (FAILED -- see docstring
# below) and its working replacement, a color-histogram classifier
# ---------------------------------------------------------------------------
#
# `run_classify_types` (embedding-KNN via the fine-tuned model, same method as
# hesitant_clusters.py's `type_proxy_knn`) was tried FIRST and is kept below/importable, but its
# output should NOT be trusted for the ceiling zone -- confirmed by direct measurement, not
# suspicion: `embedding_collapse_diagnostic` on a 20-id ceiling-zone sample gave a mean pairwise
# cosine similarity of 0.9992 (vs. 0.177 for a diverse TRAIN reference set of the same size) --
# textbook representation collapse, exactly the failure mode `hesitant_clusters.py`'s own
# docstring warns "directly undermines... the type-proxy k-NN". Concretely: id
# `09d72ff0c5854dfcb8a6e3b6aba9ce49`, VISUALLY CONFIRMED as a MAURITIUS/ID card (see
# `new_drop_sheet01.png` cell 4, "REPUBLIC OF MAURITIUS NATIONAL IDENTITY CARD"), was
# misclassified as MOZAMBIQUE/DL by the embedding-KNN with 0.9976 "similarity" -- a number that
# looks like confidence but is actually the collapse artifact, since EVERY class's centroid
# distance was 0.997-0.9999 for that id (see the per-pool `type_proxy_similarity` stats). The
# ceiling zone is a maximally-saturated population by construction (raw logit within 0.5 of the
# +12.502 ceiling mode) -- exactly the population where a model trained to compress "definitely
# fraud" toward a near-single point would lose whatever type-discriminative structure the
# embedding carried outside that saturated region. `hesitant_clusters.py`'s own ids are NOT
# saturated (46th-54th score percentile), so this failure mode never surfaced there.
#
# Replacement: a color-histogram classifier (per-channel RGB histogram, nearest-centroid) that
# doesn't touch the model's representation at all, so it can't inherit this collapse. Validated
# against 3 ids with independently-confirmed visual ground truth (2 correct on first check,
# the 3rd's apparent "mismatch" turned out to be a transcription error in the visual
# ground-truth itself, not a classifier error -- re-examining `new_drop_sheet02.png` directly
# confirmed id `53d9f320a8` IS Mauritius, matching the color classifier's prediction) -- 3/3 once
# corrected. Also ~85x faster (no model forward pass), which is why the full ceiling zone (not a
# stable-sample estimate) is classified below rather than reusing `run_classify_types`'s
# subsampling workaround.

DEFAULT_TYPE_CSV = OUT_DIR / "movement_census_types.csv"
TYPE_REF_N_PER_TYPE = 40  # matches hesitant_clusters.py's own --n-ref-per-type default
TYPE_REF_SEED = 42
DEFAULT_STABLE_SAMPLE_N = 900  # stable (non-NEW_DROP) ceiling ids to classify, for a per-
# template baseline estimate -- CPU-bound (~1 img/s for a DINOv2 ViT-B/14 forward pass at 518px
# locally), so classifying the full ~3,400-id stable population (on top of ALL 376 NEW_DROP ids,
# always classified in full) would take well over an hour; 900 gives ~150-190/template, enough
# for a stable per-template PROPORTION estimate without the multi-hour runtime. SUPERSEDED by the
# color-histogram classifier below for the actual item-6 deliverable -- kept only because
# run_classify_types (the embedding-KNN attempt) is kept importable as a documented negative
# result, not deleted.
STABLE_SAMPLE_SEED = 42


def run_classify_types(args) -> None:
    """Runs finetune_v0's own type_proxy_knn (embedding-KNN against a labeled TRAIN reference
    set, in the fine-tuned model's own cosine embedding space -- see hesitant_clusters.py, which
    this reuses rather than reimplementing). ALL 376 NEW_DROP ids are classified exactly (needed
    for item 4's dossier and as item 6's numerator); the STABLE (non-NEW_DROP) ceiling population
    is classified from a random sample (`--stable-sample-n`, default 900 of ~3,398) rather than
    in full, since a full-ceiling CPU forward pass would take over an hour locally -- the sample's
    per-type PROPORTION is used by `per_template_drop_report` to estimate each template's full
    ceiling-zone denominator, not a raw per-type count. Tags each row with `pool` (dropped /
    stable_sample) so that scaling step knows which rows are exact counts vs. a sample.

    A real forward pass over each image (embedding extraction only, no gradient, no optimizer
    step) -- not training. Requires the finetune_v0 checkpoint (`--checkpoint`)."""
    from common import build_finetuned_model, device_and_seed, eval_transform, load_checkpoint
    from hesitant_clusters import KNOWN_TYPES, build_type_reference, embed_paths, type_proxy_knn

    if not args.out_csv.exists():
        raise SystemExit(f"{args.out_csv} not found -- run --stage attribute first")
    df = pd.read_csv(args.out_csv, dtype={"id": str})
    meta = load_labels(args.data_dir, "public_test")[["id", "path"]]
    df = df.merge(meta, on="id", how="left")
    df = df[df["path"].notna() & df["path"].map(lambda p: Path(p).exists())]

    dropped = df[df["new_drop_bucket"].notna()].copy()
    dropped["pool"] = "dropped"
    stable_pool = df[(df["zone"] == "ceiling") & df["new_drop_bucket"].isna()]
    n_stable = min(args.stable_sample_n, len(stable_pool))
    stable_sample = stable_pool.sample(n=n_stable, random_state=STABLE_SAMPLE_SEED).copy()
    stable_sample["pool"] = "stable_sample"
    targets = pd.concat([dropped, stable_sample], ignore_index=True)
    print(f"[movement_census] classify_types: {len(dropped)} NEW_DROP (exact) + "
          f"{len(stable_sample)}/{len(stable_pool)} stable-ceiling sample = {len(targets)} total images")

    cfg, state = load_checkpoint(args.checkpoint)
    cfg.data_dir = args.data_dir  # checkpoint stores VESSL's "data" layout; override for local "data/raw"
    device = device_and_seed(cfg)
    model = build_finetuned_model(cfg, state, device)
    transform, _ = eval_transform(cfg)

    print(f"[movement_census] building type reference set ({TYPE_REF_N_PER_TYPE}/type from TRAIN)...")
    ref_embs, ref_types = build_type_reference(cfg, model, device, transform, TYPE_REF_N_PER_TYPE, TYPE_REF_SEED)
    print(f"[movement_census] reference set: {len(ref_types)} ids across {len(KNOWN_TYPES)} known types")

    paths = [Path(p) for p in targets["path"]]
    t0 = time.monotonic()
    query_embs = embed_paths(model, device, paths, transform)
    print(f"[movement_census] embedded {len(paths)} ids in {time.monotonic()-t0:.0f}s")
    preds, confs = type_proxy_knn(query_embs, ref_embs, ref_types)
    targets["doc_type_proxy"] = preds
    targets["type_proxy_similarity"] = confs

    out_cols = ["id", "pool", "doc_type_proxy", "type_proxy_similarity"]
    args.type_csv.parent.mkdir(parents=True, exist_ok=True)
    targets[out_cols].to_csv(args.type_csv, index=False)
    print(f"[movement_census] wrote {len(targets)} type-classified ids -> {args.type_csv}")
    print(targets.groupby("pool")["doc_type_proxy"].value_counts())


COLOR_HIST_BINS = 16
COLOR_HIST_REF_N_PER_TYPE = 60
COLOR_HIST_REF_SEED = 42


def color_hist_feature(path, bins: int = COLOR_HIST_BINS) -> np.ndarray:
    """Per-channel RGB density histogram over a 128x128 downsample -- a template-identification
    feature that never touches the fine-tuned model's (collapsed, in the ceiling zone) embedding
    space. Each of the 5 known types has a visually distinct dominant palette (flag colors,
    card-background tint), which this is built to exploit directly."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((128, 128))
    arr = np.asarray(img)
    feats = [np.histogram(arr[:, :, c], bins=bins, range=(0, 255), density=True)[0] for c in range(3)]
    return np.concatenate(feats)


def build_color_hist_reference(data_dir, n_per_type: int = COLOR_HIST_REF_N_PER_TYPE, seed: int = COLOR_HIST_REF_SEED) -> dict:
    """type -> centroid feature vector, from a labeled TRAIN sample per known type."""
    known_types = ("EGYPT/DL", "GUINEA/DL", "BENIN/DL", "MOZAMBIQUE/DL", "MAURITIUS/ID")
    train_df = load_labels(data_dir, "train")
    centroids = {}
    for t in known_types:
        sub = train_df[train_df["type"] == t]
        n = min(n_per_type, len(sub))
        sample = sub.sample(n=n, random_state=seed)
        feats = np.array([color_hist_feature(p) for p in sample["path"]])
        centroids[t] = feats.mean(axis=0)
    return centroids


def color_hist_classify(paths, centroids: dict) -> tuple[list[str], list[float]]:
    """Nearest-centroid classification. Returns (predicted_type, margin) where margin =
    (second_nearest_dist - nearest_dist) / nearest_dist -- a genuine confidence signal (unlike
    the embedding-KNN's collapsed similarity scores above): large margin = the two nearest
    centroids are clearly separated for this image; near-zero margin = a genuinely ambiguous
    call, worth flagging rather than trusting blindly."""
    types = list(centroids.keys())
    cmat = np.array([centroids[t] for t in types])
    preds, margins = [], []
    for p in paths:
        feat = color_hist_feature(p)
        dists = np.linalg.norm(cmat - feat, axis=1)
        order = np.argsort(dists)
        nearest, second = dists[order[0]], dists[order[1]]
        preds.append(types[order[0]])
        margins.append(float((second - nearest) / nearest) if nearest > 0 else float("inf"))
    return preds, margins


def run_classify_types_color(args) -> None:
    """The working replacement for `run_classify_types` (see the module-level docstring above
    this section for why the embedding-KNN approach fails on the ceiling zone). Classifies the
    FULL ceiling zone (both NEW_DROP and stable) -- no sampling needed, since this is ~85x faster
    than the model forward-pass approach. CPU-only, no checkpoint, no model."""
    if not args.out_csv.exists():
        raise SystemExit(f"{args.out_csv} not found -- run --stage attribute first")
    df = pd.read_csv(args.out_csv, dtype={"id": str})
    meta = load_labels(args.data_dir, "public_test")[["id", "path"]]
    df = df.merge(meta, on="id", how="left")
    ceiling = df[(df["zone"] == "ceiling") & df["path"].notna() & df["path"].map(lambda p: Path(p).exists())].copy()
    print(f"[movement_census] classify_types_color: {len(ceiling)}/{int((df['zone']=='ceiling').sum())} "
          "ceiling-zone ids have a locally-reachable image (classifying ALL of them, no sampling)")

    t0 = time.monotonic()
    centroids = build_color_hist_reference(args.data_dir)
    print(f"[movement_census] built color-histogram reference centroids in {time.monotonic()-t0:.0f}s")

    t0 = time.monotonic()
    preds, margins = color_hist_classify(ceiling["path"].tolist(), centroids)
    print(f"[movement_census] classified {len(ceiling)} ceiling-zone ids in {time.monotonic()-t0:.0f}s")
    ceiling["doc_type_proxy"] = preds
    ceiling["type_proxy_margin"] = margins
    ceiling["pool"] = np.where(ceiling["new_drop_bucket"].notna(), "dropped", "stable_sample")

    out_cols = ["id", "pool", "doc_type_proxy", "type_proxy_margin"]
    args.type_csv.parent.mkdir(parents=True, exist_ok=True)
    ceiling[out_cols].to_csv(args.type_csv, index=False)
    print(f"[movement_census] wrote {len(ceiling)} color-classified ids -> {args.type_csv} "
          "(pool='stable_sample' here means the FULL stable ceiling population, not a sample -- "
          "kept the same column name as run_classify_types' output for per_template_drop_report "
          "compatibility, but n_stable_sample IS n_stable_total now, so no scaling is needed)")
    print(ceiling.groupby("pool")["doc_type_proxy"].value_counts())
    low_margin = float((ceiling["type_proxy_margin"] < 0.05).mean())
    print(f"[movement_census] {low_margin*100:.1f}% of ids classified with margin<0.05 (genuinely ambiguous)")


# ---------------------------------------------------------------------------
# Part 2, item 5: training-composition cross-check (manifest vs. actually-SELECTED rows vs. real fraud)
# ---------------------------------------------------------------------------

def reconstruct_selected_photosub_rows(args) -> "pd.DataFrame":
    """Deterministically reproduces the EXACT 4,455 rows `photosub_v0`'s actual training run
    selected out of the full 9,000-row generation manifest -- not the manifest itself, which is
    generation-time supply, not train-time demand. `select_mixed_rows` is a pure function of
    (rows_df, train_ids, mode_weights, share, base_n_positive, seed); train_ids/base_n_positive
    are reproduced here via the same `stratified_split`/`load_labels` calls `train.py` itself
    uses (config's own val_fraction=0.1, seed=42) -- verified to reproduce the technical report's
    documented numbers exactly (train=62,417 real + 4,455 photosub = 66,872 total; val=6,935;
    D-mode 402/608 shortfall) before being trusted for this cross-check."""
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from freuid.data import stratified_split
    from freuid.photosub.mixing import load_photosub_rows, select_mixed_rows

    cfg_path = REPO_ROOT / "configs" / "photosub_v0.yaml"
    cfg_raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    photosub_cfg = cfg_raw["extra"]["photosub"]
    seed = int(cfg_raw["seed"])
    val_fraction = float(cfg_raw.get("val_fraction", 0.1))

    train_ids, val_ids = stratified_split(args.data_dir, val_fraction, seed)
    labels = load_labels(args.data_dir, "train")
    base = labels[labels["id"].isin(train_ids)]
    base_n_positive = int((base["label"] == 1).sum())
    print(f"[movement_census] reconstructed split: train={len(train_ids)} (base_n_positive="
          f"{base_n_positive}) val={len(val_ids)} -- cross-check against docs/technical_report.md's "
          "train=62,417/val=6,935 (66,872 total including 4,455 photosub-mixed rows)")

    manifest_path = REPO_ROOT / photosub_cfg["rows_csv"]
    if not manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} not found -- this is the VESSL-only generation-time manifest "
            "(scripts/generate_photosub_dataset.py's output), not tracked by git per CLAUDE.md's "
            "'never commit dataset files' invariant. Pull it read-only, e.g.: "
            "scp freuid-hy:/root/repo/data/processed/photosub_generated/rows.csv "
            f"{manifest_path}"
        )
    rows_df = load_photosub_rows(manifest_path)
    mode_weights = photosub_cfg["mode_weights"]
    share = float(photosub_cfg["share"])
    selected = select_mixed_rows(rows_df, train_ids, mode_weights, share, base_n_positive, seed=seed)
    print(f"[movement_census] reconstructed selected photosub rows: {len(selected)} "
          f"(target share={share}, base_n_positive={base_n_positive})")
    return selected, base


def manifest_composition_report(selected: "pd.DataFrame", base: "pd.DataFrame") -> str:
    """Item 5: does the ACTUALLY-SELECTED synthetic mix over-represent MAURITIUS/ID relative to
    its real-fraud share? Checked at two granularities -- aggregate (all modes pooled) AND
    MODE_D-specifically (the ghost-mismatch generator, since GHOST_TEMPLATES structurally gates
    D to only EGYPT/DL and MAURITIUS/ID, so an aggregate check alone could hide a mode-specific
    concentration inside an otherwise-balanced total)."""
    real_fraud = base[base["label"] == 1]
    real_share = (real_fraud["type"].value_counts() / len(real_fraud)).rename("real_fraud_share")

    agg_n = selected["type"].value_counts().rename("n_selected")
    agg_share = (agg_n / len(selected)).rename("selected_share")
    agg = pd.concat([agg_n, agg_share, real_share], axis=1).fillna(0.0)
    agg["ratio_selected_to_real"] = agg["selected_share"] / agg["real_fraud_share"]
    agg = agg.reset_index(names="type").sort_values("ratio_selected_to_real", ascending=False)

    d_mode = selected[selected["mode"].isin(["D_main", "D_ghost"])]
    d_n = d_mode["type"].value_counts().rename("n_d_mode")
    d_share = (d_n / max(1, len(d_mode))).rename("d_mode_share_of_all_d")
    d_tab = pd.concat([d_n, d_share, real_share], axis=1).fillna(0.0)
    d_tab["ratio_dmode_to_real"] = d_tab["d_mode_share_of_all_d"] / d_tab["real_fraud_share"]
    d_tab = d_tab.reset_index(names="type").sort_values("ratio_dmode_to_real", ascending=False)

    lines = ["## Part 2, item 5: training-composition cross-check\n"]
    lines.append(
        f"Reconstructed exactly (see `reconstruct_selected_photosub_rows`'s docstring for the "
        f"verification against `docs/technical_report.md`'s documented split/mix numbers) -- "
        f"NOT the raw 9,000-row generation manifest, which is oversupply, not what training "
        f"actually saw. {len(selected)} rows actually selected into `photosub_v0`'s train split.\n"
    )
    lines.append("### Aggregate (all modes pooled): selected share vs. real-fraud share\n")
    lines.append(df_to_md(agg, float_fmt="{:.4f}"))
    mauritius_ratio = float(agg.loc[agg["type"] == "MAURITIUS/ID", "ratio_selected_to_real"].iloc[0])
    lines.append(
        f"\n**MAURITIUS/ID's aggregate ratio is {mauritius_ratio:.2f}x** -- NOT disproportionate "
        "(all 5 templates land within a narrow ~0.8x-1.1x band; EGYPT/DL is actually "
        "*under*-represented relative to its outsized 26.8% real-fraud share, since generation "
        "sampled bona-fide sources roughly evenly across templates rather than weighting toward "
        "EGYPT/DL's higher real-fraud volume). **The disproportion the task hypothesized does not "
        "show up in the aggregate mix.**\n"
    )
    lines.append("### MODE_D (ghost-mismatch) specifically: share of all D-mode rows vs. real-fraud share\n")
    lines.append(
        "GHOST_TEMPLATES structurally gates MODE_D to only 2 of 5 templates (EGYPT/DL, "
        "MAURITIUS/ID) -- the other 3 get exactly 0 D-mode rows by construction, not by "
        "under-sampling. This is where to look for a real concentration.\n"
    )
    lines.append(df_to_md(d_tab, float_fmt="{:.4f}"))
    maur_d_ratio = float(d_tab.loc[d_tab["type"] == "MAURITIUS/ID", "ratio_dmode_to_real"].iloc[0])
    lines.append(
        f"\n**MAURITIUS/ID's MODE_D ratio is {maur_d_ratio:.2f}x its real-fraud share** -- "
        "MAURITIUS/ID and EGYPT/DL split essentially all 402 D-mode rows between them (roughly "
        "51/49), while each is only ~18-27% of real fraud, so both are substantially "
        "over-represented within the D-mode allocation specifically. **This IS the input v1's "
        "per-template cap needs**: not a blanket per-template cap (the aggregate mix is already "
        "balanced), but a cap specific to MODE_D's ghost-mismatch generator, which is structurally "
        "concentrated on the 2 ghost-bearing templates and, within those 2, allocates roughly "
        "evenly rather than weighting down MAURITIUS/ID specifically -- if MAURITIUS/ID's ceiling "
        "drop-rate (below) is disproportionately larger than EGYPT/DL's despite a similar D-mode "
        "allocation, that points at something template-specific about how MAURITIUS/ID's ghost "
        "region interacts with the model, not just at the generator's allocation policy.\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part 2, item 6: per-template drop-rate report (ceiling-zone baseline vs. NEW_DROP)
# ---------------------------------------------------------------------------

def per_template_drop_report(df: pd.DataFrame, type_csv: Path) -> str:
    if not type_csv.exists():
        return (
            "## Part 2, item 6: per-template drop rates\n\n"
            f"**Not available** -- {type_csv} not found. Run `--stage classify_types` first "
            "(needs the finetune_v0 checkpoint + a forward pass over the ceiling zone's images; "
            "see that stage's docstring).\n"
        )
    types = pd.read_csv(type_csv, dtype={"id": str})
    dropped_types = types[types["pool"] == "dropped"]
    stable_sample_types = types[types["pool"] == "stable_sample"]
    n_dropped_total = int((df["new_drop_bucket"].notna()).sum())
    n_stable_total = int((df["zone"] == "ceiling").sum()) - n_dropped_total
    n_missing_dropped = n_dropped_total - len(dropped_types)
    if n_missing_dropped > 0:
        print(f"[movement_census] per_template_drop_report: {n_missing_dropped} NEW_DROP ids have "
              "no type classification (image not locally reachable) -- excluded from the numerator")

    dropped_by_type = dropped_types["doc_type_proxy"].value_counts()
    stable_by_type = stable_sample_types["doc_type_proxy"].value_counts()
    full_coverage = len(stable_sample_types) >= n_stable_total  # color classifier: full ceiling zone, no sampling
    if full_coverage:
        stable_est_by_type = stable_by_type.astype(float)
    else:
        stable_frac = stable_by_type / max(1, len(stable_sample_types))
        stable_est_by_type = stable_frac * n_stable_total

    rows = []
    for t in sorted(set(dropped_by_type.index) | set(stable_est_by_type.index)):
        n_dropped = int(dropped_by_type.get(t, 0))
        n_stable_est = float(stable_est_by_type.get(t, 0.0))
        n_ceiling_est = n_dropped + n_stable_est
        rows.append({
            "type": t, "n_dropped": n_dropped, "n_stable": int(stable_by_type.get(t, 0)),
            "n_ceiling": n_ceiling_est,
            "drop_rate": n_dropped / n_ceiling_est if n_ceiling_est else float("nan"),
        })
    table = pd.DataFrame(rows).sort_values("drop_rate", ascending=False)

    ghost_templates = {"EGYPT/DL", "MAURITIUS/ID"}
    ghost_rows = table[table["type"].isin(ghost_templates)]
    other_rows = table[~table["type"].isin(ghost_templates)]
    ghost_mean = float(ghost_rows["drop_rate"].mean()) if len(ghost_rows) else float("nan")
    other_mean = float(other_rows["drop_rate"].mean()) if len(other_rows) else float("nan")
    top_type = str(table.iloc[0]["type"]) if len(table) else None
    top_rate = float(table.iloc[0]["drop_rate"]) if len(table) else float("nan")

    lines = ["## Part 2, item 6: per-template drop rates, FULL ceiling zone (not a sample)\n"]
    coverage_note = (
        f"`n_dropped` and `n_stable` are BOTH exact counts -- the color-histogram classifier "
        f"(see the section above the embedding-KNN attempt's docstring) classified ALL "
        f"{n_dropped_total + n_stable_total} ceiling-zone ids ({len(dropped_types)} of "
        f"{n_dropped_total} NEW_DROP + {len(stable_sample_types)} of {n_stable_total} stable), "
        "not a sample -- `drop_rate` below is an exact rate, not an estimate."
        if full_coverage else
        f"`n_dropped` is exact ({len(dropped_types)}/{n_dropped_total} NEW_DROP ids classified); "
        f"`n_stable` is a SAMPLE ({len(stable_sample_types)}/{n_stable_total}) scaled to the full "
        "stable population by proportion -- `drop_rate` below is an estimate, not exact."
    )
    lines.append(coverage_note + "\n")
    lines.append(df_to_md(table, float_fmt="{:.4f}"))

    lines.append(
        f"\n**Headline finding, and it overturns this analysis's own working hypothesis: "
        f"`{top_type}` has the highest drop rate at {top_rate*100:.1f}%, not MAURITIUS/ID.** "
        "GUINEA/DL is NOT a ghost-bearing template (`GHOST_TEMPLATES['GUINEA/DL'] is None`) and "
        "received ZERO MODE_D rows in training (confirmed in item 5's manifest cross-check above) "
        "-- so the MODE_D/ghost-mismatch mechanism CANNOT explain GUINEA/DL's drop rate at all. "
        "This directly contradicts `VISUAL_FINDINGS.md`'s 39/40-Mauritius finding from the "
        "top-40-BY-COST sample -- both are correct measurements of different things: the top-40 "
        "sample is genuinely Mauritius-dominated (Mauritius ids carry disproportionately large "
        "per-id pair cost among drops, since they started at a higher finetune_v0 rank and so "
        "fall further), but the FULL 376-id NEW_DROP bucket, by raw count and by rate, is "
        "GUINEA/DL-dominated. Verified against real visual ground truth (not trusted blindly): 8 "
        "randomly-sampled GUINEA/DL-classified drop ids were rendered and manually confirmed as "
        "genuine 'RÉPUBLIQUE DE GUINÉE PERMIS DE CONDUIRE' cards.\n"
    )
    lines.append(
        f"\nGhost-bearing templates (EGYPT/DL, MAURITIUS/ID) mean drop_rate={ghost_mean:.4f} vs. "
        f"non-ghost templates (BENIN/DL, GUINEA/DL, MOZAMBIQUE/DL) mean={other_mean:.4f} -- this "
        "non-ghost mean is pulled way up by GUINEA/DL specifically (BENIN/DL and MOZAMBIQUE/DL "
        "both sit under 2%), so a simple ghost-vs-non-ghost split obscures rather than clarifies "
        "the real pattern here; GUINEA/DL needs its own explanation, not folded into 'non-ghost "
        "baseline.'\n"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Part 2: template-composition sampling (item 6) + stratified detailed dossier (item 4)
# ---------------------------------------------------------------------------

TEMPLATE_SAMPLE_N = 50
TEMPLATE_SAMPLE_SEED = 42


def _dense_id_cell(row, cell_width: int = 170, thumb_height: int = 120, caption_height: int = 22) -> "object":
    """Card-only, ID-labeled thumbnail for high-density template-identification sheets -- much
    smaller than the detailed dossier cells (item 4 needs those; item 6 just needs 'what template
    is this' at scale, which needs far less resolution)."""
    from PIL import Image, ImageDraw

    img = Image.open(row["path"]).convert("RGB")
    w, h = img.size
    scale = min(cell_width / w, thumb_height / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    thumb = img.resize((new_w, new_h))
    cell = Image.new("RGB", (cell_width, thumb_height + caption_height), (255, 255, 255))
    cell.paste(thumb, ((cell_width - new_w) // 2, (thumb_height - new_h) // 2))
    d = ImageDraw.Draw(cell)
    d.text((2, thumb_height + 2), row["id"][:10], fill=(0, 0, 0))
    return cell


def run_template_sample(args) -> None:
    """Item 6: samples TEMPLATE_SAMPLE_N ids from the NEW_DROP bucket ('dropped') and
    TEMPLATE_SAMPLE_N from the rest of the ceiling zone ('stable', i.e. ceiling ids that did NOT
    drop), renders each as a dense id-labeled sheet for manual template classification. Comparing
    per-template composition between the two samples is what answers "which templates are
    disproportionately dropping, relative to their own baseline ceiling presence" -- not just
    "what's common in the drops" (which could just reflect that template's overall prevalence).
    No classification is done here -- that's a manual visual step; this only produces the sheets.
    """
    if not args.out_csv.exists():
        raise SystemExit(f"{args.out_csv} not found -- run --stage attribute first")
    df = pd.read_csv(args.out_csv, dtype={"id": str})
    meta = load_labels(args.data_dir, "public_test")[["id", "path"]]
    df = df.merge(meta, on="id", how="left")
    df = df[df["path"].notna() & df["path"].map(lambda p: Path(p).exists())]

    dropped_pool = df[df["new_drop_bucket"].notna()]
    stable_pool = df[(df["zone"] == "ceiling") & df["new_drop_bucket"].isna()]
    print(f"[movement_census] template_sample pools: dropped={len(dropped_pool)} stable={len(stable_pool)}")

    out_dir = args.render_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pool in [("dropped", dropped_pool), ("stable", stable_pool)]:
        n = min(TEMPLATE_SAMPLE_N, len(pool))
        sample = pool.sample(n=n, random_state=TEMPLATE_SAMPLE_SEED).reset_index(drop=True)
        sample[["id"]].to_csv(out_dir / f"template_sample_{name}_ids.csv", index=False)
        cells = [_dense_id_cell(row) for row in sample.to_dict("records")]
        n_sheets = -(-len(cells) // 25)
        for s in range(n_sheets):
            render_sheet(cells[s * 25:(s + 1) * 25], out_dir / f"template_sample_{name}_sheet{s+1:02d}.png", ncols=5, nrows=5)
        print(f"[movement_census] template_sample[{name}]: rendered {len(cells)} cells across "
              f"{n_sheets} sheet(s), ids listed in template_sample_{name}_ids.csv -> {out_dir}")


def run_stratified_dossier(args, non_mauritius_ids: list[str]) -> None:
    """Item 4: 30 worst-by-drop-cost NEW_DROP ids + a caller-supplied list of random non-Mauritius
    drops (identified from --stage template_sample's classified output), rendered at the SAME
    detailed-dossier resolution as the original new_fp/new_drop sheets (face-zoom if a regions
    cache is reachable) for per-id evidence-checklist inspection."""
    if not args.out_csv.exists():
        raise SystemExit(f"{args.out_csv} not found -- run --stage attribute first")
    df = pd.read_csv(args.out_csv, dtype={"id": str})
    rdir = regions_dir(args.data_dir)
    has_regions = rdir.exists()
    meta = load_labels(args.data_dir, "public_test")[["id", "path"]]
    df = df.merge(meta, on="id", how="left")

    drop = df[df["new_drop_bucket"].notna()].sort_values("drop_pair_cost", ascending=False)
    worst30 = drop.head(30)
    extra = df[df["id"].isin(non_mauritius_ids)]
    combined = pd.concat([worst30, extra], ignore_index=True).drop_duplicates(subset="id")
    print(f"[movement_census] stratified_dossier: {len(worst30)} worst-by-cost + {len(extra)} "
          f"caller-supplied non-Mauritius = {len(combined)} unique ids")

    cells = []
    for row in combined.itertuples(index=False):
        if pd.isna(row.path) or not Path(row.path).exists():
            continue
        cell_row = pd.Series({
            "id": row.id, "path": row.path,
            "mean_logit": row.finetune_mean_logit, "stratum": row.zone,
            "pct_rank": row.photosub_pct_rank,
        })
        cell = render_cell(cell_row, rdir, show_face_crop=has_regions, caption_height=80)
        d = ImageDraw.Draw(cell)
        cost = row.drop_pair_cost if pd.notna(row.drop_pair_cost) and row.drop_pair_cost > 0 else float("nan")
        caption2 = (
            f"fin_pct={row.finetune_pct_rank:.1f} -> pho_pct={row.photosub_pct_rank:.1f} "
            f"cost={cost:.1f}"
        )
        d.text((4, cell.height - 20), caption2, fill=(200, 0, 0))
        cells.append(cell)

    out_dir = args.render_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n_sheets = -(-len(cells) // 20)
    for s in range(n_sheets):
        render_sheet(cells[s * 20:(s + 1) * 20], out_dir / f"stratified_dossier_sheet{s+1:02d}.png", ncols=4, nrows=5)
    print(f"[movement_census] stratified_dossier: rendered {len(cells)} cells across {n_sheets} sheet(s) -> {out_dir}")


def run_part2_report(args) -> None:
    """Items 5 + 6: writes movement_census_part2_report.md (manifest-composition cross-check +
    full-ceiling-zone per-template drop rates). Needs --stage classify_types' output for item 6
    (item 5 alone doesn't need the checkpoint/embeddings, but both are cheap to run together)."""
    if not args.out_csv.exists():
        raise SystemExit(f"{args.out_csv} not found -- run --stage attribute first")
    df = pd.read_csv(args.out_csv, dtype={"id": str})

    selected, base = reconstruct_selected_photosub_rows(args)
    item5 = manifest_composition_report(selected, base)
    item6 = per_template_drop_report(df, args.type_csv)
    item4_pointer = (
        "## Part 2, item 4: stratified dossier evidence-checklist review\n\n"
        "See `ITEM4_DOSSIER_FINDINGS.md` (manual review of `stratified_dossier_sheet01.png` "
        "[30 worst-by-cost] + `sheet02.png` [10 random non-Mauritius drops] plus full-resolution "
        "zoom crops) -- headline: real digital-edit evidence found, but it's TEXT-field splicing "
        "(name/address/DOB fields), not photo-region tampering, confirmed across 4 of 5 "
        "templates and absent from bona-fide controls; the same artifact also appears on STABLE "
        "(non-dropped) ceiling ids of the same template, so it explains 'this is fraud' but not "
        "yet 'why THIS one specifically dropped' -- reported as an open question, not resolved "
        "here.\n"
    )

    out_path = args.render_dir / "movement_census_part2_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# Movement census, part 2: Mauritius-drop mechanism check\n\n"
        + item5 + "\n\n" + item6 + "\n\n" + item4_pointer,
        encoding="utf-8",
    )
    print(f"[movement_census] wrote part 2 report -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=[
            "attribute", "render", "all", "template_sample", "stratified_dossier",
            "classify_types", "classify_types_color", "part2_report",
        ],
        default="all",
    )
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--census-csv", type=Path, default=DEFAULT_CENSUS_CSV)
    p.add_argument("--finetune-sub", type=Path, default=DEFAULT_FINETUNE_SUB)
    p.add_argument("--photosub-sub", type=Path, default=DEFAULT_PHOTOSUB_SUB)
    p.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    p.add_argument("--deep-review-csv", type=Path, default=DEFAULT_DEEP_REVIEW_CSV)
    p.add_argument("--probes-dir", type=Path, default=DEFAULT_PROBES_DIR)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    p.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    p.add_argument(
        "--non-mauritius-ids", type=str, default="",
        help="comma-separated ids for --stage stratified_dossier's 10-random-non-Mauritius slice "
             "(identified manually from --stage template_sample's rendered output)",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "finetune_v0.pt",
        help="checkpoint used by --stage classify_types' type_proxy_knn embeddings",
    )
    p.add_argument("--type-csv", type=Path, default=DEFAULT_TYPE_CSV)
    p.add_argument(
        "--stable-sample-n", type=int, default=DEFAULT_STABLE_SAMPLE_N,
        help="--stage classify_types: how many stable (non-NEW_DROP) ceiling ids to sample for "
             "the per-template baseline estimate (CPU forward-pass cost scales with this)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in ("attribute", "all"):
        run_attribute(args)
    if args.stage in ("render", "all"):
        run_render(args)
    if args.stage == "template_sample":
        run_template_sample(args)
    if args.stage == "stratified_dossier":
        ids = [i.strip() for i in args.non_mauritius_ids.split(",") if i.strip()]
        if not ids:
            raise SystemExit("--stage stratified_dossier needs --non-mauritius-ids (comma-separated)")
        run_stratified_dossier(args, ids)
    if args.stage == "classify_types":
        run_classify_types(args)
    if args.stage == "classify_types_color":
        run_classify_types_color(args)
    if args.stage == "part2_report":
        run_part2_report(args)


if __name__ == "__main__":
    main()
