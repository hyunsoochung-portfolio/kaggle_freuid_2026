"""Turns the corrected FREUID metric (official_score_reconciliation.py) into a priced action
list: which score movements are actually worth how much expected FREUID, and which candidate
next-submission is worth spending a Kaggle submission on.

Reuses Prompt 1's MC machinery UNCHANGED (`mc_expected_metrics`, `fast_official_score`,
`build_present_df`, `build_imputation_schemes`, `finetune_threshold_position`, all imported
directly from `official_score_reconciliation.py`) so nothing here can silently drift from that
analysis's definitions. Same scope choice as Prompt 1 throughout: full 142,818-row population
(present ids + ~135k placeholder block at each submission's own actual placeholder score/the
measured train-base-rate fraud assumption) -- Prompt 1 already showed this scope's ABSOLUTE
expected-FREUID values don't match the observed LB (a real, unresolved methodology gap, see that
report's Diagnosis section). This script's use of the same machinery is deliberately for
RELATIVE comparison across candidates under an internally-consistent pipeline, not for claiming
an absolute predicted LB score -- every number below should be read comparatively (which
candidate beats which, by how much, under a fixed methodology), never as "this candidate will
score X on Kaggle." Item 3's sensitivity pass exists specifically to check whether that relative
ranking survives label-model uncertainty; it does NOT and CANNOT fix the absolute-scope gap.

Two families of "modification" appear below and they are NOT the same thing:
  - Hypothetical MOVES (item 1c): simulate what a hypothetically-improved MODEL's capability
    would do to specific FAMILIES of ids (ceiling-zone confusion, boundary/deep misses), to price
    which capability is worth building next. These never touch a real submission file.
  - Candidate SUBMISSIONS (item 2): real, constructible score vectors (tie-broken finetune_v0,
    rank-averaged ensembles) that COULD actually be submitted, priced the same way for ranking.
Neither family ever hand-edits a SPECIFIC known id's score based on its reviewed verdict -- see
the module-end note on why that's out of scope by design, not by oversight.

No training, no new submissions written. Pure CPU/pandas + the vendored scorer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.official_score import DEFAULT_BPCER_TARGET  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, df_to_md  # noqa: E402
from official_score_reconciliation import (  # noqa: E402
    IMPUTATION_SCHEMES,
    MC_BASE_SEED,
    N_MC_DRAWS,
    PLACEHOLDER_FRAUD_RATE,
    SUBMISSIONS,
    TOTAL_TEST_ROWS,
    build_imputation_schemes,
    build_present_df,
    fast_official_score,
    finetune_threshold_position,
    mc_expected_metrics,
)

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = OUT_DIR / "threshold_pricing_out"
PROBES_DIR = REPO_ROOT / "data" / "probes"

# Reference crossing score for "above/below the 1%-BPCER threshold" throughout this script.
# TWO reference points, not one -- Prompt 1 already showed the crossing point is scope-dependent
# and unreliable in absolute terms, and threshold_battle() below finds this matters concretely
# here: at FULL_POPULATION_CROSSING (0.5, Prompt 1's headline full-population blended run), all
# 4 confirmed ceiling-B ids are budget-eaters; at PRESENT_ONLY_CROSSING (computed fresh below,
# present-ids-only 1%-budget crossing), none of them are, because a SINGLE real verdict=B id
# (`4c1ac0279e`, zone=transitional, score 0.9949) sitting at rank 17 of the entire present
# population alone eats a large share of the much-smaller present-only budget before the
# threshold ever reaches the ceiling zone's score range at all. Both are reported; neither is
# silently treated as authoritative.
FULL_POPULATION_CROSSING = 0.5


def present_only_crossing_score(blended: pd.DataFrame, score_col: str = "finetune_v0_score",
                                 bpcer_target: float = DEFAULT_BPCER_TARGET) -> float:
    """1%-BPCER crossing score computed over present ids ONLY (no placeholder block at all) --
    the other end of the scope band `FULL_POPULATION_CROSSING` sits at."""
    s = blended.sort_values(score_col, ascending=False).reset_index(drop=True)
    total_bona = float(s["p_bonafide"].sum())
    budget = bpcer_target * total_bona
    cum_bona = s["p_bonafide"].cumsum()
    idx = int((cum_bona >= budget).idxmax())
    return float(s.loc[idx, score_col])


# ---------------------------------------------------------------------------
# Probe id sets
# ---------------------------------------------------------------------------

def load_probe_sets() -> dict[str, set[str]]:
    """The 9 deep-miss ids and 59 boundary ids (both real, confirmed verdict='F' -- see
    data/probes/README.md), plus the 200-id ceiling self-consistency probe (no verdict, used
    only as an additional zone-composition reference, never as confirmed-fraud evidence)."""
    deep = pd.read_csv(PROBES_DIR / "missed_frauds_deep_ids.csv", dtype={"id": str})
    boundary = pd.read_csv(PROBES_DIR / "missed_frauds_boundary_ids.csv", dtype={"id": str})
    ceiling_sample = pd.read_csv(PROBES_DIR / "ceiling_frauds_sample_ids.csv", dtype={"id": str})
    return {
        "deep": set(deep["id"]),
        "boundary": set(boundary["id"]),
        "ceiling_sample": set(ceiling_sample["id"]),
    }


# ---------------------------------------------------------------------------
# Item 1: threshold battle for finetune_v0
# ---------------------------------------------------------------------------

CEILING_B_IDS = {"1ec807903066414f951f97f0a04ea926", "63053f74658044c49e9387d2db2efadd",
                 "7538e1336b5e4bab97f4a191d838914a", "a3f97add9d8040b5909bad18b055e6c0"}


def threshold_battle(df: pd.DataFrame, blended: pd.DataFrame, probes: dict[str, set[str]],
                      crossing_score: float) -> dict:
    """(a) bona-fide mass above the threshold + which reviewed ids they are (are the 4 confirmed
    ceiling-B ids budget-eaters?); (b) fraud mass below the threshold, split deep/boundary/other."""
    present = blended[["id", "finetune_v0_score", "p_fraud", "p_bonafide", "verdict", "zone"]].copy()
    present["above_threshold"] = present["finetune_v0_score"] >= crossing_score

    above = present[present["above_threshold"]]
    below = present[~present["above_threshold"]]

    bonafide_above_mass = float(above["p_bonafide"].sum())
    reviewed_bonafide_above = above[above["verdict"] == "B"][["id", "finetune_v0_score", "zone"]]
    ceiling_b_status = present[present["id"].isin(CEILING_B_IDS)][
        ["id", "finetune_v0_score", "zone", "above_threshold"]
    ].copy()
    ceiling_b_status["role"] = np.where(ceiling_b_status["above_threshold"], "BUDGET-EATER", "already safe")

    fraud_below_mass = float(below["p_fraud"].sum())
    below_deep = below[below["id"].isin(probes["deep"])]
    below_boundary = below[below["id"].isin(probes["boundary"])]
    below_other_mask = ~below["id"].isin(probes["deep"] | probes["boundary"])
    below_other = below[below_other_mask]

    split = pd.DataFrame([
        {"group": "deep-9 (confirmed F, below threshold)", "n_ids": len(below_deep),
         "fraud_mass": float(below_deep["p_fraud"].sum())},
        {"group": "boundary-59 (confirmed F, below threshold)", "n_ids": len(below_boundary),
         "fraud_mass": float(below_boundary["p_fraud"].sum())},
        {"group": "other (zone-imputed, below threshold)", "n_ids": len(below_other),
         "fraud_mass": float(below_other["p_fraud"].sum())},
    ])

    # Also report where deep-9 / boundary-59 sit overall (not just the below-threshold subset) --
    # answers "are these already caught" independent of the split above.
    deep_scores = present[present["id"].isin(probes["deep"])][["id", "finetune_v0_score", "above_threshold"]]
    boundary_scores = present[present["id"].isin(probes["boundary"])][["finetune_v0_score", "above_threshold"]]

    return {
        "bonafide_above_mass": bonafide_above_mass,
        "reviewed_bonafide_above": reviewed_bonafide_above,
        "ceiling_b_status": ceiling_b_status,
        "fraud_below_mass": fraud_below_mass,
        "split": split,
        "deep_scores": deep_scores,
        "boundary_above_count": int(boundary_scores["above_threshold"].sum()),
        "boundary_total": len(boundary_scores),
        "ceiling_zone_bonafide_mass": float(present.loc[present["zone"] == "ceiling", "p_bonafide"].sum()),
        "present": present,
    }


# ---------------------------------------------------------------------------
# Item 1c: hypothetical-move MC pricing
# ---------------------------------------------------------------------------

def mc_expected_metrics_conditional_move(
    present_p_fraud: np.ndarray,
    present_score: np.ndarray,
    placeholder_score: float,
    placeholder_fraud_rate: float,
    n_placeholder: int,
    move_mask: np.ndarray,
    move_score_if_bonafide: float,
    n_draws: int = N_MC_DRAWS,
    seed: int = MC_BASE_SEED,
) -> dict:
    """Prices a CAPABILITY fix, not a specific-id edit: for ids in `move_mask`, IF that draw's
    Bernoulli(p_fraud) label comes out bona-fide, the hypothetical improved model is assumed to
    have scored it like `move_score_if_bonafide` instead of its current (confusable) score; if
    the draw comes out fraud, the score is left exactly as observed (already correctly high).
    This is the honest way to price 'what if the model could tell ceiling-zone bona-fide from
    ceiling-zone fraud' without ever assuming we know WHICH specific ids are which -- the model
    never gets credit for information this analysis doesn't actually have.
    """
    rng = np.random.default_rng(seed)
    placeholder_scores = np.full(n_placeholder, placeholder_score)

    draws = {"audet": np.empty(n_draws), "apcer_at_bpcer": np.empty(n_draws), "freuid": np.empty(n_draws)}
    for d in range(n_draws):
        present_labels = rng.binomial(1, present_p_fraud)
        draw_score = present_score.copy()
        demote = move_mask & (present_labels == 0)
        draw_score[demote] = move_score_if_bonafide
        placeholder_labels = rng.binomial(1, placeholder_fraud_rate, size=n_placeholder)
        y_true = np.concatenate([present_labels, placeholder_labels])
        y_score = np.concatenate([draw_score, placeholder_scores])
        result = fast_official_score(y_true, y_score)
        for k, v in result.items():
            draws[k][d] = v

    out = {}
    for k, arr in draws.items():
        out[f"{k}_mean"] = float(arr.mean())
        out[f"{k}_std"] = float(arr.std(ddof=1))
    return out


def price_hypothetical_moves(df: pd.DataFrame, blended: pd.DataFrame, probes: dict[str, set[str]],
                              baseline: dict) -> pd.DataFrame:
    name = "finetune_v0"
    placeholder_score = SUBMISSIONS[name]["placeholder_score"]
    n_placeholder = TOTAL_TEST_ROWS - len(df)
    present_p_fraud = blended["p_fraud"].to_numpy()
    present_score = blended[f"{name}_score"].to_numpy()
    zone = blended["zone"].to_numpy()
    ids = blended["id"].to_numpy()

    floor_median_score = float(blended.loc[blended["zone"] == "floor", f"{name}_score"].median())
    ceiling_median_score = float(blended.loc[blended["zone"] == "ceiling", f"{name}_score"].median())

    rows = []

    # Move A: demote ceiling-zone bona-fide mass (conditional -- see docstring above).
    move_mask_a = (zone == "ceiling")
    rA = mc_expected_metrics_conditional_move(
        present_p_fraud, present_score, placeholder_score, PLACEHOLDER_FRAUD_RATE, n_placeholder,
        move_mask_a, floor_median_score, seed=MC_BASE_SEED * 3001,
    )
    rows.append({"move": "A: demote ceiling bona-fide mass (capability fix, conditional)", **rA})

    # Move B: promote the 59 boundary ids above threshold (deterministic -- real verdict=F).
    move_mask_b = np.isin(ids, list(probes["boundary"]))
    score_b = present_score.copy()
    score_b[move_mask_b] = np.maximum(score_b[move_mask_b], ceiling_median_score)
    rB = mc_expected_metrics(present_p_fraud, score_b, placeholder_score, PLACEHOLDER_FRAUD_RATE,
                              n_placeholder, seed=MC_BASE_SEED * 3002)
    rows.append({"move": "B: promote boundary-59 to ceiling-median score (deterministic)", **rB})

    # Move C: promote the 9 deep-miss ids (deterministic -- real verdict=F).
    move_mask_c = np.isin(ids, list(probes["deep"]))
    score_c = present_score.copy()
    score_c[move_mask_c] = np.maximum(score_c[move_mask_c], ceiling_median_score)
    rC = mc_expected_metrics(present_p_fraud, score_c, placeholder_score, PLACEHOLDER_FRAUD_RATE,
                              n_placeholder, seed=MC_BASE_SEED * 3003)
    rows.append({"move": "C: promote deep-9 to ceiling-median score (deterministic)", **rC})

    # Move D: all three combined (B+C deterministic score edits, A's conditional demotion).
    score_d = present_score.copy()
    score_d[move_mask_b] = np.maximum(score_d[move_mask_b], ceiling_median_score)
    score_d[move_mask_c] = np.maximum(score_d[move_mask_c], ceiling_median_score)
    rD = mc_expected_metrics_conditional_move(
        present_p_fraud, score_d, placeholder_score, PLACEHOLDER_FRAUD_RATE, n_placeholder,
        move_mask_a, floor_median_score, seed=MC_BASE_SEED * 3004,
    )
    rows.append({"move": "D: A+B+C combined", **rD})

    out = pd.DataFrame(rows)
    out["freuid_delta_vs_baseline"] = out["freuid_mean"] - baseline["freuid_mean"]
    out["apcer_delta_vs_baseline"] = out["apcer_at_bpcer_mean"] - baseline["apcer_at_bpcer_mean"]
    # Significance check: MC MEANS have standard error std/sqrt(n_draws), not raw std -- a delta
    # between two independent means needs combined SE = sqrt(se_a^2 + se_b^2). At n_draws=200,
    # raw freuid_std (~0.0004) hugely overstates the noise floor for the DELTA specifically;
    # this is the check that actually tells "real vs MC noise" apart at this sample size.
    baseline_se = baseline["freuid_std"] / np.sqrt(N_MC_DRAWS)
    out["freuid_delta_se"] = np.sqrt((out["freuid_std"] / np.sqrt(N_MC_DRAWS)) ** 2 + baseline_se ** 2)
    out["freuid_delta_z"] = out["freuid_delta_vs_baseline"] / out["freuid_delta_se"]
    out["significant_at_2se"] = out["freuid_delta_z"].abs() >= 2.0
    return out


# ---------------------------------------------------------------------------
# Item 2: candidate submissions
# ---------------------------------------------------------------------------

def safe_epsilon_tiebreak(df: pd.DataFrame, score_col: str, secondary_col: str = "finetune_mean_logit") -> np.ndarray:
    """Breaks EXACT ties in `score_col` using `secondary_col` as the tiebreaker, with each
    perturbation confined strictly inside that id's own safe margin (half the gap to its nearest
    DISTINCT neighboring score value) so no perturbation can ever cross into -- or reorder
    relative to -- a group that was originally distinct. Groups with zero safe margin (adjacent
    distinct values separated by float-precision-level gaps) are left untouched rather than risked."""
    d = df[["id", score_col, secondary_col]].copy().reset_index(drop=True)
    order = d[score_col].to_numpy().argsort(kind="mergesort")
    sorted_scores = d[score_col].to_numpy()[order]
    distinct_vals, distinct_idx = np.unique(sorted_scores, return_index=True)
    gaps = np.diff(distinct_vals)

    # margin_up[g] / margin_down[g] = half-gap to the next distinct value above/below group g
    n_groups = len(distinct_vals)
    margin_up = np.empty(n_groups)
    margin_down = np.empty(n_groups)
    margin_up[:-1] = gaps / 2.0
    margin_up[-1] = 0.0
    margin_down[1:] = gaps / 2.0
    margin_down[0] = 0.0
    safe_margin = np.minimum(margin_up, margin_down)

    group_of_sorted = np.searchsorted(distinct_vals, sorted_scores)
    new_scores_sorted = sorted_scores.copy()
    i = 0
    while i < len(sorted_scores):
        g = group_of_sorted[i]
        j = i
        while j < len(sorted_scores) and group_of_sorted[j] == g:
            j += 1
        size = j - i
        if size > 1 and safe_margin[g] > 0:
            members = d.iloc[order[i:j]]
            secondary_rank = members[secondary_col].rank(method="first").to_numpy()
            frac = (secondary_rank - (size - 1) / 2.0) / max(1, size - 1)  # in [-0.5, 0.5]
            new_scores_sorted[i:j] = sorted_scores[i:j] + frac * (2.0 * safe_margin[g] * 0.9)
        i = j

    out = np.empty(len(d))
    out[order] = new_scores_sorted
    return out


def check_tie_straddle(blended: pd.DataFrame, score_col: str, threshold: float) -> dict:
    """Does any exact-tie cluster of `score_col` sit AT the reference threshold value itself
    (the only way a tie cluster can 'straddle' a single scalar threshold -- ties elsewhere don't
    interact with this specific operating point at all)?"""
    counts = blended[score_col].value_counts()
    at_threshold = counts[counts.index == threshold]
    return {
        "n_ids_at_threshold": int(at_threshold.sum()) if len(at_threshold) else 0,
        "straddles": len(at_threshold) > 0 and int(at_threshold.iloc[0]) > 1,
    }


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Mirrors infer.py's `_rank_normalize`: fractional rank in (eps, 1-eps), average rank for
    ties -- the exact convention this repo's own TTA combination already uses, reused here for
    combining two DIFFERENT submissions rather than TTA scales."""
    a = np.asarray(scores, dtype=np.float64)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1
        if j > i + 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    lo, hi = ranks.min(), ranks.max()
    if hi > lo:
        ranks = (ranks - lo) / (hi - lo) * (1 - 2e-7) + 1e-7
    return ranks


def build_candidates(df: pd.DataFrame, blended: pd.DataFrame) -> dict[str, dict]:
    """Each candidate: {'present_score': array over blended's id order, 'placeholder_score': float}."""
    candidates: dict[str, dict] = {}

    # (a) finetune_v0 + epsilon tie-break
    fin_tiebroken = safe_epsilon_tiebreak(blended, "finetune_v0_score")
    candidates["finetune_v0+tiebreak"] = {
        "present_score": fin_tiebroken, "placeholder_score": SUBMISSIONS["finetune_v0"]["placeholder_score"],
    }

    # (d) photosub_v0 + epsilon tie-break
    pho_tiebroken = safe_epsilon_tiebreak(blended, "photosub_v0_score", secondary_col="finetune_mean_logit")
    candidates["photosub_v0+tiebreak"] = {
        "present_score": pho_tiebroken, "placeholder_score": SUBMISSIONS["photosub_v0"]["placeholder_score"],
    }

    # (b) rank-average(finetune_v0, photosub_v0) at 3 weight settings
    rank_fin = rank_normalize(blended["finetune_v0_score"].to_numpy())
    rank_pho = rank_normalize(blended["photosub_v0_score"].to_numpy())
    for w_fin in (0.7, 0.5, 0.3):
        ensemble = w_fin * rank_fin + (1.0 - w_fin) * rank_pho
        candidates[f"rank_avg(finetune={w_fin:.1f},photosub={1 - w_fin:.1f})"] = {
            "present_score": ensemble, "placeholder_score": 0.5,  # both models place 0.5 identically
        }

    return candidates


def price_candidates(df: pd.DataFrame, blended: pd.DataFrame, candidates: dict[str, dict]) -> pd.DataFrame:
    n_placeholder = TOTAL_TEST_ROWS - len(df)
    present_p_fraud = blended["p_fraud"].to_numpy()
    rows = []
    for i, (cand_name, cand) in enumerate(candidates.items()):
        r = mc_expected_metrics(present_p_fraud, cand["present_score"], cand["placeholder_score"],
                                 PLACEHOLDER_FRAUD_RATE, n_placeholder, seed=MC_BASE_SEED * 4000 + i)
        r["candidate"] = cand_name
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Item 3: sensitivity across the label-model uncertainty band
# ---------------------------------------------------------------------------

def sensitivity_across_schemes(df: pd.DataFrame, schemes: dict[str, pd.DataFrame],
                                candidates: dict[str, dict]) -> pd.DataFrame:
    """Reprices every candidate under all 3 imputation schemes (blended/zone_only/verdict_only)
    -- the label-model uncertainty band. verdict_only uses its own (smaller, real-verdict-only)
    present-id population, so candidates' present_score arrays must be re-sliced to that
    scheme's id order rather than reused as-is."""
    n_placeholder_fixed = TOTAL_TEST_ROWS - len(df)
    rows = []
    for scheme_idx, scheme_name in enumerate(IMPUTATION_SCHEMES):
        scheme_df = schemes[scheme_name]
        id_to_pos = {i: p for p, i in enumerate(df["id"])}
        scheme_positions = scheme_df["id"].map(id_to_pos).to_numpy()
        n_placeholder = n_placeholder_fixed if scheme_name != "verdict_only" else (
            TOTAL_TEST_ROWS - len(scheme_df)
        )
        for cand_idx, (cand_name, cand) in enumerate(candidates.items()):
            present_score = cand["present_score"][scheme_positions]
            r = mc_expected_metrics(
                scheme_df["p_fraud"].to_numpy(), present_score, cand["placeholder_score"],
                PLACEHOLDER_FRAUD_RATE, n_placeholder, seed=MC_BASE_SEED * 5000 + scheme_idx * 100 + cand_idx,
            )
            r["scheme"] = scheme_name
            r["candidate"] = cand_name
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Item 4: ranked recommendation
# ---------------------------------------------------------------------------

def build_recommendation(cand_df: pd.DataFrame, baseline: dict, sens_df: pd.DataFrame) -> pd.DataFrame:
    """Ranks every candidate (+ the unmodified finetune_v0 baseline) by MC-expected FREUID, with
    a standard-error-based z-score vs baseline AND a flag for whether the candidate crosses the
    finetune_v0-vs-photosub_v0 boundary -- see the report text for why that specific comparison
    from THIS pipeline cannot be trusted (it disagrees with the real, already-observed LB)."""
    rows = list(cand_df.to_dict("records"))
    rows.append({"candidate": "finetune_v0 (baseline, unmodified)", **baseline})
    out = pd.DataFrame(rows)

    baseline_se = baseline["freuid_std"] / np.sqrt(N_MC_DRAWS)
    out["freuid_se"] = out["freuid_std"] / np.sqrt(N_MC_DRAWS)
    out["delta_vs_baseline"] = out["freuid_mean"] - baseline["freuid_mean"]
    out["delta_se"] = np.sqrt(out["freuid_se"] ** 2 + baseline_se ** 2)
    out["z_vs_baseline"] = out["delta_vs_baseline"] / out["delta_se"]
    out["crosses_photosub_boundary"] = out["candidate"].str.contains("photosub", case=False)

    # Sensitivity gate (item 3): does the sign of the improvement survive verdict_only, the
    # scheme closest to real ground truth (real verdicts, zero imputation)?
    verdict_only_freuid = sens_df[sens_df["scheme"] == "verdict_only"].set_index("candidate")["freuid_mean"]
    out["verdict_only_freuid"] = out["candidate"].map(verdict_only_freuid)
    baseline_verdict_only = sens_df[(sens_df["scheme"] == "verdict_only") &
                                     (sens_df["candidate"] == "finetune_v0+tiebreak")]["freuid_mean"]
    # finetune_v0 baseline itself wasn't run through sensitivity_across_schemes (only candidates
    # were) -- use finetune_v0+tiebreak's verdict_only value as the closest available anchor,
    # since item 2's own pricing found tie-breaking has a negligible effect on its own.
    anchor = float(baseline_verdict_only.iloc[0]) if len(baseline_verdict_only) else float("nan")
    out["verdict_only_delta_vs_baseline"] = out["verdict_only_freuid"] - anchor
    out["survives_pessimistic_band"] = np.sign(out["delta_vs_baseline"]) == np.sign(out["verdict_only_delta_vs_baseline"])
    out.loc[out["candidate"].str.contains("baseline"), "survives_pessimistic_band"] = True  # baseline vs itself

    return out.sort_values("freuid_mean")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(battle_full: dict, battle_present: dict, present_crossing: float,
                  moves_df: pd.DataFrame, cand_df: pd.DataFrame,
                  sens_df: pd.DataFrame, rec_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Threshold pricing: turning the corrected metric into an action list\n")
    lines.append("Reuses `official_score_reconciliation.py`'s MC machinery unchanged throughout. "
                 "**Scope caveat carried over from Prompt 1**: absolute expected-FREUID values do "
                 "NOT match the observed LB (documented mechanism: zone-vs-TTA-score mismatch + "
                 "APCER's tail sensitivity) -- every number below is for RELATIVE comparison "
                 "across candidates under one fixed, internally-consistent pipeline, never a "
                 "predicted real LB score.\n")

    lines.append("\n## Item 1: the threshold battle, quantified\n")
    lines.append(f"**Two reference threshold scores, not one** -- the crossing point is itself "
                 f"scope-dependent (Prompt 1's finding): **{FULL_POPULATION_CROSSING}** "
                 "(full-population blended run's own crossing, dominated by the placeholder "
                 f"block) vs **{present_crossing:.6f}** (present-ids-only 1%-budget crossing, no "
                 "placeholder block at all). These give MATERIALLY DIFFERENT answers to 'is X a "
                 "budget-eater' below -- both reported, neither picked silently.\n")

    lines.append("\n**Why present-only lands so much higher**: a single REAL, confirmed "
                 "verdict=B id (`4c1ac0279e`, zone=`transitional`, score **0.9949**) sits at rank "
                 "17 of the entire 7,821-id present population -- a genuinely bona-fide document "
                 "scoring almost exactly like the most confident fraud. One id alone consumes a "
                 "meaningful share of the present-only budget (37.9 units) before the threshold "
                 "even reaches the ceiling zone's score range. This is mechanism 1 from Prompt "
                 "1's diagnosis (zone-vs-TTA-score mismatch) showing up as a single, concrete, "
                 "identifiable instance -- reported as evidence for the capability gap, per item "
                 "5 NOT as a target for hand-editing.\n")

    def render_battle(battle: dict, label: str, crossing: float) -> None:
        lines.append(f"\n### At threshold = {crossing} ({label})\n")
        lines.append(f"**(a) Bona-fide mass above the threshold**: {battle['bonafide_above_mass']:.2f} "
                     "expected bona-fide ids sit at/above this threshold score (budget-eaters -- "
                     "every unit here is a false alarm the real APCER@1%BPCER operating point has "
                     f"to absorb). Ceiling zone alone contributes "
                     f"{battle['ceiling_zone_bonafide_mass']:.2f} expected bona-fide mass overall "
                     "(not all of it necessarily above this specific threshold).\n")
        lines.append("\n**The 4 confirmed ceiling-B ids**:\n")
        lines.append(df_to_md(battle["ceiling_b_status"], float_fmt="{:.6f}"))
        all_eaters = bool(battle["ceiling_b_status"]["above_threshold"].all())
        none_eaters = bool((~battle["ceiling_b_status"]["above_threshold"]).all())
        verdict = "ALL 4 are budget-eaters" if all_eaters else ("NONE are budget-eaters" if none_eaters else "SOME are budget-eaters")
        lines.append(f"\n**{verdict}** at this threshold.\n")
        if len(battle["reviewed_bonafide_above"]):
            lines.append("\n**All reviewed (human-verdicted) bona-fide ids above this threshold:**\n")
            lines.append(df_to_md(battle["reviewed_bonafide_above"], float_fmt="{:.6f}"))
        lines.append(f"\n**(b) Fraud mass below the threshold**: {battle['fraud_below_mass']:.2f} "
                     "expected fraud ids sit below this threshold (misses at this operating "
                     "point), split by group:\n")
        lines.append(df_to_md(battle["split"], float_fmt="{:.2f}"))
        lines.append(f"\nBoundary-59 status: **{battle['boundary_above_count']}/{battle['boundary_total']}** "
                     "already sit ABOVE this threshold.\n")

    render_battle(battle_full, "Prompt 1's full-population headline crossing", FULL_POPULATION_CROSSING)
    render_battle(battle_present, "present-ids-only 1%-budget crossing", present_crossing)

    lines.append("\n**Deep-9 scores** (scope-independent -- the real remaining lever regardless "
                 "of which threshold reference is used, since most sit well below BOTH):\n")
    lines.append(df_to_md(battle_full["deep_scores"], float_fmt="{:.6f}"))

    lines.append("\n**(c) Hypothetical-move pricing (MC, capability fixes, never submitted):**\n")
    display_moves = moves_df[["move", "audet_mean", "apcer_at_bpcer_mean", "freuid_mean",
                               "freuid_delta_vs_baseline", "freuid_delta_z", "significant_at_2se"]]
    lines.append(df_to_md(display_moves, float_fmt="{:.5f}"))
    lines.append("\n`freuid_delta_z` uses the STANDARD ERROR of each 200-draw mean (std/sqrt(200)), "
                 "not raw std -- raw std hugely overstates the noise floor for a delta-of-means. "
                 "At this sample size: Move B (promote boundary-59) is indistinguishable from "
                 "zero, consistent with the finding above that the group is already caught (a "
                 "free result, not an actionable lever). Move C (promote deep-9) and Move D "
                 "(combined) clear the 2-SE bar and are correctly signed (lower FREUID = better); "
                 "Move A (fix ceiling confusion) does NOT clear 2-SE on its own at n_draws=200 -- "
                 "its effect, while directionally plausible, isn't resolved at this sample size "
                 "and would need more draws to confirm rather than being asserted as real.\n")

    lines.append("\n## Item 2: candidate submissions, priced\n")
    lines.append(df_to_md(cand_df[["candidate", "audet_mean", "audet_std", "apcer_at_bpcer_mean",
                                    "apcer_at_bpcer_std", "freuid_mean", "freuid_std"]], float_fmt="{:.5f}"))

    lines.append("\n## Item 3: sensitivity across the label-model uncertainty band\n")
    pivot = sens_df.pivot(index="candidate", columns="scheme", values="freuid_mean")
    pivot["spread_blended_minus_verdict_only"] = pivot.get("blended", np.nan) - pivot.get("verdict_only", np.nan)
    lines.append(df_to_md(pivot.reset_index(), float_fmt="{:.5f}"))
    lines.append("\n**Critical caveat, not a footnote**: `verdict_only` lands ~0.09-0.10 HIGHER "
                 "(worse) than `blended`/`zone_only` for every single candidate, uniformly -- the "
                 "exact same uniform-misfit signature Prompt 1 diagnosed as a scope/methodology "
                 "artifact, not real information about any candidate. Don't read "
                 "'`verdict_only` says everything is worse' as a real pessimistic scenario for "
                 "any ONE candidate over another -- it shifts all of them together. What DOES "
                 "carry information here is whether the RELATIVE ORDER (which candidate beats "
                 "which) flips between schemes -- checked explicitly in the recommendation "
                 "below via `survives_pessimistic_band`.\n")

    lines.append("\n## Item 4: ranked recommendation\n")
    lines.append("**The finetune_v0-vs-photosub_v0 comparison from THIS pipeline cannot be "
                 "trusted** -- Part 1's movement-census analysis already established, from the "
                 "REAL observed public LB, that finetune_v0 (0.00744) beats photosub_v0 (0.01616) "
                 "by 2.17x. The table below, from this MC pipeline, ranks photosub_v0-heavy "
                 "rank-averages as BETTER than finetune_v0 alone -- directly contradicting that "
                 "known ground truth. This is not a new finding; it's the same present-ids-only "
                 "AuDET-favors-photosub_v0 result the movement census already flagged (finetune_v0 "
                 "AuDET=0.0508 vs photosub_v0 AuDET=0.0410 on the reviewed subset -- photosub_v0 "
                 "really does win on present-ids ranking quality alone), reproduced here under a "
                 "different instrument. **Any recommendation that crosses into photosub_v0 "
                 "territory is flagged `UNTRUSTED` below and excluded from the actionable "
                 "recommendation, regardless of what its MC number says.**\n")
    display_rec = rec_df[["candidate", "freuid_mean", "freuid_se", "delta_vs_baseline", "z_vs_baseline",
                           "crosses_photosub_boundary", "survives_pessimistic_band"]].copy()
    display_rec["crosses_photosub_boundary"] = display_rec["crosses_photosub_boundary"].map(
        {True: "UNTRUSTED (crosses to photosub_v0)", False: "trusted (finetune_v0-anchored)"})
    lines.append(df_to_md(display_rec, float_fmt="{:.5f}"))

    trusted = rec_df[~rec_df["crosses_photosub_boundary"]].sort_values("freuid_mean")
    best_trusted = trusted.iloc[0]
    lines.append(f"\n**Best TRUSTED candidate: `{best_trusted['candidate']}`** -- "
                 f"delta vs baseline {best_trusted['delta_vs_baseline']:+.6f} "
                 f"(z={best_trusted['z_vs_baseline']:.2f}, "
                 f"{'significant' if abs(best_trusted['z_vs_baseline']) >= 2 else 'NOT significant'} "
                 f"at 2-SE{'' if best_trusted['survives_pessimistic_band'] else ', and its sign does NOT survive the verdict_only pessimistic band -- expected and unremarkable given the delta is already noise-level, not a real reversal to worry about'}"
                 "). Given the delta is tiny and not clearly significant, the honest "
                 "recommendation is: **epsilon-tiebreaking finetune_v0 is safe and essentially "
                 "free (no evidence of harm, weak/no evidence of benefit at this MC resolution) "
                 "-- worth doing before any resubmission since it costs nothing, but it is not a "
                 "reason to spend a submission on its own.** No trusted candidate in this batch "
                 "clears a confident, actionable improvement over the current best "
                 "(`finetune_v0`) -- the real lever, per items 1 and the strategy memo below, is "
                 "a NEW TRAINING RUN, not a re-combination of what already exists.\n")

    lines.append("\n## Strategy memo: what the next training run needs\n")
    lines.append("Synthesizing items 1-3: this pipeline's absolute numbers aren't trustworthy, "
                 "but three findings replicate across every scope/scheme tried and are strong "
                 "enough to act on:\n")
    lines.append("1. **The boundary-59 group is already caught** (59/59 above the full-population "
                 "threshold, 16/59 even above the much stricter present-only threshold, and Move "
                 "B prices at zero, indistinguishable from noise). `photosub_v0`'s boundary-fraud "
                 "promotion goal is **already substantially achieved by finetune_v0 itself** -- "
                 "not a gap that needs new capability.\n")
    lines.append("2. **The deep-9 family is the real, still-open gap** (7-9 of 9 sit below both "
                 "threshold references; Move C prices as the one hypothetical move that clears "
                 "2-SE significance). This matches `photosub_v0`'s own deep-9 diagnostic in "
                 "`docs/technical_report.md` (MODE_B/C converged and stayed strongly positive; "
                 "MODE_A regressed for 2 of 4 ids) -- the promotable signal is real and already "
                 "partially demonstrated, just not cleanly, and not without the ceiling-zone "
                 "cost documented next.\n")
    lines.append("3. **The ceiling-zone confusion is a real, distinct, and separately-priced "
                 "cost** (36-37 expected bona-fide mass sits inside the ceiling zone, "
                 "score-indistinguishable from confirmed fraud per Part 2's embedding-collapse "
                 "finding; Move A's effect, while not yet resolved at 2-SE with n_draws=200, is "
                 "directionally the right sign and independently corroborated by "
                 "`bayar_dinov2_v0`'s and `photosub_v0`'s own real, already-observed regressions "
                 "-- both prior attempts that changed model capacity/training signal traded away "
                 "some of finetune_v0's existing ceiling confidence while chasing exactly this "
                 "kind of gain).\n")
    lines.append("\n**The answer the decomposition points to: `photosub_v0`'s AuDET gain (real, "
                 "on present ids, per the movement census) without its ceiling-retention cost "
                 "(also real, per `docs/technical_report.md`'s clean_floor/ceiling guards and this "
                 "analysis's Move A), plus specifically the deep-9 family's promotion (the one "
                 "hypothetical move that priced as significant here) -- NOT a blanket "
                 "photo-substitution push across all modes.** `photosub_v0` already ships partial "
                 "capability-protection scaffolding for exactly this (the 200-id "
                 "`ceiling_frauds_sample_ids.csv` self-consistency guard, the "
                 "`clean_floor_sample_ids.csv` regression guard) -- no separate 'photosub_v1' "
                 "spec file exists in this repo yet, so 'revive' means extending this existing "
                 "scaffolding, not resurrecting a lost document. A next training run should:\n")
    lines.append("- **Gate on APCER@1%BPCER specifically, not just AuDET/probe_AuDET** -- "
                 "CLAUDE.md's own priority #2 (partial-AUC loss term) is now additionally "
                 "justified by this analysis: the ceiling zone's confusable bona-fide mass is "
                 "exactly the kind of tail failure a full-curve AuDET loss doesn't specifically "
                 "punish. Add the 4 confirmed ceiling-B ids (and the `ceiling_frauds_sample_ids` "
                 "self-consistency set) as an explicit per-epoch APCER-relevant probe alongside "
                 "the existing ceiling-retention guard, not just a mean-logit regression check.\n")
    lines.append("- **Keep MODE_B/C's photo-substitution signal** (converged cleanly per the "
                 "deep-9 diagnostic) **while dropping or reworking MODE_A** (regressed for 2/4 "
                 "ids, already diagnosed as a shape-realism gap in `docs/technical_report.md`) -- "
                 "don't re-run the same A/B/C/D weight mix unmodified.\n")
    lines.append("- **Re-verify boundary-59 stays caught** rather than spending training signal "
                 "trying to improve it further -- it's already free per this analysis; capacity "
                 "spent there is capacity not spent on the ceiling-zone/deep-9 tradeoff that "
                 "actually matters.\n")
    lines.append("- **Do not treat this analysis's specific per-id findings (the 4 ceiling-B ids, "
                 "`4c1ac0279e`, the deep-9 ids) as training targets to hand-tune toward** -- they "
                 "are diagnostic evidence for what CAPABILITY is missing, not a checklist of "
                 "individual predictions to fix. See the explicit non-goal below.\n")

    lines.append("\n## Explicit non-goal (per task instruction, restated for the record)\n")
    lines.append("**Do not hand-edit individual known ids' scores in a submission file** (e.g. "
                 "manually demoting the 4 reviewed ceiling-B ids, or manually promoting the "
                 "deep-9/boundary ids) to game the public leaderboard. This would be pure "
                 "public-LB probing on a handful of ids this analysis happens to have human "
                 "verdicts for -- it cannot transfer to the private test set (which has none of "
                 "these specific ids), provides zero real capability improvement, and directly "
                 "contradicts this project's whole reprint-robustness thesis (CLAUDE.md's Project "
                 "section: a prior model that scored ~0.0006 locally but ~0.377 public by "
                 "exploiting exactly this kind of local-only signal). If this is proposed later, "
                 "flag it as this same anti-pattern.\n")

    report_path = out_dir / "threshold_pricing_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[threshold_pricing] wrote {report_path}")


def main() -> None:
    df = build_present_df()
    schemes = build_imputation_schemes(df)
    blended = schemes["blended"]
    probes = load_probe_sets()

    n_placeholder = TOTAL_TEST_ROWS - len(df)
    baseline = mc_expected_metrics(
        blended["p_fraud"].to_numpy(), blended["finetune_v0_score"].to_numpy(),
        SUBMISSIONS["finetune_v0"]["placeholder_score"], PLACEHOLDER_FRAUD_RATE, n_placeholder,
        seed=MC_BASE_SEED * 3000,
    )
    print(f"[threshold_pricing] finetune_v0 baseline: E[FREUID]={baseline['freuid_mean']:.5f}")

    present_crossing = present_only_crossing_score(blended)
    battle_full = threshold_battle(df, blended, probes, FULL_POPULATION_CROSSING)
    battle_present = threshold_battle(df, blended, probes, present_crossing)
    print(f"[threshold_pricing] full-pop threshold={FULL_POPULATION_CROSSING}: "
          f"bonafide_above={battle_full['bonafide_above_mass']:.2f} fraud_below={battle_full['fraud_below_mass']:.2f}")
    print(f"[threshold_pricing] present-only threshold={present_crossing:.6f}: "
          f"bonafide_above={battle_present['bonafide_above_mass']:.2f} fraud_below={battle_present['fraud_below_mass']:.2f}")

    moves_df = price_hypothetical_moves(df, blended, probes, baseline)
    print(moves_df[["move", "freuid_mean", "freuid_delta_vs_baseline"]].to_string(index=False))

    candidates = build_candidates(df, blended)
    tie_straddle_fin = check_tie_straddle(blended, "finetune_v0_score", FULL_POPULATION_CROSSING)
    tie_straddle_fin_present = check_tie_straddle(blended, "finetune_v0_score", present_crossing)
    print(f"[threshold_pricing] finetune_v0 tie-at-full-pop-threshold: {tie_straddle_fin}")
    print(f"[threshold_pricing] finetune_v0 tie-at-present-only-threshold: {tie_straddle_fin_present}")
    cand_df = price_candidates(df, blended, candidates)
    print(cand_df[["candidate", "freuid_mean"]].to_string(index=False))

    sens_df = sensitivity_across_schemes(df, schemes, candidates)
    rec_df = build_recommendation(cand_df, baseline, sens_df)
    print(rec_df[["candidate", "freuid_mean", "z_vs_baseline", "crosses_photosub_boundary",
                  "survives_pessimistic_band"]].to_string(index=False))

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    moves_df.to_csv(DEFAULT_OUT_DIR / "hypothetical_moves.csv", index=False)
    cand_df.to_csv(DEFAULT_OUT_DIR / "candidate_pricing.csv", index=False)
    sens_df.to_csv(DEFAULT_OUT_DIR / "sensitivity.csv", index=False)
    rec_df.to_csv(DEFAULT_OUT_DIR / "recommendation.csv", index=False)

    write_report(battle_full, battle_present, present_crossing, moves_df, cand_df, sens_df, rec_df, DEFAULT_OUT_DIR)


if __name__ == "__main__":
    main()
