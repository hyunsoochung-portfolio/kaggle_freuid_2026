"""The full-corpus diff gate: candidate submission vs reference submission, before spending a
Kaggle submission on the candidate.

Generalizes two prior one-off analyses into a reusable, repeatable tool (any two submission CSVs,
not just the finetune_v0-vs-photosub_v0 pair those were built around):
  - movement_census.py's crossing-count/bucket logic (NEW_FP / NEW_DROP -- present-id ranking
    movement between two submissions).
  - official_score_reconciliation.py's MC machinery (expected AuDET/APCER@1%/FREUID under the
    frozen zone/verdict label model, Monte Carlo'd since APCER@1% has no closed form under
    probabilistic labels).

Ground truth for the 7,821 present public-test ids is unavailable (same caveat as both scripts
above) -- every fraud/bona-fide judgment is either a real human-review verdict or a per-zone-
imputed probability, reused verbatim from movement_census.py so this can never silently diverge
from either prior analysis's definition of "p_fraud".

**Scope caveat, inherited from official_score_reconciliation.py's own diagnosis**: this script's
absolute expected-FREUID values do NOT match observed public-LB scores (documented mechanism:
the zone framework's raw-logit definitions vs the actual TTA-rank-averaged submission score
disagree at exactly the margin APCER@1%BPCER cares about). This does NOT make the gate useless --
see "Gate logic" below for why RELATIVE, same-methodology comparison survives that caveat where
absolute values don't, and why the gate additionally requires PASSING IN EVERY IMPUTATION SCHEME
(not just the headline one) before it trusts a relative comparison at all.

Gate logic: a candidate PASSES only if its expected FREUID is <= the reference's in ALL THREE
imputation schemes (blended, zone-imputed-only, verdict-only) -- the "pessimistic band" the task
asked for. Passing in only some schemes is reported as a WARN, not a PASS: on its own that's
exactly the kind of scheme-dependent result Prompt 1/2 already showed can mean "this pipeline's
own methodology, not the model" is driving the difference, not necessarily "avoid this
candidate" -- read the per-scheme breakdown before deciding, don't just read the verdict.

Usage:
    python scripts/analysis/diff_gate.py --candidate submissions/photosub_v0.csv \\
        --reference submissions/finetune_v0.csv

No training, no new submissions. Pure CPU/pandas + the vendored scorer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, df_to_md  # noqa: E402
from movement_census import (  # noqa: E402
    DEFAULT_CENSUS_CSV,
    DEFAULT_DEEP_REVIEW_CSV,
    DEFAULT_REVIEW_CSV,
    SOFT_BAND_PCT,
    discordant_mass_below_above,
    load_human_verdicts,
    load_submission_rank,
    load_zones,
)
from official_score_reconciliation import (  # noqa: E402
    IMPUTATION_SCHEMES,
    MC_BASE_SEED,
    N_MC_DRAWS,
    PLACEHOLDER_FRAUD_RATE,
    TOTAL_TEST_ROWS,
    build_imputation_schemes,
    load_placeholder_score,
    mc_expected_metrics,
)

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = OUT_DIR / "diff_gate_out"
DEFAULT_REFERENCE = REPO_ROOT / "submissions" / "finetune_v0.csv"


# ---------------------------------------------------------------------------
# Generic crossing-count / bucket logic (parameterized column names -- the
# movement_census.py originals hardcode "finetune"/"photosub", these don't).
# ---------------------------------------------------------------------------

def build_buckets_generic(
    df: pd.DataFrame, ref_col: str, cand_col: str,
    ceiling_lower_edge: float, soft_band: float = SOFT_BAND_PCT,
) -> pd.DataFrame:
    out = df.copy()
    out["rank_delta"] = out[f"{cand_col}_pct_rank"] - out[f"{ref_col}_pct_rank"]
    out["new_fp_bucket"] = None
    out["new_drop_bucket"] = None

    floor_pop = (out["zone"] == "floor") & (out["verdict"] != "F")
    full_cross_fp = floor_pop & (out[f"{cand_col}_pct_rank"] >= ceiling_lower_edge)
    soft_fp = floor_pop & ~full_cross_fp & (out["rank_delta"] >= soft_band)
    out.loc[full_cross_fp, "new_fp_bucket"] = "full"
    out.loc[soft_fp, "new_fp_bucket"] = "soft"

    ceiling_pop = (out["zone"] == "ceiling") & (out["verdict"] != "B")
    full_cross_drop = ceiling_pop & (out[f"{cand_col}_pct_rank"] < ceiling_lower_edge)
    soft_drop = ceiling_pop & ~full_cross_drop & (out["rank_delta"] <= -soft_band)
    out.loc[full_cross_drop, "new_drop_bucket"] = "full"
    out.loc[soft_drop, "new_drop_bucket"] = "soft"

    out["recovered_deep_miss"] = (
        (out["verdict"] == "F") & (out["zone"] != "ceiling") & (out["rank_delta"] > 0)
    )
    out["corrected_false_positive"] = (
        (out["verdict"] == "B") & (out["zone"] == "ceiling") & (out["rank_delta"] < 0)
    )
    return out


def compute_pair_costs_generic(df: pd.DataFrame, ref_col: str, cand_col: str) -> pd.DataFrame:
    out = df.copy()
    ref_mass = discordant_mass_below_above(out, f"{ref_col}_pct_rank")
    cand_mass = discordant_mass_below_above(out, f"{cand_col}_pct_rank")
    out = out.join(ref_mass).join(cand_mass)

    frauds_below_cand = out[f"frauds_below_{cand_col}_pct_rank"]
    frauds_below_ref = out[f"frauds_below_{ref_col}_pct_rank"]
    out["fp_pair_cost"] = np.where(
        out["new_fp_bucket"].notna(),
        out["p_bonafide"] * (frauds_below_cand - frauds_below_ref),
        0.0,
    )
    bonafides_above_cand = out[f"bonafides_above_{cand_col}_pct_rank"]
    bonafides_above_ref = out[f"bonafides_above_{ref_col}_pct_rank"]
    out["drop_pair_cost"] = np.where(
        out["new_drop_bucket"].notna(),
        out["p_fraud"] * (bonafides_above_cand - bonafides_above_ref),
        0.0,
    )
    return out


def crossing_counts(df: pd.DataFrame, ref_col: str, cand_col: str) -> dict:
    ceiling_lower_edge = float(df.loc[df["zone"] == "ceiling", f"{ref_col}_pct_rank"].min())
    bucketed = build_buckets_generic(df, ref_col, cand_col, ceiling_lower_edge)
    costed = compute_pair_costs_generic(bucketed, ref_col, cand_col)
    return {
        "ceiling_lower_edge": ceiling_lower_edge,
        "n_new_fp_full": int((costed["new_fp_bucket"] == "full").sum()),
        "n_new_fp_soft": int((costed["new_fp_bucket"] == "soft").sum()),
        "n_new_drop_full": int((costed["new_drop_bucket"] == "full").sum()),
        "n_new_drop_soft": int((costed["new_drop_bucket"] == "soft").sum()),
        "n_recovered_deep_miss": int(costed["recovered_deep_miss"].sum()),
        "n_corrected_false_positive": int(costed["corrected_false_positive"].sum()),
        "total_fp_pair_cost": float(costed["fp_pair_cost"].sum()),
        "total_drop_pair_cost": float(costed["drop_pair_cost"].sum()),
        "df": costed,
    }


# ---------------------------------------------------------------------------
# Data assembly + MC pricing
# ---------------------------------------------------------------------------

def build_pair_df(candidate_path: Path, reference_path: Path) -> tuple[pd.DataFrame, dict]:
    zones, floor_mode, ceiling_mode = load_zones(DEFAULT_CENSUS_CSV)
    present_ids = set(zones["id"])
    print(f"[diff_gate] {len(present_ids)} present ids | floor_mode={floor_mode:.3f} "
          f"ceiling_mode={ceiling_mode:.3f}")

    verdicts = load_human_verdicts(DEFAULT_REVIEW_CSV, DEFAULT_DEEP_REVIEW_CSV)
    df = zones.merge(verdicts, on="id", how="left")

    placeholder_scores = {}
    for role, path in (("reference", reference_path), ("candidate", candidate_path)):
        sub_scores = load_submission_rank(path, present_ids, role)
        df = df.merge(sub_scores[["id", f"{role}_score", f"{role}_pct_rank"]], on="id", how="left")
        placeholder_scores[role] = load_placeholder_score(path, present_ids)
        print(f"[diff_gate] {role} ({path.name}): placeholder score = {placeholder_scores[role]}")

    return df, placeholder_scores


def price_both(
    df: pd.DataFrame, schemes: dict[str, pd.DataFrame], placeholder_scores: dict,
) -> pd.DataFrame:
    n_placeholder = TOTAL_TEST_ROWS - len(df)
    rows = []
    for scheme_idx, scheme_name in enumerate(IMPUTATION_SCHEMES):
        scheme_df = schemes[scheme_name]
        for role_idx, role in enumerate(("reference", "candidate")):
            r = mc_expected_metrics(
                scheme_df["p_fraud"].to_numpy(), scheme_df[f"{role}_score"].to_numpy(),
                placeholder_scores[role], PLACEHOLDER_FRAUD_RATE, n_placeholder,
                seed=MC_BASE_SEED * 6000 + scheme_idx * 10 + role_idx,
            )
            r["scheme"] = scheme_name
            r["role"] = role
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gate verdict
# ---------------------------------------------------------------------------

def gate_verdict(mc_df: pd.DataFrame, n_draws: int = N_MC_DRAWS, z: float = 2.0) -> dict:
    """PASS only if candidate's expected FREUID is not SIGNIFICANTLY worse than reference's
    (delta > z * standard-error-of-the-delta) in EVERY imputation scheme (the pessimistic band)
    -- WARN if it passes in some but not all, FAIL if it's significantly worse in the headline
    (blended) scheme. See module docstring for why partial agreement is a WARN, not an automatic
    FAIL: scheme-dependence on its own has already been shown (Prompt 1/2) to sometimes reflect
    this pipeline's own methodology rather than the candidate's real quality.

    Uses the STANDARD ERROR of each 200-draw mean (std/sqrt(n_draws)), not a strict delta<=0 --
    a strict inequality fails ~50% of the time on a truly-tied or even IDENTICAL candidate/
    reference pair purely from independent MC sampling noise (verified: comparing a submission
    against itself gave a raw delta<=0 failure in 2/3 schemes before this fix). A tiny negative
    OR positive delta that isn't distinguishable from zero must not drive the verdict.
    """
    per_scheme = []
    for scheme_name in IMPUTATION_SCHEMES:
        sub = mc_df[mc_df["scheme"] == scheme_name]
        ref = sub[sub["role"] == "reference"].iloc[0]
        cand = sub[sub["role"] == "candidate"].iloc[0]
        delta = cand["freuid_mean"] - ref["freuid_mean"]
        delta_se = float(np.sqrt((ref["freuid_std"] / np.sqrt(n_draws)) ** 2 +
                                  (cand["freuid_std"] / np.sqrt(n_draws)) ** 2))
        per_scheme.append({
            "scheme": scheme_name, "reference_freuid": ref["freuid_mean"],
            "candidate_freuid": cand["freuid_mean"], "delta": delta, "delta_se": delta_se,
            "delta_z": delta / delta_se if delta_se > 0 else 0.0,
            "candidate_not_worse": bool(delta <= z * delta_se),
        })
    per_scheme_df = pd.DataFrame(per_scheme)
    n_pass = int(per_scheme_df["candidate_not_worse"].sum())
    if n_pass == len(per_scheme_df):
        verdict = "PASS"
    elif per_scheme_df.loc[per_scheme_df["scheme"] == "blended", "candidate_not_worse"].iloc[0]:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    return {
        "verdict": verdict, "n_pass": n_pass, "n_schemes": len(per_scheme_df),
        "per_scheme": per_scheme_df,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(candidate_path: Path, reference_path: Path, mc_df: pd.DataFrame,
                  gate: dict, crossing: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Full-corpus diff gate\n")
    lines.append(f"**Candidate**: `{candidate_path}`  \n**Reference**: `{reference_path}`\n")
    lines.append("\n**Scope caveat** (inherited from `official_score_reconciliation.py`): absolute "
                 "expected-FREUID values below do not match observed public-LB scores -- read "
                 "them as RELATIVE (candidate vs reference under one fixed pipeline), never as a "
                 "predicted real LB score.\n")

    lines.append(f"\n## Gate verdict: **{gate['verdict']}**\n")
    lines.append(f"Candidate not-worse-than-reference in {gate['n_pass']}/{gate['n_schemes']} "
                 "imputation schemes.\n")
    lines.append(df_to_md(gate["per_scheme"], float_fmt="{:.5f}"))
    if gate["verdict"] == "PASS":
        lines.append("\n**PASS**: candidate's expected FREUID is not worse than the reference's "
                     "in every imputation scheme (the pessimistic band) -- safe to consider for "
                     "submission on this instrument's own terms.\n")
    elif gate["verdict"] == "WARN":
        lines.append("\n**WARN**: candidate looks not-worse under the headline (blended) scheme "
                     "but disagrees under at least one other scheme -- this is exactly the kind "
                     "of scheme-dependence Prompt 1/2 already showed CAN mean 'pipeline "
                     "methodology,' not 'real candidate quality.' Read the per-scheme table "
                     "above and the crossing counts below before deciding; do not auto-submit "
                     "on a WARN.\n")
    else:
        lines.append("\n**FAIL**: candidate's expected FREUID is worse than the reference's even "
                     "under the headline (blended) scheme -- do not submit without understanding "
                     "why.\n")

    lines.append("\n## Monte Carlo expected metrics (200 draws/cell)\n")
    display = mc_df[["scheme", "role", "audet_mean", "audet_std", "apcer_at_bpcer_mean",
                      "apcer_at_bpcer_std", "freuid_mean", "freuid_std"]]
    lines.append(df_to_md(display, float_fmt="{:.5f}"))

    lines.append("\n## Crossing counts (present-ids-only, movement_census.py convention)\n")
    lines.append(f"Ceiling zone's reference-rank lower edge: "
                 f"{crossing['ceiling_lower_edge']:.2f} pctile\n")
    summary = pd.DataFrame([{
        "new_fp_full": crossing["n_new_fp_full"], "new_fp_soft": crossing["n_new_fp_soft"],
        "new_drop_full": crossing["n_new_drop_full"], "new_drop_soft": crossing["n_new_drop_soft"],
        "recovered_deep_miss": crossing["n_recovered_deep_miss"],
        "corrected_false_positive": crossing["n_corrected_false_positive"],
        "total_fp_pair_cost": crossing["total_fp_pair_cost"],
        "total_drop_pair_cost": crossing["total_drop_pair_cost"],
    }])
    lines.append(df_to_md(summary, float_fmt="{:.2f}"))

    report_path = out_dir / "diff_gate_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[diff_gate] wrote {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, help="Path to candidate submission CSV")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE),
                         help="Path to reference submission CSV")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    candidate_path, reference_path = Path(args.candidate), Path(args.reference)
    out_dir = Path(args.out_dir)

    df, placeholder_scores = build_pair_df(candidate_path, reference_path)
    schemes = build_imputation_schemes(df)

    mc_df = price_both(df, schemes, placeholder_scores)
    for _, row in mc_df.iterrows():
        print(f"[diff_gate] {row['scheme']}/{row['role']}: "
              f"E[FREUID]={row['freuid_mean']:.5f}+-{row['freuid_std']:.5f}")

    gate = gate_verdict(mc_df)
    print(f"[diff_gate] GATE VERDICT: {gate['verdict']} "
          f"({gate['n_pass']}/{gate['n_schemes']} schemes)")

    blended = schemes["blended"]
    crossing = crossing_counts(blended, "reference", "candidate")
    print(f"[diff_gate] crossing: new_fp={crossing['n_new_fp_full']}(full)+"
          f"{crossing['n_new_fp_soft']}(soft) new_drop={crossing['n_new_drop_full']}(full)+"
          f"{crossing['n_new_drop_soft']}(soft)")

    out_dir.mkdir(parents=True, exist_ok=True)
    mc_df.to_csv(out_dir / "mc_grid.csv", index=False)
    gate["per_scheme"].to_csv(out_dir / "gate_per_scheme.csv", index=False)

    write_report(candidate_path, reference_path, mc_df, gate, crossing, out_dir)

    if gate["verdict"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
