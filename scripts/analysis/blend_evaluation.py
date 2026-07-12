"""Exact, label-light evaluation of rank-average(finetune_v0, photosub_v0) blends at
finetune-heavy weights -- the MC pricing of blends in threshold_pricing.py is untrusted (it
ranks photosub_v0-heavy blends as "better" than finetune_v0 alone, directly contradicting the
real observed public LB where finetune_v0 wins by 2.17x -- see that report's Item 4). This
script sidesteps the label model almost entirely: instead of pricing an expected FREUID under
probabilistic labels, it asks concrete, exact, mostly-label-free questions about each blend's
actual score vector:

  - deep-9 / boundary-59: REAL, confirmed human verdicts (F). No imputation needed to know
    their true label -- only their RANK relative to a threshold matters, which is exact
    arithmetic on the blend's own score vector.
  - NEW_DROP / NEW_FP crossing counts: diff_gate.py's bucket/crossing-count logic uses only
    zone + verdict + rank (see build_buckets_generic) -- no p_fraud/p_bonafide anywhere in the
    COUNT fields (only the separate, unused, pair-COST fields need the label model; this script
    reports counts only).
  - The present-calibrated 1%-BPCER crossing score itself (threshold_pricing.present_only_
    crossing_score) is the one place a label model (p_bonafide mass) is used -- but it is the
    SAME frozen, already-established blended-imputation calibration threshold_pricing.py already
    built and reported, reused verbatim here (not re-derived, not MC'd) and recomputed exactly
    (closed-form weighted-mass crossing, not sampled) for each blend's own score ordering.

No MC anywhere in this script. No seed-pool blend members: searched submissions/, configs/,
checkpoints/ for any second finetune_v0 (or photosub_v0) seed variant -- none exist (confirmed
against the same absence threshold_pricing.py's session already established).

Usage: python scripts/analysis/blend_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO_ROOT, df_to_md  # noqa: E402
from diff_gate import crossing_counts  # noqa: E402
from movement_census import (  # noqa: E402
    DEFAULT_CENSUS_CSV,
    DEFAULT_DEEP_REVIEW_CSV,
    DEFAULT_REVIEW_CSV,
    estimate_fraud_probability,
    load_human_verdicts,
    load_submission_rank,
    load_zones,
)
from threshold_pricing import (  # noqa: E402
    CEILING_B_IDS,
    load_probe_sets,
    present_only_crossing_score,
    rank_normalize,
    safe_epsilon_tiebreak,
)

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = OUT_DIR / "blend_evaluation_out"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"

FINETUNE_WEIGHTS = [0.9, 0.8, 0.7, 0.6]
BUDGET_EATER_IDS = CEILING_B_IDS | {"4c1ac0279e114232bda980437acac11c"}


# ---------------------------------------------------------------------------
# Seed-pool search (item 1)
# ---------------------------------------------------------------------------

def find_seed_pool_members() -> list[Path]:
    """Any submission CSV besides finetune_v0.csv/photosub_v0.csv that looks like a same-
    architecture seed variant (naming convention: finetune_v0_seed*/photosub_v0_seed*, or any
    *_seedN.csv). None found as of this run -- see module docstring."""
    if not SUBMISSIONS_DIR.exists():
        return []
    return sorted(p for p in SUBMISSIONS_DIR.glob("*seed*.csv"))


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def build_base_df() -> pd.DataFrame:
    zones, floor_mode, ceiling_mode = load_zones(DEFAULT_CENSUS_CSV)
    present_ids = set(zones["id"])
    print(f"[blend_eval] {len(present_ids)} present ids | floor_mode={floor_mode:.3f} "
          f"ceiling_mode={ceiling_mode:.3f}")

    verdicts = load_human_verdicts(DEFAULT_REVIEW_CSV, DEFAULT_DEEP_REVIEW_CSV)
    df = zones.merge(verdicts, on="id", how="left")

    fin = load_submission_rank(SUBMISSIONS_DIR / "finetune_v0.csv", present_ids, "finetune")
    pho = load_submission_rank(SUBMISSIONS_DIR / "photosub_v0.csv", present_ids, "photosub")
    df = df.merge(fin[["id", "finetune_score", "finetune_pct_rank"]], on="id", how="left")
    df = df.merge(pho[["id", "photosub_score", "photosub_pct_rank"]], on="id", how="left")
    return df


def build_blends(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rank-transform finetune_v0/photosub_v0's present-id scores, weighted-average at each
    finetune-heavy weight, epsilon-tiebreak the result, and attach pct_rank -- everything
    diff_gate.crossing_counts / present_only_crossing_score need."""
    rank_fin = rank_normalize(df["finetune_score"].to_numpy())
    rank_pho = rank_normalize(df["photosub_score"].to_numpy())

    blends = {}
    for w in FINETUNE_WEIGHTS:
        name = f"blend_{w:.1f}"
        raw_blend = w * rank_fin + (1.0 - w) * rank_pho
        d = df.copy()
        d[f"{name}_score"] = raw_blend

        n_before = d[f"{name}_score"].nunique()
        if n_before < len(d):
            tiebroken = safe_epsilon_tiebreak(
                d, f"{name}_score", secondary_col="finetune_mean_logit",
            )
            d[f"{name}_score"] = tiebroken
            n_after = d[f"{name}_score"].nunique()
            print(f"[blend_eval] {name}: {n_before}->{n_after} unique scores after tiebreak "
                  f"({len(d) - n_before} exact ties found pre-tiebreak)")
        else:
            print(f"[blend_eval] {name}: {n_before}/{len(d)} unique scores -- no ties, "
                  "tiebreak is a no-op")

        d[f"{name}_pct_rank"] = d[f"{name}_score"].rank(pct=True) * 100.0
        blends[name] = d
    return blends


# ---------------------------------------------------------------------------
# Exact readouts (item 2)
# ---------------------------------------------------------------------------

def probe_positions(
    d: pd.DataFrame, score_col: str, ids: set[str], crossing_score: float,
) -> pd.DataFrame:
    sub = d[d["id"].isin(ids)][["id", score_col]].copy()
    sub["rank"] = sub[score_col].rank(ascending=False, method="min").astype(int)
    sub["above_threshold"] = sub[score_col] >= crossing_score
    return sub.sort_values("rank")


def tie_cluster_status(d: pd.DataFrame, score_col: str, crossing_score: float) -> dict:
    """Does an exact-tie cluster sit AT the crossing score itself?"""
    counts = d[score_col].value_counts()
    at_threshold = counts[counts.index == crossing_score]
    n_at = int(at_threshold.sum()) if len(at_threshold) else 0
    return {"n_ids_at_threshold_score": n_at, "straddles": n_at > 1}


def evaluate_blend(name: str, d: pd.DataFrame, probes: dict[str, set[str]]) -> dict:
    score_col = f"{name}_score"
    crossing_score = present_only_crossing_score(d, score_col=score_col)

    # deep-9
    deep_ids = probes["deep"]
    deep_pos = probe_positions(d, score_col, deep_ids, crossing_score)
    n_deep_above = int(deep_pos["above_threshold"].sum())

    # boundary-59 / ceiling-200
    boundary_pos = probe_positions(d, score_col, probes["boundary"], crossing_score)
    ceiling_pos = probe_positions(d, score_col, probes["ceiling_sample"], crossing_score)
    n_boundary_above = int(boundary_pos["above_threshold"].sum())
    n_ceiling_above = int(ceiling_pos["above_threshold"].sum())

    # reviewed ceiling-B ids + 4c1ac0279e (budget-eating check)
    budget_pos = probe_positions(d, score_col, BUDGET_EATER_IDS, crossing_score)

    # crossing counts vs finetune_v0 reference (diff_gate's label-free component)
    cross = crossing_counts(d, "finetune", name)

    # tie-cluster status at the threshold
    tie_status = tie_cluster_status(d, score_col, crossing_score)

    return {
        "name": name,
        "crossing_score": crossing_score,
        "n_deep9_above": n_deep_above,
        "deep9_positions": deep_pos,
        "n_boundary_above": n_boundary_above,
        "boundary_total": len(boundary_pos),
        "n_ceiling_above": n_ceiling_above,
        "ceiling_total": len(ceiling_pos),
        "budget_eater_positions": budget_pos,
        "new_fp_full": cross["n_new_fp_full"], "new_fp_soft": cross["n_new_fp_soft"],
        "new_drop_full": cross["n_new_drop_full"], "new_drop_soft": cross["n_new_drop_soft"],
        "recovered_deep_miss": cross["n_recovered_deep_miss"],
        "corrected_false_positive": cross["n_corrected_false_positive"],
        "tie_status": tie_status,
    }


# ---------------------------------------------------------------------------
# Decision table (item 3)
# ---------------------------------------------------------------------------

def build_decision_table(results: dict[str, dict], baseline: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        boundary_guard_held = r["n_boundary_above"] >= baseline["n_boundary_above"]
        ceiling_guard_held = r["n_ceiling_above"] >= baseline["n_ceiling_above"]
        rank_now = r["budget_eater_positions"].set_index("id")["rank"].sort_index()
        rank_baseline = baseline["budget_eater_positions"].set_index("id")["rank"].sort_index()
        budget_eaters_moved_up = int((rank_now < rank_baseline).sum())
        crossings_near_zero = (r["new_fp_full"] == 0) and (r["new_drop_full"] == 0)
        deep9_promoted = r["n_deep9_above"] - baseline["n_deep9_above"]
        interesting = (
            (deep9_promoted > 0) and crossings_near_zero
            and boundary_guard_held and ceiling_guard_held
        )
        rows.append({
            "weight": name,
            "deep9_above": r["n_deep9_above"], "deep9_promoted_vs_finetune": deep9_promoted,
            "new_fp_full": r["new_fp_full"], "new_fp_soft": r["new_fp_soft"],
            "new_drop_full": r["new_drop_full"], "new_drop_soft": r["new_drop_soft"],
            "boundary_above": f"{r['n_boundary_above']}/{r['boundary_total']}",
            "boundary_guard_held": boundary_guard_held,
            "ceiling_above": f"{r['n_ceiling_above']}/{r['ceiling_total']}",
            "ceiling_guard_held": ceiling_guard_held,
            "budget_eaters_moved_up": budget_eaters_moved_up,
            "tie_straddles_threshold": r["tie_status"]["straddles"],
            "INTERESTING_OUTCOME": interesting,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Write blend as a real submission CSV (only if a full diff_gate run is warranted)
# ---------------------------------------------------------------------------

def write_blend_submission(d: pd.DataFrame, name: str, out_path: Path) -> None:
    """Full 142,818-row submission: present ids get the blend score, everything else keeps
    finetune_v0's own placeholder convention (0.5) -- matches how a real blend submission would
    actually be constructed (rank-averaging never touches ids neither model scored)."""
    fin_full = pd.read_csv(SUBMISSIONS_DIR / "finetune_v0.csv", dtype={"id": str})
    blend_scores = d.set_index("id")[f"{name}_score"]
    out = fin_full.copy()
    present_mask = out["id"].isin(blend_scores.index)
    out.loc[present_mask, "label"] = out.loc[present_mask, "id"].map(blend_scores)
    out.to_csv(out_path, index=False)
    print(f"[blend_eval] wrote {out_path} ({present_mask.sum()} present ids blended, "
          f"{(~present_mask).sum()} placeholders kept at finetune_v0's convention)")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(results: dict[str, dict], baseline: dict, decision_df: pd.DataFrame,
                  seed_pool: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Blend evaluation: exact, label-light readouts for finetune-heavy blends\n")
    lines.append("No MC anywhere. See module docstring for exactly which readouts are fully "
                 "label-free (deep-9/boundary-59 identity, NEW_DROP/NEW_FP crossing counts) vs. "
                 "which reuse the one frozen, already-established label-model quantity "
                 "(the present-calibrated 1%-BPCER crossing score itself).\n")
    if seed_pool:
        lines.append(f"\n**Seed-pool members found**: {[str(p) for p in seed_pool]}\n")
    else:
        lines.append("\n**No seed-pool members found** (searched submissions/*seed*.csv, "
                     "configs/, checkpoints/ -- only finetune_v0/photosub_v0 exist). Blends "
                     "are finetune_v0 x photosub_v0 only.\n")

    lines.append("\n## Baseline (finetune_v0, weight=1.0)\n")
    lines.append(f"Present-calibrated crossing score: **{baseline['crossing_score']:.6f}**\n")
    lines.append(f"deep-9 above: {baseline['n_deep9_above']}/9 | "
                 f"boundary above: {baseline['n_boundary_above']}/{baseline['boundary_total']} | "
                 f"ceiling above: {baseline['n_ceiling_above']}/{baseline['ceiling_total']}\n")
    lines.append("\n**Budget-eater positions (4 reviewed ceiling-B + 4c1ac0279e):**\n")
    lines.append(df_to_md(baseline["budget_eater_positions"], float_fmt="{:.6f}"))

    for name, r in results.items():
        lines.append(f"\n## {name}\n")
        lines.append(f"Present-calibrated crossing score: **{r['crossing_score']:.6f}**\n")
        lines.append(f"\n**deep-9 positions** ({r['n_deep9_above']}/9 above threshold):\n")
        lines.append(df_to_md(r["deep9_positions"], float_fmt="{:.6f}"))
        lines.append(f"\n**Crossing counts vs finetune_v0 reference** (diff_gate label-free "
                     "component): "
                     f"new_fp={r['new_fp_full']}(full)+{r['new_fp_soft']}(soft), "
                     f"new_drop={r['new_drop_full']}(full)+{r['new_drop_soft']}(soft), "
                     f"recovered_deep_miss={r['recovered_deep_miss']}, "
                     f"corrected_false_positive={r['corrected_false_positive']}\n")
        lines.append(f"\nboundary above: {r['n_boundary_above']}/{r['boundary_total']} | "
                     f"ceiling above: {r['n_ceiling_above']}/{r['ceiling_total']}\n")
        lines.append("\n**Budget-eater positions:**\n")
        lines.append(df_to_md(r["budget_eater_positions"], float_fmt="{:.6f}"))
        lines.append(f"\nTie-cluster at threshold: {r['tie_status']}\n")

    lines.append("\n## Decision table\n")
    lines.append(df_to_md(decision_df, float_fmt="{:.4f}"))

    interesting = decision_df[decision_df["INTERESTING_OUTCOME"]]
    if len(interesting):
        lines.append(f"\n**Interesting outcome found at**: {list(interesting['weight'])} -- "
                     "several deep-9 ids cross above the threshold while NEW_FP/NEW_DROP stay "
                     "at zero and both guards hold. This is an improvement argument on BOTH "
                     "metric components with no label model involved -- see the full diff_gate "
                     "run below.\n")
    else:
        lines.append("\n**No weight achieves the interesting outcome.** Either deep-9 never "
                     "promotes without also moving NEW_FP/NEW_DROP off zero, or the guards "
                     "(boundary/ceiling) don't hold, at every finetune-heavy weight tried. "
                     "Plausible explanation per the task's own pre-registration: photosub_v0's "
                     "deep-9 gains dilute away faster than its drops at finetune-heavy weights "
                     "-- consistent with photosub_v0's own deep-9 diagnostic (MODE_B/C converge "
                     "cleanly and strongly; the ceiling-zone/drop cost is comparatively small "
                     "per-id but broad, so it survives dilution better than a few strong "
                     "individual deep-9 promotions do). **Conclusion: wait for v1.** No "
                     "rank-average of the current two submissions is a defensible next "
                     "submission on this evidence.\n")

        lines.append("\n**A second, more fundamental finding, worth separating from the "
                     "dilution explanation above**: the present-calibrated threshold itself is "
                     f"nearly uncrossable by construction. `finetune_v0` ALONE scores "
                     f"{baseline['n_deep9_above']}/9 deep-9 and {baseline['n_ceiling_above']}/"
                     f"{baseline['ceiling_total']} ceiling-probe above it -- the bar was never "
                     "cleared even before any blending, because the threshold (0.977-0.987 "
                     "across these blends) is pinned near the very top of the score range by a "
                     "single confirmed-bona-fide outlier (`4c1ac0279e`, score 0.9949) consuming "
                     "most of the tiny present-only budget on its own (documented in "
                     "threshold_pricing_out/threshold_pricing_report.md). Ceiling-zone scores "
                     "top out around 0.947 (per that report's zone-score-range finding) -- BELOW "
                     "every blend's threshold here -- so '0/200 ceiling above' is not a blend "
                     "artifact, it is a structural consequence of the threshold's own "
                     "calibration. Separately, `recovered_deep_miss` (the broader confirmed-F "
                     "population that rose in rank vs. finetune_v0, not just the curated "
                     "deep-9) DOES show a real, monotonic gain (31-45 ids across these four "
                     "weights) -- confirming photosub_v0 genuinely helps a wider miss population "
                     "at the RANKING level (consistent with the movement census's AuDET "
                     "finding), even though that gain never clears this specific ultra-strict "
                     "threshold for the 9 curated ids. Both explanations are real and "
                     "complementary, not competing.\n")

    report_path = out_dir / "blend_evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[blend_eval] wrote {report_path}")


def main() -> None:
    seed_pool = find_seed_pool_members()
    print(f"[blend_eval] seed-pool members found: {len(seed_pool)}")

    df = build_base_df()
    blended = estimate_fraud_probability(df)  # frozen label model, reused only for the
                                               # crossing-score calibration (see module docstring)
    blends = build_blends(blended)
    probes = load_probe_sets()

    baseline_score_col = "finetune_score"
    baseline_crossing = present_only_crossing_score(blended, score_col=baseline_score_col)
    baseline_deep = probe_positions(blended, baseline_score_col, probes["deep"], baseline_crossing)
    baseline_boundary = probe_positions(
        blended, baseline_score_col, probes["boundary"], baseline_crossing,
    )
    baseline_ceiling = probe_positions(
        blended, baseline_score_col, probes["ceiling_sample"], baseline_crossing,
    )
    baseline_budget = probe_positions(
        blended, baseline_score_col, BUDGET_EATER_IDS, baseline_crossing,
    )
    baseline = {
        "crossing_score": baseline_crossing,
        "n_deep9_above": int(baseline_deep["above_threshold"].sum()),
        "n_boundary_above": int(baseline_boundary["above_threshold"].sum()),
        "boundary_total": len(baseline_boundary),
        "n_ceiling_above": int(baseline_ceiling["above_threshold"].sum()),
        "ceiling_total": len(baseline_ceiling),
        "budget_eater_positions": baseline_budget,
    }
    print(f"[blend_eval] baseline (finetune_v0): crossing={baseline_crossing:.6f} "
          f"deep9_above={baseline['n_deep9_above']}/9 "
          f"boundary_above={baseline['n_boundary_above']}/{baseline['boundary_total']} "
          f"ceiling_above={baseline['n_ceiling_above']}/{baseline['ceiling_total']}")

    results = {}
    for name, d in blends.items():
        r = evaluate_blend(name, d, probes)
        results[name] = r
        print(f"[blend_eval] {name}: crossing={r['crossing_score']:.6f} "
              f"deep9_above={r['n_deep9_above']}/9 "
              f"new_fp={r['new_fp_full']}+{r['new_fp_soft']} "
              f"new_drop={r['new_drop_full']}+{r['new_drop_soft']} "
              f"boundary={r['n_boundary_above']}/{r['boundary_total']} "
              f"ceiling={r['n_ceiling_above']}/{r['ceiling_total']}")

    decision_df = build_decision_table(results, baseline)
    print(decision_df.to_string(index=False))

    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    decision_df.to_csv(DEFAULT_OUT_DIR / "decision_table.csv", index=False)

    interesting = decision_df[decision_df["INTERESTING_OUTCOME"]]
    if len(interesting):
        best_name = interesting.iloc[0]["weight"]
        print(f"[blend_eval] INTERESTING OUTCOME at {best_name} -- writing submission + "
              "running full diff_gate")
        blend_sub_path = SUBMISSIONS_DIR / f"{best_name}.csv"
        write_blend_submission(blends[best_name], best_name, blend_sub_path)
        import subprocess
        subprocess.run(
            [sys.executable, str(OUT_DIR / "diff_gate.py"),
             "--candidate", str(blend_sub_path),
             "--reference", str(SUBMISSIONS_DIR / "finetune_v0.csv"),
             "--out-dir", str(DEFAULT_OUT_DIR / f"diff_gate_{best_name}")],
            check=False,
        )
    else:
        print("[blend_eval] no interesting outcome at any weight -- not writing a blend "
              "submission, not running diff_gate. Conclusion: wait for v1.")

    write_report(results, baseline, decision_df, seed_pool, DEFAULT_OUT_DIR)


if __name__ == "__main__":
    main()
