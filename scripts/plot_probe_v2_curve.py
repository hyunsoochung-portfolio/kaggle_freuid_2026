"""Robustness curve: AuDET on probe_v2 across severities, for both arms' best checkpoints.

    python scripts/plot_probe_v2_curve.py

Reads reports/ab_recapture/eval_matrix.csv (written by scripts/eval_ab_matrix.py) and plots
AuDET vs. probe_v2 severity (mild/default/harsh) for arm_a_best and arm_b_best (arm_b_last
included as a lighter reference line). Saves reports/ab_recapture/probe_v2_robustness_curve.png.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "ab_recapture"
SEVERITY_ORDER = ["mild", "default", "harsh"]
LINES = {
    "arm_a_best": dict(color="#CC3311", marker="o", linewidth=2.5, label="arm A best (recapture, control)"),
    "arm_b_best": dict(color="#4477AA", marker="s", linewidth=2.5, label="arm B best (no recapture)"),
    "arm_b_last": dict(color="#4477AA", marker="s", linewidth=1.2, linestyle="--", alpha=0.5,
                        label="arm B last epoch (reference)"),
}


def main() -> None:
    csv_path = REPORT_DIR / "eval_matrix.csv"
    df = pd.read_csv(csv_path)
    df = df[df["instrument"].str.startswith("probe_v2_")]
    df["severity"] = df["instrument"].str.replace("probe_v2_", "", regex=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_probe_v2_curve] matplotlib not available -- skipped")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for variant, style in LINES.items():
        sub = df[df["variant"] == variant].set_index("severity").reindex(SEVERITY_ORDER)
        if sub["audet"].isna().all():
            print(f"[plot_probe_v2_curve] {variant}: no probe_v2 rows found -- skipping line")
            continue
        ax.plot(SEVERITY_ORDER, sub["audet"], **style)

    ax.set_yscale("log")
    ax.set_ylabel("AuDET on probe_v2 (log scale, lower = better)")
    ax.set_xlabel("probe_v2 severity")
    ax.set_title("Recapture A/B: robustness to an independent degradation instrument")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = REPORT_DIR / "probe_v2_robustness_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"[plot_probe_v2_curve] wrote {out_path}")


if __name__ == "__main__":
    main()
