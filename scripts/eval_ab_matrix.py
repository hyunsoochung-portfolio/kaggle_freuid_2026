"""Build the recapture A/B evaluation matrix (reports/ab_recapture/README.md's core data).

    python scripts/eval_ab_matrix.py

Scores every available checkpoint variant against every available instrument and writes
reports/ab_recapture/eval_matrix.csv (+ a markdown rendering). Variants and instruments
are each best-effort: a variant whose checkpoint file doesn't exist, or an instrument
whose cache directory doesn't exist, is skipped with a printed note instead of erroring
-- see reports/ab_recapture/README.md for which of these fired.

Variants (checkpoint files):
    arm_a_best   checkpoints/finetune_v0.pt              (control, never retrained)
    arm_a_last   checkpoints/finetune_v0_last.pt          (not retained by arm A's run -- expected absent)
    arm_b_best   checkpoints/ablate_no_recapture.pt
    arm_b_last   checkpoints/ablate_no_recapture_last.pt

Instruments:
    clean_val    each variant's own held-out val split (sanity check, not the verdict)
    probe_v1     recapture_transforms applied to val ids, same instrument arm A trained
                 against (circular in arm A's favor -- descriptive context only)
    probe_v2_*   scripts/build_probe_v2.py's independent degradation chain, 3 severities

All four variants are expected to share the same val split (same data_dir/seed/
val_fraction across finetune_v0.yaml and ablate_no_recapture.yaml) -- this is asserted,
not assumed.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
from common import (  # noqa: E402
    REPO_ROOT,
    build_finetuned_model,
    device_and_seed,
    eval_transform,
    get_split_ids,
    load_checkpoint,
    score_paths,
    split_dataframe,
)
from freuid.augment import recapture_transforms  # noqa: E402
from freuid.metrics import evaluate  # noqa: E402
from freuid.transforms import resolve_data_config  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports" / "ab_recapture"
CKPT_DIR = REPO_ROOT / "checkpoints"
PROBE_V2_DIR = REPO_ROOT / "data" / "processed" / "probe_v2"
REAL_PRINT_DIR = REPO_ROOT / "data" / "processed" / "real_print_set"
SEVERITIES = ["mild", "default", "harsh"]

VARIANTS: dict[str, Path] = {
    "arm_a_best": CKPT_DIR / "finetune_v0.pt",
    "arm_a_last": CKPT_DIR / "finetune_v0_last.pt",
    "arm_b_best": CKPT_DIR / "ablate_no_recapture.pt",
    "arm_b_last": CKPT_DIR / "ablate_no_recapture_last.pt",
}


def _score_probe_v1(model, cfg, val_ids, device) -> dict[str, float]:
    df = split_dataframe(cfg, val_ids)
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    tf = recapture_transforms(data_cfg["image_size"], data_cfg["mean"], data_cfg["std"])
    seed = cfg.extra.get("recapture_probe_seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    scores = score_paths(model, df["path"].tolist(), tf, device, batch_size=cfg.batch_size)
    return evaluate(scores, df["label"].to_numpy())


def _score_probe_v2(model, cfg, val_ids, device, severity: str) -> dict[str, float] | None:
    sev_dir = PROBE_V2_DIR / severity
    if not sev_dir.exists():
        return None
    df = split_dataframe(cfg, val_ids)
    df = df[df["id"].map(lambda i: (sev_dir / f"{i}.png").exists())]
    if df.empty:
        return None
    paths = [sev_dir / f"{i}.png" for i in df["id"]]
    tf, _ = eval_transform(cfg)  # plain resize+normalize -- degradation is already baked into the png
    scores = score_paths(model, paths, tf, device, batch_size=cfg.batch_size)
    return evaluate(scores, df["label"].to_numpy())


def _score_clean_val(model, cfg, val_ids, device) -> dict[str, float]:
    df = split_dataframe(cfg, val_ids)
    tf, _ = eval_transform(cfg)
    scores = score_paths(model, df["path"].tolist(), tf, device, batch_size=cfg.batch_size)
    return evaluate(scores, df["label"].to_numpy())


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    reference_val_ids: set[str] | None = None

    if REAL_PRINT_DIR.exists():
        print(f"[eval_ab_matrix] real_print_set found at {REAL_PRINT_DIR} -- add scoring for it")
    else:
        print(f"[eval_ab_matrix] real_print_set NOT found at {REAL_PRINT_DIR} -- skipping 4th column")

    for variant, ckpt_path in VARIANTS.items():
        if not ckpt_path.exists():
            print(f"[eval_ab_matrix] {variant}: {ckpt_path} not found -- skipping")
            continue

        cfg, state = load_checkpoint(ckpt_path)
        device = device_and_seed(cfg)
        model = build_finetuned_model(cfg, state, device)
        _, val_ids = get_split_ids(cfg)

        if reference_val_ids is None:
            reference_val_ids = val_ids
        elif val_ids != reference_val_ids:
            raise SystemExit(
                f"[eval_ab_matrix] {variant}'s val split differs from the reference "
                "(data_dir/seed/val_fraction mismatch) -- matrix would compare apples to oranges"
            )

        ckpt_epoch = state.get("epoch", "?")
        print(f"[eval_ab_matrix] {variant}: {ckpt_path.name} (epoch={ckpt_epoch}, n_val={len(val_ids)})")

        clean_m = _score_clean_val(model, cfg, val_ids, device)
        rows.append({"variant": variant, "instrument": "clean_val", **clean_m})
        print(f"    clean_val   AuDET={clean_m['audet']:.6f} APCER@1%BPCER={clean_m['apcer_at_1pct_bpcer']:.6f}")

        probe1_m = _score_probe_v1(model, cfg, val_ids, device)
        rows.append({"variant": variant, "instrument": "probe_v1", **probe1_m})
        print(f"    probe_v1    AuDET={probe1_m['audet']:.6f} APCER@1%BPCER={probe1_m['apcer_at_1pct_bpcer']:.6f}")

        for severity in SEVERITIES:
            m = _score_probe_v2(model, cfg, val_ids, device, severity)
            if m is None:
                print(f"    probe_v2_{severity}: cache not found -- skipping")
                continue
            rows.append({"variant": variant, "instrument": f"probe_v2_{severity}", **m})
            print(f"    probe_v2_{severity:<8s} AuDET={m['audet']:.6f} APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.6f}")

    if not rows:
        raise SystemExit("[eval_ab_matrix] no variants found -- nothing to write")

    result = pd.DataFrame(rows)
    out_csv = REPORT_DIR / "eval_matrix.csv"
    result.to_csv(out_csv, index=False)
    print(f"\n[eval_ab_matrix] wrote {out_csv}")

    pivot = result.pivot(index="variant", columns="instrument", values="audet")
    print("\n=== AuDET matrix (lower is better) ===")
    print(pivot.to_string(float_format=lambda v: f"{v:.6f}"))
    pivot.to_csv(REPORT_DIR / "eval_matrix_audet_pivot.csv")


if __name__ == "__main__":
    main()
