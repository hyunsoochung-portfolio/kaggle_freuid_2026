"""Per-slice diagnostic metrics for the finetune_v0 checkpoint.

Scores finetune_v0's own training-time validation split (plain, undegraded transform --
matches what train.py's val_loader used, NOT the recapture probe) and reports AuDET /
APCER@1%BPCER overall, per document type, and per is_digital.

Also states explicitly whether val positives are circular (produced by SynthTamperWrapper)
by construction: train.py's build_loaders only wraps train_ds in SynthTamperWrapper when
synth_tamper_prob > 0 -- val_ds is always a plain FreuidDataset built straight from
train_labels.csv, so every val positive is a genuine, dataset-labeled fraud sample, never a
synthetic tamper edit. This script does not re-derive that from runtime behavior (there is
no per-sample "was this synthetic" flag once written to the val split -- the wrapper info
is structural/architectural), so this is a documented code-reading fact, not a measurement.

Also flags that train_labels.csv has no attack/fraud-type column (only id, image_path,
label, is_digital, type) -- so a true per-attack-type slice (physical tamper / GenAI edit /
print-capture) is not derivable from metadata. is_digital is reported as the closest
available proxy for attack channel (non-digital implies the sample has already been through
some capture/print channel).

Usage: python scripts/analysis/per_slice_metrics.py [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    build_finetuned_model,
    device_and_seed,
    df_to_md,
    ensure_report_dir,
    eval_transform,
    get_split_ids,
    load_checkpoint,
    score_paths,
    split_dataframe,
)
from freuid.metrics import evaluate  # noqa: E402

VAL_CIRCULARITY_NOTE = (
    "val positives are NOT synthetic: train.py's build_loaders() only wraps train_ds in "
    "SynthTamperWrapper (synth_tamper_prob > 0); val_ds is always a plain FreuidDataset "
    "built directly from train_labels.csv. Every label=1 val sample is a genuine, "
    "dataset-provided fraud example -- val is not circular with respect to synth_tamper."
)
ATTACK_TYPE_NOTE = (
    "train_labels.csv has no attack/fraud-type column (columns: id, image_path, label, "
    "is_digital, type) -- a true per-attack-type (physical tamper / GenAI edit / "
    "print-capture) slice is not derivable from metadata alone. is_digital is reported "
    "as the closest available proxy for attack channel."
)


def _slice_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(group_col):
        labels = group["label"].to_numpy()
        if len(set(labels)) < 2:
            rows.append({group_col: key, "n": len(group), "n_fraud": int(labels.sum()),
                         "audet": float("nan"), "apcer_at_1pct_bpcer": float("nan"), "freuid": float("nan"),
                         "note": "single-class slice, AuDET/APCER/FREUID undefined"})
            continue
        m = evaluate(group["score"].to_numpy(), labels)
        rows.append({group_col: key, "n": len(group), "n_fraud": int(labels.sum()),
                     "audet": m["audet"], "apcer_at_1pct_bpcer": m["apcer_at_1pct_bpcer"], "freuid": m["freuid"],
                     "note": ""})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    from common import DEFAULT_CHECKPOINT
    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT

    cfg, state = load_checkpoint(ckpt_path)
    device = device_and_seed(cfg)
    print(f"[per_slice] checkpoint={ckpt_path} backbone={cfg.backbone} device={device}")

    train_ids, val_ids = get_split_ids(cfg)
    df = split_dataframe(cfg, val_ids)
    print(f"[per_slice] val n={len(df)} (train n={len(train_ids)}) fraud_rate={df['label'].mean():.4f}")

    model = build_finetuned_model(cfg, state, device)
    transform, data_cfg = eval_transform(cfg)
    print(f"[per_slice] scoring at image_size={data_cfg['image_size']} (single pass, no TTA)")

    scores = score_paths(model, df["path"].tolist(), transform, device, batch_size=32)
    df = df.copy()
    df["score"] = scores

    out_dir = ensure_report_dir()
    df[["id", "type", "is_digital", "label", "score"]].to_csv(out_dir / "per_slice_scores.csv", index=False)

    overall = evaluate(df["score"].to_numpy(), df["label"].to_numpy())
    by_type = _slice_metrics(df, "type")
    by_digital = _slice_metrics(df, "is_digital")

    by_type.to_csv(out_dir / "per_slice_by_type.csv", index=False)
    by_digital.to_csv(out_dir / "per_slice_by_is_digital.csv", index=False)

    summary_path = out_dir / "per_slice_summary.md"
    lines = [
        "# Per-slice metrics -- finetune_v0\n",
        f"Val set: n={len(df)}, fraud_rate={df['label'].mean():.4f} "
        f"(plain eval transform, image_size={data_cfg['image_size']}, no TTA, no recapture degradation)\n",
        f"**Overall**: AuDET={overall['audet']:.6f}  APCER@1%BPCER={overall['apcer_at_1pct_bpcer']:.6f}  "
        f"FREUID={overall['freuid']:.6f}\n",
        f"> **Val circularity**: {VAL_CIRCULARITY_NOTE}\n",
        f"> **Attack-type slicing**: {ATTACK_TYPE_NOTE}\n",
        "## By document type\n",
        df_to_md(by_type),
        "\n## By is_digital (attack-channel proxy)\n",
        df_to_md(by_digital),
        "\n",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[per_slice] wrote {out_dir / 'per_slice_scores.csv'}, per_slice_by_type.csv, "
          f"per_slice_by_is_digital.csv, per_slice_summary.md")
    print(f"[per_slice] overall AuDET={overall['audet']:.6f} APCER@1%BPCER={overall['apcer_at_1pct_bpcer']:.6f} "
          f"FREUID={overall['freuid']:.6f}")


if __name__ == "__main__":
    main()
