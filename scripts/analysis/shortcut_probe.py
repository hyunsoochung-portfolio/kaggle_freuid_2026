"""Metadata-shortcut probe: can trivial, non-forensic image statistics predict fraud?

Extracts per-image features that carry no document-content information -- width, height,
file size in bytes, a hash + mean of the JPEG quantization table, and per-channel pixel
mean/std (computed on a fast low-res JPEG DCT decode, not the full-resolution image) -- for
both the train and val ids of finetune_v0's own split. Fits a LogisticRegression and a
HistGradientBoostingClassifier on TRAIN features/labels (mirroring the real training
regime), and evaluates AuDET on the held-out VAL split.

If either probe scores well below 0.5 AuDET (better than random), the dataset has an
exploitable metadata/statistics shortcut independent of document content, and the real
model (which sees full pixel content) may be partially or wholly riding on the same
correlate rather than genuine forensic/tamper evidence.

Resumable by design (the VESSL workspace's container can silently recycle mid-run -- see
project memory): feature extraction is chunked and appended to
shortcut_features_{train,val}.csv incrementally; ids already present on disk are skipped on
a re-run, so a kill only loses progress since the last completed chunk (default 2000 ids).

Usage:
    python scripts/analysis/shortcut_probe.py [--checkpoint PATH] [--limit N] [--chunk-size N]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_CHECKPOINT, ensure_report_dir, get_split_ids, load_checkpoint, split_dataframe  # noqa: E402
from freuid.metrics import evaluate  # noqa: E402

FEATURE_COLS = [
    "width", "height", "aspect_ratio", "file_size_bytes", "quant_hash", "quant_mean",
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
]


def _quant_features(img: Image.Image) -> tuple[int, float]:
    """(stable hash of quantization tables, mean quant value) -- 0/0.0 if not a JPEG."""
    q = getattr(img, "quantization", None)
    if not q:
        return 0, 0.0
    flat = tuple(v for table in q.values() for v in table)
    h = int(hashlib.md5(str(flat).encode()).hexdigest()[:8], 16) % 100000
    return h, float(np.mean(flat))


def _extract_one(path: str) -> dict:
    p = Path(path)
    file_size = p.stat().st_size
    with Image.open(p) as img:
        width, height = img.size
        quant_hash, quant_mean = _quant_features(img)
        img.draft("RGB", (64, 64))
        small = img.convert("RGB").resize((64, 64))
        arr = np.asarray(small, dtype=np.float32)
    return {
        "width": width, "height": height, "aspect_ratio": width / max(height, 1),
        "file_size_bytes": file_size, "quant_hash": quant_hash, "quant_mean": quant_mean,
        "mean_r": arr[:, :, 0].mean(), "mean_g": arr[:, :, 1].mean(), "mean_b": arr[:, :, 2].mean(),
        "std_r": arr[:, :, 0].std(), "std_g": arr[:, :, 1].std(), "std_b": arr[:, :, 2].std(),
    }


def extract_features_resumable(df: pd.DataFrame, out_csv: Path, chunk_size: int) -> pd.DataFrame:
    """df must have columns id, path, label. Resumes from out_csv if it already has some ids."""
    done_ids: set[str] = set()
    header_written = out_csv.exists()
    if header_written:
        done_ids = set(pd.read_csv(out_csv, usecols=["id"])["id"].astype(str))

    todo = df[~df["id"].astype(str).isin(done_ids)]
    print(f"[shortcut] {out_csv.name}: {len(done_ids)} already extracted, {len(todo)} remaining")

    for start in range(0, len(todo), chunk_size):
        chunk = todo.iloc[start:start + chunk_size]
        rows = [dict(id=row.id, label=row.label, **_extract_one(row.path)) for row in chunk.itertuples(index=False)]
        chunk_df = pd.DataFrame(rows)
        chunk_df.to_csv(out_csv, mode="a", header=not header_written, index=False)
        header_written = True
        print(f"[shortcut] {out_csv.name}: extracted {min(start + chunk_size, len(todo))}/{len(todo)}"
              f" (chunk of {len(chunk)}) -- checkpointed")

    return pd.read_csv(out_csv, dtype={"id": str})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap train ids for speed")
    parser.add_argument("--chunk-size", type=int, default=2000, help="ids per checkpointed extraction chunk")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    cfg, _ = load_checkpoint(ckpt_path)
    train_ids, val_ids = get_split_ids(cfg)

    train_df = split_dataframe(cfg, train_ids)
    val_df = split_dataframe(cfg, val_ids)
    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=cfg.seed).reset_index(drop=True)
    print(f"[shortcut] train n={len(train_df)} val n={len(val_df)}")

    out_dir = ensure_report_dir()
    train_feat = extract_features_resumable(train_df, out_dir / "shortcut_features_train.csv", args.chunk_size)
    val_feat = extract_features_resumable(val_df, out_dir / "shortcut_features_val.csv", args.chunk_size)

    # restrict to exactly the ids we intended this run (in case a stale CSV has extra rows from a
    # previous --limit setting)
    train_feat = train_feat[train_feat["id"].isin(train_df["id"].astype(str))].reset_index(drop=True)
    val_feat = val_feat[val_feat["id"].isin(val_df["id"].astype(str))].reset_index(drop=True)

    X_train, y_train = train_feat[FEATURE_COLS].to_numpy(), train_feat["label"].to_numpy()
    X_val, y_val = val_feat[FEATURE_COLS].to_numpy(), val_feat["label"].to_numpy()

    results = []

    scaler = StandardScaler().fit(X_train)
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(scaler.transform(X_train), y_train)
    val_scores_lr = logreg.predict_proba(scaler.transform(X_val))[:, 1]
    m_lr = evaluate(val_scores_lr, y_val)
    results.append({"model": "LogisticRegression", **m_lr})
    print(f"[shortcut] LogisticRegression  AuDET={m_lr['audet']:.4f} APCER@1%BPCER={m_lr['apcer_at_1pct_bpcer']:.4f}")

    gbt = HistGradientBoostingClassifier(random_state=cfg.seed)
    gbt.fit(X_train, y_train)
    val_scores_gbt = gbt.predict_proba(X_val)[:, 1]
    m_gbt = evaluate(val_scores_gbt, y_val)
    results.append({"model": "HistGradientBoostingClassifier", **m_gbt})
    print(f"[shortcut] HistGBT             AuDET={m_gbt['audet']:.4f} APCER@1%BPCER={m_gbt['apcer_at_1pct_bpcer']:.4f}")

    pd.DataFrame(results).to_csv(out_dir / "shortcut_probe_results.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from sklearn.inspection import permutation_importance
        r = permutation_importance(gbt, X_val, y_val, n_repeats=5, random_state=cfg.seed, scoring="roc_auc")
        importances = r.importances_mean
        order = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([FEATURE_COLS[i] for i in order][::-1], importances[order][::-1], color="#4477AA")
        ax.set_xlabel("importance")
        ax.set_title("shortcut_probe: HistGBT feature importance (permutation, val AUC)")
        fig.tight_layout()
        fig.savefig(out_dir / "shortcut_feature_importance.png", dpi=150)
        print(f"[shortcut] wrote {out_dir / 'shortcut_feature_importance.png'}")
    except ImportError:
        print("[shortcut] matplotlib not available -- skipped plot")

    print(f"[shortcut] wrote {out_dir / 'shortcut_probe_results.csv'}")


if __name__ == "__main__":
    main()
