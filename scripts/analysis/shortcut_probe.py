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

Usage: python scripts/analysis/shortcut_probe.py [--checkpoint PATH] [--limit N]
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


def _quant_features(img: Image.Image) -> tuple[int, float]:
    """(stable hash of quantization tables, mean quant value) -- 0/0.0 if not a JPEG."""
    q = getattr(img, "quantization", None)
    if not q:
        return 0, 0.0
    flat = tuple(v for table in q.values() for v in table)
    h = int(hashlib.md5(str(flat).encode()).hexdigest()[:8], 16) % 100000
    return h, float(np.mean(flat))


def extract_features(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        path = Path(p)
        file_size = path.stat().st_size
        with Image.open(path) as img:
            width, height = img.size
            quant_hash, quant_mean = _quant_features(img)
            # fast low-res decode (JPEG DCT-domain downscale) for pixel stats -- these
            # features are meant to be trivial/superficial, not full-resolution forensics
            img.draft("RGB", (64, 64))
            small = img.convert("RGB").resize((64, 64))
            arr = np.asarray(small, dtype=np.float32)
        rows.append({
            "width": width, "height": height, "aspect_ratio": width / max(height, 1),
            "file_size_bytes": file_size, "quant_hash": quant_hash, "quant_mean": quant_mean,
            "mean_r": arr[:, :, 0].mean(), "mean_g": arr[:, :, 1].mean(), "mean_b": arr[:, :, 2].mean(),
            "std_r": arr[:, :, 0].std(), "std_g": arr[:, :, 1].std(), "std_b": arr[:, :, 2].std(),
        })
    return pd.DataFrame(rows)


FEATURE_COLS = [
    "width", "height", "aspect_ratio", "file_size_bytes", "quant_hash", "quant_mean",
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap train ids for speed")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    cfg, _ = load_checkpoint(ckpt_path)
    train_ids, val_ids = get_split_ids(cfg)

    train_df = split_dataframe(cfg, train_ids)
    val_df = split_dataframe(cfg, val_ids)
    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=cfg.seed).reset_index(drop=True)
    print(f"[shortcut] train n={len(train_df)} val n={len(val_df)} -- extracting trivial metadata features...")

    train_feat = extract_features(train_df["path"].tolist())
    val_feat = extract_features(val_df["path"].tolist())

    out_dir = ensure_report_dir()
    train_feat.assign(id=train_df["id"].values, label=train_df["label"].values).to_csv(
        out_dir / "shortcut_features_train.csv", index=False)
    val_feat.assign(id=val_df["id"].values, label=val_df["label"].values).to_csv(
        out_dir / "shortcut_features_val.csv", index=False)

    X_train, y_train = train_feat[FEATURE_COLS].to_numpy(), train_df["label"].to_numpy()
    X_val, y_val = val_feat[FEATURE_COLS].to_numpy(), val_df["label"].to_numpy()

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

        importances = gbt.feature_importances_ if hasattr(gbt, "feature_importances_") else None
        if importances is None:
            # HistGradientBoostingClassifier has no feature_importances_; use permutation importance
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
