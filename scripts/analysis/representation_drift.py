"""Representation-drift analysis: what did fine-tuning actually change in the ViT?

For each of the 12 transformer blocks, compares the PRETRAINED (untouched DINOv2) backbone
against the FINE-TUNED finetune_v0 checkpoint on two axes, using the CLS-token hidden state
at each block's output:

  1. Linear-probe AuDET: fit a per-block logistic regression on a TRAIN-id subsample's
     CLS features, evaluate AuDET on a VAL-id subsample -- for both model variants. Shows
     at which depth fraud/bona-fide separability emerges, and whether fine-tuning moved
     that depth earlier (more forensic/low-level) or later (more semantic/high-level).
  2. Linear CKA (Kornblith et al. 2019) between pretrained-block-i and finetuned-block-i
     features on the SAME val images -- shows which blocks' representations moved the most
     (CKA near 1 = barely changed, near 0 = substantially rewired).

Never modifies the checkpoint or trains anything beyond a cheap linear probe used purely
for measurement. Subsamples train/val for tractability (default 1500 each, stratified by
label) -- pass --full to use the entire split (slower, more memory).

Usage: python scripts/analysis/representation_drift.py [--checkpoint PATH] [--n-samples 1500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    build_finetuned_model,
    build_pretrained_model,
    device_and_seed,
    ensure_report_dir,
    eval_transform,
    get_split_ids,
    load_checkpoint,
    split_dataframe,
)
from freuid.metrics import evaluate  # noqa: E402


class BlockCLSExtractor:
    """Registers forward hooks on every model.blocks[i], captures the CLS token (position 0)
    of each block's output. Call `run(model, imgs)` per batch, then `stack()` once done."""

    def __init__(self, model: torch.nn.Module) -> None:
        if not hasattr(model, "blocks"):
            raise AttributeError(f"{type(model).__name__} has no .blocks -- ViT-only analysis")
        self.n_blocks = len(model.blocks)
        self._captured: dict[int, torch.Tensor] = {}
        self._handles = []
        for i, block in enumerate(model.blocks):
            self._handles.append(block.register_forward_hook(self._make_hook(i)))
        self._per_block_batches: list[list[np.ndarray]] = [[] for _ in range(self.n_blocks)]

    def _make_hook(self, idx: int):
        def hook(module, inp, out):
            self._captured[idx] = out
        return hook

    @torch.no_grad()
    def run(self, model, imgs: torch.Tensor) -> None:
        model(imgs)
        for i in range(self.n_blocks):
            cls = self._captured[i][:, 0, :].float().cpu().numpy()  # CLS token
            self._per_block_batches[i].append(cls)

    def stack(self) -> np.ndarray:
        """Returns (n_samples, n_blocks, dim)."""
        per_block = [np.concatenate(b, axis=0) for b in self._per_block_batches]
        return np.stack(per_block, axis=1)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


@torch.no_grad()
def extract_block_features(model, paths: list[str], transform, device, batch_size: int = 32) -> np.ndarray:
    extractor = BlockCLSExtractor(model)
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        extractor.run(model, imgs)
    feats = extractor.stack()
    extractor.remove()
    return feats


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) between two (n_samples, n_features) matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    denom = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic / denom) if denom > 0 else float("nan")


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.reset_index(drop=True)
    frac = n / len(df)
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .reset_index(drop=True)
    )


def probe_audet(train_feat: np.ndarray, train_y: np.ndarray, val_feat: np.ndarray, val_y: np.ndarray, seed: int) -> dict:
    scaler = StandardScaler().fit(train_feat)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(scaler.transform(train_feat), train_y)
    scores = clf.predict_proba(scaler.transform(val_feat))[:, 1]
    return evaluate(scores, val_y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-samples", type=int, default=1500, help="train/val subsample size each")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    cfg, state = load_checkpoint(ckpt_path)
    device = device_and_seed(cfg)

    train_ids, val_ids = get_split_ids(cfg)
    train_df = stratified_subsample(split_dataframe(cfg, train_ids), args.n_samples, cfg.seed)
    val_df = stratified_subsample(split_dataframe(cfg, val_ids), args.n_samples, cfg.seed)
    print(f"[drift] train subsample n={len(train_df)} val subsample n={len(val_df)}")

    transform, _ = eval_transform(cfg)

    print("[drift] loading pretrained model...")
    pretrained = build_pretrained_model(cfg, device)
    print("[drift] extracting pretrained block features (train)...")
    pre_train = extract_block_features(pretrained, train_df["path"].tolist(), transform, device)
    print("[drift] extracting pretrained block features (val)...")
    pre_val = extract_block_features(pretrained, val_df["path"].tolist(), transform, device)
    del pretrained
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[drift] loading fine-tuned model...")
    finetuned = build_finetuned_model(cfg, state, device)
    print("[drift] extracting fine-tuned block features (train)...")
    ft_train = extract_block_features(finetuned, train_df["path"].tolist(), transform, device)
    print("[drift] extracting fine-tuned block features (val)...")
    ft_val = extract_block_features(finetuned, val_df["path"].tolist(), transform, device)
    del finetuned

    n_blocks = pre_train.shape[1]
    train_y, val_y = train_df["label"].to_numpy(), val_df["label"].to_numpy()

    rows = []
    for b in range(n_blocks):
        m_pre = probe_audet(pre_train[:, b, :], train_y, pre_val[:, b, :], val_y, cfg.seed)
        m_ft = probe_audet(ft_train[:, b, :], train_y, ft_val[:, b, :], val_y, cfg.seed)
        cka = linear_cka(pre_val[:, b, :], ft_val[:, b, :])
        rows.append({
            "block": b,
            "pretrained_audet": m_pre["audet"], "pretrained_apcer": m_pre["apcer_at_1pct_bpcer"],
            "finetuned_audet": m_ft["audet"], "finetuned_apcer": m_ft["apcer_at_1pct_bpcer"],
            "cka_pretrained_vs_finetuned": cka,
        })
        print(f"[drift] block {b:2d}  pretrained AuDET={m_pre['audet']:.4f}  "
              f"finetuned AuDET={m_ft['audet']:.4f}  CKA={cka:.4f}")

    result_df = pd.DataFrame(rows)
    out_dir = ensure_report_dir()
    result_df.to_csv(out_dir / "representation_drift.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        ax1.plot(result_df["block"], result_df["pretrained_audet"], "o-", color="#4477AA", label="pretrained (frozen)")
        ax1.plot(result_df["block"], result_df["finetuned_audet"], "o-", color="#CC3311", label="fine-tuned (finetune_v0)")
        ax1.set_ylabel("linear-probe AuDET (lower=better)")
        ax1.set_title("Per-block linear-probe separability: pretrained vs. fine-tuned")
        ax1.legend()

        ax2.bar(result_df["block"], result_df["cka_pretrained_vs_finetuned"], color="#228833")
        ax2.set_xlabel("transformer block")
        ax2.set_ylabel("linear CKA\n(pretrained vs fine-tuned)")
        ax2.set_ylim(0, 1)
        ax2.set_title("Representation similarity per block (1=unchanged, 0=fully rewired)")

        fig.tight_layout()
        fig.savefig(out_dir / "representation_drift.png", dpi=150)
        print(f"[drift] wrote {out_dir / 'representation_drift.png'}")
    except ImportError:
        print("[drift] matplotlib not available -- skipped plot")

    print(f"[drift] wrote {out_dir / 'representation_drift.csv'}")


if __name__ == "__main__":
    main()
