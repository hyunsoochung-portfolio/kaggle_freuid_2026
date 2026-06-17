"""Training entrypoint.

    uv run python -m freuid.train --config configs/baseline.yaml

Trains a binary fraud classifier (BCEWithLogitsLoss), validates each epoch with the
competition metrics, and checkpoints the best AuDET to checkpoints/<name>.pt.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freuid.config import Config, load_config
from freuid.data import FreuidDataset, stratified_split
from freuid.metrics import evaluate
from freuid.models import build_model
from freuid.transforms import build_transforms
from freuid.utils import pick_device, seed_everything


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass. With an optimizer it trains; without, it evaluates.

    Returns (mean_loss, scores, labels) where scores = P(fraud).
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, all_scores, all_labels = 0.0, [], []
    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        targets = labels.float().unsqueeze(1).to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(imgs)
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        all_scores.append(torch.sigmoid(logits).detach().squeeze(1).float().cpu())
        all_labels.append(labels)
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / len(loader.dataset), scores, labels


def build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    train_ids, val_ids = stratified_split(cfg.data_dir, cfg.val_fraction, cfg.seed)
    train_ds = FreuidDataset(cfg.data_dir, "train", build_transforms(cfg.image_size, True), ids=train_ids)
    val_ds = FreuidDataset(cfg.data_dir, "train", build_transforms(cfg.image_size, False), ids=val_ids)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = pick_device()
    print(f"[train] config '{cfg.name}' | device={device} | backbone={cfg.backbone}")

    train_loader, val_loader = build_loaders(cfg)
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_model(cfg.backbone, cfg.pretrained).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Path("checkpoints").mkdir(exist_ok=True)
    best_audet = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_scores, val_labels = run_epoch(model, val_loader, device, criterion)
        m = evaluate(val_scores, val_labels)
        print(
            f"epoch {epoch:>2}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"AuDET={m['audet']:.4f} APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.4f}"
        )
        if m["audet"] < best_audet:
            best_audet = m["audet"]
            ckpt = Path("checkpoints") / f"{cfg.name}.pt"
            torch.save(
                {"model": model.state_dict(), "config": vars(cfg), "epoch": epoch, "metrics": m},
                ckpt,
            )
            print(f"  ↳ saved {ckpt} (AuDET={best_audet:.4f})")


if __name__ == "__main__":
    main()
