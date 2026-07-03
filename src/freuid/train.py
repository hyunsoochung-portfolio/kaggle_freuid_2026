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
from freuid.transforms import build_transforms, resolve_data_config
from freuid.utils import pick_device, seed_everything


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    """One pass. With an optimizer it trains; without, it evaluates.

    Returns (mean_loss, scores, labels) where scores = P(fraud). In train mode the
    scores/labels are not collected (they would force a GPU->CPU sync every batch and
    are unused), so both are returned as None.

    scaler: torch.cuda.amp.GradScaler, only used (and only has effect) when training
    with use_amp=True. autocast wraps the forward+loss in reduced precision; the
    scaler keeps the backward pass numerically stable under fp16.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_seen, all_scores, all_labels = 0.0, 0, [], []
    use_amp = scaler is not None and scaler.is_enabled()
    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = labels.float().unsqueeze(1).to(device, non_blocking=True)
        # eval 모드에서는 불필요한 그래디언트 계산을 끄는 컨텍스트 매니저
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                # 모델 forward() 호출. imgs [B, 3, H, W] -> logits [B, 1] (B=batch_size)
                logits = model(imgs)
                # B개의 샘플에 대한 BCEWithLogitsLoss 계산. logits [B, 1], targets [B, 1] -> loss [1]
                loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                # 이 순간: model의 모든 파라미터 p에 대해 p.grad 가 채워짐 (기울기 계산 완료)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n_seen += bs
        if not is_train:
            # logits [B, 1] -> scores [B] (P(fraud) in [0, 1]).
            # 나중에 이걸 다 모아 AuDET 계산(metrics.py)에 씀.
            all_scores.append(torch.sigmoid(logits).squeeze(1).float().cpu())
            all_labels.append(labels)
    mean_loss = total_loss / max(n_seen, 1)
    if is_train:
        return mean_loss, None, None
    return mean_loss, torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def build_loaders(cfg: Config, data_cfg: dict) -> tuple[DataLoader, DataLoader]:
    train_ids, val_ids = stratified_split(cfg.data_dir, cfg.val_fraction, cfg.seed)
    if cfg.limit:
        # deterministic subset (sorted by id) for fast dev/smoke runs
        train_ids = set(sorted(train_ids)[: cfg.limit])
        val_ids = set(sorted(val_ids)[: max(1, cfg.limit // 5)])
    size, mean, std = data_cfg["image_size"], data_cfg["mean"], data_cfg["std"]
    train_tf = build_transforms(size, True, mean, std)
    val_tf = build_transforms(size, False, mean, std)
    train_ds = FreuidDataset(cfg.data_dir, "train", train_tf, ids=train_ids)
    val_ds = FreuidDataset(cfg.data_dir, "train", val_tf, ids=val_ids)
    pin_memory = torch.cuda.is_available()  # unsupported/no-op on MPS, only helps CUDA
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    # speed knobs, set via unknown yaml keys (land in cfg.extra automatically) —
    # no dataclass change needed. deterministic=True is the safe default for a run
    # you'll actually submit/report; set false in a config for fast iteration.
    deterministic = cfg.extra.get("deterministic", True)
    seed_everything(cfg.seed, deterministic=deterministic)
    device = pick_device()
    use_amp = cfg.extra.get("use_amp", True) and device.type == "cuda"
    # Pull normalization + input size from the backbone itself (cfg.image_size overrides
    # the native resolution when set) so preprocessing always matches the pretrained model.
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    print(
        f"[train] config '{cfg.name}' | device={device} | backbone={cfg.backbone} | "
        f"image_size={data_cfg['image_size']} mean={data_cfg['mean']} | "
        f"deterministic={deterministic} use_amp={use_amp}"
    )

    train_loader, val_loader = build_loaders(cfg, data_cfg)
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_model(cfg.backbone, cfg.pretrained).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Path("checkpoints").mkdir(exist_ok=True)
    best_audet = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_loss, val_scores, val_labels = run_epoch(model, val_loader, device, criterion)
        m = evaluate(val_scores, val_labels)
        print(
            f"epoch {epoch:>2}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"AuDET={m['audet']:.4f} APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.4f}"
        )
        if m["audet"] < best_audet:
            best_audet = m["audet"]
            ckpt = Path("checkpoints") / f"{cfg.name}.pt"
            # model 가중치 + config + epoch + metrics를 한 딕셔너리로 저장 (checkpoints/<name>.pt)
            torch.save(
                {"model": model.state_dict(), "config": vars(cfg), "epoch": epoch, "metrics": m},
                ckpt,
            )
            print(f"  ↳ saved {ckpt} (AuDET={best_audet:.4f})")


if __name__ == "__main__":
    main()
