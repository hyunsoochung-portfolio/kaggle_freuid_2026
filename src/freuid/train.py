"""Training entrypoint.

    uv run python -m freuid.train --config configs/baseline.yaml

Trains a binary fraud classifier (BCEWithLogitsLoss), validates each epoch with the
competition metrics, and checkpoints the best AuDET to checkpoints/<name>.pt.
"""

from __future__ import annotations

import argparse
import contextlib
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freuid.config import Config, load_config
from freuid.data import FreuidDataset, domain_holdout_split, stratified_split
from freuid.metrics import evaluate
from freuid.models import build_model, llrd_param_groups
from freuid.transforms import build_transforms, resolve_data_config
from freuid.utils import pick_device, seed_everything


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None,
              use_amp=False):
    """One pass. With an optimizer it trains; without, it evaluates.

    Returns (mean_loss, scores, labels) where scores = P(fraud). In train mode the
    scores/labels are not collected (they would force a GPU->CPU sync every batch and
    are unused), so both are returned as None.

    scheduler steps per-batch (warmup/cosine is a per-step schedule); scaler + use_amp
    enable mixed precision. When use_amp is False everything reduces to the plain path.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_seen, all_scores, all_labels = 0.0, 0, [], []
    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        targets = labels.float().unsqueeze(1).to(device)
        # AMP(혼합정밀): forward/loss를 float16으로 계산해 속도·메모리 절약. use_amp=False면 무효.
        amp_ctx = torch.autocast(device_type="cuda", enabled=True) if use_amp \
            else contextlib.nullcontext()
        # eval 모드에서는 불필요한 그래디언트 계산을 끄는 컨텍스트 매니저
        with torch.set_grad_enabled(is_train), amp_ctx:
            # 모델 forward() 호출. imgs [B, 3, H, W] -> logits [B, 1] (B=batch_size)
            logits = model(imgs)
            # B개의 샘플에 대한 BCEWithLogitsLoss 계산. logits [B, 1], targets [B, 1] -> loss [1]
            loss = criterion(logits, targets)
        if is_train:
            optimizer.zero_grad()
            # scaler: AMP underflow 방지 (loss 키워 backward, step서 복원). 꺼지면 통과.
            scaler.scale(loss).backward()
            # 이 순간: model의 모든 파라미터 p에 대해 p.grad 가 채워짐 (기울기 계산 완료)
            # 단, p 값(가중치) 자체는 아직 그대로
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()  # per-batch LR 갱신 (warmup→cosine)
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


def build_loaders(cfg: Config, data_cfg: dict, batch_size: int) -> tuple[DataLoader, DataLoader]:
    if cfg.val_types:  # cross-domain: hold out whole document types as validation
        train_ids, val_ids = domain_holdout_split(cfg.data_dir, cfg.val_types)
    else:
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
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers,
    )

    # DataLoader가 for문을 돌 때 막후에서 하는 일 (개념 코드):
    #   indices = sampler(dataset)        # 어떤 순서로 꺼낼지 인덱스 정함 (shuffle이면 섞음)
    #   batch = []
    #   for i in indices_for_this_batch:  # 이번 배치에 쓸 인덱스들
    #       sample = dataset[i]           # dataset[i] = dataset.__getitem__(i) 자동 호출
    #       batch.append(sample)
    #   imgs, labels = collate(batch)     # batch_size개를 텐서로 쌓음
    #   yield imgs, labels                # for문에 배치 하나 넘김

    # DataLoader 자체는 "이터러블"(반복 가능 객체)을 반환한다. 데이터를 지금 읽지는 않고,
    # `for batch in loader:` 로 돌 때마다 배치 하나씩 만들어 내놓는다(게으른 로딩).
    # 각 배치 = Dataset.__getitem__ 으로 받은 batch_size개의 (img, label)을 쌓은 튜플:
    #     imgs:   FloatTensor [B, 3, H, W]   (B=batch_size, 마지막 배치는 drop_last로 버려져 항상 B)
    #     labels: LongTensor  [B]            (각 0/1)
    # 예) batch_size=32, image_size=384 -> imgs [32, 3, 384, 384], labels [32]
    return train_loader, val_loader


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Linear warmup for warmup_epochs, then cosine decay to lr_min. None if warmup_epochs<=0.

    A per-step LambdaLR that multiplies EVERY param group's base LR by the same factor, so
    LLRD's per-layer ratios are preserved through warmup and decay.
    """
    if not cfg.warmup_epochs or cfg.warmup_epochs <= 0:
        return None
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = max(1, int(steps_per_epoch * cfg.warmup_epochs))
    min_ratio = cfg.lr_min / cfg.lr if cfg.lr else 0.0

    def lr_factor(step: int) -> float:
        if step < warmup_steps:  # 0 → 1.0 선형 상승
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1.0 → 0.0 코사인 하강
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


def _run_training(cfg: Config, data_cfg: dict, device, batch_size: int) -> None:
    """One full training run at a given batch size. main() retries this with a smaller
    batch on CUDA OOM (the GPU/VESSL do NOT auto-manage memory — 80GB is a hard wall)."""
    train_loader, val_loader = build_loaders(cfg, data_cfg, batch_size)
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"batch={batch_size}")

    model = build_model(
        cfg.backbone, cfg.pretrained, cfg.head_dropout, cfg.pool, data_cfg["image_size"]
    ).to(device)
    if cfg.grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing()  # recompute activations in backward → big memory saving
    if cfg.compile:
        try:
            model = torch.compile(model)  # JIT graph fusion; ~1.3-2x on ViT
        except Exception as e:  # best-effort: fall back to eager if compile isn't available
            print(f"[train] torch.compile failed ({e}); running eager")
    criterion = torch.nn.BCEWithLogitsLoss()

    # optimizer: LLRD → earlier layers get a smaller LR (llrd_param_groups); else one uniform group.
    if cfg.llrd_decay:
        param_groups = llrd_param_groups(model, cfg.lr, cfg.weight_decay, cfg.llrd_decay)
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_amp = cfg.amp and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # torch >= 2.4
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # torch 2.3.x (the VESSL image)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    print(
        f"[train] amp={use_amp} llrd={cfg.llrd_decay} warmup_epochs={cfg.warmup_epochs} "
        f"lr={cfg.lr} groups={len(optimizer.param_groups)} head_dropout={cfg.head_dropout} "
        f"pool={cfg.pool} grad_ckpt={cfg.grad_checkpointing}"
    )

    Path("checkpoints").mkdir(exist_ok=True)
    best_audet = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, device, criterion, optimizer,
                                   scheduler=scheduler, scaler=scaler, use_amp=use_amp)
        val_loss, val_scores, val_labels = run_epoch(model, val_loader, device, criterion,
                                                     use_amp=use_amp)
        m = evaluate(val_scores, val_labels)
        print(
            f"epoch {epoch:>2}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"AuDET={m['audet']:.4f} APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.4f}"
        )
        if m["audet"] < best_audet:
            best_audet = m["audet"]
            ckpt = Path("checkpoints") / f"{cfg.name}.pt"
            # model 가중치 + config + epoch + metrics를 한 딕셔너리로 저장 (checkpoints/<name>.pt)
            # torch.compile 로 감싸면 state_dict 키에 '_orig_mod.' 가 붙으므로 원본을 꺼내 저장.
            state = getattr(model, "_orig_mod", model).state_dict()
            torch.save(
                {"model": state, "config": vars(cfg), "epoch": epoch, "metrics": m},
                ckpt,
            )
            print(f"  ↳ saved {ckpt} (AuDET={best_audet:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = pick_device()
    # Pull normalization + input size from the backbone itself (cfg.image_size overrides
    # the native resolution when set) so preprocessing always matches the pretrained model.
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    print(
        f"[train] config '{cfg.name}' | device={device} | backbone={cfg.backbone} | "
        f"image_size={data_cfg['image_size']} mean={data_cfg['mean']}"
    )

    # Retry with a smaller batch on CUDA OOM. grad_checkpointing (config) is the main defense;
    # this loop just self-corrects if the batch guess is still too big for 80GB.
    batch_size = cfg.batch_size
    while True:
        try:
            _run_training(cfg, data_cfg, device, batch_size)
            return
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 4:
                raise
            batch_size //= 2
            print(f"[oom] CUDA out of memory — retrying with batch_size={batch_size}")


if __name__ == "__main__":
    main()
