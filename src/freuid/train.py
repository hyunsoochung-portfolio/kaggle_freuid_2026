"""Training entrypoint.

    uv run python -m freuid.train --config configs/baseline.yaml

Trains a binary fraud classifier (BCEWithLogitsLoss), validates each epoch with the
competition metrics, and checkpoints the best AuDET to checkpoints/<name>.pt.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freuid.config import Config, load_config
from freuid.data import FreuidDataset, stratified_split, unpack_batch
from freuid.loss import combined_loss
from freuid.metrics import evaluate
from freuid.models import build_model
from freuid.transforms import build_transforms, resolve_data_config
from freuid.utils import pick_device, seed_everything


def run_epoch(model, loader, device, criterion, optimizer=None, auc_weight: float = 0.0,
              scaler=None):
    """One pass. With an optimizer it trains; without, it evaluates.

    Returns (mean_loss, scores, labels) where scores = P(fraud). In train mode the
    scores/labels are not collected (they would force a GPU->CPU sync every batch and
    are unused), so both are returned as None.

    auc_weight > 0 adds a pairwise soft-AUC term to the BCE loss (train only).
    auc_weight = 0.0 is bit-for-bit identical to plain BCE.

    ``scaler`` (a ``torch.cuda.amp.GradScaler``) enables AMP (autocast + loss scaling)
    when non-None and ``scaler.is_enabled()``; omitting it (the default for every
    existing config) reproduces the original FP32 path exactly -- a disabled GradScaler
    is a documented no-op passthrough for scale/step/update.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_seen, all_scores, all_labels = 0.0, 0, [], []
    use_amp = scaler is not None and scaler.is_enabled()
    for batch in tqdm(loader, leave=False):
        imgs, labels, face_meta = unpack_batch(batch)
        imgs = imgs.to(device)
        labels_dev = labels.to(device)
        face_meta_dev = face_meta.to(device) if face_meta is not None else None
        # eval 모드에서는 불필요한 그래디언트 계산을 끄는 컨텍스트 매니저
        with torch.set_grad_enabled(is_train):
            amp_ctx = (
                torch.autocast(device_type="cuda", enabled=use_amp)
                if device.type == "cuda" else contextlib.nullcontext()
            )
            with amp_ctx:
                # 모델 forward() 호출. imgs [B, 3, H, W] -> logits [B, 1] (B=batch_size)
                logits = model(imgs, face_meta_dev) if face_meta_dev is not None else model(imgs)
                # BCE + optional pairwise AUC term (train only; val always uses plain BCE)
                _aw = auc_weight if is_train else 0.0
                loss = combined_loss(logits, labels_dev, criterion, _aw)
            if is_train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # 이 순간: model의 모든 파라미터 p에 대해 p.grad 가 채워짐 (기울기 계산 완료)
                    # 단, p 값(가중치) 자체는 아직 그대로
                    optimizer.step()
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n_seen += bs
        if not is_train:
            # logits [B, 1] -> scores [B] (P(fraud) in [0, 1]).
            # 나중에 이걸 다 모아 AuDET 계산(metrics.py)에 씀.
            all_scores.append(torch.sigmoid(logits).squeeze(1).float().cpu())
            all_labels.append(labels_dev.cpu())
    mean_loss = total_loss / max(n_seen, 1)
    if is_train:
        return mean_loss, None, None
    return mean_loss, torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def build_loaders(cfg: Config, data_cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Train/val loaders. By default both are analog-doubled: {all images, clean} ∪
    {is_digital images, recapture}. Each digital doc is seen as-is AND print-and-recaptured
    (label preserved), teaching the model the digital and analog appearance (the digital→
    physical shift the hidden test probes). Validation analog copies are seeded per-sample
    (cfg.seed) so they degrade identically every epoch -- val AuDET is then a fair,
    harder-than-clean checkpoint metric (no separate probe needed).

    model_type=="consistency" (frozen-backbone) keeps the plain, undoubled FreuidDataset path
    because it needs face-region metadata and its own transforms.
    """
    from freuid.augment import AnalogDoubleDataset, recapture_transforms

    train_ids, val_ids = stratified_split(cfg.data_dir, cfg.val_fraction, cfg.seed)
    if cfg.limit:  # deterministic subset (sorted by id) for fast dev/smoke runs
        train_ids = set(sorted(train_ids)[: cfg.limit])
        val_ids = set(sorted(val_ids)[: max(1, cfg.limit // 5)])
    size, mean, std = data_cfg["image_size"], data_cfg["mean"], data_cfg["std"]
    clean_tf = build_transforms(size, False, mean, std)  # originals: resize + normalize

    if cfg.extra.get("model_type") == "consistency":
        _rdir: Path | None = None
        if cfg.extra.get("use_rectify", False):
            from freuid.preprocess import regions_dir as _get_rdir
            _rdir = _get_rdir(cfg.data_dir)
            _rdir = _rdir if _rdir.exists() else None
        rfm = bool(cfg.extra.get("use_face_region", False))
        train_tf = build_transforms(size, True, mean, std, augment=cfg.extra.get("augment"))
        train_ds = FreuidDataset(cfg.data_dir, "train", train_tf, ids=train_ids,
                                 regions_dir=_rdir, return_face_meta=rfm)
        val_ds = FreuidDataset(cfg.data_dir, "train", clean_tf, ids=val_ids,
                               regions_dir=_rdir, return_face_meta=rfm)
    elif cfg.extra.get("analog_double", True):
        # partial (not a local closure) so DataLoader workers can pickle it under 'spawn'
        # (macOS default); calling it with a seed -> recapture pipeline (None=random, int=fixed).
        make_analog = functools.partial(recapture_transforms, size, mean, std)
        base_train = FreuidDataset(cfg.data_dir, "train", None, ids=train_ids)
        base_val = FreuidDataset(cfg.data_dir, "train", None, ids=val_ids)
        train_ds = AnalogDoubleDataset(base_train, clean_tf, make_analog)
        # val: deterministic_seed makes the analog copies identical every epoch
        val_ds = AnalogDoubleDataset(base_val, clean_tf, make_analog, deterministic_seed=cfg.seed)
        print(
            f"[train] analog_double: train {len(base_train.samples)}+{len(train_ds.analog_idx)}"
            f"={len(train_ds)} | val {len(base_val.samples)}+{len(val_ds.analog_idx)}={len(val_ds)}"
        )
    else:
        # analog_double=False: plain single-copy path (reproduces the pre-analog-double
        # baseline like dinov2_v1) -- standard train aug + clean val, no recapture doubling.
        # For A/B isolation of the analog-double augmentation.
        train_tf = build_transforms(size, True, mean, std)
        train_ds = FreuidDataset(cfg.data_dir, "train", train_tf, ids=train_ids)
        val_ds = FreuidDataset(cfg.data_dir, "train", clean_tf, ids=val_ids)
        print(f"[train] analog_double=OFF (plain): train {len(train_ds)} | val {len(val_ds)}")

    pin_memory = torch.cuda.is_available()  # unsupported/no-op on MPS, only helps CUDA
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
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


def _check_init_loss(model, loader, device, criterion, tol: float = 0.3) -> None:
    """Assert that BCE on the first train batch ≈ ln(2) before any weight update.

    A fresh classifier head (bias=0, small weights) outputs logits ≈ 0, so
    sigmoid → 0.5 and BCE → ln(2) ≈ 0.693 on any class mix. Failing this usually
    means labels are on the wrong scale, the head bias was initialised incorrectly,
    or the loss function is mis-wired.
    """
    model.eval()
    imgs, labels, face_meta = unpack_batch(next(iter(loader)))
    with torch.no_grad():
        imgs = imgs.to(device)
        logits = model(imgs, face_meta.to(device)) if face_meta is not None else model(imgs)
        loss = criterion(logits, labels.float().unsqueeze(1).to(device)).item()
    model.train()
    expected = math.log(2)  # ≈ 0.693
    assert abs(loss - expected) < tol, (
        f"init BCE={loss:.4f} expected ≈{expected:.4f} (tol={tol}). "
        "Check: labels not shuffled/inverted, loss not pre-averaged with wrong sign, "
        "head bias not set to a constant."
    )
    print(f"[sanity] init BCE={loss:.4f} ~= ln2={expected:.4f} (tol={tol}) OK")


def _sanity_overfit(model, loader, device, criterion, steps: int = 100,
                    target: float = 0.02) -> None:
    """Overfit a single batch to near-zero loss; asserts the forward+backward path works.

    Runs on a COPY of the model so the real training weights are untouched.
    Uses SGD (no momentum) so convergence is purely the model's capacity.
    """
    import copy
    m = copy.deepcopy(model)
    imgs, labels, face_meta = unpack_batch(next(iter(loader)))
    imgs = imgs.to(device)
    face_meta = face_meta.to(device) if face_meta is not None else None
    targets = labels.float().unsqueeze(1).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m.train()

    def _forward():
        return m(imgs, face_meta) if face_meta is not None else m(imgs)

    for _ in range(steps):
        opt.zero_grad()
        criterion(_forward(), targets).backward()
        opt.step()
    final = criterion(_forward(), targets).item()
    assert final < target, (
        f"sanity overfit: loss={final:.4f} after {steps} steps (target <{target}). "
        "Check: gradient flow not blocked, model has enough capacity for one batch."
    )
    print(f"[sanity] single-batch overfit: loss={final:.6f} after {steps} steps OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--sanity", action="store_true",
        help="run init-loss check + single-batch overfit check, then exit",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap train/val dataset sizes for quick smoke runs",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.limit = args.limit
    seed_everything(cfg.seed)
    device = pick_device()
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    print(
        f"[train] config '{cfg.name}' | device={device} | backbone={cfg.backbone} | "
        f"image_size={data_cfg['image_size']} mean={data_cfg['mean']}"
    )

    train_loader, val_loader = build_loaders(cfg, data_cfg)
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    # model_type dispatch: add new model types here (e.g. model_type="consistency")
    model_type = cfg.extra.get("model_type", "baseline")
    if model_type == "consistency":
        from freuid.models import build_consistency_model
        model = build_consistency_model(cfg).to(device)
    elif model_type == "joint":
        # full-FT attention-pool backbone + parallel patch-consistency branch, fused.
        from freuid.models import build_joint_model
        model = build_joint_model(cfg).to(device)
    else:
        model = build_model(
            cfg.backbone, cfg.pretrained,
            pool=cfg.extra.get("pool"),
            head_dropout=float(cfg.extra.get("head_dropout", 0.0)),
        ).to(device)
        # Fine-tuning knobs (baseline/ViT path only; frozen consistency path untouched).
        train_last_k = cfg.extra.get("train_last_k_blocks")
        if train_last_k is not None:
            from freuid.optim import freeze_all_but_last_k_blocks
            freeze_all_but_last_k_blocks(model, int(train_last_k))
        if cfg.extra.get("grad_checkpointing", False):
            if hasattr(model, "set_grad_checkpointing"):
                model.set_grad_checkpointing(True)
                print("[train] gradient checkpointing enabled")
            else:
                print(f"[train] WARNING: grad_checkpointing=True but {cfg.backbone} "
                      "has no set_grad_checkpointing")
    criterion = torch.nn.BCEWithLogitsLoss()

    # Always check init loss before any weight updates.
    _check_init_loss(model, train_loader, device, criterion)

    if args.sanity:
        _sanity_overfit(model, train_loader, device, criterion)
        print("[sanity] all checks passed — exiting")
        return

    llrd_cfg = cfg.extra.get("llrd") or {}
    if llrd_cfg.get("enabled", False):
        from freuid.optim import build_llrd_param_groups, build_warmup_cosine_scheduler
        param_groups = build_llrd_param_groups(
            model, base_lr=cfg.lr, weight_decay=cfg.weight_decay,
            decay=float(llrd_cfg.get("decay", 0.7)),
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = build_warmup_cosine_scheduler(
            optimizer, cfg.epochs, warmup_epochs=int(llrd_cfg.get("warmup_epochs", 2)),
        )
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        # Optional linear warmup (extra.warmup_epochs) then cosine decay; else plain cosine.
        # Warmup stabilises full fine-tuning when LLRD is off (e.g. the joint model). Cosine
        # anneals LR toward 0 by the final epoch so the backbone settles.
        warmup_e = int(cfg.extra.get("warmup_epochs", 0))
        if warmup_e > 0:
            from freuid.optim import build_warmup_cosine_scheduler
            scheduler = build_warmup_cosine_scheduler(optimizer, cfg.epochs, warmup_epochs=warmup_e)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    auc_weight = float(cfg.extra.get("auc_loss_weight", 0.0))
    if auc_weight > 0.0:
        print(f"[train] auc_loss_weight={auc_weight} (pairwise soft-AUC term active)")

    # AMP: disabled GradScaler is a documented no-op passthrough, so this is safe to
    # always construct and thread through run_epoch -- every config without
    # extra.amp=True gets scaler.is_enabled()==False and reproduces the exact FP32 path.
    amp_enabled = bool(cfg.extra.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None
    if amp_enabled:
        print("[train] AMP enabled (autocast + GradScaler)")

    # Checkpoint on the val AuDET (the val set is analog-doubled, so this already reflects
    # performance under the recapture shift). Ties broken by APCER@1%BPCER (secondary metric).

    Path("checkpoints").mkdir(exist_ok=True)
    best_metric = float("inf")
    best_tiebreak = float("inf")
    # Early stopping: stop once val_loss rises for `early_stop_patience` epochs in a row
    # (0 = disabled). Guards against overtraining when running to a large epochs ceiling.
    es_patience = int(cfg.extra.get("early_stop_patience", 0))
    prev_val_loss = float("inf")
    val_rises = 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, device, criterion, optimizer,
                                   auc_weight, scaler=scaler)
        val_loss, val_scores, val_labels = run_epoch(model, val_loader, device, criterion,
                                                     scaler=scaler)
        m = evaluate(val_scores, val_labels)
        last_lrs = scheduler.get_last_lr()
        lr = last_lrs[0] if len(last_lrs) == 1 else max(last_lrs)
        scheduler.step()

        lr_str = (f"lr={lr:.2e}" if len(last_lrs) == 1
                  else f"lr_head={lr:.2e} lr_min={min(last_lrs):.2e}")
        print(
            f"\n[epoch {epoch:>2}/{cfg.epochs}] {lr_str} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} AuDET={m['audet']:.4f} "
            f"APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.4f}"
        )

        current, current_tie = m["audet"], m["apcer_at_1pct_bpcer"]
        improved = current < best_metric or (current == best_metric and current_tie < best_tiebreak)
        if improved:
            best_metric = current
            best_tiebreak = current_tie
            ckpt = Path("checkpoints") / f"{cfg.name}.pt"
            torch.save(
                {"model": model.state_dict(), "config": vars(cfg), "epoch": epoch, "metrics": m},
                ckpt,
            )
            print(f"  -> saved {ckpt} (val_AuDET={best_metric:.6f})")

        # Optionally also keep the LATEST epoch's weights (overwritten each epoch). Useful when
        # the val metric saturates and best-val locks onto an early/undertrained epoch -- then
        # the last, more-trained checkpoint can generalise better to the harder hidden test.
        if cfg.extra.get("save_last", False):
            last_ckpt = Path("checkpoints") / f"{cfg.name}_last.pt"
            torch.save(
                {"model": model.state_dict(), "config": vars(cfg), "epoch": epoch, "metrics": m},
                last_ckpt,
            )
            print(f"  -> saved {last_ckpt} (last, epoch={epoch})")

        # Early stop on consecutive val_loss rises (checked after saving, so the last
        # checkpoint includes this epoch). val_loss is noisy here, so patience>=3 is advised.
        if es_patience > 0:
            val_rises = val_rises + 1 if val_loss > prev_val_loss else 0
            prev_val_loss = val_loss
            if val_rises >= es_patience:
                print(f"[early-stop] val_loss rose {val_rises} epochs in a row "
                      f"(patience={es_patience}) -> stopping at epoch {epoch}")
                break


if __name__ == "__main__":
    main()
