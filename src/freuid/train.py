"""Training entrypoint.

    uv run python -m freuid.train --config configs/baseline.yaml

Trains a binary fraud classifier (BCEWithLogitsLoss), validates each epoch with the
competition metrics, and checkpoints the best AuDET to checkpoints/<name>.pt.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freuid.config import Config, load_config
from freuid.data import FreuidDataset, lodo_split, stratified_split
from freuid.metrics import evaluate
from freuid.models import build_model
from freuid.transforms import build_transforms, resolve_data_config
from freuid.utils import pick_device, seed_everything


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass. With an optimizer it trains; without, it evaluates.

    Returns (mean_loss, scores, labels) where scores = P(fraud). In train mode the
    scores/labels are not collected (they would force a GPU->CPU sync every batch and
    are unused), so both are returned as None.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_seen, all_scores, all_labels = 0.0, 0, [], []
    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device)
        targets = labels.float().unsqueeze(1).to(device)
        # eval 모드에서는 불필요한 그래디언트 계산을 끄는 컨텍스트 매니저
        with torch.set_grad_enabled(is_train):
            # 모델 forward() 호출. imgs [B, 3, H, W] -> logits [B, 1] (B=batch_size)
            logits = model(imgs)
            # B개의 샘플에 대한 BCEWithLogitsLoss 계산. logits [B, 1], targets [B, 1] -> loss [1]
            loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad()
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
            all_labels.append(labels)
    mean_loss = total_loss / max(n_seen, 1)
    if is_train:
        return mean_loss, None, None
    return mean_loss, torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def _split_ids(cfg: Config) -> tuple[set[str], set[str]]:
    """Train/val id split: Leave-One-Domain-Out if val_doc_type is set, else stratified."""
    if cfg.val_doc_type:
        return lodo_split(cfg.data_dir, cfg.val_doc_type)
    return stratified_split(cfg.data_dir, cfg.val_fraction, cfg.seed)


def build_loaders(
    cfg: Config, data_cfg: dict
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    # model_type dispatch: add new model types here (e.g. model_type="consistency")
    train_ids, val_ids = _split_ids(cfg)
    if cfg.limit:
        # deterministic subset (sorted by id) for fast dev/smoke runs
        train_ids = set(sorted(train_ids)[: cfg.limit])
        val_ids = set(sorted(val_ids)[: max(1, cfg.limit // 5)])
    size, mean, std = data_cfg["image_size"], data_cfg["mean"], data_cfg["std"]
    augment = cfg.extra.get("augment")
    train_tf = build_transforms(size, True, mean, std, augment=augment)
    val_tf = build_transforms(size, False, mean, std)

    synth_prob = float(cfg.extra.get("synth_tamper_prob", 0.0))
    if synth_prob > 0.0:
        from freuid.augment import SynthTamperWrapper, recapture_transforms
        _base_train_ds = FreuidDataset(cfg.data_dir, "train", None, ids=train_ids)
        _tamper_tf = recapture_transforms(size, mean, std)
        train_ds = SynthTamperWrapper(
            _base_train_ds,
            clean_transform=train_tf,
            tamper_transform=_tamper_tf,
            prob=synth_prob,
            seed=cfg.seed,
        )
        _n_bona = sum(1 for s in _base_train_ds.samples if s.label == 0)
        print(
            f"[train] synth_tamper: prob={synth_prob:.2f} "
            f"| {_n_bona} bona-fide → ~{int(_n_bona * synth_prob)} synthetic positives/epoch "
            f"| donor_pool={len(train_ds._donor_pool)}"
        )
    else:
        train_ds = FreuidDataset(cfg.data_dir, "train", train_tf, ids=train_ids)
    val_ds = FreuidDataset(cfg.data_dir, "train", val_tf, ids=val_ids)
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

    # Recapture probe: same val ids, recapture augmentation, deterministic per-epoch seed.
    # num_workers=0 so numpy/random seeding in the main process controls augmentation.
    probe_loader: DataLoader | None = None
    if cfg.extra.get("use_recapture_probe"):
        from freuid.augment import recapture_transforms
        probe_tf = recapture_transforms(size, mean, std)
        probe_ds = FreuidDataset(cfg.data_dir, "train", probe_tf, ids=val_ids)
        probe_loader = DataLoader(
            probe_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        )

    return train_loader, val_loader, probe_loader


def _run_probe(model, probe_loader, device, criterion, seed: int) -> dict[str, float]:
    """Evaluate the degraded probe with a fixed seed so augmentation is the same each epoch."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    _, scores, labels = run_epoch(model, probe_loader, device, criterion)
    return evaluate(scores, labels)


def _check_init_loss(model, loader, device, criterion, tol: float = 0.3) -> None:
    """Assert that BCE on the first train batch ≈ ln(2) before any weight update.

    A fresh classifier head (bias=0, small weights) outputs logits ≈ 0, so
    sigmoid → 0.5 and BCE → ln(2) ≈ 0.693 on any class mix. Failing this usually
    means labels are on the wrong scale, the head bias was initialised incorrectly,
    or the loss function is mis-wired.
    """
    model.eval()
    imgs, labels = next(iter(loader))
    with torch.no_grad():
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.float().unsqueeze(1).to(device)).item()
    model.train()
    expected = math.log(2)  # ≈ 0.693
    assert abs(loss - expected) < tol, (
        f"init BCE={loss:.4f} expected ≈{expected:.4f} (tol={tol}). "
        "Check: labels not shuffled/inverted, loss not pre-averaged with wrong sign, "
        "head bias not set to a constant."
    )
    print(f"[sanity] init BCE={loss:.4f} ~= ln2={expected:.4f} (tol={tol}) OK")


def _sanity_overfit(model, loader, device, criterion, steps: int = 100, target: float = 0.02) -> None:
    """Overfit a single batch to near-zero loss; asserts the forward+backward path works.

    Runs on a COPY of the model so the real training weights are untouched.
    Uses SGD (no momentum) so convergence is purely the model's capacity.
    """
    import copy
    m = copy.deepcopy(model)
    imgs, labels = next(iter(loader))
    imgs = imgs.to(device)
    targets = labels.float().unsqueeze(1).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m.train()
    for _ in range(steps):
        opt.zero_grad()
        criterion(m(imgs), targets).backward()
        opt.step()
    final = criterion(m(imgs), targets).item()
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = pick_device()
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    print(
        f"[train] config '{cfg.name}' | device={device} | backbone={cfg.backbone} | "
        f"image_size={data_cfg['image_size']} mean={data_cfg['mean']}"
    )

    train_loader, val_loader, probe_loader = build_loaders(cfg, data_cfg)
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    if probe_loader is not None:
        print(f"[train] probe={len(probe_loader.dataset)} (recapture, seed={cfg.extra.get('recapture_probe_seed', 0)})")

    # model_type dispatch: add new model types here (e.g. model_type="consistency")
    model = build_model(cfg.backbone, cfg.pretrained).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Always check init loss before any weight updates.
    _check_init_loss(model, train_loader, device, criterion)

    if args.sanity:
        _sanity_overfit(model, train_loader, device, criterion)
        print("[sanity] all checks passed — exiting")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine decay over the run: anneals LR toward 0 by the final epoch. Helps the
    # pretrained RGB backbone settle rather than oscillating at a flat LR.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Checkpoint criterion: "probe_audet" (recapture compass) or "audet" (in-domain).
    ckpt_key = cfg.extra.get("checkpoint_metric", "audet")
    probe_seed = cfg.extra.get("recapture_probe_seed", 0)

    Path("checkpoints").mkdir(exist_ok=True)
    best_metric = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_scores, val_labels = run_epoch(model, val_loader, device, criterion)
        m = evaluate(val_scores, val_labels)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        probe_str = ""
        if probe_loader is not None:
            pm = _run_probe(model, probe_loader, device, criterion, probe_seed)
            m["probe_audet"] = pm["audet"]
            m["probe_apcer_at_1pct_bpcer"] = pm["apcer_at_1pct_bpcer"]
            probe_str = f" probe_AuDET={m['probe_audet']:.6f}"

        print(
            f"\n[epoch {epoch:>2}/{cfg.epochs}] lr={lr:.2e} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"AuDET={m['audet']:.4f} APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.4f}{probe_str}"
        )

        current = m.get(ckpt_key, m["audet"])
        if current < best_metric:
            best_metric = current
            ckpt = Path("checkpoints") / f"{cfg.name}.pt"
            torch.save(
                {"model": model.state_dict(), "config": vars(cfg), "epoch": epoch, "metrics": m},
                ckpt,
            )
            print(f"  -> saved {ckpt} ({ckpt_key}={best_metric:.6f})")


if __name__ == "__main__":
    main()
