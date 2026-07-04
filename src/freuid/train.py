"""Training entrypoint.

model=twostream (extra) selects the two-stream (full+face+ela) path, kept as an
isolated side path with its own simpler loop. Everything else (baseline/consistency,
LLRD, sanity checks, recapture probe, synth tamper) is develop's existing pipeline,
unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from freuid.config import Config, load_config
from freuid.data import (
    FreuidDataset,
    TwoStreamDataset,
    domain_holdout_split,
    lodo_split,
    stratified_split,
    unpack_batch,
)
from freuid.loss import combined_loss
from freuid.metrics import evaluate
from freuid.models import build_model, build_twostream_model
from freuid.transforms import build_transforms, resolve_data_config
from freuid.utils import pick_device, seed_everything


def run_epoch(model, loader, device, criterion, optimizer=None, auc_weight: float = 0.0, scaler=None):
    """One pass over FreuidDataset-shaped batches. With an optimizer it trains;
    without, it evaluates. auc_weight > 0 adds a pairwise soft-AUC term to BCE
    (train only); auc_weight=0.0 is bit-for-bit identical to plain BCE.
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
        with torch.set_grad_enabled(is_train):
            amp_ctx = (
                torch.autocast(device_type="cuda", enabled=use_amp)
                if device.type == "cuda" else contextlib.nullcontext()
            )
            with amp_ctx:
                logits = model(imgs, face_meta_dev) if face_meta_dev is not None else model(imgs)
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
                    optimizer.step()
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n_seen += bs
        if not is_train:
            all_scores.append(torch.sigmoid(logits).squeeze(1).float().cpu())
            all_labels.append(labels_dev.cpu())
    mean_loss = total_loss / max(n_seen, 1)
    if is_train:
        return mean_loss, None, None
    return mean_loss, torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def run_epoch_twostream(model, loader, device, criterion, optimizer=None, scaler=None):
    """One pass over (full, face, ela, label) batches -- the twostream side path."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_seen, all_scores, all_labels = 0.0, 0, [], []
    use_amp = scaler is not None and scaler.is_enabled()
    for full, face, ela, labels in tqdm(loader, leave=False):
        full = full.to(device, non_blocking=True)
        face = face.to(device, non_blocking=True)
        ela = ela.to(device, non_blocking=True)
        targets = labels.float().unsqueeze(1).to(device, non_blocking=True)
        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(full, face, ela)
                loss = criterion(logits, targets)
            if is_train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
        bs = full.size(0)
        total_loss += loss.item() * bs; n_seen += bs
        if not is_train:
            all_scores.append(torch.sigmoid(logits).squeeze(1).float().cpu())
            all_labels.append(labels)
    mean_loss = total_loss / max(n_seen, 1)
    if is_train:
        return mean_loss, None, None
    return mean_loss, torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy()


def _split_ids(cfg: Config) -> tuple[set[str], set[str]]:
    """Train/val id split for the baseline/consistency path: LODO if val_doc_type
    is set, else stratified."""
    if cfg.val_doc_type:
        return lodo_split(cfg.data_dir, cfg.val_doc_type)
    return stratified_split(cfg.data_dir, cfg.val_fraction, cfg.seed)


def _resolve_split_ids(cfg: Config):
    """Train/val id split for the twostream path (extra.split / extra.holdout_types)."""
    split_kind = cfg.extra.get("split", "stratified")
    if split_kind == "domain_holdout":
        holdout_types = cfg.extra.get("holdout_types")
        if not holdout_types:
            raise ValueError("split: domain_holdout requires extra.holdout_types")
        train_ids, val_ids = domain_holdout_split(cfg.data_dir, holdout_types)
        print(f"[train] split=domain_holdout holdout_types={holdout_types}")
    elif split_kind == "stratified":
        train_ids, val_ids = stratified_split(cfg.data_dir, cfg.val_fraction, cfg.seed)
    else:
        raise ValueError(f"unknown split kind: {split_kind!r}")
    if cfg.limit:
        train_ids = set(sorted(train_ids)[: cfg.limit])
        val_ids = set(sorted(val_ids)[: max(1, cfg.limit // 5)])
    return train_ids, val_ids


def build_loaders(cfg: Config, data_cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    train_ids, val_ids = _split_ids(cfg)
    if cfg.limit:
        train_ids = set(sorted(train_ids)[: cfg.limit])
        val_ids = set(sorted(val_ids)[: max(1, cfg.limit // 5)])
    size, mean, std = data_cfg["image_size"], data_cfg["mean"], data_cfg["std"]
    augment = cfg.extra.get("augment")
    train_tf = build_transforms(size, True, mean, std, augment=augment)
    val_tf = build_transforms(size, False, mean, std)

    _rdir: Path | None = None
    if cfg.extra.get("use_rectify", False):
        from freuid.preprocess import regions_dir as _get_rdir
        _rdir = _get_rdir(cfg.data_dir)
        if _rdir.exists():
            print(f"[train] use_rectify=True -> loading from {_rdir}")
        else:
            print(f"[train] WARNING: use_rectify=True but cache not found at {_rdir}; using raw images")
            _rdir = None

    model_type = cfg.extra.get("model_type", "baseline")
    return_face_meta = model_type == "consistency" and bool(cfg.extra.get("use_face_region", False))

    synth_double = bool(cfg.extra.get("synth_tamper_double", False))
    synth_prob = float(cfg.extra.get("synth_tamper_prob", 0.0))
    if synth_double:
        from torch.utils.data import ConcatDataset
        from freuid.augment import SynthTamperDataset, recapture_transforms
        _base_train_ds = FreuidDataset(
            cfg.data_dir, "train", train_tf, ids=train_ids, regions_dir=_rdir,
            return_face_meta=return_face_meta,
        )
        _tamper_tf = recapture_transforms(size, mean, std)
        _synth_ds = SynthTamperDataset(_base_train_ds, tamper_transform=_tamper_tf, seed=cfg.seed)
        train_ds = ConcatDataset([_base_train_ds, _synth_ds])
        print(
            f"[train] synth_tamper_double: {len(_base_train_ds)} original + "
            f"{len(_synth_ds)} synthetic (from bona-fide) = {len(train_ds)} total/epoch"
        )
    elif synth_prob > 0.0:
        from freuid.augment import SynthTamperWrapper, recapture_transforms
        _base_train_ds = FreuidDataset(
            cfg.data_dir, "train", None, ids=train_ids, regions_dir=_rdir,
            return_face_meta=return_face_meta,
        )
        _tamper_tf = recapture_transforms(size, mean, std)
        train_ds = SynthTamperWrapper(
            _base_train_ds, clean_transform=train_tf, tamper_transform=_tamper_tf,
            prob=synth_prob, seed=cfg.seed,
        )
        _n_bona = sum(1 for s in _base_train_ds.samples if s.label == 0)
        print(
            f"[train] synth_tamper: prob={synth_prob:.2f} "
            f"| {_n_bona} bona-fide -> ~{int(_n_bona * synth_prob)} synthetic positives/epoch "
            f"| donor_pool={len(train_ds._donor_pool)}"
        )
    else:
        train_ds = FreuidDataset(
            cfg.data_dir, "train", train_tf, ids=train_ids, regions_dir=_rdir,
            return_face_meta=return_face_meta,
        )
    val_ds = FreuidDataset(
        cfg.data_dir, "train", val_tf, ids=val_ids, regions_dir=_rdir,
        return_face_meta=return_face_meta,
    )
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
    )

    probe_loader: DataLoader | None = None
    if cfg.extra.get("use_recapture_probe"):
        from freuid.augment import recapture_transforms
        probe_tf = recapture_transforms(size, mean, std)
        probe_ds = FreuidDataset(
            cfg.data_dir, "train", probe_tf, ids=val_ids, regions_dir=_rdir,
            return_face_meta=return_face_meta,
        )
        probe_loader = DataLoader(probe_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, probe_loader


def build_loaders_twostream(cfg, data_cfg):
    train_ids, val_ids = _resolve_split_ids(cfg)
    size, mean, std = data_cfg["image_size"], data_cfg["mean"], data_cfg["std"]
    face_size = cfg.extra.get("face_size", 160)
    ela_size = cfg.extra.get("ela_size", size)
    full_train_tf = build_transforms(size, True, mean, std)
    full_val_tf = build_transforms(size, False, mean, std)
    face_train_tf = build_transforms(face_size, True, mean, std)
    face_val_tf = build_transforms(face_size, False, mean, std)
    ela_tf = build_transforms(ela_size, False, mean, std)
    train_ds = TwoStreamDataset(cfg.data_dir, "train", ids=train_ids,
        full_transform=full_train_tf, face_transform=face_train_tf, ela_transform=ela_tf)
    val_ds = TwoStreamDataset(cfg.data_dir, "train", ids=val_ids,
        full_transform=full_val_tf, face_transform=face_val_tf, ela_transform=ela_tf)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=True,
        persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    return train_loader, val_loader


def _run_probe(model, probe_loader, device, criterion, seed: int, scaler=None) -> dict[str, float]:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    _, scores, labels = run_epoch(model, probe_loader, device, criterion, scaler=scaler)
    return evaluate(scores, labels)


def _check_init_loss(model, loader, device, criterion, tol: float = 0.3) -> None:
    model.eval()
    imgs, labels, face_meta = unpack_batch(next(iter(loader)))
    with torch.no_grad():
        imgs = imgs.to(device)
        logits = model(imgs, face_meta.to(device)) if face_meta is not None else model(imgs)
        loss = criterion(logits, labels.float().unsqueeze(1).to(device)).item()
    model.train()
    expected = math.log(2)
    assert abs(loss - expected) < tol, (
        f"init BCE={loss:.4f} expected ~{expected:.4f} (tol={tol}). "
        "Check: labels not shuffled/inverted, loss not pre-averaged with wrong sign, "
        "head bias not set to a constant."
    )
    print(f"[sanity] init BCE={loss:.4f} ~= ln2={expected:.4f} (tol={tol}) OK")


def _sanity_overfit(model, loader, device, criterion, steps: int = 100, target: float = 0.02) -> None:
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



def _safe_save(state: dict, ckpt_path: Path, retries: int = 3, delay: float = 2.0) -> bool:
    """torch.save with retries -- a transient Drive-symlink hiccup (e.g. right as
    the Colab runtime disconnects) would otherwise crash the whole run and lose
    that epoch's result. Falls back to a local copy if Drive keeps failing so the
    checkpoint isn't lost outright."""
    import time
    for attempt in range(retries):
        try:
            torch.save(state, ckpt_path)
            return True
        except (RuntimeError, OSError) as e:
            print(f"[train] WARNING: checkpoint save failed (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(delay)
    fallback = Path("/content") / ckpt_path.name
    try:
        torch.save(state, fallback)
        print(f"[train] WARNING: saved fallback checkpoint locally at {fallback} (Drive save failed)")
    except Exception as e:
        print(f"[train] ERROR: fallback save also failed: {e}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sanity", action="store_true",
        help="run init-loss check + single-batch overfit check, then exit")
    parser.add_argument("--limit", type=int, default=None,
        help="cap train/val dataset sizes for quick smoke runs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.limit = args.limit

    deterministic = cfg.extra.get("deterministic", True)
    seed_everything(cfg.seed, deterministic=deterministic)
    device = pick_device()
    model_kind = cfg.extra.get("model", "single")
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)

    # --- twostream: isolated side path (own dataset shape, own simpler loop) ---
    if model_kind == "twostream":
        use_amp = cfg.extra.get("use_amp", True) and device.type == "cuda"
        print(f"[train] config '{cfg.name}' | device={device} | model=twostream | "
              f"backbone={cfg.backbone} | image_size={data_cfg['image_size']} mean={data_cfg['mean']} | "
              f"deterministic={deterministic} use_amp={use_amp}")
        train_loader, val_loader = build_loaders_twostream(cfg, data_cfg)
        freeze_aux = cfg.extra.get("freeze_aux", False)
        model = build_twostream_model(
            full_backbone=cfg.backbone,
            face_backbone=cfg.extra.get("face_backbone", "resnet18"),
            ela_backbone=cfg.extra.get("ela_backbone", "resnet18"),
            pretrained=cfg.pretrained, freeze_aux=freeze_aux).to(device)
        print(f"[train] freeze_aux={freeze_aux}")
        print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                       lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        Path("checkpoints").mkdir(exist_ok=True)
        best_audet = float("inf")
        for epoch in range(1, cfg.epochs + 1):
            train_loss, *_ = run_epoch_twostream(model, train_loader, device, criterion, optimizer, scaler)
            val_loss, val_scores, val_labels = run_epoch_twostream(model, val_loader, device, criterion)
            m = evaluate(val_scores, val_labels)
            print(f"epoch {epoch:>2}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"AuDET={m['audet']:.4f} APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.4f}")
            if m["audet"] < best_audet:
                best_audet = m["audet"]
                ckpt = Path("checkpoints") / f"{cfg.name}.pt"
                _safe_save({"model": model.state_dict(), "config": vars(cfg), "epoch": epoch,
                            "metrics": m, "model_kind": model_kind}, ckpt)
                print(f"  -> saved {ckpt} (AuDET={best_audet:.4f})")
        return

    # --- baseline / consistency path (develop's existing pipeline, unchanged) ---
    train_loader, val_loader, probe_loader = build_loaders(cfg, data_cfg)
    print(f"[train] config '{cfg.name}' | device={device} | backbone={cfg.backbone} | "
          f"image_size={data_cfg['image_size']} mean={data_cfg['mean']}")
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    if probe_loader is not None:
        print(f"[train] probe={len(probe_loader.dataset)} (recapture, seed={cfg.extra.get('recapture_probe_seed', 0)})")

    model_type = cfg.extra.get("model_type", "baseline")
    if model_type == "consistency":
        from freuid.models import build_consistency_model
        model = build_consistency_model(cfg).to(device)
    else:
        model = build_model(cfg.backbone, cfg.pretrained).to(device)
        train_last_k = cfg.extra.get("train_last_k_blocks")
        if train_last_k is not None:
            from freuid.optim import freeze_all_but_last_k_blocks
            freeze_all_but_last_k_blocks(model, int(train_last_k))
        if cfg.extra.get("grad_checkpointing", False):
            if hasattr(model, "set_grad_checkpointing"):
                model.set_grad_checkpointing(True)
                print("[train] gradient checkpointing enabled")
            else:
                print(f"[train] WARNING: grad_checkpointing=True but {cfg.backbone} has no set_grad_checkpointing")

    criterion = torch.nn.BCEWithLogitsLoss()
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    auc_weight = float(cfg.extra.get("auc_loss_weight", 0.0))
    if auc_weight > 0.0:
        print(f"[train] auc_loss_weight={auc_weight} (pairwise soft-AUC term active)")

    amp_enabled = (bool(cfg.extra.get("amp", False)) or bool(cfg.extra.get("use_amp", False))) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None
    if amp_enabled:
        print("[train] AMP enabled (autocast + GradScaler)")

    ckpt_key = cfg.extra.get("checkpoint_metric", "audet")
    tie_key = {"probe_audet": "probe_apcer_at_1pct_bpcer", "audet": "apcer_at_1pct_bpcer"}.get(ckpt_key)
    probe_seed = cfg.extra.get("recapture_probe_seed", 0)

    Path("checkpoints").mkdir(exist_ok=True)
    best_metric = float("inf")
    best_tiebreak = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, device, criterion, optimizer, auc_weight, scaler=scaler)
        val_loss, val_scores, val_labels = run_epoch(model, val_loader, device, criterion, scaler=scaler)
        m = evaluate(val_scores, val_labels)
        last_lrs = scheduler.get_last_lr()
        lr = last_lrs[0] if len(last_lrs) == 1 else max(last_lrs)
        scheduler.step()

        probe_str = ""
        if probe_loader is not None:
            pm = _run_probe(model, probe_loader, device, criterion, probe_seed, scaler=scaler)
            m["probe_audet"] = pm["audet"]
            m["probe_apcer_at_1pct_bpcer"] = pm["apcer_at_1pct_bpcer"]
            probe_str = f" probe_AuDET={m['probe_audet']:.6f}"

        lr_str = f"lr={lr:.2e}" if len(last_lrs) == 1 else f"lr_head={lr:.2e} lr_min={min(last_lrs):.2e}"
        print(
            f"\n[epoch {epoch:>2}/{cfg.epochs}] {lr_str} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"AuDET={m['audet']:.4f} APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.4f}{probe_str}"
        )

        current = m.get(ckpt_key, m["audet"])
        current_tie = m.get(tie_key, float("inf")) if tie_key else float("inf")
        improved = current < best_metric or (current == best_metric and current_tie < best_tiebreak)
        if improved:
            best_metric = current
            best_tiebreak = current_tie
            ckpt = Path("checkpoints") / f"{cfg.name}.pt"
            _safe_save(
                {"model": model.state_dict(), "config": vars(cfg), "epoch": epoch, "metrics": m},
                ckpt,
            )
            print(f"  -> saved {ckpt} ({ckpt_key}={best_metric:.6f})")


if __name__ == "__main__":
    main()
