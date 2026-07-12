"""Training entrypoint.

    uv run python -m freuid.train --config configs/baseline.yaml

Trains a binary fraud classifier (BCEWithLogitsLoss), validates each epoch with the
competition metrics, and checkpoints the best AuDET to checkpoints/<name>.pt.
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
    forward_with_extras,
    lodo_split,
    stratified_split,
    unpack_and_move,
)
from freuid.loss import combined_loss
from freuid.metrics import evaluate
from freuid.models import build_model
from freuid.transforms import build_transforms, resolve_data_config
from freuid.utils import pick_device, seed_everything


def run_epoch(
    model, loader, device, criterion, optimizer=None, auc_weight: float = 0.0, scaler=None,
    pair_hinge_weight: float = 0.0, pair_margin: float = 1.0,
):
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

    ``pair_hinge_weight`` > 0 (photosub_v0 only, train only) switches the batch-unpacking to
    freuid.photosub.mixing.unpack_photosub_batch (imgs, labels, pair_ids) instead of the usual
    unpack_and_move, and adds freuid.loss.pair_hinge_loss to the loss. 0.0 (every other config)
    is bit-for-bit identical to the original path -- face_meta/face_crop are never combined
    with photosub mixing (model_type=baseline only), so this is a clean either/or, not a merge
    of the two unpacking conventions. Returns a 3rd loop-level value, ``n_pairs_found``
    (summed over batches, train mode only) -- purely for the smoke-run "twin pairs verified by
    assertion" log line; 0 whenever pair_hinge_weight == 0.0.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n_seen, all_scores, all_labels = 0.0, 0, [], []
    n_pairs_found = 0
    use_amp = scaler is not None and scaler.is_enabled()
    use_pairs = pair_hinge_weight > 0.0 and is_train
    for batch in tqdm(loader, leave=False):
        if use_pairs:
            from freuid.photosub.mixing import unpack_photosub_batch
            imgs, labels_dev, pair_ids_dev = unpack_photosub_batch(batch, device)
            face_meta_dev = face_crop_dev = None
        else:
            imgs, labels, face_meta_dev, face_crop_dev = unpack_and_move(batch, device)
            labels_dev = labels.to(device)
        # eval 모드에서는 불필요한 그래디언트 계산을 끄는 컨텍스트 매니저
        with torch.set_grad_enabled(is_train):
            amp_ctx = (
                torch.autocast(device_type="cuda", enabled=use_amp)
                if device.type == "cuda" else contextlib.nullcontext()
            )
            with amp_ctx:
                # 모델 forward() 호출. imgs [B, 3, H, W] -> logits [B, 1] (B=batch_size)
                logits = forward_with_extras(model, imgs, face_meta_dev, face_crop_dev)
                # BCE + optional pairwise AUC term (train only; val always uses plain BCE)
                _aw = auc_weight if is_train else 0.0
                loss = combined_loss(logits, labels_dev, criterion, _aw)
                if use_pairs:
                    from freuid.loss import pair_hinge_loss
                    from freuid.photosub.mixing import count_pairs_in_batch
                    loss = loss + pair_hinge_weight * pair_hinge_loss(
                        logits, labels_dev, pair_ids_dev, margin=pair_margin
                    )
                    n_pairs_found += count_pairs_in_batch(labels_dev, pair_ids_dev)
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
        return mean_loss, None, None, n_pairs_found
    return mean_loss, torch.cat(all_scores).numpy(), torch.cat(all_labels).numpy(), 0


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

    # model_type dispatch: add new model types here (e.g. model_type="bayar_fusion")
    model_type = cfg.extra.get("model_type", "baseline")
    overlay_cfg = cfg.extra.get("overlay", {}) if model_type == "bayar_fusion" else {}
    return_face_crop = model_type == "bayar_fusion"
    face_crop_size = int(overlay_cfg.get("crop_size", 224))
    face_crop_margin = float(overlay_cfg.get("crop_margin", 0.75))

    # Regions cache: used when extra.use_rectify=True (consistency path with card
    # rectification) OR model_type=bayar_fusion (needs the cached face box for crops,
    # regardless of use_rectify -- see use_rectified_as_main below).
    _rdir: Path | None = None
    if cfg.extra.get("use_rectify", False) or model_type == "bayar_fusion":
        from freuid.preprocess import regions_dir as _get_rdir
        _rdir = _get_rdir(cfg.data_dir)
        if _rdir.exists():
            print(f"[train] loading regions cache from {_rdir}")
        else:
            print(f"[train] WARNING: regions cache not found at {_rdir}; using raw images/no face data")
            _rdir = None

    # bayar_fusion keeps DINOv2's main input as the RAW image (matching the baseline/
    # finetune path exactly, so this experiment isolates "add an overlay branch" without
    # also silently switching DINOv2 to see rectified cards) -- only use_rectify (the
    # consistency path) wants the rectified card as the main image.
    use_rectified_as_main = cfg.extra.get("use_rectify", False)

    # Face-region head needs (id, valid) face boxes alongside each batch. Only the
    # consistency path knows how to consume the extra tensor, so gate on model_type too.
    return_face_meta = model_type == "consistency" and bool(cfg.extra.get("use_face_region", False))

    _ds_face_kwargs = dict(
        return_face_meta=return_face_meta,
        return_face_crop=return_face_crop,
        face_crop_size=face_crop_size,
        face_crop_margin=face_crop_margin,
        use_rectified_as_main=use_rectified_as_main,
    )
    # No degradation on the 224px face-crop view (bayar_fusion only): recapture-style
    # augmentation erases the fine noise residue BayarConv2d depends on, defeating the
    # branch before it can learn anything from it. Bare ToTensor -> [0,1]-scaled float;
    # OverlayStream normalizes internally for its own RGB branch (see its docstring) --
    # nothing upstream should pre-normalize this tensor.
    face_crop_transform = None
    if return_face_crop:
        from torchvision.transforms import ToTensor as _ToTensor
        face_crop_transform = _ToTensor()

    synth_prob = float(cfg.extra.get("synth_tamper_prob", 0.0))
    if synth_prob > 0.0:
        from freuid.augment import SynthTamperWrapper, recapture_transforms
        _base_train_ds = FreuidDataset(
            cfg.data_dir, "train", None, ids=train_ids, regions_dir=_rdir,
            **_ds_face_kwargs,
        )
        _tamper_tf = recapture_transforms(size, mean, std)
        train_ds = SynthTamperWrapper(
            _base_train_ds,
            clean_transform=train_tf,
            tamper_transform=_tamper_tf,
            prob=synth_prob,
            seed=cfg.seed,
            face_crop_transform=face_crop_transform,
        )
        _n_bona = sum(1 for s in _base_train_ds.samples if s.label == 0)
        print(
            f"[train] synth_tamper: prob={synth_prob:.2f} "
            f"| {_n_bona} bona-fide → ~{int(_n_bona * synth_prob)} synthetic positives/epoch "
            f"| donor_pool={len(train_ds._donor_pool)}"
        )
    else:
        train_ds = FreuidDataset(
            cfg.data_dir, "train", train_tf, ids=train_ids, regions_dir=_rdir,
            face_crop_transform=face_crop_transform,
            **_ds_face_kwargs,
        )
    val_ds = FreuidDataset(
        cfg.data_dir, "train", val_tf, ids=val_ids, regions_dir=_rdir,
        face_crop_transform=face_crop_transform,
        **_ds_face_kwargs,
    )
    pin_memory = torch.cuda.is_available()  # unsupported/no-op on MPS, only helps CUDA

    # Photosub mixing (photosub_v0): adds offline-generated photo-substitution rows to the
    # train split + twin-pair batching. Gated on extra.photosub.enabled so every other config
    # is completely unaffected. model_type=baseline only (photosub_v0 doesn't combine with the
    # consistency/bayar_fusion face_meta/face_crop paths).
    photosub_cfg = cfg.extra.get("photosub", {}) or {}
    if photosub_cfg.get("enabled", False):
        if model_type != "baseline":
            raise ValueError("photosub mixing only supports model_type=baseline")
        from freuid.photosub.mixing import (
            PhotosubTwinDataset,
            TwinPairBatchSampler,
            load_photosub_rows,
            select_mixed_rows,
        )
        from freuid.photosub.pipeline import photosub_generated_dir

        rows_csv = Path(photosub_cfg.get("rows_csv", photosub_generated_dir(cfg.data_dir) / "rows.csv"))
        rows_df = load_photosub_rows(rows_csv)
        base_samples = train_ds.samples if hasattr(train_ds, "samples") else train_ds.base.samples
        base_n_positive = sum(1 for s in base_samples if s.label == 1)
        mode_weights = photosub_cfg.get("mode_weights", {"A": 0.44, "B": 0.33, "C": 0.22, "D": 0.0})
        share = float(photosub_cfg.get("share", 0.15))
        per_template_cap = photosub_cfg.get("per_template_cap")  # None = no cap (photosub_v0 behavior)
        selected = select_mixed_rows(
            rows_df, train_ids, mode_weights, share, base_n_positive, seed=cfg.seed,
            per_template_cap=per_template_cap,
        )
        if selected.empty:
            print(
                f"[train] WARNING: photosub.enabled=True but 0 rows selected from {rows_csv} "
                f"({len(rows_df)} rows total in file) -- check rows_csv / share / mode_weights"
            )
        else:
            mode_counts = selected["mode"].value_counts().to_dict()
            print(
                f"[train] photosub mixing: base_n_positive={base_n_positive} share={share} -> "
                f"{len(selected)} rows selected {mode_counts} from {rows_csv}"
            )
        twin_ds = PhotosubTwinDataset(train_ds, selected, train_tf)
        twin_pair_prob = float(photosub_cfg.get("twin_pair_prob", 0.5))
        batch_sampler = TwinPairBatchSampler(
            twin_ds, batch_size=cfg.batch_size, twin_pair_prob=twin_pair_prob, seed=cfg.seed,
        )
        print(f"[train] photosub twin_pair_prob={twin_pair_prob} (n_base={twin_ds.n_base}, n_photosub={len(selected)})")
        train_loader = DataLoader(
            twin_ds, batch_sampler=batch_sampler, num_workers=cfg.num_workers, pin_memory=pin_memory,
        )
    else:
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
        probe_ds = FreuidDataset(
            cfg.data_dir, "train", probe_tf, ids=val_ids, regions_dir=_rdir,
            face_crop_transform=face_crop_transform,
            **_ds_face_kwargs,
        )
        probe_loader = DataLoader(
            probe_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        )

    return train_loader, val_loader, probe_loader


def _run_probe(model, probe_loader, device, criterion, seed: int, scaler=None) -> dict[str, float]:
    """Evaluate the degraded probe with a fixed seed so augmentation is the same each epoch."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    _, scores, labels, _ = run_epoch(model, probe_loader, device, criterion, scaler=scaler)
    return evaluate(scores, labels)


def _check_init_loss(model, loader, device, criterion, tol: float = 0.3, photosub_batches: bool = False) -> None:
    """Assert that BCE on the first train batch ≈ ln(2) before any weight update.

    A fresh classifier head (bias=0, small weights) outputs logits ≈ 0, so
    sigmoid → 0.5 and BCE → ln(2) ≈ 0.693 on any class mix. Failing this usually
    means labels are on the wrong scale, the head bias was initialised incorrectly,
    or the loss function is mis-wired.

    ``photosub_batches=True`` unpacks via freuid.photosub.mixing.unpack_photosub_batch
    (imgs, labels, pair_ids) instead of unpack_and_move -- the pair_id 3rd element must never
    be handed to forward_with_extras as a face_meta tensor (model_type=baseline has no 2nd
    forward arg at all).
    """
    model.eval()
    if photosub_batches:
        from freuid.photosub.mixing import unpack_photosub_batch
        imgs, labels, _ = unpack_photosub_batch(next(iter(loader)), device)
        face_meta = face_crop = None
    else:
        imgs, labels, face_meta, face_crop = unpack_and_move(next(iter(loader)), device)
    with torch.no_grad():
        logits = forward_with_extras(model, imgs, face_meta, face_crop)
        loss = criterion(logits, labels.float().unsqueeze(1).to(device)).item()
    model.train()
    expected = math.log(2)  # ≈ 0.693
    assert abs(loss - expected) < tol, (
        f"init BCE={loss:.4f} expected ≈{expected:.4f} (tol={tol}). "
        "Check: labels not shuffled/inverted, loss not pre-averaged with wrong sign, "
        "head bias not set to a constant."
    )
    print(f"[sanity] init BCE={loss:.4f} ~= ln2={expected:.4f} (tol={tol}) OK")


def _sanity_overfit(
    model, loader, device, criterion, steps: int = 100, target: float = 0.02,
    photosub_batches: bool = False,
) -> None:
    """Overfit a single batch to near-zero loss; asserts the forward+backward path works.

    Runs on a COPY of the model so the real training weights are untouched.
    Uses SGD (no momentum) so convergence is purely the model's capacity.

    See ``_check_init_loss`` for what ``photosub_batches`` changes.
    """
    import copy
    m = copy.deepcopy(model)
    if photosub_batches:
        from freuid.photosub.mixing import unpack_photosub_batch
        imgs, labels, _ = unpack_photosub_batch(next(iter(loader)), device)
        face_meta = face_crop = None
    else:
        imgs, labels, face_meta, face_crop = unpack_and_move(next(iter(loader)), device)
    targets = labels.float().unsqueeze(1).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    m.train()

    def _step_forward():
        return forward_with_extras(m, imgs, face_meta, face_crop)

    for _ in range(steps):
        opt.zero_grad()
        criterion(_step_forward(), targets).backward()
        opt.step()
    final = criterion(_step_forward(), targets).item()
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

    train_loader, val_loader, probe_loader = build_loaders(cfg, data_cfg)
    photosub_enabled = bool((cfg.extra.get("photosub") or {}).get("enabled", False))
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    if probe_loader is not None:
        print(f"[train] probe={len(probe_loader.dataset)} (recapture, seed={cfg.extra.get('recapture_probe_seed', 0)})")

    # model_type dispatch: add new model types here (e.g. model_type="consistency")
    model_type = cfg.extra.get("model_type", "baseline")
    if model_type == "consistency":
        from freuid.models import build_consistency_model
        model = build_consistency_model(cfg).to(device)
    elif model_type == "bayar_fusion":
        from freuid.models import build_bayar_fusion_model
        model = build_bayar_fusion_model(cfg).to(device)
        if cfg.extra.get("grad_checkpointing", False):
            if hasattr(model.dino, "set_grad_checkpointing"):
                model.dino.set_grad_checkpointing(True)
                print("[train] gradient checkpointing enabled (dino sub-module)")
            else:
                print(f"[train] WARNING: grad_checkpointing=True but {cfg.backbone} has no set_grad_checkpointing")
    else:
        model = build_model(cfg.backbone, cfg.pretrained).to(device)
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
                print(f"[train] WARNING: grad_checkpointing=True but {cfg.backbone} has no set_grad_checkpointing")
    criterion = torch.nn.BCEWithLogitsLoss()

    # Always check init loss before any weight updates.
    _check_init_loss(model, train_loader, device, criterion, photosub_batches=photosub_enabled)

    if args.sanity:
        _sanity_overfit(model, train_loader, device, criterion, photosub_batches=photosub_enabled)
        print("[sanity] all checks passed — exiting")
        return

    llrd_cfg = cfg.extra.get("llrd") or {}
    if llrd_cfg.get("enabled", False):
        from freuid.optim import build_warmup_cosine_scheduler
        if model_type == "bayar_fusion":
            from freuid.models import build_bayar_fusion_param_groups
            param_groups = build_bayar_fusion_param_groups(
                model, base_lr=cfg.lr, weight_decay=cfg.weight_decay,
                decay=float(llrd_cfg.get("decay", 0.7)),
            )
        else:
            from freuid.optim import build_llrd_param_groups
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
        # Cosine decay over the run: anneals LR toward 0 by the final epoch. Helps the
        # pretrained RGB backbone settle rather than oscillating at a flat LR.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    auc_weight = float(cfg.extra.get("auc_loss_weight", 0.0))
    if auc_weight > 0.0:
        print(f"[train] auc_loss_weight={auc_weight} (pairwise soft-AUC term active)")

    # Photosub twin-pair hinge (photosub_v0 only) -- see build_loaders' photosub_cfg block and
    # freuid.loss.pair_hinge_loss. 0.0 (every other config, or photosub.enabled=False) is
    # bit-for-bit identical to the pre-photosub run_epoch path.
    photosub_cfg = cfg.extra.get("photosub", {}) or {}
    pair_hinge_weight = float(photosub_cfg.get("pair_hinge_weight", 0.0)) if photosub_cfg.get("enabled", False) else 0.0
    pair_margin = float(photosub_cfg.get("pair_margin", 1.0))
    if pair_hinge_weight > 0.0:
        print(f"[train] photosub pair_hinge_weight={pair_hinge_weight} margin={pair_margin}")
    run_photosub_probes = photosub_cfg.get("enabled", False) and photosub_cfg.get("probe_hooks", True)
    if run_photosub_probes:
        from freuid.photosub.probes import run_probe_hooks
        probe_hook_tf = build_transforms(data_cfg["image_size"], False, data_cfg["mean"], data_cfg["std"])

    # Gate-composite checkpoint selection (photosub_v1, opt-in -- see
    # freuid.photosub.checkpoint_select and docs/photosub_v1_spec.md's pre-registered gates).
    # Requires run_photosub_probes (the gates read off that stage's per-epoch output).
    ckpt_sel_cfg = photosub_cfg.get("checkpoint_selection", {}) or {}
    run_checkpoint_selection = run_photosub_probes and ckpt_sel_cfg.get("enabled", False)
    if run_checkpoint_selection:

        import pandas as _pd

        from freuid.photosub.checkpoint_select import (
            CheckpointTracker,
            GateThresholds,
            average_state_dicts,
            evaluate_gates,
            extract_boundary_baseline,
            extract_ceiling_baseline,
        )
        from freuid.photosub.probes import probes_dir
        _deep_csv = probes_dir(cfg.data_dir) / "missed_frauds_deep_ids.csv"
        _deep_df = _pd.read_csv(_deep_csv, dtype={"id": str})
        _deep9_baseline_logits = dict(zip(_deep_df["id"], _deep_df["logit"]))
        gate_thresholds = GateThresholds(
            deep9_majority_frac=float(ckpt_sel_cfg.get("deep9_majority_frac", 0.5)),
            ceiling_drop_tolerance=float(ckpt_sel_cfg.get("ceiling_drop_tolerance", 0.20)),
            stability_window=int(ckpt_sel_cfg.get("stability_window", 2)),
        )
        checkpoint_tracker = CheckpointTracker(thresholds=gate_thresholds)
        _ceiling_baseline = None
        _boundary_baseline = None
        last_k = int(ckpt_sel_cfg.get("last_k_averaging", 0))  # 0 = disabled
        retain_window = max(gate_thresholds.stability_window, last_k, 1)
        _recent_state_dicts: dict[int, dict] = {}  # epoch -> CPU state_dict, ring buffer
        print(f"[train] gate-composite checkpoint selection enabled: {gate_thresholds} "
              f"last_k_averaging={last_k or 'off'} retain_window={retain_window}")

    # AMP: disabled GradScaler is a documented no-op passthrough, so this is safe to
    # always construct and thread through run_epoch -- every config without
    # extra.amp=True gets scaler.is_enabled()==False and reproduces the exact FP32 path.
    amp_enabled = bool(cfg.extra.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None
    if amp_enabled:
        print("[train] AMP enabled (autocast + GradScaler)")

    # Checkpoint criterion: "probe_audet" (recapture compass) or "audet" (in-domain).
    # Ties on the primary metric are broken by the matching APCER@1%BPCER (secondary
    # competition metric) so two epochs with identical AuDET don't checkpoint arbitrarily.
    ckpt_key = cfg.extra.get("checkpoint_metric", "audet")
    tie_key = {"probe_audet": "probe_apcer_at_1pct_bpcer", "audet": "apcer_at_1pct_bpcer"}.get(ckpt_key)
    probe_seed = cfg.extra.get("recapture_probe_seed", 0)

    Path("checkpoints").mkdir(exist_ok=True)
    best_metric = float("inf")
    best_tiebreak = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss, _, _, n_pairs_found = run_epoch(
            model, train_loader, device, criterion, optimizer, auc_weight, scaler=scaler,
            pair_hinge_weight=pair_hinge_weight, pair_margin=pair_margin,
        )
        if pair_hinge_weight > 0.0:
            print(f"  [photosub] twin pairs found this epoch: {n_pairs_found}")
        val_loss, val_scores, val_labels, _ = run_epoch(model, val_loader, device, criterion, scaler=scaler)
        m = evaluate(val_scores, val_labels)
        last_lrs = scheduler.get_last_lr()
        lr = last_lrs[0] if len(last_lrs) == 1 else max(last_lrs)
        scheduler.step()

        probe_str = ""
        if probe_loader is not None:
            pm = _run_probe(model, probe_loader, device, criterion, probe_seed, scaler=scaler)
            m["probe_audet"] = pm["audet"]
            m["probe_apcer_at_1pct_bpcer"] = pm["apcer_at_1pct_bpcer"]
            m["probe_freuid"] = pm["freuid"]
            probe_str = f" probe_AuDET={m['probe_audet']:.6f} probe_FREUID={m['probe_freuid']:.6f}"

        if run_photosub_probes:
            probe_metrics = run_probe_hooks(model, cfg.data_dir, probe_hook_tf, device, val_scores, val_labels)
            for d in probe_metrics["probe_deep_detail"]:
                print(f"  [probe_deep] {d['id']} logit={d['logit']:.4f} score={d['score']:.4f} pct_rank={d['pct_rank']:.2f}")
            print(
                "  [probe] deep(9): mean_logit={:.4f} mean_pct_rank={:.2f} | "
                "boundary(59): mean_logit={:.4f} mean_pct_rank={:.2f} | "
                "clean_floor(295): mean_logit={:.4f} | ceiling(200): mean_logit={:.4f}".format(
                    probe_metrics["probe_deep_mean_logit"], probe_metrics["probe_deep_mean_pct_rank"],
                    probe_metrics["probe_boundary_mean_logit"], probe_metrics["probe_boundary_mean_pct_rank"],
                    probe_metrics["probe_clean_floor_mean_logit"], probe_metrics["probe_ceiling_mean_logit"],
                )
            )
            print(
                "  [probe_threshold] score={:.4f} rank={}/{} (n_bonafide={}) | "
                "boundary missed: {}/{} below threshold | ceiling exposure: {}/{} above threshold".format(
                    probe_metrics["probe_threshold_score"], probe_metrics["probe_threshold_rank"],
                    val_scores.shape[0], probe_metrics["probe_n_val_bonafide"],
                    probe_metrics["probe_boundary_below_threshold"], probe_metrics["probe_boundary_total"],
                    probe_metrics["probe_ceiling_above_threshold"], probe_metrics["probe_ceiling_total"],
                )
            )
            ceiling_by_template = probe_metrics["probe_ceiling_by_template"]
            if ceiling_by_template is not None:
                template_str = ", ".join(
                    f"{r.template}={int(r.n_above)}/{int(r.n_total)}"
                    for r in ceiling_by_template.itertuples()
                )
                print(f"  [probe_threshold] ceiling budget exposure by template: {template_str}")

        if run_checkpoint_selection:
            gate_result = evaluate_gates(
                epoch, probe_metrics, _deep9_baseline_logits, _ceiling_baseline, _boundary_baseline,
                gate_thresholds,
            )
            checkpoint_tracker.record_epoch(gate_result)
            if epoch == 1:
                _ceiling_baseline = extract_ceiling_baseline(probe_metrics)
                _boundary_baseline = extract_boundary_baseline(probe_metrics)
            print(
                f"  [gate] composite={gate_result.composite_pass} "
                f"deep9={gate_result.deep9_pass} ceiling={gate_result.ceiling_pass} "
                f"boundary={gate_result.boundary_pass} | "
                f"eligible_epochs={checkpoint_tracker.eligible_epochs()}"
            )
            _recent_state_dicts[epoch] = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            for stale_epoch in [e for e in _recent_state_dicts if e <= epoch - retain_window]:
                del _recent_state_dicts[stale_epoch]

        lr_str = f"lr={lr:.2e}" if len(last_lrs) == 1 else f"lr_head={lr:.2e} lr_min={min(last_lrs):.2e}"
        print(
            f"\n[epoch {epoch:>2}/{cfg.epochs}] {lr_str} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"AuDET={m['audet']:.4f} APCER@1%BPCER={m['apcer_at_1pct_bpcer']:.4f} FREUID={m['freuid']:.4f}{probe_str}"
        )

        current = m.get(ckpt_key, m["audet"])
        current_tie = m.get(tie_key, float("inf")) if tie_key else float("inf")
        improved = current < best_metric or (current == best_metric and current_tie < best_tiebreak)
        if improved:
            best_metric = current
            best_tiebreak = current_tie
            ckpt = Path("checkpoints") / f"{cfg.name}.pt"
            torch.save(
                {"model": model.state_dict(), "config": vars(cfg), "epoch": epoch, "metrics": m},
                ckpt,
            )
            print(f"  -> saved {ckpt} ({ckpt_key}={best_metric:.6f})")

    if run_checkpoint_selection:
        _finalize_checkpoint_selection(
            checkpoint_tracker, _recent_state_dicts, cfg, last_k, average_state_dicts,
        )


def _finalize_checkpoint_selection(
    checkpoint_tracker, recent_state_dicts: dict[int, dict], cfg: Config, last_k: int,
    average_state_dicts_fn,
) -> None:
    """Post-training selection: gate-composite latest-stable, plus optional last-k averaging.
    ADDITIVE artifacts alongside the existing metric-based `checkpoints/<name>.pt` -- never
    overwrites or replaces it, so this feature can never change behavior for a run that opts in
    but whose gates never confirm eligible, or for any other config that doesn't opt in at all.
    """
    eligible = checkpoint_tracker.eligible_epochs()
    latest_stable = checkpoint_tracker.latest_stable_epoch()
    print(f"[train] checkpoint selection: eligible_epochs={eligible} latest_stable={latest_stable} "
          f"retained_in_buffer={sorted(recent_state_dicts)}")

    if latest_stable is not None and latest_stable in recent_state_dicts:
        ckpt = Path("checkpoints") / f"{cfg.name}_gate_selected.pt"
        torch.save(
            {
                "model": recent_state_dicts[latest_stable], "config": vars(cfg),
                "epoch": latest_stable, "selection": "gate_composite_latest_stable",
            },
            ckpt,
        )
        print(f"  -> saved {ckpt} (epoch {latest_stable}, gate-composite latest-stable)")
    elif latest_stable is not None:
        print(f"  latest_stable=epoch {latest_stable} fell outside the retained checkpoint "
              "buffer -- increase retain_window / last_k_averaging to capture it next run")
    else:
        print("  no latest-stable epoch found (gates never confirmed a stable eligible window) "
              "-- falling back to the metric-based checkpoint only")

    if last_k > 0:
        eligible_in_buffer = [e for e in eligible if e in recent_state_dicts]
        pool = (
            eligible_in_buffer[-last_k:] if eligible_in_buffer
            else sorted(recent_state_dicts)[-last_k:]
        )
        if pool:
            averaged = average_state_dicts_fn([recent_state_dicts[e] for e in pool])
            ckpt = Path("checkpoints") / f"{cfg.name}_lastk_avg.pt"
            torch.save(
                {"model": averaged, "config": vars(cfg), "epochs_averaged": pool,
                 "selection": f"last_{last_k}_averaging"},
                ckpt,
            )
            print(f"  -> saved {ckpt} (averaged epochs {pool})")
        else:
            print("  last_k_averaging enabled but no checkpoints retained to average")


if __name__ == "__main__":
    main()
