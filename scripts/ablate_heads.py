"""S3 head ablation harness.

Trains the consistency model for a few epochs under each head combination (global-only,
patch-only, face-only, all-fused) on the same data/split/seed, and reports the best
probe_audet reached by each -- the S3 gate requires each head to help alone, and the
full fusion to be at least as good as the best single head.

Usage (run from repo root on the VESSL workspace, where the regions cache + data + GPU live):

    python scripts/ablate_heads.py --config configs/consistency_v1.yaml \\
        --epochs 3 --limit 2048

``--limit`` caps train/val/probe set size (see Config.limit) so the sweep is cheap; raise
it for a more trustworthy signal once the wiring is confirmed to work.
"""

from __future__ import annotations

import argparse
import copy

import torch

from freuid.config import load_config
from freuid.models import build_consistency_model
from freuid.train import _run_probe, build_loaders, run_epoch
from freuid.transforms import resolve_data_config
from freuid.utils import pick_device, seed_everything

HEAD_COMBOS: dict[str, tuple[bool, bool]] = {
    # name -> (use_patch_consistency, use_face_region)
    "global_only": (False, False),
    "patch_only": (True, False),
    "face_only": (False, True),
    "fusion_all": (True, True),
}


def run_combo(name: str, use_patch: bool, use_face: bool, base_cfg, epochs: int) -> float:
    cfg = copy.deepcopy(base_cfg)
    cfg.name = f"{base_cfg.name}_ablate_{name}"
    cfg.epochs = epochs
    cfg.extra["use_patch_consistency"] = use_patch
    cfg.extra["use_face_region"] = use_face

    seed_everything(cfg.seed)
    device = pick_device()
    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    train_loader, val_loader, probe_loader = build_loaders(cfg, data_cfg)
    if probe_loader is None:
        raise SystemExit("ablation harness requires extra.use_recapture_probe: true in the config")

    model = build_consistency_model(cfg).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    probe_seed = cfg.extra.get("recapture_probe_seed", 0)

    best_probe = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        train_loss, *_ = run_epoch(model, train_loader, device, criterion, optimizer)
        pm = _run_probe(model, probe_loader, device, criterion, probe_seed)
        best_probe = min(best_probe, pm["audet"])
        print(
            f"  [{name}] epoch {epoch}/{cfg.epochs} train_loss={train_loss:.4f} "
            f"probe_AuDET={pm['audet']:.6f} (best={best_probe:.6f})"
        )
    return best_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument(
        "--combos", nargs="+", default=list(HEAD_COMBOS),
        help=f"subset of {list(HEAD_COMBOS)} to run",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    base_cfg.limit = args.limit

    results: dict[str, float] = {}
    for name in args.combos:
        use_patch, use_face = HEAD_COMBOS[name]
        print(f"\n=== {name} (patch={use_patch} face={use_face}) ===")
        results[name] = run_combo(name, use_patch, use_face, base_cfg, args.epochs)

    print("\n=== S3 ablation summary (lower probe_AuDET is better) ===")
    for name, score in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:<12} probe_AuDET={score:.6f}")

    single_heads = {k: v for k, v in results.items() if k != "fusion_all"}
    if single_heads and "fusion_all" in results:
        best_single = min(single_heads.values())
        gate = results["fusion_all"] <= best_single
        print(
            f"\n[gate] fusion_all={results['fusion_all']:.6f} vs best_single={best_single:.6f} "
            f"-> {'PASS' if gate else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
