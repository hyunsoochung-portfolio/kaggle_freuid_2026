"""Attention/attribution visualization + localization audit for the finetune_v0 checkpoint.

Four complementary map types for a given input image, all overlaid on the image and saved
side by side:
  1. CLS-to-patch attention (last 3 blocks, mean over heads; per-head grid for the last block).
  2. Grad-CAM (pytorch-grad-cam), target layer = final block's norm1, w.r.t. the fraud logit.
  3. Occlusion sensitivity (model-agnostic): batched sliding 56px gray patch, stride 28.
  4. Patch-token PCA-to-RGB (DINOv2-style), pretrained vs. fine-tuned side by side.

Then a quantitative attribution-vs-ground-truth audit: 200 synthetic tampers (from val
bona-fide images, via scripts/analysis/tamper_bbox.py's replicated synth_tamper + the full
recapture degradation chain with bbox tracking) plus 20 clean controls, scored on IoU@best-
threshold and pointing-game accuracy for the Grad-CAM and occlusion maps.

Token layout (CRITICAL, asserted at runtime): this is vit_base_patch14_reg4_dinov2 --
sequence order is [CLS, 4 register tokens, patch tokens]. At image_size=518, patch_size=14,
grid=37x37=1369 patches, total tokens = 1+4+1369 = 1374. Every function below excludes the
5 prefix tokens (CLS + registers) before reshaping to the spatial grid.

`timm.layers.set_fused_attn(False)` is called at import time -- REQUIRED for map type 1:
timm's Attention module bakes `self.fused_attn` into the instance at __init__ time,
and when fused (torch's scaled_dot_product_attention), `attn_drop` is never called with
explicit post-softmax weights, so a forward hook on it would silently capture nothing. This
must happen before any model is constructed, hence it runs at module import.

No training-code changes: augment.py's tamper/recapture logic is replicated (not modified)
in tamper_bbox.py; everything here only reads the existing finetune_v0.pt checkpoint.

Usage:
    python scripts/analysis/visualize.py --mode figures --n-per-group 10
    python scripts/analysis/visualize.py --mode quant --n-synth 200 --n-clean 20
    python scripts/analysis/visualize.py --mode both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import timm.layers
timm.layers.set_fused_attn(False)  # noqa: E402 -- must precede any model construction, see docstring

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    REPORT_DIR,
    build_finetuned_model,
    build_pretrained_model,
    device_and_seed,
    eval_transform,
    get_split_ids,
    load_checkpoint,
    split_dataframe,
)
from tamper_bbox import BBox, recapture_transforms_with_bbox, synth_tamper_bbox  # noqa: E402

PATCH_SIZE = 14
NUM_REGISTERS = 4
NUM_PREFIX = 1 + NUM_REGISTERS  # CLS + registers
ATTN_DIR = REPORT_DIR / "attn"


def grid_size_for(image_size: int) -> int:
    assert image_size % PATCH_SIZE == 0, f"image_size {image_size} not divisible by patch_size {PATCH_SIZE}"
    return image_size // PATCH_SIZE


def assert_token_layout(n_tokens: int, grid: int) -> None:
    expected = NUM_PREFIX + grid * grid
    assert n_tokens == expected, (
        f"token count {n_tokens} != expected {expected} (1 CLS + {NUM_REGISTERS} registers + "
        f"{grid}x{grid} patches) -- token-layout assumption violated"
    )


# ---------------------------------------------------------------------------
# 1. CLS -> patch attention (forward-hook on attn.attn_drop's INPUT, i.e. the real
#    post-softmax weights -- only works because set_fused_attn(False) was called above)
# ---------------------------------------------------------------------------

class AttentionCapture:
    def __init__(self, model: torch.nn.Module, block_indices: list[int]) -> None:
        self._captured: dict[int, torch.Tensor] = {}
        self._handles = []
        for i in block_indices:
            blk = model.blocks[i]
            self._handles.append(blk.attn.attn_drop.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx: int):
        def hook(module, inp, out):
            self._captured[idx] = inp[0].detach()  # (B, heads, N, N), post-softmax, pre-dropout
        return hook

    def get(self, idx: int) -> torch.Tensor:
        return self._captured[idx]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()


@torch.no_grad()
def cls_patch_attention(model: torch.nn.Module, img: torch.Tensor, grid: int, last_n: int = 3) -> dict:
    """Returns {block_idx: {"mean": (B,g,g), "per_head": (B,heads,g,g)}} for the last `last_n` blocks."""
    n_blocks = len(model.blocks)
    block_idxs = list(range(n_blocks - last_n, n_blocks))
    cap = AttentionCapture(model, block_idxs)
    model(img)
    out = {}
    for i in block_idxs:
        attn = cap.get(i)  # (B, heads, N, N)
        assert_token_layout(attn.shape[-1], grid)
        cls_to_patch = attn[:, :, 0, NUM_PREFIX:]  # (B, heads, grid*grid) -- registers excluded
        mean_heads = cls_to_patch.mean(dim=1)
        out[i] = {
            "mean": mean_heads.reshape(-1, grid, grid).cpu().numpy(),
            "per_head": cls_to_patch.reshape(cls_to_patch.size(0), cls_to_patch.size(1), grid, grid).cpu().numpy(),
        }
    cap.remove()
    return out


# ---------------------------------------------------------------------------
# 2. Grad-CAM (pytorch-grad-cam), target layer = final block's norm1
# ---------------------------------------------------------------------------

def build_gradcam(model: torch.nn.Module, grid: int):
    from pytorch_grad_cam import GradCAM

    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        n = tensor.shape[1]
        assert_token_layout(n, grid)
        result = tensor[:, NUM_PREFIX:, :].reshape(tensor.size(0), grid, grid, tensor.size(-1))
        return result.permute(0, 3, 1, 2)  # (B, D, grid, grid)

    target_layer = model.blocks[-1].norm1
    return GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)


def gradcam_map(cam, img: torch.Tensor) -> np.ndarray:
    """Returns (B, H, W) in [0,1], already resized to match `img`'s spatial size.

    Since finetune_v0's backbone is fully trainable (requires_grad=True everywhere), each
    GradCAM backward pass accumulates into every parameter's .grad; zero it out afterward so
    repeated calls across hundreds of images don't slowly bloat GPU memory with unused grads.
    """
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    targets = [ClassifierOutputTarget(0) for _ in range(img.size(0))]
    result = cam(input_tensor=img, targets=targets)
    cam.model.zero_grad(set_to_none=True)
    return result


# ---------------------------------------------------------------------------
# 3. Occlusion sensitivity (model-agnostic; batched for speed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def occlusion_map(
    model: torch.nn.Module, img: torch.Tensor, device, mean, std,
    patch: int = 56, stride: int = 28, batch_size: int = 64,
) -> np.ndarray:
    """img: (1, 3, H, W), single image. Returns (H, W) importance map (baseline_prob - occluded_prob,
    upsampled from the coarse position grid to the full image resolution via bilinear interpolation).

    Fills occluded regions with mid-gray (0.5 in raw [0,1] pixel space) transformed through the
    same per-channel normalization as the model's input -- a fixed, image-independent baseline,
    not each image's own mean (which would drift per image and no longer be "gray")."""
    assert img.size(0) == 1, "occlusion_map processes one image at a time"
    H, W = img.shape[-2:]
    baseline_logit = model(img).item()
    baseline_prob = torch.sigmoid(torch.tensor(baseline_logit)).item()

    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    mean_t = torch.tensor(mean, device=img.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=img.device).view(1, 3, 1, 1)
    gray = (0.5 - mean_t) / std_t  # (1,3,1,1), the normalized-space value of raw mid-gray

    positions = [(y, x) for y in ys for x in xs]
    drops = np.zeros(len(positions), dtype=np.float32)
    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        batch = img.repeat(len(chunk), 1, 1, 1).clone()
        for i, (y, x) in enumerate(chunk):
            batch[i, :, y:y + patch, x:x + patch] = gray
        logits = model(batch.to(device))
        probs = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
        drops[start:start + len(chunk)] = baseline_prob - probs

    grid_h, grid_w = len(ys), len(xs)
    coarse = drops.reshape(grid_h, grid_w)
    coarse_t = torch.from_numpy(coarse)[None, None].float()
    full = F.interpolate(coarse_t, size=(H, W), mode="bilinear", align_corners=False)
    return full[0, 0].numpy()


# ---------------------------------------------------------------------------
# 4. Patch-token PCA -> RGB (DINOv2-style), computed on the final block's raw output
# ---------------------------------------------------------------------------

@torch.no_grad()
def patch_token_pca_rgb(model: torch.nn.Module, img: torch.Tensor, grid: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    captured = {}

    def hook(module, inp, out):
        captured["out"] = out.detach()

    h = model.blocks[-1].register_forward_hook(hook)
    model(img)
    h.remove()

    tokens = captured["out"][0]  # (N, D)
    assert_token_layout(tokens.shape[0], grid)
    patch_tokens = tokens[NUM_PREFIX:].float().cpu().numpy()  # (grid*grid, D)

    pca = PCA(n_components=3)
    comps = pca.fit_transform(patch_tokens)  # (grid*grid, 3)
    comps = comps - comps.min(axis=0, keepdims=True)
    span = comps.max(axis=0, keepdims=True)
    comps = comps / np.where(span > 0, span, 1.0)
    return comps.reshape(grid, grid, 3)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _to_display_image(img_tensor: torch.Tensor, mean, std) -> np.ndarray:
    """img_tensor: (3,H,W) normalized -> (H,W,3) uint8-range float in [0,1] for imshow."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    x = (img_tensor.cpu() * std_t + mean_t).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def _overlay(ax, base_img: np.ndarray, heatmap: np.ndarray, title: str, bbox: BBox | None = None) -> None:
    hm = heatmap.astype(np.float32)
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    ax.imshow(base_img)
    ax.imshow(hm, cmap="jet", alpha=0.45, extent=(0, base_img.shape[1], base_img.shape[0], 0))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="lime", facecolor="none"))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def make_comparison_figure(
    model_ft, model_pre, cam, img: torch.Tensor, mean, std, grid: int, device, out_dir: Path, tag: str,
    bbox: BBox | None = None,
) -> None:
    """Main 6-panel figure: original, CLS-attn(last-3 mean), Grad-CAM, occlusion, PCA(pretrained), PCA(fine-tuned).

    `cam` is a GradCAM instance built once by the caller (build_gradcam(model_ft, grid)) and
    reused across images -- constructing a fresh one per image would re-register forward/
    backward hooks on the target layer each time, relying on garbage-collection timing (which
    can be delayed by hook-closure reference cycles) to release the old ones.
    """
    base_img = _to_display_image(img[0], mean, std)

    attn_maps = cls_patch_attention(model_ft, img, grid, last_n=3)
    blocks = sorted(attn_maps)
    attn_last3_mean = np.mean([attn_maps[b]["mean"][0] for b in blocks], axis=0)

    gc_map = gradcam_map(cam, img)[0]

    occ_map = occlusion_map(model_ft, img, device, mean, std)

    pca_ft = patch_token_pca_rgb(model_ft, img, grid)
    pca_pre = patch_token_pca_rgb(model_pre, img, grid)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    axes[0, 0].imshow(base_img)
    axes[0, 0].set_title(f"{tag}: original", fontsize=9)
    axes[0, 0].axis("off")
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        axes[0, 0].add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="lime", facecolor="none"))

    _overlay(axes[0, 1], base_img, attn_last3_mean, "CLS attn (mean, last 3 blocks)", bbox)
    _overlay(axes[0, 2], base_img, gc_map, "Grad-CAM (block[-1].norm1)", bbox)
    _overlay(axes[1, 0], base_img, occ_map, "Occlusion sensitivity (56px/28stride)", bbox)

    axes[1, 1].imshow(pca_pre)
    axes[1, 1].set_title("Patch-token PCA (pretrained DINOv2)", fontsize=9)
    axes[1, 1].axis("off")
    axes[1, 2].imshow(pca_ft)
    axes[1, 2].set_title("Patch-token PCA (fine-tuned finetune_v0)", fontsize=9)
    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_maps.png", dpi=140)
    plt.close(fig)

    # Companion figure: last-3-blocks individual mean maps
    fig2, axes2 = plt.subplots(1, len(blocks), figsize=(4 * len(blocks), 4))
    if len(blocks) == 1:
        axes2 = [axes2]
    for ax, b in zip(axes2, blocks, strict=True):
        _overlay(ax, base_img, attn_maps[b]["mean"][0], f"block {b} (mean over heads)", bbox)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{tag}_attn_blocks.png", dpi=140)
    plt.close(fig2)

    # Companion figure: per-head grid for the LAST block only
    last_block = blocks[-1]
    per_head = attn_maps[last_block]["per_head"][0]  # (heads, g, g)
    n_heads = per_head.shape[0]
    ncols = 4
    nrows = (n_heads + ncols - 1) // ncols
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes3 = np.atleast_2d(axes3)
    for i in range(nrows * ncols):
        ax = axes3[i // ncols, i % ncols]
        if i < n_heads:
            _overlay(ax, base_img, per_head[i], f"head {i}", bbox)
        else:
            ax.axis("off")
    fig3.suptitle(f"{tag}: block {last_block} per-head CLS attention", fontsize=11)
    fig3.tight_layout()
    fig3.savefig(out_dir / f"{tag}_heads.png", dpi=140)
    plt.close(fig3)


# ---------------------------------------------------------------------------
# Quantitative eval: IoU@best-threshold + pointing-game vs. ground-truth bbox
# ---------------------------------------------------------------------------

def _bbox_mask(bbox: BBox, h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def iou_at_best_threshold(heatmap: np.ndarray, gt_mask: np.ndarray, n_thresh: int = 17) -> float:
    hm = heatmap - heatmap.min()
    hm = hm / (hm.max() + 1e-8)
    best = 0.0
    for t in np.linspace(0.1, 0.9, n_thresh):
        pred = hm >= t
        inter = (pred & gt_mask).sum()
        union = (pred | gt_mask).sum()
        if union > 0:
            best = max(best, inter / union)
    return float(best)


def pointing_game_hit(heatmap: np.ndarray, gt_mask: np.ndarray) -> bool:
    idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return bool(gt_mask[idx])


def concentration_score(heatmap: np.ndarray, top_frac: float = 0.10) -> float:
    """Fraction of total (non-negative) map energy contained in the top `top_frac` pixels."""
    hm = np.clip(heatmap, 0, None).ravel()
    total = hm.sum()
    if total <= 0:
        return 0.0
    k = max(1, int(len(hm) * top_frac))
    top_sum = np.partition(hm, -k)[-k:].sum()
    return float(top_sum / total)


def build_donor_pool(df: pd.DataFrame, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    bona = df[df["label"] == 0]
    paths = rng.choice(bona["path"].to_numpy(), size=min(n, len(bona)), replace=False)
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def make_tampered_sample(
    src_path: str, donor_pool: list[np.ndarray], rng: np.random.Generator,
    pipeline, max_retries: int = 5,
) -> tuple[torch.Tensor, BBox] | None:
    """Returns (normalized_image_tensor (3,H,W), ground-truth bbox in image_size-space), or
    None if the tamper region got clipped below min_visibility on every retry."""
    arr = np.array(Image.open(src_path).convert("RGB"))
    donor = donor_pool[int(rng.integers(len(donor_pool)))] if donor_pool else None
    for _ in range(max_retries):
        tampered, _name, bbox = synth_tamper_bbox(arr, rng, donor)
        result = pipeline(image=tampered, bboxes=[bbox], category_ids=[0])
        if result["bboxes"]:
            return result["image"], tuple(result["bboxes"][0][:4])
    return None


def run_quant_eval(cfg, state, device, n_synth: int, n_clean: int, seed: int) -> None:
    ATTN_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ATTN_DIR / "quant_eval.csv"
    done_tags: set[str] = set()
    if out_csv.exists():
        done_tags = set(pd.read_csv(out_csv)["tag"])
        print(f"[visualize] quant eval: {len(done_tags)} samples already done, resuming")

    _, val_ids = get_split_ids(cfg)
    val_df = split_dataframe(cfg, val_ids)
    bona_df = val_df[val_df["label"] == 0].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    donor_pool = build_donor_pool(bona_df, 50, rng)

    transform, data_cfg = eval_transform(cfg)
    mean, std = data_cfg["mean"], data_cfg["std"]
    image_size = data_cfg["image_size"]
    grid = grid_size_for(image_size)
    tamper_pipeline = recapture_transforms_with_bbox(image_size, mean, std)

    model = build_finetuned_model(cfg, state, device)
    cam = build_gradcam(model, grid)

    src_paths = bona_df["path"].sample(n=min(n_synth, len(bona_df)), random_state=seed).tolist()
    rows = []
    for i, src_path in enumerate(src_paths[:n_synth]):
        tag = f"synth_{i:04d}"
        if tag in done_tags:
            continue
        sample = make_tampered_sample(src_path, donor_pool, rng, tamper_pipeline)
        if sample is None:
            continue
        img_tensor, bbox = sample
        img = img_tensor.unsqueeze(0).to(device)
        gt_mask = _bbox_mask(bbox, image_size, image_size)

        gc = gradcam_map(cam, img)[0]
        occ = occlusion_map(model, img, device, mean, std)

        row = {
            "tag": tag, "kind": "synth", "bbox": str(bbox),
            "gradcam_iou": iou_at_best_threshold(gc, gt_mask),
            "gradcam_hit": pointing_game_hit(gc, gt_mask),
            "gradcam_concentration": concentration_score(gc),
            "occlusion_iou": iou_at_best_threshold(occ, gt_mask),
            "occlusion_hit": pointing_game_hit(occ, gt_mask),
            "occlusion_concentration": concentration_score(occ),
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        if (i + 1) % 20 == 0:
            print(f"[visualize] quant eval: synth {i + 1}/{n_synth} -- checkpointed")

    clean_paths = bona_df["path"].sample(n=min(n_clean, len(bona_df)), random_state=seed + 1).tolist()
    for i, src_path in enumerate(clean_paths[:n_clean]):
        tag = f"clean_{i:04d}"
        if tag in done_tags:
            continue
        img = transform(Image.open(src_path).convert("RGB")).unsqueeze(0).to(device)
        gc = gradcam_map(cam, img)[0]
        occ = occlusion_map(model, img, device, mean, std)
        row = {
            "tag": tag, "kind": "clean", "bbox": "",
            "gradcam_iou": float("nan"), "gradcam_hit": float("nan"),
            "gradcam_concentration": concentration_score(gc),
            "occlusion_iou": float("nan"), "occlusion_hit": float("nan"),
            "occlusion_concentration": concentration_score(occ),
        }
        rows.append(row)
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        if (i + 1) % 10 == 0:
            print(f"[visualize] quant eval: clean {i + 1}/{n_clean} -- checkpointed")

    print(f"[visualize] quant eval done -> {out_csv}")


# ---------------------------------------------------------------------------
# Qualitative figures: 10 real-fraud, 10 synthetic, 10 bona-fide, 30 borderline
# ---------------------------------------------------------------------------

def run_figures(cfg, state, device, n_per_group: int, seed: int) -> None:
    fig_dir = ATTN_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    transform, data_cfg = eval_transform(cfg)
    mean, std = data_cfg["mean"], data_cfg["std"]
    image_size = data_cfg["image_size"]
    grid = grid_size_for(image_size)
    tamper_pipeline = recapture_transforms_with_bbox(image_size, mean, std)

    model_ft = build_finetuned_model(cfg, state, device)
    model_pre = build_pretrained_model(cfg, device)
    cam = build_gradcam(model_ft, grid)

    _, val_ids = get_split_ids(cfg)
    val_df = split_dataframe(cfg, val_ids)
    fraud_df = val_df[val_df["label"] == 1]
    bona_df = val_df[val_df["label"] == 0]

    rng = np.random.default_rng(seed)

    def _skip(tag: str) -> bool:
        return (fig_dir / f"{tag}_maps.png").exists()

    # 10 real val fraud
    for i, row in enumerate(fraud_df.sample(n=min(n_per_group, len(fraud_df)), random_state=seed).itertuples()):
        tag = f"realfraud_{i:02d}"
        if _skip(tag):
            continue
        img = transform(Image.open(row.path).convert("RGB")).unsqueeze(0).to(device)
        make_comparison_figure(model_ft, model_pre, cam, img, mean, std, grid, device, fig_dir, tag)
        print(f"[visualize] figure: {tag} done")

    # 10 synthetic tampers (with GT bbox drawn)
    donor_pool = build_donor_pool(bona_df, 50, rng)
    bona_paths = bona_df["path"].sample(n=min(n_per_group, len(bona_df)), random_state=seed + 1).tolist()
    for i, src_path in enumerate(bona_paths):
        tag = f"synthfig_{i:02d}"
        if _skip(tag):
            continue
        sample = make_tampered_sample(src_path, donor_pool, rng, tamper_pipeline)
        if sample is None:
            continue
        img_tensor, bbox = sample
        img = img_tensor.unsqueeze(0).to(device)
        make_comparison_figure(model_ft, model_pre, cam, img, mean, std, grid, device, fig_dir, tag, bbox=bbox)
        print(f"[visualize] figure: {tag} done")

    # 10 bona-fide (clean)
    for i, row in enumerate(bona_df.sample(n=min(n_per_group, len(bona_df)), random_state=seed + 2).itertuples()):
        tag = f"bonafide_{i:02d}"
        if _skip(tag):
            continue
        img = transform(Image.open(row.path).convert("RGB")).unsqueeze(0).to(device)
        make_comparison_figure(model_ft, model_pre, cam, img, mean, std, grid, device, fig_dir, tag)
        print(f"[visualize] figure: {tag} done")

    # 30 borderline public-test ids, if the directory exists (produced by score_distribution_audit.py)
    borderline_dir = REPORT_DIR / "borderline"
    if borderline_dir.exists():
        for i, path in enumerate(sorted(borderline_dir.glob("*.jpeg"))):
            tag = f"borderline_{i:02d}"
            if _skip(tag):
                continue
            img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            make_comparison_figure(model_ft, model_pre, cam, img, mean, std, grid, device, fig_dir, tag)
            print(f"[visualize] figure: {tag} done")
    else:
        print(f"[visualize] {borderline_dir} not found -- skipping the borderline group "
              "(run scripts/analysis/score_distribution_audit.py first)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mode", choices=["figures", "quant", "both"], default="both")
    parser.add_argument("--n-per-group", type=int, default=10)
    parser.add_argument("--n-synth", type=int, default=200)
    parser.add_argument("--n-clean", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINT
    cfg, state = load_checkpoint(ckpt_path)
    device = device_and_seed(cfg)
    seed = args.seed if args.seed is not None else cfg.seed

    if args.mode in ("figures", "both"):
        run_figures(cfg, state, device, args.n_per_group, seed)
    if args.mode in ("quant", "both"):
        run_quant_eval(cfg, state, device, args.n_synth, args.n_clean, seed)


if __name__ == "__main__":
    main()
