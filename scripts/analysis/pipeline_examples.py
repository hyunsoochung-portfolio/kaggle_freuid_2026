"""Illustrative before/after example images for docs/pipeline_finetune.md.

Generates:
  1. recapture_examples.png -- one source image, original vs. three independent draws of
     recapture_transforms (it's stochastic, so each draw looks different).
  2. tamper_examples.png -- one bona-fide source image, original vs. each of the three
     synthetic tamper types (copy_move, field_smudge, local_splice), edited region boxed.
  3. tamper_then_recapture.png -- the same three tampered images, additionally passed
     through the full recapture chain (tamper-then-recapture is the order a synthetic
     positive would take). Illustrative only: the synthetic-tamper training path was
     removed when analog-double became the default, so the model no longer trains on these.

Reuses augment.py's recapture_transforms directly (no changes to training code) and
tamper_bbox.py's replicated tamper functions (for the bbox overlay only -- illustrative,
not a training input). Output goes to reports/pipeline_examples/, which is gitignored:
every image here is a real dataset image, same non-commercial-license concern as
reports/analysis_v0/borderline/ and attn/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_CHECKPOINT, REPO_ROOT, get_split_ids, load_checkpoint, split_dataframe  # noqa: E402
from tamper_bbox import _copy_move, _field_smudge, _local_splice, recapture_transforms_with_bbox  # noqa: E402
from freuid.augment import recapture_transforms  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "pipeline_examples"


def _denorm(t: torch.Tensor, mean, std) -> np.ndarray:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    x = (t.cpu() * std_t + mean_t).clamp(0, 1)
    return x.permute(1, 2, 0).numpy()


def main() -> None:
    cfg, _ = load_checkpoint(DEFAULT_CHECKPOINT)
    _, val_ids = get_split_ids(cfg)
    val_df = split_dataframe(cfg, val_ids)
    bona_df = val_df[val_df["label"] == 0].reset_index(drop=True)

    image_size = 384  # readable figure size; the actual pipeline uses 518, same transforms
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. recapture_transforms: same source image, three independent draws
    # ------------------------------------------------------------------
    src_path = bona_df["path"].iloc[3]
    orig = Image.open(src_path).convert("RGB")
    tf = recapture_transforms(image_size, mean, std)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(orig.resize((image_size, image_size)))
    axes[0].set_title("original")
    axes[0].axis("off")
    for i in range(1, 4):
        torch.manual_seed(i)
        np.random.seed(i)
        disp = _denorm(tf(orig), mean, std)
        axes[i].imshow(disp)
        axes[i].set_title(f"recapture_transforms draw {i}")
        axes[i].axis("off")
    fig.suptitle("recapture_transforms: simulating the print-and-recapture chain (stochastic -- differs every call)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "recapture_examples.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'recapture_examples.png'}")

    # ------------------------------------------------------------------
    # 2. Synthetic tamper types (raw, before recapture degradation)
    # ------------------------------------------------------------------
    tamper_src_path = bona_df["path"].iloc[10]
    orig_arr = np.array(Image.open(tamper_src_path).convert("RGB"))
    donor_arr = np.array(Image.open(bona_df["path"].iloc[20]).convert("RGB"))

    cm_arr, cm_bbox = _copy_move(orig_arr, np.random.default_rng(2))
    fs_arr, fs_bbox = _field_smudge(orig_arr, np.random.default_rng(3))
    ls_arr, ls_bbox = _local_splice(orig_arr, donor_arr, np.random.default_rng(4))
    tampered = [("copy_move", cm_arr, cm_bbox), ("field_smudge", fs_arr, fs_bbox), ("local_splice", ls_arr, ls_bbox)]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(orig_arr)
    axes[0].set_title("original (bona-fide)")
    axes[0].axis("off")
    for ax, (name, arr, bbox) in zip(axes[1:], tampered, strict=True):
        ax.imshow(arr)
        x1, y1, x2, y2 = bbox
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="lime", facecolor="none", linewidth=2))
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Synthetic tamper types (green = edited region) -- raw, before recapture degradation")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tamper_examples.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'tamper_examples.png'}")

    # ------------------------------------------------------------------
    # 3. Tamper + recapture together -- what the model actually trains on
    # ------------------------------------------------------------------
    pipeline = recapture_transforms_with_bbox(image_size, mean, std)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(Image.open(tamper_src_path).convert("RGB").resize((image_size, image_size)))
    axes[0].set_title("original (bona-fide)")
    axes[0].axis("off")
    for ax, (name, arr, bbox) in zip(axes[1:], tampered, strict=True):
        result = pipeline(image=arr, bboxes=[bbox], category_ids=[0])
        disp = _denorm(result["image"], mean, std)
        ax.imshow(disp)
        if result["bboxes"]:
            x1, y1, x2, y2 = result["bboxes"][0][:4]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="lime", facecolor="none", linewidth=2))
        ax.set_title(f"{name} + recapture")
        ax.axis("off")
    fig.suptitle("What the model actually trains on: tamper always seen through the analog hole")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tamper_then_recapture.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'tamper_then_recapture.png'}")


if __name__ == "__main__":
    main()
