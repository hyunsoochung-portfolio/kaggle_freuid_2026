"""Attention-rollout saliency maps for finetune_v0 -- the visual counterpart to the
face-occlusion sensitivity experiment (occlusion_test.py / occlusion_report.md).

Where occlusion_test.py asks "does removing the face region change the score", this asks
"where does the model's own attention actually go". The two should agree (see the
face-attention-ratio vs. |Δface| correlation this script appends to occlusion_report.md); if
they don't, that's a red flag that one of the two instruments is broken, not that the model
does something exotic.

Method
------
1. Primary: attention rollout (Abnar & Zuidema, 2020) over all 12 ViT-B/14 blocks --
   per-block attention averaged over heads, corrected for the residual stream via
   `0.5*A + 0.5*I`, then chained across blocks by left-multiplication. The CLS row of the
   final rolled-out matrix gives each patch token's attribution to the model's decision.
   `vit_base_patch14_reg4_dinov2.lvd142m` (this repo's backbone) has a CLS token AND 4
   register tokens as prefix tokens (`class_token=True, reg_tokens=4, no_embed_class=True` in
   timm's `vit_base_patch14_reg4_dinov2` factory) -- token order is
   `[CLS, reg0..reg3, patch0..patch(N-1)]`. Getting this wrong silently shifts every
   patch-token index by up to 4, which would misalign every heatmap without erroring. This
   reuses `scripts/analysis/visualize.py`'s already-verified token-layout constants
   (`PATCH_SIZE/NUM_REGISTERS/NUM_PREFIX/grid_size_for/assert_token_layout`) and its
   `AttentionCapture` hook (forward hook on `attn.attn_drop`, capturing the real post-softmax
   weights -- only possible because `timm.layers.set_fused_attn(False)` runs at import time,
   BEFORE any Attention module is constructed; fused attention (`F.scaled_dot_product_attention`)
   never materializes explicit weights for a hook to see) instead of re-deriving either.
2. Secondary sanity check (10 images): gradient×input on `model.patch_embed`'s raw output
   (captured via a forward hook + `retain_grad()`, backward from the raw logit). With
   `dynamic_img_size=True` timm's `PatchEmbed` emits NHWC `(1, grid, grid, C)` directly (no
   prefix tokens are in this tensor at all -- they're concatenated later in `_pos_embed`), so
   no index arithmetic is needed here, only `(act * grad).sum(channel).abs()`.
3. Same 3 groups, same ids as the occlusion experiment: loads `occlusion_results.csv` for the
   id list (30/group here) and reuses its `logit_clean` / `abs_delta_face` columns directly
   (rather than rescoring) so the two experiments are numerically comparable, not just
   thematically similar.
4. Face box comes from the same regions-cache reader and the same 10%-per-side expansion as
   occlusion_test.py (`read_face_box` / `expand_box`, imported, not reimplemented) -- the
   face-attention ratio is computed against the EXACT region occlusion_test.py occluded. The
   box DRAWN in the rendered panels is the raw (unexpanded) SCRFD box, since that's what the
   task asked to render; the ratio computation separately uses the expanded box, documented
   here to avoid ambiguity.

No training-code changes: read-only against the existing finetune_v0.pt checkpoint. Nothing
under `src/freuid/` or the regions cache is modified.

Usage:
    python scripts/analysis/saliency_maps.py --checkpoint checkpoints/finetune_v0.pt \\
        --occlusion-csv scripts/analysis/occlusion_results.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import timm.layers

timm.layers.set_fused_attn(False)  # noqa: E402 -- must precede model construction, see docstring

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.data import load_labels  # noqa: E402
from freuid.preprocess import regions_dir  # noqa: E402
from freuid.transforms import build_transforms, resolve_data_config  # noqa: E402
from freuid.utils import pick_device  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_CHECKPOINT, build_finetuned_model, df_to_md, load_checkpoint  # noqa: E402
from occlusion_test import expand_box, read_face_box  # noqa: E402
from visualize import (  # noqa: E402
    NUM_PREFIX,
    AttentionCapture,
    assert_token_layout,
    grid_size_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OCCLUSION_CSV = Path(__file__).resolve().parent / "occlusion_results.csv"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "saliency_out"
DEFAULT_SALIENCY_CSV = Path(__file__).resolve().parent / "saliency_results.csv"
DEFAULT_REPORT = Path(__file__).resolve().parent / "occlusion_report.md"

GROUPS = ("TRAIN_FRAUD_FACE", "TRAIN_BONAFIDE", "HESITANT_TEST")


# ---------------------------------------------------------------------------
# Attention rollout
# ---------------------------------------------------------------------------

def attention_rollout(model: torch.nn.Module, img: torch.Tensor, grid: int) -> np.ndarray:
    """Abnar & Zuidema attention rollout across every block. Returns a (grid, grid) array,
    the CLS row's attribution over patch tokens only (CLS + 4 register prefix tokens
    excluded), normalized to sum to 1."""
    n_blocks = len(model.blocks)
    cap = AttentionCapture(model, list(range(n_blocks)))
    with torch.no_grad():
        model(img)
    attn_per_block = [cap.get(i) for i in range(n_blocks)]  # each (1, heads, N, N)
    cap.remove()

    n_tokens = attn_per_block[0].shape[-1]
    assert_token_layout(n_tokens, grid)
    device = attn_per_block[0].device
    eye = torch.eye(n_tokens, device=device)
    result = eye.clone()
    for attn in attn_per_block:
        a = attn[0].mean(dim=0)  # (N, N) -- average over heads
        a_hat = 0.5 * a + 0.5 * eye  # residual-stream correction
        a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)  # guard fp drift, rows already sum to ~1
        result = a_hat @ result  # left-multiply: later blocks compose on top of earlier ones
    cls_row = result[0]  # attribution flowing into the CLS token after all 12 blocks
    patch_mass = cls_row[NUM_PREFIX:]
    patch_mass = patch_mass / patch_mass.sum()
    return patch_mass.reshape(grid, grid).cpu().numpy()


def grad_input_saliency(model: torch.nn.Module, img: torch.Tensor) -> np.ndarray:
    """Gradient x input on model.patch_embed's raw output. With dynamic_img_size=True timm's
    PatchEmbed emits NHWC (1, grid, grid, C) with no prefix tokens mixed in, so no reg/CLS
    index arithmetic is needed here at all. Returns a (grid, grid) array normalized to sum 1."""
    captured: dict[str, torch.Tensor] = {}

    def hook(module, inp, out):
        out.retain_grad()
        captured["out"] = out

    h = model.patch_embed.register_forward_hook(hook)
    model.zero_grad(set_to_none=True)
    logit = model(img)
    h.remove()
    logit.sum().backward()

    act = captured["out"].detach()
    grad = captured["out"].grad
    sal = (act * grad).sum(dim=-1).squeeze(0).abs()  # (grid, grid)
    model.zero_grad(set_to_none=True)
    sal = sal / sal.sum().clamp_min(1e-12)
    return sal.cpu().numpy()


# ---------------------------------------------------------------------------
# Face-attention ratio
# ---------------------------------------------------------------------------

def face_attention_ratio(
    grid_map: np.ndarray, face_box_orig: tuple[int, int, int, int], orig_w: int, orig_h: int,
    resized_size: int,
) -> tuple[float, float, float]:
    """(rollout/saliency mass inside the face box) / (face box area as a fraction of card
    area). ``grid_map`` must already be normalized to sum to 1 over the patch grid.

    Box overlap is computed in the model's actual RESIZED input space (patches live there),
    but the area fraction is scale-invariant under any resize (uniform or not), so it's
    computed directly from the original-image box for simplicity. Returns
    (mass_in_box, area_fraction, ratio).
    """
    g = grid_map.shape[0]
    sx, sy = resized_size / orig_w, resized_size / orig_h
    x1, y1, x2, y2 = face_box_orig
    rx1, ry1, rx2, ry2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
    cell = resized_size / g

    mass = 0.0
    for i in range(g):
        cy1, cy2 = i * cell, (i + 1) * cell
        oy = min(cy2, ry2) - max(cy1, ry1)
        if oy <= 0:
            continue
        for j in range(g):
            cx1, cx2 = j * cell, (j + 1) * cell
            ox = min(cx2, rx2) - max(cx1, rx1)
            if ox <= 0:
                continue
            mass += grid_map[i, j] * (ox * oy) / (cell * cell)

    area_fraction = ((x2 - x1) * (y2 - y1)) / (orig_w * orig_h)
    area_fraction = max(area_fraction, 1e-9)
    return float(mass), float(area_fraction), float(mass / area_fraction)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def upsample_heatmap(grid_map: np.ndarray, w: int, h: int) -> np.ndarray:
    hm = grid_map - grid_map.min()
    hm = hm / (hm.max() + 1e-12)
    img = Image.fromarray((hm * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def render_group_montage(items: list[dict], group: str, out_dir: Path) -> Path:
    """items: list of dicts with keys id, base_img (HWC float [0,1]), box_raw, rollout_grid,
    clean_logit, ratio. 6 image-pairs per row (original+box | heatmap overlay)."""
    n = len(items)
    ncols_pairs = 6
    nrows = max(1, (n + ncols_pairs - 1) // ncols_pairs)
    fig, axes = plt.subplots(nrows, ncols_pairs * 2, figsize=(ncols_pairs * 3.6, nrows * 2.5))
    axes = np.atleast_2d(axes)

    for idx, d in enumerate(items):
        row, pair = divmod(idx, ncols_pairs)
        ax_img, ax_hm = axes[row, pair * 2], axes[row, pair * 2 + 1]

        ax_img.imshow(d["base_img"])
        x1, y1, x2, y2 = d["box_raw"]
        ax_img.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.2, edgecolor="lime", facecolor="none"))
        ax_img.set_title(f"{d['id'][:8]} logit={d['clean_logit']:.2f}", fontsize=6)
        ax_img.axis("off")

        h, w = d["base_img"].shape[:2]
        hm_up = upsample_heatmap(d["rollout_grid"], w, h)
        ax_hm.imshow(d["base_img"])
        ax_hm.imshow(hm_up, cmap="jet", alpha=0.5)
        ax_hm.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.2, edgecolor="lime", facecolor="none"))
        ax_hm.set_title(f"ratio={d['ratio']:.2f}", fontsize=6)
        ax_hm.axis("off")

    for idx in range(n, nrows * ncols_pairs):
        row, pair = divmod(idx, ncols_pairs)
        axes[row, pair * 2].axis("off")
        axes[row, pair * 2 + 1].axis("off")

    fig.suptitle(f"{group}: attention-rollout saliency (n={n}) -- green=raw SCRFD face box", fontsize=11)
    fig.tight_layout()
    out_path = out_dir / f"{group}_montage.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_image_tensor(path: Path, transform) -> tuple[torch.Tensor, Image.Image]:
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0), img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--occlusion-csv", default=str(DEFAULT_OCCLUSION_CSV))
    parser.add_argument("--n-per-group", type=int, default=30)
    parser.add_argument("--n-sanity", type=int, default=10)
    parser.add_argument("--expand-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--csv-out", default=str(DEFAULT_SALIENCY_CSV))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.checkpoint or DEFAULT_CHECKPOINT
    cfg, state = load_checkpoint(ckpt_path)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    device = pick_device()
    model = build_finetuned_model(cfg, state, device)

    data_cfg = resolve_data_config(cfg.backbone, cfg.image_size)
    mean, std = data_cfg["mean"], data_cfg["std"]
    image_size = data_cfg["image_size"]
    grid = grid_size_for(image_size)
    n_patches = grid * grid
    print(
        f"[saliency] backbone={cfg.backbone} image_size={image_size} patch_size=14 "
        f"grid={grid}x{grid}={n_patches} patches + 1 CLS + 4 reg = {n_patches + NUM_PREFIX} tokens"
    )
    transform = build_transforms(image_size, False, mean, std)
    rdir = regions_dir(cfg.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- needed for face boxes")

    occ_df = pd.read_csv(args.occlusion_csv, dtype={"id": str})
    train_df = load_labels(cfg.data_dir, "train").set_index("id")
    test_df = load_labels(cfg.data_dir, "public_test").set_index("id")

    rng = np.random.default_rng(args.seed)
    per_image_rows = []
    items_by_group: dict[str, list[dict]] = {g: [] for g in GROUPS}
    sanity_pool: list[dict] = []

    # Distribute the sanity-check sample evenly across groups (e.g. n_sanity=10 -> [4,3,3]),
    # rather than a flat per-group cutoff that can silently starve whichever group is
    # processed last once the overall list gets truncated to n_sanity.
    base, extra = divmod(args.n_sanity, len(GROUPS))
    sanity_per_group = {g: base + (1 if i < extra else 0) for i, g in enumerate(GROUPS)}

    t_start = time.time()
    for group in GROUPS:
        g_df = occ_df[occ_df["group"] == group]
        n = min(args.n_per_group, len(g_df))
        sample = g_df.sample(n=n, random_state=args.seed)
        meta_df = train_df if group != "HESITANT_TEST" else test_df

        for i, row in enumerate(sample.itertuples()):
            cid = row.id
            path = Path(meta_df.loc[cid, "path"])
            img_t, pil_img = load_image_tensor(path, transform)
            img_t = img_t.to(device)
            w, h = pil_img.size

            fb = read_face_box(rdir, cid)
            box_raw = (int(fb["x1"]), int(fb["y1"]), int(fb["x2"]), int(fb["y2"]))
            box_exp = expand_box(fb, args.expand_frac, w, h)

            rollout_grid = attention_rollout(model, img_t, grid)
            mass, area_frac, ratio = face_attention_ratio(rollout_grid, box_exp, w, h, image_size)

            per_image_rows.append({
                "id": cid, "group": group,
                "clean_logit": float(row.logit_clean),
                "abs_delta_face": float(row.abs_delta_face),
                "abs_delta_control": float(row.abs_delta_control),
                "rollout_mass_in_box": mass,
                "face_area_fraction": area_frac,
                "rollout_ratio": ratio,
            })
            items_by_group[group].append({
                "id": cid, "base_img": np.array(pil_img).astype(np.float32) / 255.0,
                "box_raw": box_raw, "rollout_grid": rollout_grid,
                "clean_logit": float(row.logit_clean), "ratio": ratio,
            })
            if i < sanity_per_group[group]:
                sanity_pool.append({
                    "id": cid, "group": group, "img_t": img_t, "box_exp": box_exp,
                    "w": w, "h": h, "rollout_ratio": ratio,
                })
        print(f"[saliency] {group}: {n} images scored ({time.time() - t_start:.1f}s elapsed)")

    print(f"[saliency] rollout pass done in {(time.time() - t_start) / 60:.1f} min")

    # Render montages
    for group in GROUPS:
        out_path = render_group_montage(items_by_group[group], group, out_dir)
        print(f"[saliency] wrote montage -> {out_path}")

    per_image_df = pd.DataFrame(per_image_rows)
    csv_path = Path(args.csv_out)
    per_image_df.to_csv(csv_path, index=False)
    print(f"[saliency] wrote per-image csv -> {csv_path} ({len(per_image_df)} rows)")

    # Secondary sanity check: gradient x input on a subset, compare ratios
    sanity_pool = sanity_pool[:args.n_sanity]
    sanity_rows = []
    for d in sanity_pool:
        sal_grid = grad_input_saliency(model, d["img_t"])
        _, _, gi_ratio = face_attention_ratio(sal_grid, d["box_exp"], d["w"], d["h"], image_size)
        sanity_rows.append({
            "id": d["id"], "group": d["group"],
            "rollout_ratio": d["rollout_ratio"], "grad_input_ratio": gi_ratio,
        })
    sanity_df = pd.DataFrame(sanity_rows)
    print(f"[saliency] sanity check ({len(sanity_df)} images):\n{sanity_df.to_string(index=False)}")

    write_report_section(Path(args.report), per_image_df, sanity_df, grid, n_patches, image_size)


def write_report_section(
    report_path: Path, per_image_df: pd.DataFrame, sanity_df: pd.DataFrame,
    grid: int, n_patches: int, image_size: int,
) -> None:
    from scipy.stats import pearsonr, spearmanr

    lines = ["\n---\n", "# Saliency-map report (attention rollout)\n"]
    lines.append(
        "Visual counterpart to the occlusion experiment above: where does `finetune_v0`'s "
        "own attention actually go, and does that agree with the occlusion deltas already "
        "measured?\n"
    )
    lines.append(
        f"- Token layout: {grid}x{grid}={n_patches} patches + 1 CLS + 4 register tokens = "
        f"{n_patches + NUM_PREFIX} tokens at image_size={image_size} (matches the expected "
        "arithmetic, asserted at runtime against the actual captured attention tensor shape)."
    )
    lines.append(
        "- Rollout = Abnar & Zuidema (2020): per-block attention averaged over heads, "
        "`0.5*A + 0.5*I` residual correction, chained by left-multiplication across all 12 "
        "blocks; CLS row's attribution over patch tokens, normalized to sum to 1."
    )
    lines.append(
        "- Face-attention ratio = (rollout mass inside the 10%-expanded face box, the SAME "
        "box occlusion_test.py occluded) / (that box's area as a fraction of the card). "
        "Ratio ≈ 1: no special attention on the face. Ratio ≫ 1: attention concentrates "
        "there.\n"
    )

    lines.append("## Face-attention ratio distribution per group\n")
    lines.append("| group | n | mean ratio | median ratio | std | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    for group, g in per_image_df.groupby("group"):
        r = g["rollout_ratio"]
        lines.append(
            f"| {group} | {len(g)} | {r.mean():.3f} | {r.median():.3f} | {r.std():.3f} | "
            f"{r.min():.3f} | {r.max():.3f} |"
        )

    lines.append("\n## Correlation: face-attention ratio vs. occlusion |Δface|\n")
    lines.append("| group | n | Pearson r | p | Spearman rho | p |")
    lines.append("|---|---|---|---|---|---|")
    for group, g in list(per_image_df.groupby("group")) + [("ALL (pooled)", per_image_df)]:
        if len(g) < 3 or g["rollout_ratio"].std() == 0 or g["abs_delta_face"].std() == 0:
            lines.append(f"| {group} | {len(g)} | n/a | n/a | n/a | n/a |")
            continue
        pr, pp = pearsonr(g["rollout_ratio"], g["abs_delta_face"])
        sr, sp = spearmanr(g["rollout_ratio"], g["abs_delta_face"])
        lines.append(f"| {group} | {len(g)} | {pr:.3f} | {pp:.3g} | {sr:.3f} | {sp:.3g} |")

    lines.append(
        "\nIf rollout and occlusion disagree sharply within a group (e.g. high ratio but "
        "tiny |Δface|, or the reverse), flag it: it means one of the two instruments is "
        "measuring something other than intended for that group, most likely due to the "
        "ceiling-saturation and flat-patch confounds already documented above for the "
        "occlusion numbers.\n"
    )

    lines.append(f"## Sanity check: gradient x input vs. rollout ({len(sanity_df)} images)\n")
    lines.append(df_to_md(sanity_df.round(4)))
    if len(sanity_df) >= 3 and sanity_df["rollout_ratio"].std() > 0 and sanity_df["grad_input_ratio"].std() > 0:
        sr, sp = spearmanr(sanity_df["rollout_ratio"], sanity_df["grad_input_ratio"])
        lines.append(f"\nSpearman rho between the two methods' ratios: {sr:.3f} (p={sp:.3g}).")
    lines.append("")

    lines.append("## Verdict: tying the occlusion and saliency experiments together\n")
    lines.append(_verdict_text(per_image_df))

    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[saliency] appended saliency section -> {report_path}")


def _verdict_text(df: pd.DataFrame) -> str:
    means = df.groupby("group")["rollout_ratio"].mean()
    parts = []
    for group in GROUPS:
        if group in means.index:
            parts.append(f"{group} mean ratio={means[group]:.2f}")
    summary = "; ".join(parts)
    return (
        f"Ratios observed this run: {summary}. Read this alongside the occlusion section's "
        "ceiling-saturation finding above (TRAIN_FRAUD_FACE and HESITANT_TEST clean logits "
        "both near the model's output ceiling; TRAIN_BONAFIDE at the floor with real "
        "headroom). A ratio near 1 with a small |Δface| in the same group is consistent "
        "evidence of no face-specific reliance; a ratio well above 1 that ALSO comes with a "
        "small |Δface| (expected for the ceiling-saturated groups) should be read as "
        "attention-without-consequence -- the model may look at the face without that look "
        "changing its already-saturated output -- not as contradictory evidence. Only a high "
        "ratio paired with a genuinely large |Δface| in a non-saturated group is clean "
        "evidence of face-specific reliance. Update this paragraph by hand once the real "
        "per-group numbers above are in, rather than trusting this templated summary alone."
    )


if __name__ == "__main__":
    main()
