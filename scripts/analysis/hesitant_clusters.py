"""Size the error families in finetune_v0's most uncertain public-test predictions.

Goal: figure out how much public-LB AuDET is plausibly at stake in each attack category
(GenAI face edits, print-and-recapture, unseen document types) among the ids the model is
least sure about, so follow-up work can be prioritized by expected payoff rather than by
which failure mode is easiest to imagine.

Method
------
1. "Hesitant" is defined by RANK, not raw score. `infer.py`'s submission score is itself an
   average of per-scale fractional ranks (`_rank_normalize`), so it's already close to uniform
   on (0,1) by construction (see `hesitant_test_images.py`'s own docstring) -- but "close to"
   isn't "exactly", so this re-derives an explicit percentile rank of the FINAL score among the
   ~7,821 present ids and takes the 500 closest to the 50th percentile (they land in roughly the
   46th-54th percentile, comfortably inside a 35-65 band). Cross-checked against
   `reports/hesitant_test_100.csv` if present.
2. Per-id: document-type PROXY, SCRFD face confidence, and four cheap pixel-level degradation
   stats (Laplacian-variance blur, an FFT periodic-peak moire/halftone proxy, a JPEG-blockiness
   estimate, min-side resolution).
3. Document type is NOT ground truth here -- `sample_submission.csv` (public_test's label
   source) has only `id, label` columns, no `type`/`is_digital` field at all, so
   `scrfd_coverage.py`'s per-type breakdown (which reads `load_labels(...)["type"]`) only ever
   works for the labeled `train` split; for public_test that column is all-NaN. This script
   still needs *something* to group by, so it builds a PROXY: a 5-NN majority vote against a
   small labeled reference set (drawn from train, stratified across its only 5 known types --
   EGYPT/DL, GUINEA/DL, BENIN/DL, MOZAMBIQUE/DL, MAURITIUS/ID), in the SAME fine-tuned cosine
   embedding space used for clustering. This can only ever output one of those 5 labels, so it
   structurally cannot recognize a genuinely unseen document type -- which is exactly the
   failure mode most worth sizing. The mitigation: a cluster whose nearest-neighbor similarity
   is uniformly low is itself evidence of "doesn't look like any known training type", reported
   alongside the (possibly wrong) majority-vote label. Treat `type_proxy` as a hint, never as
   ground truth.
4. Clustering: L2-normalized penultimate pooled features (`forward_head(..., pre_logits=True)`,
   NOT the fraud logit) from the 500 hesitant ids, via `sklearn.cluster.HDBSCAN`; if HDBSCAN's
   noise fraction exceeds 50%, refit with KMeans(k=8) instead.
5. Enrichment: same stats + type_proxy on a 500-id CONFIDENT control (top + bottom rank
   deciles), compared against the hesitant set.
6. This script does NOT decide which cluster is "face-edit" / "recapture" / "unseen-type" --
   that's a manual call from the rendered montage grids. It only flags obvious quantitative
   signals (e.g. low face confidence, high moire score) per cluster.

No training, no submissions, no modification of existing source files or the regions cache.

Usage:
    python scripts/analysis/hesitant_clusters.py --checkpoint checkpoints/finetune_v0.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from freuid.data import load_labels  # noqa: E402
from freuid.preprocess import regions_dir  # noqa: E402
from freuid.transforms import build_transforms, resolve_data_config  # noqa: E402
from freuid.utils import pick_device  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_CHECKPOINT, build_finetuned_model, df_to_md, load_checkpoint  # noqa: E402
from occlusion_test import read_face_box  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "hesitant_out"
DEFAULT_CSV = Path(__file__).resolve().parent / "hesitant_results.csv"
DEFAULT_REPORT = Path(__file__).resolve().parent / "hesitant_report.md"
DEFAULT_OLD_HESITANT_CSV = REPO_ROOT / "reports" / "hesitant_test_100.csv"
KNOWN_TYPES = ("EGYPT/DL", "GUINEA/DL", "BENIN/DL", "MOZAMBIQUE/DL", "MAURITIUS/ID")


# ---------------------------------------------------------------------------
# Cheap pixel-level degradation stats
# ---------------------------------------------------------------------------

def laplacian_blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian -- lower means blurrier."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fft_periodic_peak_score(gray: np.ndarray) -> float:
    """Halftone/moire proxy: how much a single frequency band's peak magnitude stands out
    above the surrounding annulus's median, in a fixed-size FFT (256x256, so the metric is
    comparable across differently-sized source images). Regular dot/line patterns (halftone
    printing, moire from a screen recapture) concentrate energy in a narrow ring; smooth
    photographic content does not."""
    g = cv2.resize(gray, (256, 256)).astype(np.float64)
    g = g - g.mean()
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    h, w = mag.shape
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(h, w) / 2)
    annulus = mag[(r >= 0.06) & (r <= 0.6)]
    if annulus.size == 0:
        return 0.0
    med = np.median(annulus)
    return float(annulus.max() / (med + 1e-6))


def jpeg_blockiness_score(gray: np.ndarray, block: int = 8) -> float:
    """Mean |gradient| at 8-pixel block-boundary columns/rows minus mean |gradient| at
    mid-block columns/rows -- a simple no-reference blockiness proxy. Higher = more visible
    8x8 block edges (heavier / repeated JPEG compression)."""
    g = gray.astype(np.float64)

    def one_axis(a: np.ndarray) -> float:
        diffs = np.abs(np.diff(a, axis=1))  # (H, W-1)
        n = diffs.shape[1]
        cols = np.arange(n)
        boundary = cols % block == (block - 1)
        interior = cols % block == (block // 2)
        if not boundary.any() or not interior.any():
            return 0.0
        return float(diffs[:, boundary].mean() - diffs[:, interior].mean())

    return float((one_axis(g) + one_axis(g.T)) / 2)


def pixel_stats(pil_img: Image.Image) -> dict:
    gray = np.array(pil_img.convert("L"))
    w, h = pil_img.size
    return {
        "blur_laplacian_var": laplacian_blur_score(gray),
        "moire_fft_score": fft_periodic_peak_score(gray),
        "blockiness_score": jpeg_blockiness_score(gray),
        "min_side_px": min(w, h),
    }


# ---------------------------------------------------------------------------
# Rank / percentile selection
# ---------------------------------------------------------------------------

def load_present_scores(submission_csv: str | Path, data_dir: str) -> pd.DataFrame:
    test_meta = load_labels(data_dir, "public_test")
    present_mask = test_meta["path"].map(lambda p: Path(p).exists())
    present = test_meta.loc[present_mask, ["id", "path"]].copy()
    sub = pd.read_csv(submission_csv, dtype={"id": str}).rename(columns={"label": "score"})
    df = present.merge(sub, on="id", how="inner")
    df["pct_rank"] = df["score"].rank(pct=True) * 100.0
    return df


def select_hesitant_and_confident(
    df: pd.DataFrame, n_hesitant: int, n_confident_each: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["dist_from_50"] = (d["pct_rank"] - 50.0).abs()
    hesitant = d.nsmallest(n_hesitant, "dist_from_50").copy()
    hesitant["set"] = "HESITANT"

    top_decile = d[d["pct_rank"] >= 90.0]
    bottom_decile = d[d["pct_rank"] <= 10.0]
    rng_state = seed
    top_sample = top_decile.sample(n=min(n_confident_each, len(top_decile)), random_state=rng_state)
    bottom_sample = bottom_decile.sample(n=min(n_confident_each, len(bottom_decile)), random_state=rng_state)
    confident = pd.concat([top_sample, bottom_sample], ignore_index=True)
    confident["set"] = "CONFIDENT"

    print(
        f"[hesitant] hesitant band: pct_rank {hesitant['pct_rank'].min():.2f}-"
        f"{hesitant['pct_rank'].max():.2f} (n={len(hesitant)}); "
        f"confident: top n={len(top_sample)} (pct>=90), bottom n={len(bottom_sample)} (pct<=10)"
    )
    return hesitant, confident


def crosscheck_old_artifact(hesitant_ids: set[str], old_csv: Path) -> str:
    if not old_csv.exists():
        return f"No earlier hesitant-predictions artifact found at `{old_csv}` -- nothing to cross-check."
    old = pd.read_csv(old_csv, dtype={"id": str})
    old_ids = set(old["id"])
    overlap = old_ids & hesitant_ids
    frac = len(overlap) / max(1, len(old_ids))
    return (
        f"Cross-check against `{old_csv}` ({len(old_ids)} ids, selected by nearest raw score to "
        f"0.5): {len(overlap)}/{len(old_ids)} ({frac * 100:.0f}%) also appear in this run's "
        f"explicit-percentile 500-id hesitant set. "
        + (
            "High overlap confirms the submission score is already close enough to uniform-rank "
            "that the two selection methods agree in practice."
            if frac >= 0.7
            else "Overlap is lower than expected -- the raw-score selection and the explicit "
            "rank-percentile selection are meaningfully different here; treat the old artifact's "
            "framing (\"score near 0.5\") as an approximation, not equivalent to rank-percentile."
        )
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_paths(model, device, paths: list[Path], transform, batch_size: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        feats = model.forward_features(imgs)
        pooled = model.forward_head(feats, pre_logits=True)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0))


def build_type_reference(cfg, model, device, transform, n_per_type: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_df = load_labels(cfg.data_dir, "train")
    rng = np.random.default_rng(seed)
    rows = []
    for t in KNOWN_TYPES:
        sub = train_df[train_df["type"] == t]
        n = min(n_per_type, len(sub))
        rows.append(sub.sample(n=n, random_state=seed))
    ref_df = pd.concat(rows, ignore_index=True)
    paths = [Path(p) for p in ref_df["path"]]
    embs = embed_paths(model, device, paths, transform)
    return embs, ref_df["type"].to_numpy()


def type_proxy_knn(query_embs: np.ndarray, ref_embs: np.ndarray, ref_types: np.ndarray, k: int = 5) -> tuple[list[str], list[float]]:
    """Cosine similarity (embeddings are already L2-normalized, so dot product = cosine) k-NN
    majority vote. Returns (predicted_type, mean_similarity_of_the_k_neighbors)."""
    sims = query_embs @ ref_embs.T  # (n_query, n_ref)
    topk_idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    preds, confs = [], []
    for i in range(query_embs.shape[0]):
        idx = topk_idx[i]
        neigh_types = ref_types[idx]
        neigh_sims = sims[i, idx]
        vals, counts = np.unique(neigh_types, return_counts=True)
        winner = vals[np.argmax(counts)]
        preds.append(str(winner))
        confs.append(float(neigh_sims[neigh_types == winner].mean()))
    return preds, confs


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def embedding_collapse_diagnostic(
    hesitant_embs: np.ndarray, confident_embs: np.ndarray, ref_embs: np.ndarray,
) -> dict:
    """Pairwise cosine-similarity spread within each embedding set (all inputs already
    L2-normalized, so dot product = cosine). A tiny spread (mean near 1.0) means the
    embeddings for that set are nearly parallel -- i.e. the fine-tuned penultimate
    representation has collapsed for that population, which directly undermines both
    clustering and the type-proxy k-NN (both rely on directional differences that a collapsed
    representation may no longer carry)."""
    def spread(embs: np.ndarray) -> dict:
        if len(embs) < 2:
            return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "std": float("nan")}
        sims = embs @ embs.T
        off = sims[~np.eye(len(sims), dtype=bool)]
        return {"mean": float(off.mean()), "min": float(off.min()), "max": float(off.max()), "std": float(off.std())}

    return {
        "hesitant": spread(hesitant_embs),
        "confident": spread(confident_embs),
        "reference_train": spread(ref_embs),
    }


def cluster_embeddings(embs: np.ndarray, min_cluster_size: int, seed: int) -> tuple[np.ndarray, str]:
    from sklearn.cluster import HDBSCAN, KMeans

    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels = hdb.fit_predict(embs)
    noise_frac = float((labels == -1).mean())
    print(f"[hesitant] HDBSCAN: {labels.max() + 1} clusters, noise_frac={noise_frac:.2f}")
    if noise_frac > 0.5:
        print("[hesitant] HDBSCAN noise fraction > 50% -- falling back to KMeans(k=8)")
        km = KMeans(n_clusters=8, random_state=seed, n_init=10)
        labels = km.fit_predict(embs)
        return labels, "kmeans_k8"
    return labels, "hdbscan"


# ---------------------------------------------------------------------------
# Montage rendering
# ---------------------------------------------------------------------------

def _relpath_or_abs(path: Path) -> str:
    """Path relative to the repo root when possible (nicer in the report); falls back to the
    absolute path when `path` lives outside the repo (e.g. a --out-dir under /tmp)."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render_montage(paths: list[Path], titles: list[str], out_path: Path, ncols: int = 5) -> None:
    n = len(paths)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.2))
    axes = np.atleast_2d(axes)
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx < n:
            img = Image.open(paths[idx]).convert("RGB")
            ax.imshow(img)
            ax.set_title(titles[idx], fontsize=6)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# AuDET-improvement bound
# ---------------------------------------------------------------------------

def audet_bound_table(
    current_audet: float, n_present: int, assumed_fraud_rate: float, k_values: list[int],
) -> pd.DataFrame:
    """Order-of-magnitude bound on AuDET improvement from perfectly re-ranking `k` currently
    hesitant, truly-fraud images. See the report's methodology section for the full assumption
    list; the headline ones: (1) public_test's fraud rate is assumed to match train's ~42% (test
    composition is deliberately different -- this could be wrong in either direction); (2) each
    flagged image, sitting at ~50th percentile, is assumed to currently rank above roughly half
    of bona-fide images that it should rank above ALL of -- i.e. it's discordant with ~50% of
    n_bonafide pairs; (3) "fixing" means moving it to rank above 100% of bona-fide (an upper
    bound on achievable gain, not a partial, realistic fix); (4) images are treated independently
    (ignores second-order overlap in which bona-fide pairs multiple fixed images share); (5)
    assumes the flagged images really are fraud -- if some are actually correctly-scored
    bona-fide, the true k is smaller and this overstates the achievable gain. This is a BOUND,
    not a prediction.
    """
    n_fraud = round(n_present * assumed_fraud_rate)
    n_bonafide = n_present - n_fraud
    total_pairs = n_fraud * n_bonafide
    current_discordant = current_audet * total_pairs

    rows = []
    for k in k_values:
        removed = min(current_discordant, k * 0.5 * n_bonafide)
        new_discordant = current_discordant - removed
        new_audet = new_discordant / total_pairs
        improvement_pct = (current_audet - new_audet) / current_audet * 100.0 if current_audet > 0 else float("nan")
        rows.append({
            "k_flagged_images": k,
            "new_audet_bound": new_audet,
            "improvement_pct": improvement_pct,
        })
    df = pd.DataFrame(rows)
    df.attrs["n_fraud"] = n_fraud
    df.attrs["n_bonafide"] = n_bonafide
    df.attrs["total_pairs"] = total_pairs
    df.attrs["current_discordant"] = current_discordant
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--submission", default="submissions/finetune_v0.csv")
    parser.add_argument("--old-hesitant-csv", default=str(DEFAULT_OLD_HESITANT_CSV))
    parser.add_argument("--n-hesitant", type=int, default=500)
    parser.add_argument("--n-confident-each", type=int, default=250)
    parser.add_argument("--n-ref-per-type", type=int, default=40)
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--montage-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--current-audet", type=float, default=0.00744)
    parser.add_argument("--assumed-fraud-rate", type=float, default=0.42)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--csv-out", default=str(DEFAULT_CSV))
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
    transform = build_transforms(image_size, False, mean, std)

    rdir = regions_dir(cfg.data_dir)
    if not rdir.exists():
        raise SystemExit(f"regions cache not found at {rdir} -- needed for face confidence")

    t0 = time.time()
    df = load_present_scores(args.submission, cfg.data_dir)
    print(f"[hesitant] {len(df)} present public-test ids loaded with scores")
    hesitant, confident = select_hesitant_and_confident(df, args.n_hesitant, args.n_confident_each, args.seed)
    crosscheck_note = crosscheck_old_artifact(set(hesitant["id"]), Path(args.old_hesitant_csv))
    print(f"[hesitant] {crosscheck_note}")

    combined = pd.concat([hesitant, confident], ignore_index=True)

    # Per-id pixel stats + face confidence
    stat_rows = []
    for row in combined.itertuples():
        pil_img = Image.open(row.path).convert("RGB")
        stats = pixel_stats(pil_img)
        fb = read_face_box(rdir, row.id)
        stats["face_score"] = float(fb.get("score", 0.0)) if fb is not None else 0.0
        stats["id"] = row.id
        stat_rows.append(stats)
    stats_df = pd.DataFrame(stat_rows)
    combined = combined.merge(stats_df, on="id")
    print(f"[hesitant] pixel/face stats computed for {len(combined)} ids ({time.time() - t0:.1f}s)")

    # Embeddings: combined (hesitant+confident) + labeled train reference set
    combined_embs = embed_paths(model, device, [Path(p) for p in combined["path"]], transform, args.batch_size)
    ref_embs, ref_types = build_type_reference(cfg, model, device, transform, args.n_ref_per_type, args.seed)
    print(f"[hesitant] embedded {len(combined)} test ids + {len(ref_types)} labeled reference ids "
          f"({time.time() - t0:.1f}s elapsed)")

    type_proxy, type_proxy_sim = type_proxy_knn(combined_embs, ref_embs, ref_types, k=5)
    combined["type_proxy"] = type_proxy
    combined["type_proxy_similarity"] = type_proxy_sim

    # Cluster the hesitant embeddings only
    hesitant_mask = (combined["set"] == "HESITANT").to_numpy()
    confident_mask = (combined["set"] == "CONFIDENT").to_numpy()
    hesitant_embs = combined_embs[hesitant_mask]

    collapse_diag = embedding_collapse_diagnostic(hesitant_embs, combined_embs[confident_mask], ref_embs)
    print(
        "[hesitant] embedding collapse diagnostic (mean pairwise cosine similarity; ~1.0 means "
        f"collapsed/non-discriminative): hesitant={collapse_diag['hesitant']['mean']:.4f} "
        f"confident={collapse_diag['confident']['mean']:.4f} "
        f"reference_train={collapse_diag['reference_train']['mean']:.4f}"
    )

    cluster_labels, cluster_method = cluster_embeddings(hesitant_embs, args.min_cluster_size, args.seed)
    combined.loc[hesitant_mask, "cluster"] = cluster_labels
    combined.loc[~hesitant_mask, "cluster"] = np.nan

    csv_path = Path(args.csv_out)
    combined.to_csv(csv_path, index=False)
    print(f"[hesitant] wrote per-image csv -> {csv_path} ({len(combined)} rows)")

    # Per-cluster montages + stats
    hesitant_rows = combined[hesitant_mask].copy()
    hesitant_embs_by_row = hesitant_embs
    cluster_stats = []
    for cl in sorted(hesitant_rows["cluster"].unique()):
        mask = hesitant_rows["cluster"] == cl
        sub = hesitant_rows[mask]
        sub_embs = hesitant_embs_by_row[mask.to_numpy()]
        label = "noise" if cl == -1 else f"cluster_{int(cl)}"

        centroid = sub_embs.mean(axis=0, keepdims=True)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        dist = 1.0 - (sub_embs @ centroid.T).ravel()
        order = np.argsort(dist)[:args.montage_size]
        chosen = sub.iloc[order]
        titles = [
            f"{r.id[:8]} {r.type_proxy[:3]} f={r.face_score:.2f}"
            for r in chosen.itertuples()
        ]
        montage_path = out_dir / f"{label}_montage.png"
        render_montage([Path(p) for p in chosen["path"]], titles, montage_path)

        type_counts = sub["type_proxy"].value_counts()
        dominant_type = type_counts.index[0]
        dominant_frac = type_counts.iloc[0] / len(sub)
        cluster_stats.append({
            "cluster": label,
            "n": len(sub),
            "dominant_type_proxy": dominant_type,
            "dominant_type_frac": dominant_frac,
            "mean_type_proxy_similarity": sub["type_proxy_similarity"].mean(),
            "median_face_score": sub["face_score"].median(),
            "median_blur": sub["blur_laplacian_var"].median(),
            "median_moire": sub["moire_fft_score"].median(),
            "median_blockiness": sub["blockiness_score"].median(),
            "median_min_side_px": sub["min_side_px"].median(),
            "montage_path": _relpath_or_abs(montage_path),
        })
        print(f"[hesitant] {label}: n={len(sub)} dominant_type_proxy={dominant_type} "
              f"({dominant_frac * 100:.0f}%) -> {montage_path}")

    cluster_df = pd.DataFrame(cluster_stats)

    # Enrichment: hesitant vs confident
    enrichment_rows = []
    hes = combined[combined["set"] == "HESITANT"]
    conf = combined[combined["set"] == "CONFIDENT"]
    for t in KNOWN_TYPES:
        hes_frac = (hes["type_proxy"] == t).mean()
        conf_frac = (conf["type_proxy"] == t).mean()
        ratio = hes_frac / conf_frac if conf_frac > 0 else float("inf")
        enrichment_rows.append({"type_proxy": t, "hesitant_frac": hes_frac, "confident_frac": conf_frac, "enrichment_ratio": ratio})
    enrichment_df = pd.DataFrame(enrichment_rows)

    from scipy.stats import mannwhitneyu
    stat_cols = ["face_score", "blur_laplacian_var", "moire_fft_score", "blockiness_score", "min_side_px", "type_proxy_similarity"]
    degradation_rows = []
    for col in stat_cols:
        h_vals, c_vals = hes[col].dropna(), conf[col].dropna()
        try:
            u, p = mannwhitneyu(h_vals, c_vals, alternative="two-sided")
        except ValueError:
            u, p = float("nan"), float("nan")
        degradation_rows.append({
            "stat": col,
            "hesitant_median": h_vals.median(),
            "confident_median": c_vals.median(),
            "ratio_h_over_c": h_vals.median() / c_vals.median() if c_vals.median() else float("inf"),
            "mannwhitney_p": p,
        })
    degradation_df = pd.DataFrame(degradation_rows)

    audet_table = audet_bound_table(
        args.current_audet, len(df), args.assumed_fraud_rate,
        # Concentrated below the saturation point (see write_report's note): under this bound's
        # assumptions, a very small k already accounts for the entire current AuDET gap, since
        # AuDET is computed over ~15M pairs while the current gap itself is tiny. Spanning
        # 1..500 in round numbers would make most of the table redundant (all reading "0, 100%").
        k_values=[1, 5, 10, 20, 30, 40, 50, 75, 100, 250, 500],
    )

    write_report(
        Path(args.report), cluster_df, enrichment_df, degradation_df, audet_table,
        crosscheck_note, cluster_method, args, len(df), collapse_diag,
    )
    print(f"[hesitant] done in {(time.time() - t0) / 60:.1f} min")


def write_report(
    report_path: Path, cluster_df: pd.DataFrame, enrichment_df: pd.DataFrame,
    degradation_df: pd.DataFrame, audet_table: pd.DataFrame, crosscheck_note: str,
    cluster_method: str, args, n_present: int, collapse_diag: dict,
) -> None:
    lines = ["# Hesitant-prediction error-family sizing report\n"]
    lines.append(
        "Sizes the error families among finetune_v0's most rank-uncertain public-test "
        "predictions, to prioritize follow-up work by estimated LB payoff rather than by "
        "which failure mode is easiest to imagine.\n"
    )

    lines.append("## Read this first: embedding collapse on the hesitant/confident sets\n")
    h, c, r = collapse_diag["hesitant"], collapse_diag["confident"], collapse_diag["reference_train"]
    lines.append(
        "Mean pairwise cosine similarity WITHIN each embedding set (1.0 = every embedding "
        "points in exactly the same direction; a healthy, discriminative set of ~500 distinct "
        "card images should NOT sit anywhere near 1.0):\n"
    )
    lines.append("| set | mean | min | max | std |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| HESITANT | {h['mean']:.4f} | {h['min']:.4f} | {h['max']:.4f} | {h['std']:.4f} |")
    lines.append(f"| CONFIDENT | {c['mean']:.4f} | {c['min']:.4f} | {c['max']:.4f} | {c['std']:.4f} |")
    lines.append(f"| reference (labeled train) | {r['mean']:.4f} | {r['min']:.4f} | {r['max']:.4f} | {r['std']:.4f} |")
    if h["mean"] > 0.99:
        specifically_hesitant = c["mean"] < 0.5 and r["mean"] < 0.5
        lines.append(
            "\n**The HESITANT set's penultimate embeddings are severely collapsed** (mean "
            "similarity far above both the CONFIDENT control's and the labeled-train reference "
            "set's healthy, spread-out values)."
            + (
                " Notably, this is NOT a general property of extreme/confident predictions -- "
                "CONFIDENT (top+bottom rank deciles, arguably even more extreme logits) shows "
                "normal, healthy diversity, comparable to the labeled-train reference set. The "
                "collapse is specific to the near-median-RANK population."
                if specifically_hesitant else ""
            )
            + " This is almost certainly the representation-space counterpart of a finding "
            "already documented for this checkpoint (see `occlusion_report.md`): 'hesitant by "
            "rank' does not mean 'hesitant by raw logit' -- most rank-median ids still carry "
            "extreme, saturated raw logits (93% of an earlier rank-median sample had clean logit "
            "> 11). If that holds here too, these 500 images are being fine-ordered relative to "
            "each other by whatever tiny residual separates otherwise near-identical, "
            "already-saturated representations -- consistent with a collapsed penultimate "
            "manifold where genuine visual differences have mostly been discarded. This was "
            "checked for a simple fix (mean-centering before renormalizing) and it did NOT "
            "recover meaningful spread, so this isn't a one-line preprocessing bug.\n\n"
            "**Practical consequence, twofold**: (1) treat `type_proxy` and the cluster "
            "assignments below as low-confidence -- both depend on directional differences a "
            "collapsed representation may no longer reliably carry, so a near-unanimous "
            "`type_proxy` vote is more likely a collapse artifact than a real shared document "
            "type; the montage grids are correspondingly MORE important, not less, as the actual "
            "arbiter of cluster content. (2) More fundamentally, if this population's fine-"
            "grained rank order is being decided by near-noise in a collapsed representation, "
            "individually 'fixing' images the visual review flags may not be achievable by "
            "simple, targeted means -- the model may currently lack the representational capacity "
            "to distinguish them at all, which would call for a training/architecture-level fix, "
            "not a post-hoc one. Weigh this against the AuDET-improvement bound below, which "
            "assumes flagged images ARE cleanly fixable.\n"
        )
    else:
        lines.append(
            "\nSimilarity levels look reasonably healthy (not obviously collapsed) -- "
            "`type_proxy` and cluster assignments below can be given more weight than the "
            "general caveats already noted for them.\n"
        )

    lines.append("## Methodology\n")
    lines.append(
        f"- Hesitant set: {args.n_hesitant} ids closest to the 50th rank-percentile of the "
        f"{n_present} present public-test ids in `{args.submission}` (percentile computed "
        "explicitly via `.rank(pct=True)`, not assumed from the raw score's near-uniformity)."
    )
    lines.append(f"- {crosscheck_note}")
    lines.append(
        f"- Confident control: {args.n_confident_each} ids from the top rank decile (pct>=90) "
        f"+ {args.n_confident_each} from the bottom decile (pct<=10), sampled with seed={args.seed}."
    )
    lines.append(
        "- **Document type is a PROXY, not ground truth.** `sample_submission.csv` (the only "
        "label source for public_test) has no `type`/`is_digital` column at all -- "
        "`scrfd_coverage.py`'s per-type breakdown only works on the labeled `train` split. "
        f"`type_proxy` here is a 5-NN majority vote against {args.n_ref_per_type}/type reference "
        f"embeddings sampled from train's only {len(KNOWN_TYPES)} known types "
        f"({', '.join(KNOWN_TYPES)}), in the fine-tuned model's own cosine embedding space. "
        "**It can only ever output one of these 5 labels** -- it structurally cannot say "
        "'unseen document type'. A cluster with uniformly low `type_proxy_similarity` is itself "
        "evidence of an unseen type, even though the majority-vote label shown for it is (by "
        "construction) one of the 5 known ones and may simply be wrong."
    )
    lines.append(
        "- Clustering: L2-normalized penultimate pooled features "
        "(`forward_head(..., pre_logits=True)`, not the fraud logit) of the 500 hesitant ids, "
        f"via `{cluster_method}` (HDBSCAN, min_cluster_size={args.min_cluster_size}; falls back "
        "to KMeans(k=8) if HDBSCAN's noise fraction exceeds 50%)."
    )
    lines.append(
        "- Degradation stats (cheap, pixel-only, no model): Laplacian-variance blur, an FFT "
        "periodic-peak moire/halftone proxy, a JPEG 8x8-block-edge blockiness estimate, and "
        "min-side resolution.\n"
    )

    lines.append(
        "## Clusters -- **visual labeling (face-edit / recapture / unseen-type) is a manual "
        "step, done from the montage grids below, not concluded here**\n"
    )
    lines.append(df_to_md(cluster_df.round(4)))
    lines.append(
        "\nFlags worth checking visually per cluster: low `median_face_score` (SCRFD didn't "
        "find/trust a face -- could mean no visible portrait, an edited/occluded face, or a "
        "genuinely different document layout); high `median_moire` (screen/print recapture "
        "signature); low `mean_type_proxy_similarity` (doesn't resemble any of the 5 known "
        "training types -- candidate unseen-document-type cluster).\n"
    )

    lines.append("## Enrichment: hesitant vs. confident, by type_proxy\n")
    lines.append(df_to_md(enrichment_df.round(4)))
    lines.append(
        "\nEnrichment ratio > 1 means that predicted type is over-represented among hesitant "
        "ids relative to confident ids (a candidate systematic weak spot); < 1 means "
        "under-represented. Remember `type_proxy` cannot detect unseen types -- an enrichment "
        "pattern here describes only the 5 known types' relative difficulty, not the unseen-type "
        "attack category directly (see the low-similarity flag above for that).\n"
    )

    lines.append("## Enrichment: hesitant vs. confident, by degradation stat\n")
    lines.append(df_to_md(degradation_df.round(4)))
    lines.append(
        "\n`ratio_h_over_c` > 1 means hesitant ids have a higher median value than confident "
        "ids for that stat (e.g. more blur, more moire, lower face_score); Mann-Whitney p tests "
        "whether the two distributions differ at all (not just their medians).\n"
    )

    lines.append("## Order-of-magnitude AuDET-improvement bound\n")
    lines.append(
        f"Current public LB AuDET = {args.current_audet} over the full graded test set; this "
        f"bound works in terms of the {n_present} locally-present ids as the task specifies, "
        f"assuming an unknown-but-guessed fraud rate of {args.assumed_fraud_rate * 100:.0f}% "
        "(train's actual rate -- public_test's true rate is hidden and deliberately harder, so "
        "this could be off in either direction). This is a **bound on achievable improvement "
        "under best-case assumptions, not a prediction** -- see the docstring in "
        "`audet_bound_table()` for the full assumption list (each flagged image assumed "
        "currently discordant with ~50% of bona-fide pairs, assumed fully fixable, assumed "
        "independent of each other, and assumed to genuinely be fraud)."
    )
    lines.append(
        f"\nn_fraud≈{audet_table.attrs['n_fraud']}, n_bonafide≈{audet_table.attrs['n_bonafide']}, "
        f"total pairs≈{audet_table.attrs['total_pairs']:,}, current discordant pairs≈"
        f"{audet_table.attrs['current_discordant']:,.0f}.\n"
    )
    lines.append(df_to_md(audet_table.round(6)))
    saturated = audet_table[audet_table["new_audet_bound"] <= 1e-9]
    if not saturated.empty:
        k_sat = int(saturated["k_flagged_images"].min())
        lines.append(
            f"\n**Note the saturation**: at k≈{k_sat} the bound already hits 0 (100% of the "
            "current gap 'explained'). That's a real consequence of AuDET being computed over "
            "millions of pairs while the current gap itself is numerically tiny -- a small "
            "number of severely-misranked positives, under this bound's assumptions, can already "
            "account for the entire observed gap. Don't read rows past the saturation point as "
            "distinguishable from each other; read them as 'this family, if it's at least this "
            "large and this badly misranked, would already fully explain the current gap.'"
        )
    lines.append(
        "\nLook up the row matching how many images you judge, after reviewing the montage "
        "grids, to be genuinely-fraud-but-misranked within a given attack category (e.g. the "
        "size of a face-edit-looking cluster, or a sum across several). That row's "
        "`improvement_pct` is the order-of-magnitude LB gain closing that specific family would "
        "be worth, under this bound's assumptions -- treat it as an upper limit to weigh "
        "against effort, not a guaranteed outcome.\n"
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[hesitant] wrote report -> {report_path}")


if __name__ == "__main__":
    main()
