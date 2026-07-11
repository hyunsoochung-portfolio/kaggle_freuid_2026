"""Per-epoch scoring hooks against the diagnostic probe id lists in data/probes/ (see that
directory's README for full provenance). These ids are NEVER training data and never influence
checkpoint selection (the checkpoint metric stays probe_audet, per CLAUDE.md) -- this module
exists so photosub_v0's pre-registered gates (configs/photosub_v0.yaml) can be read off the
training log every epoch instead of requiring a separate offline pass.

Four probes, four different aggregations (per photosub_v0's spec):
    deep         (9 ids)   -- log EACH id's logit + rank percentile individually: these are the
                              specific confirmed-fraud misses the fix targets.
    boundary     (59 ids)  -- mean logit + mean rank percentile only (aggregate boundary-mass
                              check, not individually diagnostic).
    clean_floor  (295 ids) -- mean logit only: the regression GUARD, must not rise.
    ceiling      (200 ids) -- mean logit only: the "don't trade away existing detection
                              capability" guard (see data/probes/README.md -- self-consistency,
                              no ground truth available for public_test). Also carries
                              `doc_type_proxy` (filled via
                              `scripts/analysis/classify_ceiling_probe_types.py`'s color-
                              histogram classifier, NOT embedding-KNN -- see that script's
                              docstring for why embedding classification is specifically
                              unreliable in this saturated zone), used for the per-template
                              budget breakdown below.

Rank percentile is computed against the CURRENT epoch's VAL score distribution, not against the
full public_test corpus the original review_package/logit_census pct_rank used -- a deliberately
cheap per-epoch proxy (scoring all 7821 public_test images every epoch would dominate epoch
time), fine for tracking a TREND across epochs of one training run, not comparable in absolute
terms to the pct_rank values already frozen in the probe CSVs themselves.

**Threshold-watch report** (`threshold_watch`): the tail's equivalent of the deep-9 table -- the
real APCER@1%BPCER operating point, computed EXACTLY from this epoch's VAL split (real labels,
no imputation needed, unlike the public-test-only analyses under scripts/analysis/), plus where
the frozen boundary-59 and ceiling-200 probes currently sit relative to it:
  - threshold_score/threshold_rank: the exact 1%-BPCER crossing point on VAL this epoch.
  - boundary_below_threshold: how many of the 59 confirmed-fraud boundary ids are STILL missed
    at this operating point (the APCER battle, made visible every epoch instead of only at the
    end of training).
  - ceiling_above_threshold (+ per-template breakdown): how many of the 200 ceiling
    self-consistency ids currently sit above the threshold -- since these have no ground truth,
    this is EXPOSURE, not a confirmed cost (see `scripts/analysis/threshold_pricing_out/
    threshold_pricing_report.md`'s "ceiling-zone confusion" finding on the actual submission,
    which is exactly what this per-epoch watch exists to catch developing during training,
    before it ever reaches a submission).

Fails loudly (FileNotFoundError) if a probe CSV is missing -- silently skipping a gate the
config asked for would defeat the point of a pre-registered check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from freuid.data import load_labels

_PROBE_FILES = {
    "deep": "missed_frauds_deep_ids.csv",
    "boundary": "missed_frauds_boundary_ids.csv",
    "clean_floor": "clean_floor_sample_ids.csv",
    "ceiling": "ceiling_frauds_sample_ids.csv",
}


def probes_dir(data_dir: str | Path) -> Path:
    """data/probes/ lives at the repo root, one level up from a plain ``data_dir="data"`` --
    but configs sometimes point ``data_dir`` elsewhere (VESSL: ``/root/repo/data``), so resolve
    relative to data_dir's parent rather than assuming a fixed relationship to this file."""
    return Path(data_dir).resolve().parent / "data" / "probes"


def _require_probe_csv(pdir: Path, name: str) -> pd.DataFrame:
    path = pdir / _PROBE_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"photosub probe hooks require {path} (see data/probes/README.md for provenance) -- "
            "regenerate via `python scripts/analysis/deep_miss_dossiers.py --stage freeze_probes` "
            "and/or `python scripts/analysis/build_ceiling_probe.py`, or disable probe hooks in "
            "the config."
        )
    return pd.read_csv(path, dtype={"id": str})


@torch.no_grad()
def _score_ids(
    model: torch.nn.Module,
    ids: list[str],
    data_dir: str | Path,
    transform,
    device,
    batch_size: int = 32,
) -> np.ndarray:
    """Scores public_test images by id, single-pass (no TTA -- a per-epoch training diagnostic,
    not a submission-quality score), returning RAW LOGITS. Raises if any id isn't present in
    public_test."""
    test_meta = load_labels(data_dir, "public_test").set_index("id")
    missing = [i for i in ids if i not in test_meta.index]
    if missing:
        raise ValueError(f"{len(missing)} probe ids not found in public_test, e.g. {missing[:3]!r}")

    was_training = model.training
    model.eval()
    logits_out = np.empty(len(ids), dtype=np.float64)
    for i in range(0, len(ids), batch_size):
        chunk = ids[i : i + batch_size]
        paths = [test_meta.loc[id_, "path"] for id_ in chunk]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths]).to(device)
        logits = model(imgs)
        logits_out[i : i + len(chunk)] = logits.squeeze(1).float().cpu().numpy()
    model.train(was_training)
    return logits_out


def _pct_rank(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Percentile rank of each of ``values`` within ``reference`` ("mean" convention: ties
    split the difference), vectorized -- avoids adding a scipy dependency for one function."""
    lt = (reference[None, :] < values[:, None]).mean(axis=1)
    eq = (reference[None, :] == values[:, None]).mean(axis=1)
    return 100.0 * (lt + 0.5 * eq)


def threshold_watch(val_scores: np.ndarray, val_labels: np.ndarray, bpcer_target: float = 0.01) -> dict:
    """The EXACT 1%-BPCER operating point on this epoch's VAL split (real labels -> no MC/
    imputation needed, unlike the public-test-only analyses in scripts/analysis/). Walks the
    score distribution from the top down, accumulating bona-fide count, until it crosses
    ``bpcer_target`` of the total bona-fide count -- same convention as the vendored official
    scorer's ``_det_curve``/``_apcer_at_bpcer_from_curve`` (src/freuid/official_score.py),
    reimplemented directly on hard counts here since a full DET-curve build is unnecessary
    overhead for just the crossing point.
    """
    order = np.argsort(-val_scores, kind="mergesort")
    sorted_scores = val_scores[order]
    sorted_labels = val_labels[order]
    n_bonafide = int((val_labels == 0).sum())
    if n_bonafide == 0:
        return {"threshold_score": float("nan"), "threshold_rank": 0, "n_val": len(val_scores),
                "n_val_bonafide": 0}
    budget = bpcer_target * n_bonafide
    cum_bonafide = np.cumsum(sorted_labels == 0)
    idx = int(np.searchsorted(cum_bonafide, budget, side="left"))
    idx = min(idx, len(sorted_scores) - 1)
    return {
        "threshold_score": float(sorted_scores[idx]),
        "threshold_rank": idx + 1,
        "n_val": len(val_scores),
        "n_val_bonafide": n_bonafide,
    }


def run_probe_hooks(
    model: torch.nn.Module,
    data_dir: str | Path,
    transform,
    device,
    val_scores: np.ndarray,
    val_labels: np.ndarray,
) -> dict:
    """Runs all four probes for the current epoch. ``val_scores``/``val_labels`` (this epoch's
    val-split P(fraud) scores + real labels, already computed by the normal validation pass) are
    the reference distribution for percentile rank AND the exact threshold-watch crossing -- see
    module docstring for why.

    Returns a flat metrics dict (aggregate keys only, prefixed ``probe_``) plus a
    ``probe_deep_detail`` list of per-id dicts for the 9 deep-miss ids (caller logs these
    individually per photosub_v0's spec) and a ``probe_ceiling_by_template`` DataFrame for the
    per-template budget breakdown.
    """
    pdir = probes_dir(data_dir)

    deep_df = _require_probe_csv(pdir, "deep")
    boundary_df = _require_probe_csv(pdir, "boundary")
    floor_df = _require_probe_csv(pdir, "clean_floor")
    ceiling_df = _require_probe_csv(pdir, "ceiling")

    deep_logits = _score_ids(model, deep_df["id"].tolist(), data_dir, transform, device)
    boundary_logits = _score_ids(model, boundary_df["id"].tolist(), data_dir, transform, device)
    floor_logits = _score_ids(model, floor_df["id"].tolist(), data_dir, transform, device)
    ceiling_logits = _score_ids(model, ceiling_df["id"].tolist(), data_dir, transform, device)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    deep_scores, boundary_scores = _sigmoid(deep_logits), _sigmoid(boundary_logits)
    ceiling_scores = _sigmoid(ceiling_logits)
    deep_pct = _pct_rank(deep_scores, val_scores)
    boundary_pct = _pct_rank(boundary_scores, val_scores)

    deep_detail = [
        {"id": id_, "logit": float(logit), "score": float(s), "pct_rank": float(p)}
        for id_, logit, s, p in zip(deep_df["id"], deep_logits, deep_scores, deep_pct)
    ]

    watch = threshold_watch(val_scores, val_labels)
    threshold_score = watch["threshold_score"]
    boundary_below = int((boundary_scores < threshold_score).sum()) if np.isfinite(threshold_score) else -1
    ceiling_above = int((ceiling_scores >= threshold_score).sum()) if np.isfinite(threshold_score) else -1

    ceiling_by_template = None
    if "doc_type_proxy" in ceiling_df.columns and ceiling_df["doc_type_proxy"].notna().any() and np.isfinite(threshold_score):
        tdf = ceiling_df[["doc_type_proxy"]].copy()
        tdf["above_threshold"] = ceiling_scores >= threshold_score
        ceiling_by_template = (
            tdf.groupby("doc_type_proxy")["above_threshold"]
            .agg(n_above="sum", n_total="count")
            .reset_index()
            .rename(columns={"doc_type_proxy": "template"})
        )
        ceiling_by_template["budget_share"] = ceiling_by_template["n_above"] / max(1, ceiling_above)

    return {
        "probe_deep_mean_logit": float(deep_logits.mean()),
        "probe_deep_mean_pct_rank": float(deep_pct.mean()),
        "probe_deep_detail": deep_detail,
        "probe_boundary_mean_logit": float(boundary_logits.mean()),
        "probe_boundary_mean_pct_rank": float(boundary_pct.mean()),
        "probe_clean_floor_mean_logit": float(floor_logits.mean()),
        "probe_ceiling_mean_logit": float(ceiling_logits.mean()),
        "probe_threshold_score": threshold_score,
        "probe_threshold_rank": watch["threshold_rank"],
        "probe_n_val_bonafide": watch["n_val_bonafide"],
        "probe_boundary_below_threshold": boundary_below,
        "probe_boundary_total": len(boundary_df),
        "probe_ceiling_above_threshold": ceiling_above,
        "probe_ceiling_total": len(ceiling_df),
        "probe_ceiling_by_template": ceiling_by_template,
    }
