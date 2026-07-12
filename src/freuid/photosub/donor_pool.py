"""Donor face pool for photo-substitution generators: bona-fide TRAIN portraits only.

No external face sources -- donors are drawn exclusively from this competition's own TRAINING
split (bona-fide rows only), specifically to dodge any question about external-data eligibility
under the competition rules. Noted here, not just at the call site.

Cheap attributes (grayscale-vs-color, mean luminance, face-box size, and a lightweight
appearance embedding) let ``sample_donor`` do similarity-controlled pairing without needing a
learned face-recognition model: the embedding here is a plain downsampled-grayscale intensity
vector (L2-normalized), not ArcFace/insightface -- deterministic, dependency-free, and fast
enough to build over the full bona-fide TRAIN pool on CPU. It is a coarse appearance/pose/
lighting proxy, not a true face-recognition embedding; swap in a stronger embedding (e.g.
insightface's recognition module, available on VESSL alongside the SCRFD detector already used
for face.json) via the ``embed_fn`` parameter if the render-sheet review finds the hard-case
pairing too weak.

No gender label exists anywhere in this dataset (``train_labels.csv`` has only id, image_path,
label, is_digital, type) -- "same apparent gender" hard-case matching is therefore approximated
via embedding nearest-neighbor selection alone (embedding closeness empirically tends to also
track coarse appearance including apparent gender, hairstyle, skin tone), not a dedicated
classifier. This is a real approximation, not a verified guarantee -- spot-check it in the
render-sheet human review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

EMBED_SIZE = 32  # 32x32 downsampled grayscale grid
EMBED_DIM = EMBED_SIZE * EMBED_SIZE


@dataclass
class DonorFace:
    id: str
    path: str
    type: str | None
    face_box: dict  # x1,y1,x2,y2,score -- ORIGINAL image's pixel coords (see freuid.preprocess)
    is_grayscale: bool
    mean_luminance: float  # in [0, 1]
    embedding: np.ndarray  # unit-norm-ish, shape (EMBED_DIM,), see cheap_face_embedding


def cheap_face_embedding(face_crop: Image.Image) -> np.ndarray:
    """Deterministic, dependency-free appearance embedding: downsample to a fixed small
    grayscale grid, mean-center, L2-normalize. See module docstring for what this is (and is
    not) a proxy for."""
    gray = face_crop.convert("L").resize((EMBED_SIZE, EMBED_SIZE), Image.BILINEAR)
    v = np.asarray(gray, dtype=np.float64).ravel()
    v = v - v.mean()
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros(EMBED_DIM, dtype=np.float64)
    return v / norm


def is_grayscale_crop(face_crop: Image.Image, saturation_threshold: float = 0.08) -> bool:
    """Mean HSV saturation below threshold -> effectively grayscale/desaturated print."""
    hsv = np.asarray(face_crop.convert("HSV"), dtype=np.float64)
    mean_sat = hsv[..., 1].mean() / 255.0
    return mean_sat < saturation_threshold


def mean_luminance(face_crop: Image.Image) -> float:
    gray = np.asarray(face_crop.convert("L"), dtype=np.float64)
    return float(gray.mean() / 255.0)


def face_crop_from_box(image: Image.Image, face_box: dict) -> Image.Image:
    w, h = image.size
    x1, y1 = max(0, int(face_box["x1"])), max(0, int(face_box["y1"]))
    x2, y2 = min(w, int(face_box["x2"])), min(h, int(face_box["y2"]))
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (1, 1))
    return image.crop((x1, y1, x2, y2))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _margined_crop(image: Image.Image, face_box: dict, margin: float) -> Image.Image:
    """Like face_crop_from_box, but expanded by ``margin`` on each side. A real face-recognition
    embedder (freuid.photosub.face_embedding.arcface_embedding) re-runs detection + alignment on
    whatever crop it's given, and a TIGHT (zero-margin) crop sometimes gives it too little
    context around the face to redetect reliably -- this is used only for the embedding input,
    never for the compositing donor_crop elsewhere (which stays tight, per generators.py)."""
    w, h = image.size
    x1, y1, x2, y2 = face_box["x1"], face_box["y1"], face_box["x2"], face_box["y2"]
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin
    nx1, ny1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    nx2, ny2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    if nx2 <= nx1 or ny2 <= ny1:
        return Image.new("RGB", (1, 1))
    return image.crop((nx1, ny1, nx2, ny2))


def build_donor_record(
    id_: str,
    path: str | Path,
    doc_type: str | None,
    face_box: dict,
    embed_fn: Callable[[Image.Image], np.ndarray | None] = cheap_face_embedding,
    embed_margin: float = 0.3,
) -> DonorFace | None:
    """Build one DonorFace. Returns None when ``face_box`` is the center-square fallback
    (``score <= 0``, i.e. no real SCRFD detection), the crop degenerates to nothing, or
    ``embed_fn`` itself returns None (e.g. freuid.photosub.face_embedding.arcface_embedding
    failing to (re-)detect a face in the crop -- treated as "reject this donor" rather than
    silently mixing embedding spaces within one pool, which would make cosine similarity
    meaningless)."""
    if float(face_box.get("score", 0.0)) <= 0.0:
        return None
    with Image.open(path) as img:
        img = img.convert("RGB")
        crop = face_crop_from_box(img, face_box)
        if crop.size[0] <= 1 or crop.size[1] <= 1:
            return None
        embedding = embed_fn(_margined_crop(img, face_box, embed_margin))
        if embedding is None:
            return None
        return DonorFace(
            id=str(id_),
            path=str(path),
            type=doc_type,
            face_box=face_box,
            is_grayscale=is_grayscale_crop(crop),
            mean_luminance=mean_luminance(crop),
            embedding=embedding,
        )


def build_donor_index(
    regions_dir_path: str | Path,
    rows: list,  # e.g. load_labels(data_dir, "train") filtered to label==0, .itertuples()
    embed_fn: Callable[[Image.Image], np.ndarray | None] = cheap_face_embedding,
    limit: int | None = None,
    progress_every: int | None = 200,
    label: str = "donor_index",
) -> list[DonorFace]:
    """Bona-fide TRAIN donor pool. Callers must pre-filter ``rows`` to ``label == 0`` -- this
    function does not re-check labels, so it can be reused for the probe-embedding lookup too
    (probes are neither donors nor guaranteed bona-fide from this function's point of view).

    A real face-recognition embedder (freuid.photosub.face_embedding.arcface_embedding) is slow
    enough per-image (full re-detect + align + ONNX forward pass, no cross-image batching) that
    a silent multi-thousand-item loop gives no sense of progress or ETA -- ``progress_every``
    prints elapsed/rate/ETA every N items; set None to disable."""
    import time

    items = list(rows)
    if limit is not None:
        items = items[:limit]
    donors: list[DonorFace] = []
    t0 = time.monotonic()
    for i, row in enumerate(items):
        face_path = Path(regions_dir_path) / str(row.id) / "face.json"
        if not face_path.exists() or not Path(str(row.path)).exists():
            continue
        try:
            face_box = json.loads(face_path.read_text())
        except Exception:
            continue
        rec = build_donor_record(row.id, row.path, getattr(row, "type", None), face_box, embed_fn)
        if rec is not None:
            donors.append(rec)
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            remaining = (len(items) - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"[{label}] {i + 1}/{len(items)} ({rate:.1f}/s, elapsed {elapsed:.0f}s, "
                  f"ETA {remaining:.0f}s) -- {len(donors)} kept so far")
    print(f"[{label}] done: {len(donors)}/{len(items)} kept in {time.monotonic() - t0:.0f}s")
    return donors


def exclude_probe_overlap(
    donors: list[DonorFace],
    probes: list[DonorFace],
    similarity_threshold: float = 0.9,
) -> list[DonorFace]:
    """Paranoid, cheap probe-honesty guard: drop any donor whose ``type`` matches a probe's
    type AND whose embedding cosine similarity to that probe exceeds ``similarity_threshold``.
    Keeps the frozen diagnostic probes (``data/probes/*.csv``) honest by construction -- a donor
    that's suspiciously close to a probe id (same template + near-duplicate face) could leak
    probe-specific appearance into the generated training data and inflate probe performance for
    reasons that wouldn't hold on the real, unseen test set."""
    if not probes:
        return list(donors)
    kept = []
    for d in donors:
        same_type_probes = [p for p in probes if p.type == d.type]
        if any(cosine_similarity(d.embedding, p.embedding) >= similarity_threshold for p in same_type_probes):
            continue
        kept.append(d)
    return kept


def sample_donor(
    donors: list[DonorFace],
    rng: np.random.Generator,
    target: DonorFace,
    hard_fraction: float = 0.4,
    luminance_tolerance: float = 0.15,
    hard_pool_size: int = 10,
    hard_temperature: float = 0.05,
    exclude_id: str | None = None,
) -> DonorFace:
    """Pick one donor for ``target`` (the bona-fide row being tampered).

    With probability ``hard_fraction``, picks a HARD case: among donors within
    ``luminance_tolerance`` of the target's own luminance, samples (softmax-weighted by
    embedding cosine similarity, temperature ``hard_temperature``) from the top
    ``hard_pool_size`` most similar -- approximating "same apparent gender, similar face
    embedding, matched luminance" (see module docstring's gender caveat). Otherwise picks
    uniformly at random from the whole pool (broad sampling), so the model also sees the "two
    very different people" case and doesn't overfit to only-lookalike substitutions either.

    Similarity-WEIGHTED rather than uniform-among-top-k: whenever the luminance-filtered
    candidate pool is smaller than ``hard_pool_size`` (the common case), a flat uniform choice
    across "the top-k" would just be a uniform choice across everyone, silently defeating the
    whole point of a "hard" pick. Weighting by similarity keeps the pick biased toward the
    genuinely closest match(es) regardless of how many candidates happen to be in range.

    Excludes ``exclude_id`` (the target's own id, if present in the donor pool) so a target is
    never "swapped" with its own face.
    """
    exclude_id = exclude_id or target.id
    pool = [d for d in donors if d.id != exclude_id]
    if not pool:
        raise ValueError("donor pool is empty after excluding the target itself")

    if rng.random() < hard_fraction:
        candidates = [d for d in pool if abs(d.mean_luminance - target.mean_luminance) <= luminance_tolerance]
        if not candidates:
            candidates = pool
        sims = np.array([cosine_similarity(target.embedding, d.embedding) for d in candidates])
        top_k = min(hard_pool_size, len(candidates))
        top_idx = np.argsort(-sims)[:top_k]
        top_sims = sims[top_idx]
        weights = np.exp((top_sims - top_sims.max()) / hard_temperature)
        weights = weights / weights.sum()
        choice = int(rng.choice(len(top_idx), p=weights))
        return candidates[top_idx[choice]]

    choice = int(rng.integers(len(pool)))
    return pool[choice]
