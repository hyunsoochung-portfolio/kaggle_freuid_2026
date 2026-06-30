"""Card rectification (FastSAM) and face detection (SCRFD) with disk caching.

Run once before training to fill the regions cache; training reads from cache.
Everything here runs single-process -- detectors are not fork-safe.

Cache layout under {data_dir}/processed/regions/{id}/:
    card.png   — rectified 512×512 card (warpPerspective or resize fallback)
    face.json  — portrait bbox in canonical card coords, or center-square fallback

Usage (CLI):
    python -m freuid.preprocess --config configs/consistency.yaml \\
        --splits train public_test --limit 5
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from freuid.config import Config
from freuid.data import load_labels


_CARD_SIZE = 512  # canonical output resolution

# Module-level singletons — loaded lazily, once per process.
# Sentinel string "unavailable" means we already tried and failed.
_fastsam: object = None
_scrfd: object = None


# ---------------------------------------------------------------------------
# Detector loading
# ---------------------------------------------------------------------------

def _load_fastsam():
    global _fastsam
    if _fastsam is None:
        try:
            from ultralytics import FastSAM as _FS
            _fastsam = _FS("FastSAM-s.pt")
            print("[preprocess] FastSAM-s loaded")
        except Exception as exc:
            print(f"[preprocess] FastSAM unavailable ({exc}); will use resize fallback")
            _fastsam = "unavailable"
    return None if _fastsam == "unavailable" else _fastsam


def _load_scrfd():
    global _scrfd
    if _scrfd is None:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(allowed_modules=["detection"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            _scrfd = app
            print("[preprocess] SCRFD/InsightFace loaded")
        except Exception as exc:
            print(f"[preprocess] SCRFD unavailable ({exc}); will use center-square fallback")
            _scrfd = "unavailable"
    return None if _scrfd == "unavailable" else _scrfd


# ---------------------------------------------------------------------------
# Card rectification helpers
# ---------------------------------------------------------------------------

def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 xy-points as TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _approx_quad(contour: np.ndarray) -> np.ndarray | None:
    """Approximate contour as a 4-sided polygon. Returns (4,2) float32 or None."""
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return None
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) != 4:
        return None
    return approx.reshape(4, 2).astype(np.float32)


def _extract_quad_from_results(results, img_h: int, img_w: int) -> np.ndarray | None:
    """Pull the largest quadrilateral from FastSAM results object."""
    try:
        masks_obj = getattr(results[0], "masks", None)
    except (IndexError, TypeError):
        return None
    if masks_obj is None:
        return None

    best_quad: np.ndarray | None = None
    best_area = 0.0

    # Try polygon coords first — already in original image coordinate space.
    poly_list = getattr(masks_obj, "xy", None) or []
    for poly in poly_list:
        if len(poly) < 4:
            continue
        cnt = poly.astype(np.int32).reshape(-1, 1, 2)
        q = _approx_quad(cnt)
        if q is not None:
            area = float(cv2.contourArea(q))
            if area > best_area:
                best_area = area
                best_quad = _order_quad(q)

    if best_quad is not None:
        return best_quad

    # Fallback: binary mask tensor, resize to original resolution then contour.
    mask_data = getattr(masks_obj, "data", None)
    if mask_data is not None:
        for m in mask_data.cpu().numpy():
            m_u8 = (cv2.resize(m, (img_w, img_h)) > 0.5).astype(np.uint8)
            cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            q = _approx_quad(cnt)
            if q is None:
                continue
            area = float(cv2.contourArea(q))
            if area > best_area:
                best_area = area
                best_quad = _order_quad(q)

    return best_quad


def rectify_card(image: np.ndarray, size: int = _CARD_SIZE) -> np.ndarray:
    """FastSAM → largest quad → warpPerspective to (size × size); fallback: resize.

    Args:
        image: HWC uint8 RGB array.
        size:  Output square side length (default: 512).

    Returns HWC uint8 RGB at the requested resolution.
    """
    model = _load_fastsam()
    if model is not None:
        try:
            results = model(image, imgsz=1024, conf=0.4, iou=0.9, verbose=False)
            quad = _extract_quad_from_results(results, image.shape[0], image.shape[1])
            if quad is not None:
                dst = np.array(
                    [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                    dtype=np.float32,
                )
                M = cv2.getPerspectiveTransform(quad, dst)
                return cv2.warpPerspective(image, M, (size, size))
        except Exception as exc:
            print(f"[preprocess] FastSAM inference error: {exc}")
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

def detect_face_box(image: np.ndarray) -> dict:
    """SCRFD → most-confident portrait box; fallback: center square of 0.6·min(H,W).

    Args:
        image: HWC uint8 RGB array (after card rectification).

    Returns a dict with keys x1, y1, x2, y2 (int, pixel coords) and score (float).
    """
    app = _load_scrfd()
    if app is not None:
        try:
            # InsightFace expects BGR
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            faces = app.get(bgr)
            if faces:
                best = max(faces, key=lambda f: float(f.det_score))
                x1, y1, x2, y2 = [int(v) for v in best.bbox.tolist()]
                return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": float(best.det_score)}
        except Exception as exc:
            print(f"[preprocess] SCRFD error: {exc}")
    # Center-square fallback
    h, w = image.shape[:2]
    side = int(0.6 * min(h, w))
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    return {"x1": x1, "y1": y1, "x2": min(w, x1 + side), "y2": min(h, y1 + side), "score": 0.0}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def regions_dir(data_dir: str | Path) -> Path:
    """Canonical path to the regions cache under a data root."""
    return Path(data_dir) / "processed" / "regions"


# ---------------------------------------------------------------------------
# Precaching
# ---------------------------------------------------------------------------

def precache_regions(
    cfg: Config,
    splits: list[str],
    limit: int | None = None,
) -> None:
    """Cache card.png and face.json for every id in *splits*. Idempotent.

    Skips ids where both files already exist (safe to rerun after interruption).
    Runs single-process — call before launching a multiprocessing DataLoader.

    Output layout::

        {data_dir}/processed/regions/{id}/card.png
        {data_dir}/processed/regions/{id}/face.json
    """
    cache_root = regions_dir(cfg.data_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        df = load_labels(cfg.data_dir, split)
        rows = list(df.itertuples(index=False))
        if limit is not None:
            rows = rows[:limit]

        n_done = n_skip = n_fail = 0
        for i, row in enumerate(rows):
            out_dir = cache_root / str(row.id)
            card_path = out_dir / "card.png"
            face_path = out_dir / "face.json"

            if card_path.exists() and face_path.exists():
                n_skip += 1
                continue

            try:
                bgr = cv2.imread(str(row.path))
                if bgr is None:
                    raise FileNotFoundError(row.path)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                out_dir.mkdir(exist_ok=True)

                card = rectify_card(rgb)
                cv2.imwrite(str(card_path), cv2.cvtColor(card, cv2.COLOR_RGB2BGR))

                face = detect_face_box(card)  # coords in canonical card space
                face_path.write_text(json.dumps(face))

                n_done += 1
                if (i + 1) % 500 == 0:
                    print(f"[preprocess] {split}: {i + 1}/{len(rows)} ...")
            except Exception as exc:
                print(f"[preprocess] {split}/{row.id}: FAIL {exc}")
                n_fail += 1

        print(f"[preprocess] {split}: done={n_done} skipped={n_skip} failed={n_fail}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from freuid.config import load_config

    parser = argparse.ArgumentParser(description="Precache rectified cards and face boxes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "public_test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    precache_regions(load_config(args.config), args.splits, args.limit)
