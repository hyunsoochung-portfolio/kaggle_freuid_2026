"""Augmentation pipelines for training.

Albumentations-based; gated by cfg.extra["augment"] so existing configs are unaffected.
No horizontal flip anywhere — document text and orientation must be preserved.
"""

from __future__ import annotations

import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


class _AlbumentationsTransform:
    """Wraps an albumentations Compose so it accepts PIL Images (same contract as
    torchvision transforms.Compose) and returns a CHW float32 tensor."""

    def __init__(self, pipeline: A.Compose) -> None:
        self.pipeline = pipeline

    def __call__(self, img):
        arr = np.array(img)  # PIL RGB → HWC uint8
        return self.pipeline(image=arr)["image"]  # CHW float32 tensor


def recapture_transforms(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    seed: int | None = None,
) -> _AlbumentationsTransform:
    """Print-and-recapture simulation.

    ``seed`` makes the degradation reproducible (albumentations 2.x seeds the pipeline, not
    the global RNG). Pass a seed to get a fixed, identical degradation -- used to keep the
    validation analog copies stable across epochs. Leave None for random train augmentation.

    Ordered to match the real analog degradation chain:
      1. spatial resize (to the network's input resolution)
      2. first JPEG encode (capture / upload to a system)
      3. downscale (lower-res sensor or scan at reduced DPI)
      4. second JPEG encode (double-compression artifact)
      5. optical: focus blur or motion blur
      6. sensor noise
      7. lighting & colour shifts
      8. mild perspective warp + small rotation (handling/scan tilt)
      9. ImageNet normalize + to tensor

    std_range for GaussNoise is fractional relative to uint8 max (255), so
    (0.01, 0.04) → ~2.5–10 pixel std — subtle but visible grain.
    """
    pipeline = A.Compose([
        A.Resize(image_size, image_size),
        # first JPEG compression: simulate saving/uploading the captured image
        A.ImageCompression(quality_range=(50, 95), p=0.9),
        # downscale: lower-resolution sensor or reduced-DPI scan, then bicubic back up
        A.Downscale(
            scale_range=(0.5, 0.85),
            interpolation_pair={"downscale": 2, "upscale": 2},  # INTER_CUBIC
            p=0.5,
        ),
        # second JPEG pass: double-compression creates distinctive block artifacts
        A.ImageCompression(quality_range=(60, 95), p=0.7),
        # optical degradation: defocus or motion blur from handheld capture
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=(3, 7)),
        ], p=0.5),
        # sensor / scan noise
        A.GaussNoise(std_range=(0.01, 0.04), p=0.5),
        # lighting: exposure and contrast variation from ambient / scanner lamp
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # colour: white-balance shift from mixed lighting or scanner calibration
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
        # geometry: mild perspective from non-flat capture angle (NOT a flip)
        A.Perspective(scale=(0.02, 0.05), p=0.3),
        # small rotation: phone tilt / document not perfectly square in scanner
        A.Rotate(limit=5, p=0.3),
        # ImageNet normalise + CHW tensor
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], seed=seed)
    return _AlbumentationsTransform(pipeline)


# ---------------------------------------------------------------------------
# Synthetic tamper edits
# ---------------------------------------------------------------------------

def _copy_move(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Clone a rectangular patch from one location and paste it elsewhere."""
    h, w = arr.shape[:2]
    ph = rng.integers(max(1, h // 10), max(2, h // 4))
    pw = rng.integers(max(1, w // 10), max(2, w // 4))
    y1 = int(rng.integers(0, h - ph))
    x1 = int(rng.integers(0, w - pw))
    y2 = int(rng.integers(0, h - ph))
    x2 = int(rng.integers(0, w - pw))
    out = arr.copy()
    out[y2:y2 + ph, x2:x2 + pw] = arr[y1:y1 + ph, x1:x1 + pw]
    return out


def _local_splice(arr: np.ndarray, donor_arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paste a patch from a donor image into arr (simulates photo substitution)."""
    h, w = arr.shape[:2]
    # Resize donor to match target resolution so patch pixels are compatible
    donor_resized = np.array(Image.fromarray(donor_arr).resize((w, h), Image.BILINEAR))
    ph = rng.integers(max(1, h // 6), max(2, h // 3))
    pw = rng.integers(max(1, w // 6), max(2, w // 3))
    sy = int(rng.integers(0, h - ph))
    sx = int(rng.integers(0, w - pw))
    dy = int(rng.integers(0, h - ph))
    dx = int(rng.integers(0, w - pw))
    out = arr.copy()
    out[dy:dy + ph, dx:dx + pw] = donor_resized[sy:sy + ph, sx:sx + pw]
    return out


def _field_smudge(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blur, recolor, or fill a document-field-shaped region (simulates field alteration)."""
    h, w = arr.shape[:2]
    # Typical document field: wide and short
    fh = rng.integers(max(1, h // 15), max(2, h // 5))
    fw = rng.integers(max(1, w // 5), max(2, w // 2))
    y = int(rng.integers(0, h - fh))
    x = int(rng.integers(0, w - fw))
    out = arr.copy()
    region = out[y:y + fh, x:x + fw]

    edit = int(rng.integers(3))
    if edit == 0:
        # blur: simulate out-of-focus re-photograph of a re-printed field
        blurred = Image.fromarray(region).filter(ImageFilter.GaussianBlur(radius=4))
        out[y:y + fh, x:x + fw] = np.array(blurred)
    elif edit == 1:
        # recolor: shift brightness/saturation to simulate digitally edited text
        shift = rng.integers(-50, 50, size=3).astype(np.int16)
        out[y:y + fh, x:x + fw] = np.clip(region.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    else:
        # fill + noise: white-out and reprint simulation
        fill = region.mean(axis=(0, 1)).astype(np.int16)
        noise = rng.integers(-25, 25, size=region.shape).astype(np.int16)
        out[y:y + fh, x:x + fw] = np.clip(fill + noise, 0, 255).astype(np.uint8)
    return out


class AnalogDoubleDataset(Dataset):
    """Doubles the dataset with analog (recapture) copies of the digital images.

    The training set becomes {every image, clean transform} ∪ {every is_digital=True image
    again, recapture transform}. Each digital document is thus seen both as-is and as a
    simulated print-and-recapture, with its original label preserved -- teaching the model
    the digital AND the analog appearance (the digital→physical shift the hidden test probes).
    is_digital=False images (already physically captured) are not duplicated; is_digital=None
    (unknown provenance) is likewise left un-doubled.

    The base dataset must be built with transform=None so this wrapper owns all transforms.
    Yields (image, label); the consistency face-meta path is not supported here.

    ``make_analog`` is a factory ``seed -> transform`` (e.g. a partial of recapture_transforms).
    ``deterministic_seed``: set it for VALIDATION -- each analog copy is then built from a fresh
    pipeline seeded with (deterministic_seed + sample_index), so the recaptured val images are
    identical every epoch (albumentations 2.x seeds the pipeline, not the global RNG, so this is
    the only way to fix them). Leave None for training (one shared random pipeline).
    """

    def __init__(
        self, base: Dataset, clean_transform, make_analog, deterministic_seed: int | None = None
    ) -> None:
        self.base = base
        self.clean_tf = clean_transform
        self.make_analog = make_analog          # seed (int|None) -> analog transform
        self.det_seed = deterministic_seed
        self._analog_tf = make_analog(None)     # one random pipeline, reused for train copies
        # indices of the digital samples -- these get a second, recaptured copy appended.
        # `if s.is_digital` excludes both False and None (unknown provenance -> not doubled).
        self.analog_idx = [i for i, s in enumerate(base.samples) if s.is_digital]  # type: ignore[attr-defined]
        self.n = len(base.samples)  # type: ignore[attr-defined]

    def __len__(self) -> int:
        return self.n + len(self.analog_idx)

    def __getitem__(self, idx: int):
        if idx < self.n:
            sample, tf = self.base.samples[idx], self.clean_tf  # type: ignore[attr-defined]
        else:
            orig_i = self.analog_idx[idx - self.n]
            sample = self.base.samples[orig_i]  # type: ignore[attr-defined]
            # val: fresh pipeline seeded per-sample -> identical every epoch.
            # train: the shared random pipeline -> fresh degradation each pass.
            tf = self.make_analog(self.det_seed + orig_i) if self.det_seed is not None \
                else self._analog_tf
        src = sample.card_path if sample.card_path is not None else sample.path
        img = Image.open(src).convert("RGB")
        return tf(img), sample.label


# ---------------------------------------------------------------------------
# Synthetic-fraud augmentation (data-grounded)
# ---------------------------------------------------------------------------
# Manufactures a fraud (label 1) from a bona-fide (label 0) by reproducing a
# tell observed in the real fraud data (see scripts/synth_tamper.py and the
# 100-image fraud-pattern analysis):
#   face_clean_paste    -- sharp portrait paste, hard seam, security overlay
#                          broken; on Mauritius this leaves the ghost portrait =
#                          the original person (main != ghost). All doc types.
#   face_color_on_gray  -- a colour face grafted onto a grayscale body (BENIN
#                          only: BENIN genuine photos are black-and-white).
#   field_carve         -- a value field scratched/rewritten: rough wood-grain
#                          + dark smudge + degraded digits + box boundary.
# Scalar choices come from a python ``rng`` (random.Random); array noise from an
# ``nrng`` (np.random.Generator). Seeding both makes the val probe deterministic;
# leaving them fresh gives per-epoch variety for training.

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _detect_face(img):
    """Largest frontal face (x, y, w, h) in a BGR image, or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]


def _photo_box(face, W, H):
    x, y, w, h = face
    return (max(0, int(x - 0.5 * w)), max(0, int(y - 0.7 * h)),
            min(W, int(x + w + 0.5 * w)), min(H, int(y + h + 1.15 * h)))


def _face_box(face, W, H, fx=0.18, fy=0.22):
    x, y, w, h = face
    ex, ey = int(w * fx), int(h * fy)
    return (max(0, x - ex), max(0, y - ey), min(W, x + w + ex), min(H, y + h + ey))


def _to_gray3(bgr):
    return cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _clarify(photo, rng):
    """Boost contrast + saturation + sharpness so a paste STANDS OUT (the real
    tell: fraud photos look 'too clean / on top', genuine ones are overlay-tinted)."""
    f = photo.astype(np.float32)
    m = f.mean()
    f = np.clip((f - m) * rng.uniform(1.15, 1.30) + m + rng.uniform(-3, 8), 0, 255)
    hsv = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1.15, 1.40), 0, 255)
    f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), 1.2)
    return np.clip(f + rng.uniform(0.6, 1.0) * (f - blur), 0, 255).astype(np.uint8)


def _face_clean_paste(target, donor, rng):
    ft, fd = _detect_face(target), _detect_face(donor)
    if ft is None or fd is None:
        return None
    Ht, Wt = target.shape[:2]
    Hd, Wd = donor.shape[:2]
    X0, Y0, X1, Y1 = _photo_box(ft, Wt, Ht)
    dx0, dy0, dx1, dy1 = _photo_box(fd, Wd, Hd)
    if X1 - X0 < 8 or Y1 - Y0 < 8 or dx1 - dx0 < 8 or dy1 - dy0 < 8:
        return None
    paste = cv2.resize(donor[dy0:dy1, dx0:dx1], (X1 - X0, Y1 - Y0))
    out = target.copy()
    out[Y0:Y1, X0:X1] = _clarify(paste, rng)      # hard rectangular seam
    return out, (X0, Y0, X1, Y1)


def _face_color_on_gray(target, donor, rng):
    ft, fd = _detect_face(target), _detect_face(donor)
    if ft is None or fd is None:
        return None
    Ht, Wt = target.shape[:2]
    Hd, Wd = donor.shape[:2]
    X0, Y0, X1, Y1 = _photo_box(ft, Wt, Ht)
    rw, rh = X1 - X0, Y1 - Y0
    if rw < 8 or rh < 8:
        return None
    body = _to_gray3(target[Y0:Y1, X0:X1])        # grayscale body
    fx0, fy0, fx1, fy1 = _face_box(ft, Wt, Ht)
    tx0, ty0 = max(0, fx0 - X0), max(0, fy0 - Y0)
    tx1, ty1 = min(rw, fx1 - X0), min(rh, fy1 - Y0)
    tw, th = tx1 - tx0, ty1 - ty0
    if tw < 4 or th < 4:
        return None
    dx0, dy0, dx1, dy1 = _face_box(fd, Wd, Hd)
    if dx1 - dx0 < 4 or dy1 - dy0 < 4:
        return None
    cface = _clarify(cv2.resize(donor[dy0:dy1, dx0:dx1], (tw, th)), rng)
    mask = np.zeros((th, tw), np.float32)
    cv2.ellipse(mask, (tw // 2, th // 2),
                (int(tw * 0.46), int(th * 0.5)), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, tw * 0.03))[..., None]
    body[ty0:ty1, tx0:tx1] = (
        cface * mask + body[ty0:ty1, tx0:tx1] * (1 - mask)).astype(np.uint8)
    out = target.copy()
    out[Y0:Y1, X0:X1] = body
    return out, (X0, Y0, X1, Y1)


def _find_text_region(img):
    """Most edge-dense field-sized box in the value zone (skip title / barcode)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    H, W = edges.shape
    bw, bh = int(W * 0.22), int(H * 0.075)
    if bw < 8 or bh < 6:
        return None
    best, best_s = None, -1
    for yy in range(int(H * 0.22), max(int(H * 0.22) + 1, int(H * 0.58)), max(1, bh // 2)):
        for xx in range(int(W * 0.42), max(int(W * 0.42) + 1, W - bw), max(1, bw // 3)):
            s = int(edges[yy:yy + bh, xx:xx + bw].sum())
            if s > best_s:
                best_s, best = s, (xx, yy, bw, bh)
    return best


def _digit_mask(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thr = gray.mean() - 0.5 * gray.std()
    d = cv2.dilate((gray < thr).astype(np.uint8), np.ones((2, 2), np.uint8))
    return cv2.GaussianBlur(d.astype(np.float32), (0, 0), 0.6)[..., None]


def _carve_grain(roi, rng, nrng):
    """Strong scratched/rewritten field texture: coarse horizontal wood-grain
    streaks + a dark rubbed smudge band + heavy grain, digits degraded (blurred,
    noisy, partly faded). Calibrated to the real Egypt/Mozambique examples."""
    h, w = roi.shape[:2]
    f = roi.astype(np.float32)
    _, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    streak = nrng.standard_normal((h, w)).astype(np.float32)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=w * 0.06, sigmaY=0.5)
    streak /= (np.abs(streak).max() + 1e-6)
    f += streak[..., None] * rng.uniform(26, 40)
    bc = h * rng.uniform(0.42, 0.68)
    band = np.exp(-(((yy - bc) / (h * 0.24)) ** 2))
    f -= (band * rng.uniform(16, 30))[..., None]
    f += nrng.standard_normal((h, w, 1)).astype(np.float32) * rng.uniform(8, 13)
    f = cv2.GaussianBlur(f, (0, 0), 0.6)
    f = f * rng.uniform(0.98, 1.02) + np.array([4., 2., 0.])
    d3 = _digit_mask(roi)
    digit_soft = cv2.GaussianBlur(roi, (0, 0), rng.uniform(1.0, 1.6)).astype(np.float32)
    digit_soft += nrng.standard_normal((h, w, 1)).astype(np.float32) * rng.uniform(4, 7)
    fade = rng.uniform(0.18, 0.38)
    digit_soft = digit_soft * (1 - fade) + float(f.mean()) * fade
    patch = np.clip(digit_soft * d3 + f * (1 - d3), 0, 255)
    for _ in range(rng.randint(2, 4)):
        p0 = (int(w * rng.uniform(0, 0.6)), int(h * rng.uniform(0.15, 0.85)))
        p1 = (p0[0] + int(w * rng.uniform(0.25, 0.55)),
              p0[1] + int(h * rng.uniform(-0.4, 0.4)))
        s = np.zeros((h, w), np.float32)
        cv2.line(s, p0, p1, 1.0, thickness=rng.randint(1, 2))
        s = cv2.GaussianBlur(s, (0, 0), 0.7)[..., None]
        patch = np.clip(patch + s * rng.uniform(-26, 16), 0, 255)
    return patch.astype(np.uint8)


def _field_carve(img, rng, nrng, box=None):
    reg = box if box is not None else _find_text_region(img)
    if reg is None:
        return None
    x, y, w, h = reg
    if w < 8 or h < 6:
        return None
    patch = _carve_grain(img[y:y + h, x:x + w], rng, nrng)
    edge = int(rng.uniform(170, 195))
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), (edge, edge, edge), 2)
    out = img.copy()
    out[y:y + h, x:x + w] = patch
    return out, (x, y, x + w, y + h)


def synth_tamper(img, donor, rng, nrng, text_prob=0.2, allow_color_on_gray=False):
    """Bona-fide (BGR) -> synthetic fraud (BGR). Returns (out_bgr, mode); mode is
    'none' if nothing could be applied (caller keeps the original label 0)."""
    if rng.random() < text_prob:
        res = _field_carve(img, rng, nrng)
        return (res[0], "field_carve") if res else (img, "none")
    if allow_color_on_gray and rng.random() < 0.25:
        res = _face_color_on_gray(img, donor, rng)
        if res:
            return res[0], "face_color_on_gray"
    res = _face_clean_paste(img, donor, rng)
    if res:
        return res[0], "face_clean_paste"
    res = _field_carve(img, rng, nrng)                 # no detectable face -> text
    return (res[0], "field_carve") if res else (img, "none")


def build_donor_pool(data_dir, seed: int = 42, per_type: int = 48,
                     exclude_ids: set[str] | None = None) -> dict[str, list[Path]]:
    """type -> list of bona-fide image paths, used as face-swap donors. Pass
    ``exclude_ids`` (the val ids) so no validation image ever leaks into training
    or the probe as a donor face."""
    from freuid.data import load_labels
    df = load_labels(data_dir, "train")
    df = df[df["label"] == 0]
    if exclude_ids:
        df = df[~df["id"].isin(exclude_ids)]
    rng = np.random.default_rng(seed)
    pool: dict[str, list[Path]] = {}
    for t, g in df.groupby("type"):
        rows = g.sample(n=min(per_type, len(g)), random_state=int(rng.integers(1 << 30)))
        pool[str(t)] = [Path(p) for p in rows["path"].tolist()]
    return pool


class _DonorMixin:
    """Lazily decode + cache donor BGR images from a per-type path pool."""

    donor_pool: dict[str, list[Path]]
    _donor_cache: dict

    def _donor_bgr(self, dtype, rng):
        paths = self.donor_pool.get(dtype)
        if not paths:  # unknown type -> any donor
            paths = [p for ps in self.donor_pool.values() for p in ps]
        if not paths:
            return None
        p = paths[rng.randrange(len(paths))]
        img = self._donor_cache.get(p)
        if img is None:
            img = cv2.imread(str(p))
            self._donor_cache[p] = img
        return img


class SynthTamperWrapper(Dataset, _DonorMixin):
    """Train wrapper over a transform=None base dataset. With probability ``prob``
    a BONA-FIDE (label 0) sample is turned into a synthetic fraud (label 1) by
    ``synth_tamper`` before the transform; real frauds and untampered bona-fide
    pass through unchanged. Tamper is fresh-random each epoch (train variety)."""

    def __init__(self, base, transform, donor_pool, prob=0.3, text_prob=0.2,
                 recapture_prob=0.0, seed=0):
        self.base = base
        self.tf = transform
        self.donor_pool = donor_pool
        self.prob = float(prob)
        self.text_prob = float(text_prob)
        self.recapture_prob = float(recapture_prob)
        self.seed = int(seed)
        self._donor_cache = {}

    def __len__(self):
        return len(self.base.samples)

    def __getitem__(self, idx):
        s = self.base.samples[idx]
        src = s.card_path if s.card_path is not None else s.path
        img = Image.open(src).convert("RGB")
        label = s.label
        rng = random.Random()                          # fresh entropy -> varies per epoch
        # (a) digital tell: bona-fide -> synthetic fraud
        if label == 0 and rng.random() < self.prob:
            donor = self._donor_bgr(s.type, rng)
            if donor is not None:
                bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                out, mode = synth_tamper(
                    bgr, donor, rng, np.random.default_rng(), self.text_prob,
                    allow_color_on_gray=(s.type == "BENIN/DL"))
                if mode != "none":
                    img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                    label = 1
        # (b) analog robustness: recapture ANY image (label preserved). Cap the
        # working size first -- recapture at native ~1.4k px is ~1s/img (meshgrid
        # + warp + JPEG), and the transform downsizes to the model res anyway, so
        # this keeps it ~7x cheaper without changing the look.
        if self.recapture_prob > 0 and rng.random() < self.recapture_prob:
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            m = max(bgr.shape[:2])
            if m > 640:
                sc = 640 / m
                bgr = cv2.resize(bgr, (int(bgr.shape[1] * sc), int(bgr.shape[0] * sc)),
                                 interpolation=cv2.INTER_AREA)
            bgr = _recapture_degrade(bgr, rng, np.random.default_rng())
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self.tf(img), label


class SynthProbeDataset(Dataset, _DonorMixin):
    """Deterministic val probe. Each val bona-fide contributes a CLEAN copy
    (label 0) AND a TAMPERED twin (label 1) -- genuine negatives are kept in full
    and matched by synthetic positives (50/50). The tamper is seeded per sample
    so the probe is byte-identical every epoch: a stable checkpoint compass that,
    unlike the in-domain clean val, does not saturate."""

    def __init__(self, base_bona, transform, donor_pool, text_prob=0.2, seed=0):
        self.base = base_bona                          # transform=None, val bona-fide only
        self.tf = transform
        self.donor_pool = donor_pool
        self.text_prob = float(text_prob)
        self.seed = int(seed)
        self.n = len(base_bona.samples)
        self._donor_cache = {}

    def __len__(self):
        return 2 * self.n

    def __getitem__(self, idx):
        clean = idx < self.n
        i = idx if clean else idx - self.n
        s = self.base.samples[i]
        src = s.card_path if s.card_path is not None else s.path
        img = Image.open(src).convert("RGB")
        if clean:
            return self.tf(img), 0
        rng = random.Random(self.seed + i)             # deterministic per sample
        nrng = np.random.default_rng(self.seed + i)
        donor = self._donor_bgr(s.type, rng)
        if donor is None:
            return self.tf(img), 0
        bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        out, mode = synth_tamper(bgr, donor, rng, nrng, self.text_prob,
                                 allow_color_on_gray=(s.type == "BENIN/DL"))
        if mode == "none":
            return self.tf(img), 0
        img = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        return self.tf(img), 1


# ---------------------------------------------------------------------------
# Recapture (digital -> analog) degradation, v2 -- CALIBRATED to the real
# is_digital=False images (scripts/recapture_calib.py). Matched analog/digital
# stat ratios: sharpness ~0.16x (dominant ~6x softening), illumination unevenness
# ~1.33x (smooth lighting gradient + vignette, NOT harsh glare), saturation
# ~0.90x, brightness ~0.93x, contrast ~0.93x; + mild colour cast (warm OR cool),
# sensor noise, small perspective/rotation, light JPEG. No moire/halftone -- the
# real captures are photos of PRINTS, not screens.

def _recapture_degrade(img, rng, nrng):
    """digital BGR uint8 -> analog-like BGR uint8. Scalars from python ``rng``,
    array noise from ``nrng`` -- a fixed seed gives a deterministic val probe."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    # 1. defocus blur (ALWAYS) -- softening; kept small because the warp + JPEG
    #    below also soften, and together they must land at ~6x (real) not more.
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.1, 0.35) * (w / 512.0))
    # 2. downscale-upscale (resolution loss, compounds softening)
    s = rng.uniform(0.82, 0.95)
    f = cv2.resize(cv2.resize(f, (max(8, int(w * s)), max(8, int(h * s))),
                              interpolation=cv2.INTER_AREA),
                   (w, h), interpolation=cv2.INTER_CUBIC)
    # 3. smooth illumination gradient + radial vignette (soft, not glare)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 6.283)
    grad = (xx / w - 0.5) * np.cos(ang) + (yy / h - 0.5) * np.sin(ang)
    r = np.sqrt((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
    illum = (1 + rng.uniform(0.38, 0.60) * grad) * \
            (1 - rng.uniform(0.14, 0.28) * (r / (r.max() + 1e-6)) ** 2)
    f = f * illum[..., None]
    # 4. colour cast (warm OR cool, both directions)
    f = f * np.array([rng.uniform(0.90, 1.09), rng.uniform(0.95, 1.05),
                      rng.uniform(0.90, 1.09)], np.float32)
    # 5. mildly darker + lower contrast
    f = (f - 128) * rng.uniform(0.92, 1.0) + 128
    f = f * rng.uniform(0.93, 1.0)
    # 6. desaturate ~10%
    f = np.clip(f, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.80, 0.94)
    f = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    # 7. mild sensor noise
    f = np.clip(f + nrng.standard_normal((h, w, 1)).astype(np.float32) * rng.uniform(2, 5),
                0, 255).astype(np.uint8)
    # 8. small perspective + rotation (photographed at a slight angle)
    if rng.random() < 0.7:
        m = rng.uniform(0.008, 0.03)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[rng.uniform(-m, m) * w, rng.uniform(-m, m) * h]
                                for _ in range(4)])
        f = cv2.warpPerspective(f, cv2.getPerspectiveTransform(src, dst), (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.6:
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3, 3), 1.0)
        f = cv2.warpAffine(f, rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # 9. light JPEG (capture / upload)
    ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(80, 95))])
    if ok:
        f = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return f


class _RecaptureV2Transform:
    """PIL RGB -> CHW normalized tensor via the calibrated recapture degradation.
    seed=None -> fresh randomness each call (train variety); an int seed ->
    deterministic (val probe). Same output contract as the torchvision transforms."""

    def __init__(self, image_size, mean, std, seed=None):
        self.size = int(image_size)
        self.mean = np.array(mean, np.float32)
        self.std = np.array(std, np.float32)
        self.rng = random.Random(seed)
        self.nrng = np.random.default_rng(seed if seed is not None else None)

    def __call__(self, img):
        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (self.size, self.size))
        bgr = _recapture_degrade(bgr, self.rng, self.nrng)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        return torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous().float()


def recapture_v2_transforms(image_size, mean, std, seed=None):
    """Factory mirroring recapture_transforms' signature, for the calibrated
    recapture. Selected by build_loaders when extra.recapture_version == 'v2'."""
    return _RecaptureV2Transform(image_size, mean, std, seed)
