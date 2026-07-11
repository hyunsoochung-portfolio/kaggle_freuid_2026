"""Reproduce the TWO real carve examples precisely, each with its own texture:
  carve_ripple  -> Mozambique Carta No '60682163/5' (wavy ripple, sharp digits)
  carve_grain   -> Egypt DOB '17/10/1974' (coarse horizontal wood-grain streaks)
Apply each to the matching GENUINE field and show beside the real tamper.
"""
import random

import cv2
import numpy as np

rng = random.Random(5)


def _digit_mask(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thr = gray.mean() - 0.5 * gray.std()
    d = cv2.dilate((gray < thr).astype(np.uint8), np.ones((2, 2), np.uint8))
    return cv2.GaussianBlur(d.astype(np.float32), (0, 0), 0.6)[..., None]


def carve_ripple(img, rng, box):
    """Moz look: sharp digits on a rectangular patch whose background is a wavy
    horizontal RIPPLE (stronger in the lower half), bluish, clear box."""
    x, y, w, h = box
    roi = img[y:y + h, x:x + w]
    f = roi.astype(np.float32)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # organic ripple = sum of a few horizontal sines, concentrated below baseline
    ripple = np.zeros((h, w), np.float32)
    for _ in range(3):
        period = w / rng.uniform(3.5, 8.0)
        ripple += np.sin(2 * np.pi * xx / period + rng.uniform(0, 6.28)) * rng.uniform(1.2, 2.2)
    wgt = np.clip((yy / h - 0.20) / 0.80, 0, 1)          # stronger lower down
    mapy = yy + ripple * wgt
    f = cv2.remap(f, xx.copy(), mapy.astype(np.float32), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REFLECT)
    k = max(3, int(w * 0.022)) | 1                       # horizontal smear
    f = cv2.filter2D(f, -1, np.full((1, k), 1.0 / k, np.float32))
    # distinct wavy dark smudge band just below the digit baseline
    bc = h * rng.uniform(0.55, 0.72)
    band = np.exp(-(((yy - bc) / (h * 0.13)) ** 2))
    wave = 0.5 + 0.5 * np.sin(2 * np.pi * xx / (w / rng.uniform(4, 7)) + rng.uniform(0, 6.28))
    f -= (band * (0.4 + 0.6 * wave) * rng.uniform(12, 24))[..., None]
    f += np.random.randn(h, w, 1).astype(np.float32) * rng.uniform(1.2, 2.4)
    f = f * rng.uniform(1.01, 1.04) + np.array([9., 4., 0.])   # bluish lift (BGR)
    d3 = _digit_mask(roi)
    digit_soft = cv2.GaussianBlur(roi, (0, 0), 0.3).astype(np.float32)
    patch = np.clip(digit_soft * d3 + f * (1 - d3), 0, 255).astype(np.uint8)
    edge = int(rng.uniform(196, 210))
    cv2.rectangle(patch, (1, 1), (w - 2, h - 2), (edge, edge, edge), 1)
    out = img.copy()
    out[y:y + h, x:x + w] = patch
    return out, (x, y, x + w, y + h)


def carve_grain(img, rng, box):
    """Egypt look: slightly rough digits on a coarse horizontal WOOD-GRAIN streak
    texture + grain, subtle box."""
    x, y, w, h = box
    roi = img[y:y + h, x:x + w]
    f = roi.astype(np.float32)
    # horizontal wood-grain: noise blurred long horizontally, thin vertically
    streak = np.random.randn(h, w).astype(np.float32)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=w * 0.05, sigmaY=0.5)
    streak /= (np.abs(streak).max() + 1e-6)
    f += streak[..., None] * rng.uniform(14, 20)
    f += np.random.randn(h, w, 1).astype(np.float32) * rng.uniform(4.0, 6.5)   # grain
    f = cv2.GaussianBlur(f, (0, 0), 0.5)
    f = f * rng.uniform(1.0, 1.03) + np.array([5., 3., 0.])
    d3 = _digit_mask(roi)
    digit_soft = cv2.GaussianBlur(roi, (0, 0), 0.75).astype(np.float32)
    digit_soft += np.random.randn(h, w, 1).astype(np.float32) * 2.0            # rough edges
    patch = np.clip(digit_soft * d3 + f * (1 - d3), 0, 255).astype(np.uint8)
    edge = int(rng.uniform(206, 222))
    cv2.rectangle(patch, (1, 1), (w - 2, h - 2), (edge, edge, edge), 1)
    out = img.copy()
    out[y:y + h, x:x + w] = patch
    return out, (x, y, x + w, y + h)


def crop_region(img, box, s=9):
    x, y, w, h = box
    return cv2.resize(img[y:y + h, x:x + w], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)


def crop_frac(path, fr, s=9):
    img = cv2.imread(path); H, W = img.shape[:2]
    c = img[int(H*fr[1]):int(H*fr[3]), int(W*fr[0]):int(W*fr[2])]
    return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)


def to_box(img, fr):
    H, W = img.shape[:2]
    return (int(W*fr[0]), int(H*fr[1]), int(W*(fr[2]-fr[0])), int(H*(fr[3]-fr[1])))


def label(im, txt, color):
    bar = np.full((30, im.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return np.vstack([bar, im])


def compare(real, mine, name):
    h = max(real.shape[0], mine.shape[0])
    def pad(a): return cv2.copyMakeBorder(a, 0, h-a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,)*3)
    r = label(pad(real), "REAL", (0, 0, 200))
    m = label(pad(mine), "MINE (synthetic)", (0, 130, 0))
    gap = np.full((r.shape[0], 30, 3), 255, np.uint8)
    combo = np.hstack([r, gap, m])
    if combo.shape[1] > 1600:
        sc = 1600 / combo.shape[1]
        combo = cv2.resize(combo, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"/tmp/{name}.png", combo)
    print("wrote", name)


# --- Moz ripple ---
mfr = (0.455, 0.445, 0.63, 0.505)
real_moz = crop_frac("data/fraud_only/0aeb445bb4d14381a15498962a63eaa5.jpeg", mfr)
gmoz = cv2.imread("data/train/train/000a27cd549647359deaa78721858877.jpeg")
mmoz, _ = carve_ripple(gmoz, rng, to_box(gmoz, mfr))
compare(real_moz, crop_region(mmoz, to_box(gmoz, mfr)), "cal_moz")

# --- Egypt grain ---
efr = (0.30, 0.505, 0.52, 0.575)
real_egy = crop_frac("data/fraud_only/0a6cf246b287428cbb3ac669f254bb98.jpeg", efr)
geg = cv2.imread("data/train/train/000514f0340642f3b2eb83bff862458a.jpeg")
megy, _ = carve_grain(geg, rng, to_box(geg, efr))
compare(real_egy, crop_region(megy, to_box(geg, efr)), "cal_egy")
