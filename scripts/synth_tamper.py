"""Synthetic-fraud augmentation for FREUID, grounded in a 100-image analysis of
the real fraud data. We manufacture a fake fraud (label 1) from a bona-fide
(label 0) by reproducing an *observed* tell.

Observed fraud taxonomy (share of the ~100 real frauds examined):
  FACE (≈ every fraud has a pasted portrait; the photo is the primary tell)
    - clean_paste     : sharp portrait, hard rectangular seam, studio/clean bg,
                        the card's overlay/guilloche does NOT cross it  (F1+F3)
    - color_on_gray   : a COLOR face grafted onto a GRAYSCALE body, hard tone
                        seam at the jaw/neck (seen in BENIN)             (F2)
    - (main-only swap on templates with a ghost portrait leaves the ghost
       showing the original person -> main != ghost; this happens for free
       because we only touch the main photo box)                        (F5)
  TEXT (rare in reality, ~8%; we dial it to ~20% of tampered for signal)
    - field_carve     : flat patch over a value field, background micro-texture
                        killed, digits stay sharp  (MOZAMBIQUE DOB)       (T1)
    - field_smear     : a field value locally blurred / wiped             (T2)
    - digit_overlap   : number digits doubled/ghosted from over-typing
                        (BENIN permis number)                             (T3)

Public API:
    synth_tamper(img_bgr, donor_bgr, rng, text_prob=0.20) -> (out_bgr, mode, box)
    plus each primitive, callable directly (used by the preview).
Every primitive returns (out_bgr, box) or None if it cannot apply (e.g. no face).
"""
import cv2
import numpy as np

_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ------------------------------------------------------------------ geometry --
def detect_face(img):
    """Largest frontal face as (x, y, w, h), or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = _CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(f) == 0:
        return None
    return sorted(f, key=lambda b: b[2] * b[3], reverse=True)[0]


def photo_box(face, W, H):
    """Face -> full portrait box (head + shoulders)."""
    x, y, w, h = face
    return (max(0, int(x - 0.5 * w)), max(0, int(y - 0.7 * h)),
            min(W, int(x + w + 0.5 * w)), min(H, int(y + h + 1.15 * h)))


def face_box(face, W, H, fx=0.18, fy=0.22):
    """Face -> tight face box (just the face, mild margin)."""
    x, y, w, h = face
    ex, ey = int(w * fx), int(h * fy)
    return (max(0, x - ex), max(0, y - ey),
            min(W, x + w + ex), min(H, y + h + ey))


# ------------------------------------------------------------------- helpers --
def _to_gray3(bgr):
    return cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def _clarify(photo, rng):
    """Make a photo look like a clean standalone shot that the overlay never
    touched: boost contrast + saturation + sharpness so it STANDS OUT."""
    f = photo.astype(np.float32)
    m = f.mean()
    f = np.clip((f - m) * rng.uniform(1.15, 1.30) + m + rng.uniform(-3, 8), 0, 255)
    hsv = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1.15, 1.40), 0, 255)
    f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), 1.2)
    return np.clip(f + rng.uniform(0.6, 1.0) * (f - blur), 0, 255).astype(np.uint8)


# -------------------------------------------------------------- FACE tampers --
def face_clean_paste(target, donor, rng):
    """Paste the donor's portrait into the photo box as a CLEAN, HARD-EDGED
    rectangle that stands out (no tone-match, no feather). On templates with a
    ghost portrait this leaves the ghost = original person (main != ghost)."""
    ft, fd = detect_face(target), detect_face(donor)
    if ft is None or fd is None:
        return None
    Ht, Wt = target.shape[:2]
    Hd, Wd = donor.shape[:2]
    X0, Y0, X1, Y1 = photo_box(ft, Wt, Ht)
    dx0, dy0, dx1, dy1 = photo_box(fd, Wd, Hd)
    paste = cv2.resize(donor[dy0:dy1, dx0:dx1], (X1 - X0, Y1 - Y0))
    out = target.copy()
    out[Y0:Y1, X0:X1] = _clarify(paste, rng)          # hard rectangular seam
    return out, (X0, Y0, X1, Y1)


def face_color_on_gray(target, donor, rng):
    """Graft a COLOR donor face onto a GRAYSCALE body: desaturate the photo box,
    then paste the clarified colour face -> hard tone seam at the jaw (BENIN)."""
    ft, fd = detect_face(target), detect_face(donor)
    if ft is None or fd is None:
        return None
    Ht, Wt = target.shape[:2]
    Hd, Wd = donor.shape[:2]
    X0, Y0, X1, Y1 = photo_box(ft, Wt, Ht)
    body = _to_gray3(target[Y0:Y1, X0:X1])            # grayscale body
    tx0, ty0, tx1, ty1 = face_box(ft, Wt, Ht)
    tx0, ty0, tx1, ty1 = tx0 - X0, ty0 - Y0, tx1 - X0, ty1 - Y0
    tw, th = tx1 - tx0, ty1 - ty0
    dx0, dy0, dx1, dy1 = face_box(fd, Wd, Hd)
    cface = _clarify(cv2.resize(donor[dy0:dy1, dx0:dx1], (tw, th)), rng)
    mask = np.zeros((th, tw), np.float32)
    cv2.ellipse(mask, (tw // 2, th // 2),
                (int(tw * 0.46), int(th * 0.5)), 0, 0, 360, 1, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, tw * 0.03))[..., None]
    body[ty0:ty1, tx0:tx1] = (cface * mask +
                              body[ty0:ty1, tx0:tx1] * (1 - mask)).astype(np.uint8)
    out = target.copy()
    out[Y0:Y1, X0:X1] = body
    return out, (X0, Y0, X1, Y1)


# -------------------------------------------------------------- TEXT tampers --
def find_text_region(img):
    """Most edge-dense field-sized box in the value zone (skip title / labels /
    bottom barcode) -> lands on a real value (date / number)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    H, W = edges.shape
    bw, bh = int(W * 0.22), int(H * 0.075)
    best, best_s = None, -1
    for yy in range(int(H * 0.22), int(H * 0.58), max(1, bh // 2)):
        for xx in range(int(W * 0.42), W - bw, max(1, bw // 3)):
            s = int(edges[yy:yy + bh, xx:xx + bw].sum())
            if s > best_s:
                best_s, best = s, (xx, yy, bw, bh)
    return best


def _digit_mask(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thr = gray.mean() - 0.5 * gray.std()
    d = cv2.dilate((gray < thr).astype(np.uint8), np.ones((2, 2), np.uint8))
    return cv2.GaussianBlur(d.astype(np.float32), (0, 0), 0.6)[..., None]


def _carve_grain(roi, rng):
    """Strong 'scratched / rubbed / rewritten' field texture (grounded in the
    Egypt '17/10/1974' look, pushed harder for clear visibility): coarse
    horizontal wood-grain streaks + a dark rubbed smudge band + heavy grain, and
    the digits are degraded (blurred, noisy, partly faded toward the bg)."""
    h, w = roi.shape[:2]
    f = roi.astype(np.float32)
    _, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # strong horizontal wood-grain streaks
    streak = np.random.randn(h, w).astype(np.float32)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=w * 0.06, sigmaY=0.5)
    streak /= (np.abs(streak).max() + 1e-6)
    f += streak[..., None] * rng.uniform(26, 40)
    # dark rubbed smudge band across the value
    bc = h * rng.uniform(0.42, 0.68)
    band = np.exp(-(((yy - bc) / (h * 0.24)) ** 2))
    f -= (band * rng.uniform(16, 30))[..., None]
    f += np.random.randn(h, w, 1).astype(np.float32) * rng.uniform(8, 13)   # heavy grain
    f = cv2.GaussianBlur(f, (0, 0), 0.6)
    f = f * rng.uniform(0.98, 1.02) + np.array([4., 2., 0.])
    # degrade digits: blur + noise + partial fade toward background
    d3 = _digit_mask(roi)
    digit_soft = cv2.GaussianBlur(roi, (0, 0), rng.uniform(1.0, 1.6)).astype(np.float32)
    digit_soft += np.random.randn(h, w, 1).astype(np.float32) * rng.uniform(4, 7)
    fade = rng.uniform(0.18, 0.38)
    digit_soft = digit_soft * (1 - fade) + float(f.mean()) * fade
    patch = np.clip(digit_soft * d3 + f * (1 - d3), 0, 255)
    # a few dark scratch strokes raked across the value
    for _ in range(rng.randint(2, 4)):
        p0 = (int(w * rng.uniform(0, 0.6)), int(h * rng.uniform(0.15, 0.85)))
        p1 = (p0[0] + int(w * rng.uniform(0.25, 0.55)),
              p0[1] + int(h * rng.uniform(-0.4, 0.4)))
        s = np.zeros((h, w), np.float32)
        cv2.line(s, p0, p1, 1.0, thickness=rng.randint(1, 2))
        s = cv2.GaussianBlur(s, (0, 0), 0.7)[..., None]
        patch = np.clip(patch + s * rng.uniform(-26, 16), 0, 255)
    return patch.astype(np.uint8)


def field_carve(img, rng, box=None):
    """Scratched/smudged/rewritten field tell: the value sits in a rectangular
    patch whose background micro-texture is destroyed by a strong rough grain +
    dark smudge, the digits degraded, with a visible box boundary."""
    x, y, w, h = box if box is not None else find_text_region(img)
    roi = img[y:y + h, x:x + w]
    patch = _carve_grain(roi, rng)
    edge = int(rng.uniform(170, 195))                 # darker, clearer box
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), (edge, edge, edge), 2)
    out = img.copy()
    out[y:y + h, x:x + w] = patch
    return out, (x, y, x + w, y + h)


# ----------------------------------------------------------------- dispatcher --
FACE_MODES = [("face_clean_paste", face_clean_paste, 0.75),
              ("face_color_on_gray", face_color_on_gray, 0.25)]
TEXT_MODES = [("field_carve", field_carve, 1.0)]


def _pick(rng, modes):
    r, acc = rng.random(), 0.0
    for name, fn, wt in modes:
        acc += wt
        if r <= acc:
            return name, fn
    return modes[-1][0], modes[-1][1]


def synth_tamper(img, donor, rng, text_prob=0.20):
    """Turn a bona-fide into a synthetic fraud. Returns (out_bgr, box, mode).
    ~text_prob of tampered samples get a TEXT tell, the rest a FACE tell."""
    if rng.random() < text_prob:
        name, fn = _pick(rng, TEXT_MODES)
        out, box = fn(img, rng)
        return out, box, name
    name, fn = _pick(rng, FACE_MODES)
    res = fn(img, donor, rng)
    if res is None:                                    # no face -> fall back to text
        name, fn = _pick(rng, TEXT_MODES)
        out, box = fn(img, rng)
        return out, box, name
    out, box = res
    return out, box, name


if __name__ == "__main__":
    print("FACE_MODES:", [m[0] for m in FACE_MODES])
    print("TEXT_MODES:", [m[0] for m in TEXT_MODES])
