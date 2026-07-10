"""Preview each synthetic-fraud mode against the REAL ground-truth fraud that
grounds it. One panel per mode in data/aug_preview/:

  [ GENUINE (label 0) ] | [ MY SYNTHETIC (label 1) ] | [ REAL FRAUD (label 1) ]

For the text modes a zoom row compares my synthetic field to the real tampered
field (same field). This validates that the augmentation reproduces reality.
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_tamper as st  # noqa: E402

DATA = Path("data")
IMG = DATA / "train" / "train"
OUT = DATA / "aug_preview"
OUT.mkdir(exist_ok=True)
for old in OUT.glob("*.png"):
    old.unlink()

df = pd.read_csv(DATA / "train_labels.csv")
bona = df[df.label == 0]
rng = random.Random(7)
COLW = 440


def bona_of(dtype, need_face=True, skip=0):
    ids = bona[bona.type == dtype].id.tolist()
    found = 0
    for id_ in ids:
        img = cv2.imread(str(IMG / f"{id_}.jpeg"))
        if img is None:
            continue
        if not need_face or st.detect_face(img) is not None:
            if found >= skip:
                return img
            found += 1
    return None


def rw(img, w=COLW):
    return cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))


def cap(img, text, color):
    bar = np.full((40, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return np.vstack([bar, img])


def hcat(imgs, gap=18):
    h = max(i.shape[0] for i in imgs)
    out = []
    for i in imgs:
        i = cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,) * 3)
        out.append(i)
        out.append(np.full((h, gap, 3), 255, np.uint8))
    return np.hstack(out[:-1])


def frac_box(img, fr):
    H, W = img.shape[:2]
    return (int(W * fr[0]), int(H * fr[1]), int(W * fr[2]), int(H * fr[3]))


def zoom_field(img, fr, w=620):
    x0, y0, x1, y1 = frac_box(img, fr)
    pad = int((y1 - y0) * 0.35)
    c = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    return cv2.resize(c, (w, int(c.shape[0] * w / c.shape[1])), interpolation=cv2.INTER_NEAREST)


# (title, dtype, fn, needs_donor, kind, fname, my_box_frac, real_id, real_box_frac)
DEMOS = [
    ("clean-paste (F1)", "GUINEA/DL", st.face_clean_paste, True, "face",
     "1_face_clean_paste", None, "005c4f7742584f669b62548f9180b9f0", None),
    ("colour-on-gray (F2)", "BENIN/DL", st.face_color_on_gray, True, "face",
     "2_face_color_on_gray", None, "000ee902604c4a6087c086935a7c1c20", None),
    ("swap keeps ghost (F5)", "MAURITIUS/ID", st.face_clean_paste, True, "face",
     "3_face_swap_keeps_ghost", None, "00a159cb3c564e9c8070bd510ec0c7db", None),
    ("carve (scratched/rewritten field)", "EGYPT/DL", st.field_carve, False, "text",
     "4_text_carve", (0.30, 0.505, 0.52, 0.575),
     "0a6cf246b287428cbb3ac669f254bb98", (0.30, 0.505, 0.52, 0.575)),
]

print(f"writing real-vs-synthetic previews to {OUT}/")
for title, dtype, fn, needs_donor, kind, fname, mbf, real_id, rbf in DEMOS:
    tgt = bona_of(dtype, need_face=needs_donor, skip=1)
    real = cv2.imread(str(IMG / f"{real_id}.jpeg"))
    if tgt is None or real is None:
        print("  [skip]", fname)
        continue
    if needs_donor:
        donor = bona_of(dtype, need_face=True, skip=0)
        res = fn(tgt, donor, rng)
    else:
        x0, y0, x1, y1 = frac_box(tgt, mbf)
        res = fn(tgt, rng, (x0, y0, x1 - x0, y1 - y0))
    if res is None:
        print("  [skip-noapply]", fname)
        continue
    mine, mbox = res
    mine_boxed = mine.copy()
    cv2.rectangle(mine_boxed, (mbox[0], mbox[1]), (mbox[2], mbox[3]), (0, 0, 255), 3)

    c1 = cap(rw(tgt), "GENUINE (label 0)", (0, 130, 0))
    c2 = cap(rw(mine_boxed), "MY SYNTHETIC (label 1)  " + title, (0, 0, 210))
    c3 = cap(rw(real), "REAL FRAUD (label 1, truth)", (150, 0, 120))
    panel = hcat([c1, c2, c3])

    if kind == "text":
        zc = frac_box(mine, mbf)
        zmine = zoom_field(mine, mbf)
        zreal = zoom_field(real, rbf)
        zm = cap(zmine, "MY SYNTHETIC field (zoom)", (0, 0, 210))
        zr = cap(zreal, "REAL tampered field (zoom)", (150, 0, 120))
        zoomrow = hcat([zm, zr])
        if zoomrow.shape[1] < panel.shape[1]:
            zoomrow = cv2.copyMakeBorder(zoomrow, 0, 0, 0, panel.shape[1] - zoomrow.shape[1],
                                         cv2.BORDER_CONSTANT, value=(255,) * 3)
        else:
            panel = cv2.copyMakeBorder(panel, 0, 0, 0, zoomrow.shape[1] - panel.shape[1],
                                       cv2.BORDER_CONSTANT, value=(255,) * 3)
        sep = np.full((16, panel.shape[1], 3), 255, np.uint8)
        panel = np.vstack([panel, sep, zoomrow])

    cv2.imwrite(str(OUT / f"{fname}_{dtype.replace('/', '-')}.png"), panel)
    print("  ", fname)
print("done")
