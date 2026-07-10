"""Reproduce the real 'field-erase' tamper mark (flattened rectangular patch
over a value field that kills the background micro-texture, as seen in
0abdddbb.../21-10-1968) and compare against a bona-fide.

Panel: [ REAL tampered DOB ] | [ genuine DOB ] | [ variant A ] | [ B ] | [ C ]
so we can pick the synthesis that matches the real artifact.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA = Path("data")
IMG = DATA / "train" / "train"

# --- real tampered DOB crop for reference ---
real = cv2.imread(str(DATA / "fraud_only" /
                     "0abdddbb7f004d1c8eb6f3369513ae96.jpeg"))
Hr, Wr = real.shape[:2]
real_dob = real[int(Hr * 0.235):int(Hr * 0.30), int(Wr * 0.40):int(Wr * 0.66)]

# --- a bona-fide Mozambique DOB ---
df = pd.read_csv(DATA / "train_labels.csv")
gid = df[(df.type == "MOZAMBIQUE/DL") & (df.label == 0)].id.tolist()[0]
gen = cv2.imread(str(IMG / f"{gid}.jpeg"))
Hg, Wg = gen.shape[:2]
# DOB field box (x0,y0,x1,y1) in px
bx = (int(Wg * 0.415), int(Hg * 0.24), int(Wg * 0.66), int(Hg * 0.30))
gen_dob = gen[bx[1]:bx[3], bx[0]:bx[2]].copy()


def field_erase(roi, mode):
    """Paint-over a rectangular field patch: kill the background micro-texture
    (guilloche / fine lines) while KEEPING the dark digits sharp, + a subtle
    box edge and tone shift. This is the real tamper signature."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # digit mask = dark strokes (kept sharp); dilate to catch anti-aliasing
    thr = gray.mean() - 0.6 * gray.std()
    digit = (gray < thr).astype(np.uint8)
    digit = cv2.dilate(digit, np.ones((2, 2), np.uint8))
    digit3 = cv2.GaussianBlur(digit.astype(np.float32), (0, 0), 0.7)[..., None]
    # flat paint = strongly smoothed background (texture removed)
    flat = cv2.GaussianBlur(roi, (0, 0), 3.0).astype(np.float32)
    if mode == "D":
        tone, edge = 1.02, 228          # subtle
    else:                                # E: more visible patch
        tone, edge = 1.05, 210
    flat = np.clip(flat * tone + 4, 0, 255)
    out = (roi.astype(np.float32) * digit3 + flat * (1 - digit3))
    out = np.clip(out, 0, 255).astype(np.uint8)
    # subtle rectangular box boundary (top + bottom lines)
    cv2.line(out, (0, 1), (out.shape[1], 1), (edge, edge, edge), 1)
    cv2.line(out, (0, out.shape[0] - 2), (out.shape[1], out.shape[0] - 2),
             (edge, edge, edge), 1)
    return out


def stamp_h(imgs, h=110):
    outs = []
    for im in imgs:
        w = int(im.shape[1] * h / im.shape[0])
        outs.append(cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC))
    W = max(o.shape[1] for o in outs)
    outs = [cv2.copyMakeBorder(o, 0, 0, 0, W - o.shape[1],
                               cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for o in outs]
    return outs


labels = ["REAL tamper", "genuine", "erase-D (subtle)", "erase-E (visible)"]
panels = [real_dob, gen_dob,
          field_erase(gen_dob, "D"),
          field_erase(gen_dob, "E")]
panels = stamp_h(panels, 130)
tagged = []
for lab, im in zip(labels, panels):
    bar = np.full((30, im.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, lab, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
    tagged.append(np.vstack([bar, im]))
out = np.vstack(tagged)
out = cv2.resize(out, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
cv2.imwrite("/tmp/erase_compare.png", out)
print("wrote /tmp/erase_compare.png", out.shape)
