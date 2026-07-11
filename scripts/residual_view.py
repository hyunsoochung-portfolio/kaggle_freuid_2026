"""Noise-residual (high-pass) view to expose texture/noise inconsistency that
the eye misses: a spliced/blurred/rewritten region carries a different noise
floor than its surroundings. For each fraud we show [original | residual].
A pasted face shows as a block of different noise; if TEXT was tampered, that
field would light up differently too.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA = Path("data")
IMG = DATA / "train" / "train"
OUT = Path("/tmp/survey")

df = pd.read_csv(DATA / "train_labels.csv")
fr = df[df.label == 1]
t = sys.argv[1] if len(sys.argv) > 1 else "EGYPT/DL"
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4

ids = fr[fr.type == t].id.tolist()[start:start + n]
W = 640
rows = []
for id_ in ids:
    img = cv2.imread(str(IMG / f"{id_}.jpeg"))
    h = int(img.shape[0] * W / img.shape[1])
    img = cv2.resize(img, (W, h))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)      # high-pass residual
    lv = cv2.GaussianBlur(hp * hp, (0, 0), 8.0)          # local noise energy
    lv = np.sqrt(lv)
    lv = np.clip(lv / (lv.mean() * 2.2) * 255, 0, 255).astype(np.uint8)
    res = cv2.applyColorMap(lv, cv2.COLORMAP_JET)
    rows.append(np.hstack([img, res]))

montage = np.vstack(rows)
p = OUT / f"resid_{t.replace('/', '-')}_{start}.png"
cv2.imwrite(str(p), montage)
print("wrote", p, montage.shape)
