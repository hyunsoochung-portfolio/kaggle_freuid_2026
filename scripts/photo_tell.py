"""Zoom the PHOTO region of genuine vs fraud docs, side by side, to identify
the real face-swap 'tell' (broken security overlay / seam / texture) that a
good augmentation must reproduce. Rows = samples; left=genuine, right=fraud."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA = Path("data")
IMG = DATA / "train" / "train"
OUT = Path("/tmp/survey")
OUT.mkdir(exist_ok=True)

# photo region (x0,y0,x1,y1) fractions per type
PBOX = {
    "MOZAMBIQUE/DL": (0.02, 0.17, 0.28, 0.78),
    "EGYPT/DL": (0.02, 0.28, 0.27, 0.86),
    "GUINEA/DL": (0.04, 0.30, 0.27, 0.82),
    "BENIN/DL": (0.02, 0.22, 0.24, 0.86),
    "MAURITIUS/ID": (0.03, 0.18, 0.34, 0.86),
}
df = pd.read_csv(DATA / "train_labels.csv")
t = sys.argv[1] if len(sys.argv) > 1 else "MOZAMBIQUE/DL"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
x0f, y0f, x1f, y1f = PBOX[t]

gen_ids = df[(df.type == t) & (df.label == 0)].id.tolist()[:n]
frd_ids = df[(df.type == t) & (df.label == 1)].id.tolist()[:n]


def photo(id_, W=380):
    img = cv2.imread(str(IMG / f"{id_}.jpeg"))
    H, Wi = img.shape[:2]
    c = img[int(H * y0f):int(H * y1f), int(Wi * x0f):int(Wi * x1f)]
    h = int(c.shape[0] * W / c.shape[1])
    return cv2.resize(c, (W, h), interpolation=cv2.INTER_CUBIC)


rows = []
for g, f in zip(gen_ids, frd_ids):
    pg, pf = photo(g), photo(f)
    h = max(pg.shape[0], pf.shape[0])
    pg = cv2.copyMakeBorder(pg, 0, h - pg.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,)*3)
    pf = cv2.copyMakeBorder(pf, 0, h - pf.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,)*3)
    gap = np.full((h, 30, 3), 255, np.uint8)
    rows.append(np.hstack([pg, gap, pf]))
    rows.append(np.full((8, rows[-1].shape[1], 3), 255, np.uint8))
montage = np.vstack(rows)
# header
hdr = np.full((40, montage.shape[1], 3), 255, np.uint8)
cv2.putText(hdr, "GENUINE (label 0)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 130, 0), 2)
cv2.putText(hdr, "FRAUD (label 1)", (montage.shape[1] // 2 + 20, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
p = OUT / f"phototell_{t.replace('/', '-')}.png"
cv2.imwrite(str(p), np.vstack([hdr, montage]))
print("wrote", p)
