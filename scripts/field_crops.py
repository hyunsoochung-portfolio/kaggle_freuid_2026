"""Crop+enlarge the text/field zone of frauds so subtle text tampering
(scratch-out, blur, texture mismatch) becomes legible. 2-column montage.
Usage: python3 scripts/field_crops.py TYPE START N
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA = Path("data")
IMG = DATA / "train" / "train"
OUT = Path("/tmp/survey")
OUT.mkdir(exist_ok=True)

# per-type field-zone crop box as fractions (x0,y0,x1,y1)
BOX = {
    "EGYPT/DL": (0.27, 0.26, 0.74, 0.80),
    "GUINEA/DL": (0.29, 0.14, 0.78, 0.90),
    "BENIN/DL": (0.32, 0.20, 0.82, 0.92),
    "MOZAMBIQUE/DL": (0.25, 0.13, 0.67, 0.74),
    "MAURITIUS/ID": (0.32, 0.13, 0.72, 0.90),
}

df = pd.read_csv(DATA / "train_labels.csv")
fr = df[df.label == 1]

t = sys.argv[1] if len(sys.argv) > 1 else "EGYPT/DL"
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10

ids = fr[fr.type == t].id.tolist()[start:start + n]
x0f, y0f, x1f, y1f = BOX[t]
W_TILE = 720
tiles = []
for k, id_ in enumerate(ids):
    img = cv2.imread(str(IMG / f"{id_}.jpeg"))
    if img is None:
        continue
    H, W = img.shape[:2]
    crop = img[int(H * y0f):int(H * y1f), int(W * x0f):int(W * x1f)]
    h = int(crop.shape[0] * W_TILE / crop.shape[1])
    crop = cv2.resize(crop, (W_TILE, h), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(crop, (0, 0), (58, 30), (0, 0, 0), -1)
    cv2.putText(crop, str(start + k), (4, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2)
    tiles.append(crop)

rowH = max(im.shape[0] for im in tiles)
tiles = [cv2.copyMakeBorder(im, 0, rowH - im.shape[0], 0, 0,
                            cv2.BORDER_CONSTANT, value=(255, 255, 255)) for im in tiles]
rows = []
for r in range(0, len(tiles), 2):
    row = tiles[r:r + 2]
    if len(row) < 2:
        row.append(np.full((rowH, W_TILE, 3), 255, np.uint8))
    rows.append(np.hstack(row))
montage = np.vstack(rows)
p = OUT / f"fields_{t.replace('/', '-')}_{start}.png"
cv2.imwrite(str(p), montage)
print("wrote", p, montage.shape)
