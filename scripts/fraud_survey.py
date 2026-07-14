"""Tile many fraud images into per-type montages for visual survey.
Saves /tmp/survey/<TYPE>.png and an index map /tmp/survey/index.json so we can
map a montage tile number back to its id and zoom in later."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA = Path("data")
IMG = DATA / "train" / "train"
OUT = Path("/tmp/survey")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "train_labels.csv")
fr = df[df.label == 1]
TYPES = ["EGYPT/DL", "GUINEA/DL", "BENIN/DL", "MOZAMBIQUE/DL", "MAURITIUS/ID"]

N = 20            # frauds per type
COLS = 4
TILE_W = 560

index = {}
for t in TYPES:
    ids = fr[fr.type == t].id.tolist()[:N]
    index[t] = ids
    tiles = []
    for k, id_ in enumerate(ids):
        img = cv2.imread(str(IMG / f"{id_}.jpeg"))
        if img is None:
            img = np.full((350, TILE_W, 3), 200, np.uint8)
        h = int(img.shape[0] * TILE_W / img.shape[1])
        img = cv2.resize(img, (TILE_W, h))
        # number label (top-left) so we can reference tiles
        cv2.rectangle(img, (0, 0), (46, 30), (0, 0, 0), -1)
        cv2.putText(img, str(k), (4, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        tiles.append(img)
    # pad to common height per row, build grid
    rowH = max(im.shape[0] for im in tiles)
    tiles = [cv2.copyMakeBorder(im, 0, rowH - im.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
             for im in tiles]
    rows = []
    for r in range(0, len(tiles), COLS):
        row = tiles[r:r + COLS]
        while len(row) < COLS:
            row.append(np.full((rowH, TILE_W, 3), 255, np.uint8))
        rows.append(np.hstack(row))
    montage = np.vstack(rows)
    p = OUT / f"{t.replace('/', '-')}.png"
    cv2.imwrite(str(p), montage)
    print("wrote", p, montage.shape)

(OUT / "index.json").write_text(json.dumps(index, indent=0))
print("index saved")
