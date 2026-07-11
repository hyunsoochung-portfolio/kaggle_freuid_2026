"""Preview what the recapture (analog) augmentation actually does in training.
Each row: [ ORIGINAL ] | [ + recapture_v2 (analog, label unchanged) ].
Covers a real fraud, a synth fraud, and a bona-fide across doc types, using the
EXACT functions training uses (freuid.augment). Written to data/aug_preview/.
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from freuid.augment import _recapture_degrade, synth_tamper  # noqa: E402

DATA = Path("data")
IMG = DATA / "train" / "train"
OUT = DATA / "aug_preview"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(DATA / "train_labels.csv")
rng = random.Random(4)
nrng = np.random.default_rng(4)


def load(id_, w=430):
    im = cv2.imread(str(IMG / f"{id_}.jpeg"))
    return cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))


def cap(im, txt, color):
    b = np.full((32, im.shape[1], 3), 255, np.uint8)
    cv2.putText(b, txt, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return np.vstack([b, im])


def pair(before, after, ltxt, rtxt):
    h = max(before.shape[0], after.shape[0])
    pad = lambda x: cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255,)*3)  # noqa: E731
    gap = np.full((h + 32, 18, 3), 255, np.uint8)
    return np.hstack([cap(pad(before), ltxt, (0, 130, 0)), gap,
                      cap(pad(after), rtxt, (0, 0, 210))])


def donor_bgr(dtype):
    did = df[(df.type == dtype) & (df.label == 0)].id.iloc[0]
    return cv2.imread(str(IMG / f"{did}.jpeg"))


rows = []

# 1) REAL fraud -> analog fraud
fid = df[(df.type == "EGYPT/DL") & (df.label == 1)].id.iloc[5]
b = load(fid)
rows.append(pair(b, _recapture_degrade(b.copy(), rng, nrng),
                 "REAL fraud (label 1)", "+ analog  (label 1)"))

# 2) SYNTH fraud (bona -> face-swap) -> analog synth fraud
bid = df[(df.type == "GUINEA/DL") & (df.label == 0)].id.iloc[3]
bb = load(bid)
synth, mode = synth_tamper(bb.copy(), donor_bgr("GUINEA/DL"), rng, nrng,
                           text_prob=0.0, allow_color_on_gray=False)
rows.append(pair(synth, _recapture_degrade(synth.copy(), rng, nrng),
                 f"SYNTH fraud ({mode}, label 1)", "+ analog  (label 1)"))

# 3) BONA-FIDE -> analog bona-fide
gid = df[(df.type == "MOZAMBIQUE/DL") & (df.label == 0)].id.iloc[1]
g = load(gid)
rows.append(pair(g, _recapture_degrade(g.copy(), rng, nrng),
                 "bona-fide (label 0)", "+ analog  (label 0)"))

# 4) REAL fraud MAURITIUS -> analog
mid = df[(df.type == "MAURITIUS/ID") & (df.label == 1)].id.iloc[2]
m = load(mid)
rows.append(pair(m, _recapture_degrade(m.copy(), rng, nrng),
                 "REAL fraud (label 1)", "+ analog  (label 1)"))

mw = max(r.shape[1] for r in rows)
rows = [cv2.copyMakeBorder(r, 0, 0, 0, mw - r.shape[1], cv2.BORDER_CONSTANT, value=(255,)*3)
        for r in rows]
sep = np.full((12, mw, 3), 255, np.uint8)
montage = rows[0]
for r in rows[1:]:
    montage = np.vstack([montage, sep, r])
p = OUT / "analog_examples.png"
cv2.imwrite(str(p), montage)
print("wrote", p)
