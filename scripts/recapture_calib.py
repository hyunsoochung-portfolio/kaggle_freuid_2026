"""Calibrate a candidate 'recapture' (digital->analog) degradation against the
REAL analog images. Target stats (measured on the 20 is_digital=False images at
512x320): sharp~862, illum~26.0, sat~30.6, bright~181.7, contrast~53.2.
We apply the candidate to DIGITAL genuines and check it lands on those numbers.
"""
import random

import cv2
import numpy as np
import pandas as pd

df = pd.read_csv("data/train_labels.csv")
IMG = "data/train/train"
SZ = (512, 320)


def measure(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    low = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 40)
    return np.array([cv2.Laplacian(g, cv2.CV_64F).var(), low.std(),
                     hsv[..., 1].mean(), g.mean(), g.std()])


def recapture(im, rng, nrng):
    """digital BGR (512x320) -> analog-like."""
    h, w = im.shape[:2]
    f = im.astype(np.float32)
    # 1. defocus blur (ALWAYS) -- the dominant softening (tuned to ~6x, not more)
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 0.8))
    # 2. downscale-upscale (resolution loss, compounds softening)
    s = rng.uniform(0.72, 0.9)
    f = cv2.resize(cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA),
                   (w, h), interpolation=cv2.INTER_CUBIC)
    # 3. smooth illumination gradient + radial vignette (NOT harsh glare)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 6.283)
    grad = (xx / w - 0.5) * np.cos(ang) + (yy / h - 0.5) * np.sin(ang)
    r = np.sqrt((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
    illum = (1 + rng.uniform(0.38, 0.60) * grad) * \
            (1 - rng.uniform(0.14, 0.28) * (r / r.max()) ** 2)
    f = f * illum[..., None]
    # 4. colour cast (warm OR cool, both directions)
    f = f * np.array([rng.uniform(0.90, 1.09), rng.uniform(0.95, 1.05),
                      rng.uniform(0.90, 1.09)])
    # 5. darker + lower contrast (mild)
    f = (f - 128) * rng.uniform(0.92, 1.0) + 128
    f = f * rng.uniform(0.93, 1.0)
    # 6. desaturate ~10%
    f = np.clip(f, 0, 255)
    hsv = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.80, 0.94)
    f = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    # 7. mild sensor noise
    f = f + nrng.standard_normal((h, w, 1)) * rng.uniform(2, 5)
    return np.clip(f, 0, 255).astype(np.uint8)


# real analog stats
an = df[df.is_digital == False]
real = np.mean([measure(cv2.resize(cv2.imread(f"{IMG}/{i}.jpeg"), SZ)) for i in an.id], axis=0)

# apply candidate to digital genuines of the same types
dig = df[(df.is_digital == True) & (df.type.isin(an.type.unique()))].groupby("type").head(30)
rng = random.Random(0)
nrng = np.random.default_rng(0)
digo, syn = [], []
for i in dig.id:
    im = cv2.resize(cv2.imread(f"{IMG}/{i}.jpeg"), SZ)
    digo.append(measure(im))
    syn.append(measure(recapture(im, rng, nrng)))
digo = np.mean(digo, axis=0)
syn = np.mean(syn, axis=0)

names = ["sharp", "illum", "sat", "bright", "contrast"]
print(f"{'metric':10s} {'DIGITAL':>9s} {'REAL-anlg':>10s} {'MY-synth':>9s}  {'synth/real':>10s}")
for k, d, r, s in zip(names, digo, real, syn):
    print(f"{k:10s} {d:9.1f} {r:10.1f} {s:9.1f}  {s/r:10.2f}")
