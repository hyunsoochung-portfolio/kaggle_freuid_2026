"""Sanity tests for the offline metric implementations."""

import numpy as np

from freuid.metrics import apcer_at_bpcer, audet, evaluate


def test_perfect_separation_scores_near_zero():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])  # perfectly ranked
    assert audet(scores, y) < 1e-6
    assert apcer_at_bpcer(scores, y, 0.01) < 1e-6


def test_random_is_worse_than_perfect():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    perfect = y + rng.normal(0, 1e-3, size=y.size)
    rand = rng.random(y.size)
    assert audet(perfect, y) < audet(rand, y)


def test_evaluate_keys():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    out = evaluate(s, y)
    assert set(out) == {"audet", "apcer_at_1pct_bpcer"}
