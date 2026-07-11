"""Sanity tests for the offline metric implementations."""

import numpy as np
import pytest

from freuid.metrics import apcer_at_bpcer, audet, evaluate, freuid
from freuid.official_score import official_freuid_score


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
    assert set(out) == {"audet", "apcer_at_1pct_bpcer", "freuid"}


@pytest.mark.parametrize("seed", range(15))
def test_local_metrics_match_vendored_scorer(seed):
    """The local audet()/apcer_at_bpcer()/freuid() must agree with the vendored official
    scorer to float precision -- they're not an approximation, they're the same computation
    (see metrics.py's module docstring). Exercises tied scores on even seeds."""
    rng = np.random.default_rng(seed)
    n = 300
    y = rng.integers(0, 2, size=n)
    if y.sum() == 0 or y.sum() == n:
        y[0], y[1] = 0, 1
    s = rng.integers(0, 10, size=n).astype(float) if seed % 2 == 0 else rng.random(n)

    official = official_freuid_score(y, s)
    assert audet(s, y) == pytest.approx(official["audet"], abs=1e-9)
    assert apcer_at_bpcer(s, y, 0.01) == pytest.approx(official["apcer_at_bpcer"], abs=1e-9)
    assert freuid(s, y, 0.01) == pytest.approx(official["freuid"], abs=1e-9)
