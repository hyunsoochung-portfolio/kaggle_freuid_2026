"""Tests for freuid.photosub.probes.threshold_watch -- the exact (real-label) per-epoch
1%-BPCER crossing point, independent of the vendored scorer's own _det_curve/
_apcer_at_bpcer_from_curve so the two implementations can cross-check each other."""

import numpy as np
import pytest

from freuid.official_score import _apcer_at_bpcer_from_curve, _det_curve
from freuid.photosub.probes import threshold_watch


def test_threshold_watch_perfect_separation():
    # 100 bona-fide at 0.0-0.4, 100 fraud at 0.6-1.0 -- 1% budget is 1 bona-fide.
    rng = np.random.default_rng(0)
    bona = rng.uniform(0.0, 0.4, size=100)
    fraud = rng.uniform(0.6, 1.0, size=100)
    scores = np.concatenate([bona, fraud])
    labels = np.concatenate([np.zeros(100), np.ones(100)])
    out = threshold_watch(scores, labels, bpcer_target=0.01)
    assert out["n_val_bonafide"] == 100
    # The threshold should sit within the bona-fide score range (well below the fraud cluster).
    assert out["threshold_score"] < 0.5
    # 1% of 100 bona-fide = exactly 1 -- with perfect separation all 100 frauds occupy the top
    # 100 ranks, so the threshold naturally lands right at the first (highest-scoring) bona-fide.
    assert out["threshold_rank"] == 101


def test_threshold_watch_no_bonafide_returns_nan():
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([1, 1, 1])
    out = threshold_watch(scores, labels)
    assert np.isnan(out["threshold_score"])
    assert out["n_val_bonafide"] == 0


@pytest.mark.parametrize("seed", range(10))
def test_threshold_watch_matches_vendored_det_curve_apcer_point(seed):
    """threshold_watch's crossing score should agree with the vendored scorer's own
    _apcer_at_bpcer_from_curve operating point (same DET-curve crossing, computed two
    different ways -- a cross-check that the simplified hard-count version in probes.py
    hasn't silently drifted from the vendored convention it's meant to mirror)."""
    rng = np.random.default_rng(seed)
    n = 200
    y = rng.integers(0, 2, size=n)
    if y.sum() == 0 or y.sum() == n:
        y[0], y[1] = 0, 1
    s = rng.random(n)

    watch = threshold_watch(s, y, bpcer_target=0.01)
    bpcer, apcer = _det_curve(y.astype(int), s.astype(float))
    vendored_apcer = _apcer_at_bpcer_from_curve(bpcer, apcer, 0.01)

    # Score at/above threshold_watch's crossing point should achieve an APCER no worse than
    # the vendored scorer's own operating point (both are the "tightest still-valid" threshold
    # under the same 1%-budget definition).
    n_fraud = int((y == 1).sum())
    above_or_at = s >= watch["threshold_score"]
    n_caught = int((above_or_at & (y == 1)).sum())
    apcer_at_watch_threshold = 1.0 - n_caught / n_fraud
    assert apcer_at_watch_threshold <= vendored_apcer + 1e-9
