"""Offline implementations of the FREUID competition metrics.

Lets us rank candidate models on the local validation split before spending a
Kaggle submission. Convention:

    label  1 = fraud / attack presentation
    label  0 = bona-fide (genuine document)
    score    = P(fraud), higher means more suspected fraud
    decision = flag as fraud when ``score >= threshold``

Two error rates are swept over the threshold:

    APCER(t) = P(score <  t | attack)     attack wrongly accepted as genuine
    BPCER(t) = P(score >= t | bona-fide)  genuine wrongly rejected as fraud

NOTE: the DET curve is APCER (=FNR for the fraud class) vs BPCER (=FPR). The area
under it on a *linear* axis equals ``1 - ROC AUC`` (a perfect ranker -> 0), which is
what audet() returns.

**Resolved caveat**: this module used to warn that these were only a "local proxy"
pending the official Kaggle scorer. The official scorer is now vendored verbatim at
``src/freuid/official_score.py`` (see its provenance header) -- ``audet()`` and
``apcer_at_bpcer()`` below have been checked against it directly (random data,
including tied scores) and agree to float precision (~1e-16), so they're not an
approximation, they're the same computation. What the local functions do NOT give
you is the organizers' COMBINED leaderboard score: ``FREUID = 1 - HM(1-AuDET,
1-APCER@1%BPCER)`` -- a model can look great on AuDET alone and still score badly
if APCER@1%BPCER is weak (see ``scripts/analysis/official_score_reconciliation_out/
official_score_reconciliation_report.md``). ``evaluate()`` below now returns
``"freuid"`` too, computed via the vendored scorer, so every caller gets the metric
that actually matters for ranking without having to remember to ask for it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import roc_auc_score

from freuid.official_score import DEFAULT_BPCER_TARGET, official_freuid_score


def _sweep(scores: ArrayLike, labels: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (thresholds, apcer, bpcer) swept over all candidate thresholds."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} must match")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 (bona-fide) or 1 (fraud)")

    attacks = scores[labels == 1]
    bonafide = scores[labels == 0]
    if attacks.size == 0 or bonafide.size == 0:
        raise ValueError("need at least one attack and one bona-fide sample")

    # Pad so both extremes (flag-none / flag-all) are represented.
    thr = np.unique(np.concatenate([scores, [scores.min() - 1.0, scores.max() + 1.0]]))
    apcer = np.array([(attacks < t).mean() for t in thr])
    bpcer = np.array([(bonafide >= t).mean() for t in thr])
    return thr, apcer, bpcer


def apcer_at_bpcer(scores: ArrayLike, labels: ArrayLike, bpcer_target: float = 0.01) -> float:
    """APCER at the operating point where BPCER first drops to ``bpcer_target``.

    BPCER decreases as the threshold rises; we take the lowest threshold whose
    BPCER is within budget (the strictest still-valid operating point).
    """
    thr, apcer, bpcer = _sweep(scores, labels)
    valid = bpcer <= bpcer_target
    if not valid.any():
        return 1.0  # cannot reach the target BPCER → worst case
    idx = int(np.argmax(valid))  # first (smallest) threshold meeting the budget
    return float(apcer[idx])


def audet(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the DET curve (APCER=FNR vs BPCER=FPR), linear axis.

    Equals ``1 - ROC AUC``: a perfect ranker scores 0, random ~0.5. Lower is better.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be 0 (bona-fide) or 1 (fraud)")
    return float(1.0 - roc_auc_score(labels, scores))


def freuid(scores: ArrayLike, labels: ArrayLike, bpcer_target: float = DEFAULT_BPCER_TARGET) -> float:
    """The organizers' COMBINED leaderboard score: ``1 - HM(1-AuDET, 1-APCER@bpcer_target)``.

    Computed via the vendored official scorer (``src/freuid/official_score.py``), not
    hand-recombined from the local ``audet()``/``apcer_at_bpcer()`` above -- one source of
    truth for the harmonic-mean formula rather than two implementations that could drift.
    Requires both classes present in ``labels`` (same requirement as ``audet()``/
    ``roc_auc_score``, made explicit here by the vendored scorer's own validation).
    """
    return official_freuid_score(labels, scores, bpcer_target)["freuid"]


def evaluate(scores: ArrayLike, labels: ArrayLike) -> dict[str, float]:
    """All three leaderboard-relevant metrics in one call: the two components AND the
    combined score actually used for ranking. See this module's docstring for why
    ``freuid`` is not a redundant addition -- AuDET alone can be misleading."""
    return {
        "audet": audet(scores, labels),
        "apcer_at_1pct_bpcer": apcer_at_bpcer(scores, labels, bpcer_target=0.01),
        "freuid": freuid(scores, labels, bpcer_target=0.01),
    }


if __name__ == "__main__":
    # Sanity check: a perfect ranker scores ~0 on both metrics.
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    perfect = y.astype(float) + rng.normal(0, 1e-3, size=y.size)
    print("perfect :", evaluate(perfect, y))
    print("random  :", evaluate(rng.random(y.size), y))
