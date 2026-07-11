# ---------------------------------------------------------------------------
# PROVENANCE
#
# Vendored verbatim from the FREUID Challenge 2026 organizers' official
# scoring notebook:
#     https://www.kaggle.com/code/irelic1/freuid-score?scriptVersionId=319563688
# Retrieved: 2026-07-11.
#
# DO NOT "improve", refactor, or otherwise modify the vendored code below
# (everything down to the "END VENDORED CODE" marker) beyond this header and
# import adjustments. Bit-exactness with what Kaggle actually runs is the
# entire point of vendoring this file, including quirks that look like they
# could be tightened (e.g. the tie-collapsing in ``_det_curve``, the
# trapezoid endpoints, the ``ParticipantVisibleError`` machinery that is a
# no-op outside the Kaggle harness). If the organizers publish a new script
# version, re-vendor the whole file and bump the URL/date above rather than
# hand-patching this copy.
#
# A thin, non-Kaggle numpy-array wrapper (``official_freuid_score``) is
# appended after the "END VENDORED CODE" marker for convenience use in this
# repo's own analysis scripts; it does not alter any vendored logic, it only
# adapts the ``(solution_df, submission_df, row_id_column_name)`` calling
# convention Kaggle uses to plain ``(y_true, y_score)`` arrays.
# ---------------------------------------------------------------------------

"""
FREUID Score - official combined metric for The FREUID Challenge 2026
(IJCAI-ECAI 2026, Bremen). Lower is better. Bounded in ``[0, 1]``.

The score is the harmonic mean combination of the two evaluation metrics
described on the public challenge site
(https://freuid2026.microblink.com/#evaluation):

* **AuDET** - Area under the Detection Error Trade-off curve. A single
  scalar capturing the trade-off between false-accept and false-reject
  errors across operating points.
* **APCER @ BPCER** - Attack Presentation Classification Error Rate
  measured at a fixed Bona-Fide Presentation Classification Error Rate
  (default: 1%, the production-relevant slice of the DET curve).

Both sub-metrics are bounded in ``[0, 1]`` with *lower = better*. To
combine them we:

1. Convert each to a "goodness" score ``g = 1 - m`` (so higher = better,
   bounded in ``[0, 1]``).
2. Take the harmonic mean of the goodnesses (this is the classical
   F1-style combination - it penalizes systems that fail on either
   component, unlike the arithmetic mean).
3. Convert back to a "lower = better" score: ``FREUID = 1 - HM``.

Equivalently, with ``a = AuDET`` and ``p = APCER @ BPCER``:

    FREUID = 1 - 2 (1 - a)(1 - p) / ((1 - a) + (1 - p))

Properties
----------
* ``a = 0`` and ``p = 0``  ->  ``FREUID = 0`` (perfect classifier).
* ``a = 1`` or ``p = 1``  ->  ``FREUID = 1`` (failure on either component
  is fatal; this is the key reason for the harmonic-mean combination).
* ``a = p = 0.5``  ->  ``FREUID = 0.5`` (random classifier).
* ``a = 0.0`` and ``p = 0.5``  ->  ``FREUID ~ 0.333`` (asymmetric
  performance is penalized relative to ``(0.25, 0.25)`` which scores
  exactly ``0.25``).

Submission format
-----------------
Kaggle aligns ``solution`` and ``submission`` by ``row_id_column_name``
before passing the dataframes to ``score``.

* Solution - two columns: ``id`` (row id) and ``label``
  (``0`` = bona-fide / genuine, ``1`` = attack / fraudulent).
* Submission - two columns: ``id`` (row id) and ``score``
  (real-valued attack score; higher = more confident the document is
  fraudulent).

Configuration
-------------
``bpcer_target`` controls the operating point used by the APCER@BPCER
component. The FREUID Challenge fixes it at ``0.01`` (1%); change only
if you re-purpose this metric for a different competition.
"""

import numpy as np
import pandas as pd
import pandas.api.types


# Operating point used by the FREUID Challenge leaderboard.
DEFAULT_BPCER_TARGET = 0.01


class ParticipantVisibleError(Exception):
    """Errors raised with this type are surfaced to participants on Kaggle.

    All other exceptions are hidden from participants to avoid leaking
    solution data through error messages.
    """

    pass


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_and_extract(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate dataframes and return ``(y_true, y_score)`` numpy arrays."""

    if row_id_column_name not in solution.columns:
        raise ParticipantVisibleError(
            f"Solution is missing required id column '{row_id_column_name}'."
        )
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(
            f"Submission is missing required id column '{row_id_column_name}'."
        )

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    if solution.shape[1] != 1:
        raise ParticipantVisibleError(
            "Solution must contain exactly one label column besides the id column; "
            f"found {solution.shape[1]} columns."
        )
    if submission.shape[1] != 1:
        raise ParticipantVisibleError(
            "Submission must contain exactly one score column besides the id column; "
            f"found {submission.shape[1]} columns."
        )

    y_true_series = solution.iloc[:, 0]
    y_score_series = submission.iloc[:, 0]

    if not pandas.api.types.is_numeric_dtype(y_score_series):
        raise ParticipantVisibleError(
            f"Submission column '{y_score_series.name}' must be numeric."
        )
    if not pandas.api.types.is_numeric_dtype(y_true_series):
        raise ParticipantVisibleError(
            f"Solution column '{y_true_series.name}' must be numeric (0/1)."
        )

    y_score = y_score_series.to_numpy(dtype=float)
    if not np.isfinite(y_score).all():
        raise ParticipantVisibleError(
            "Submission scores must be finite numbers (no NaN or Inf allowed)."
        )

    y_true = y_true_series.to_numpy()
    unique_labels = set(np.unique(y_true).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ParticipantVisibleError(
            f"Solution labels must be in {{0, 1}}; found {sorted(unique_labels)}."
        )
    y_true = y_true.astype(int)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ParticipantVisibleError(
            "Both bona-fide (0) and attack (1) samples are required to "
            "compute the FREUID score."
        )

    return y_true, y_score


# ---------------------------------------------------------------------------
# DET curve (shared by AuDET and APCER @ BPCER)
# ---------------------------------------------------------------------------
#
# Convention (consistent with PAD / ISO/IEC 30107-3):
#
#   At a decision threshold ``tau`` we predict "attack" iff ``score >= tau``.
#
#   * BPCER(tau) = mean(score[bona-fide] >= tau)
#       Bona-fide Presentation Classification Error Rate (false alarm on
#       genuine documents).
#   * APCER(tau) = mean(score[attack]    <  tau)
#       Attack Presentation Classification Error Rate (attack misses).
#
# The DET curve is the locus { (BPCER(tau), APCER(tau)) : tau in R }.

def _det_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bpcer, apcer)`` arrays sorted by ``bpcer`` ascending.

    The arrays cover the full sweep from ``BPCER = 0`` (every sample
    classified as bona-fide) to ``BPCER = 1`` (every sample classified
    as attack), with one (BPCER, APCER) point per unique score
    threshold.
    """
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    # Sort by *descending* score so that, walking down the array, we sweep
    # the threshold from "predict bona-fide for everyone" (tau = +inf)
    # towards "predict attack for everyone" (tau = -inf).
    order = np.argsort(-y_score, kind="mergesort")
    s_sorted = y_score[order]
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)

    # Collapse adjacent thresholds with identical score (otherwise tied
    # samples would produce spurious vertical/horizontal segments).
    distinct = np.r_[np.diff(s_sorted) != 0, True]
    tp_cum = tp_cum[distinct]
    fp_cum = fp_cum[distinct]

    bpcer = fp_cum / n_neg
    apcer = 1.0 - tp_cum / n_pos

    # Prepend the (BPCER=0, APCER=1) endpoint that corresponds to tau = +inf.
    bpcer = np.concatenate(([0.0], bpcer))
    apcer = np.concatenate(([1.0], apcer))
    return bpcer, apcer


# ---------------------------------------------------------------------------
# Sub-metrics
# ---------------------------------------------------------------------------

def _audet_from_curve(bpcer: np.ndarray, apcer: np.ndarray) -> float:
    """Area under the DET curve in linear ``[0, 1] x [0, 1]`` space.

    Equals ``1 - AUROC`` with attacks treated as the positive class.
    """
    return float(np.trapezoid(apcer, bpcer))


def _apcer_at_bpcer_from_curve(
    bpcer: np.ndarray, apcer: np.ndarray, bpcer_target: float
) -> float:
    """Smallest APCER attainable while keeping ``BPCER <= bpcer_target``.

    APCER is monotonically non-increasing in BPCER, so we walk the curve
    rightwards until the BPCER budget is exhausted and read off the APCER
    at that operating point.
    """
    eps = 1e-12
    feasible = bpcer <= bpcer_target + eps
    if not feasible.any():
        # The (BPCER=0, APCER=1) endpoint is always feasible whenever
        # bpcer_target >= 0, so this branch is purely defensive.
        return 1.0
    idx = int(np.flatnonzero(feasible).max())
    return float(apcer[idx])


# ---------------------------------------------------------------------------
# DET-F1 combination
# ---------------------------------------------------------------------------

def _combine_det_f1(audet: float, apcer_at_bpcer: float) -> float:
    """Combine two ``[0, 1]`` lower-is-better metrics via a DET-F1 score.

    ``DET-F1 = 1 - HM(1 - audet, 1 - apcer)``

    Returns a value in ``[0, 1]`` where lower is better. A failure on
    either component (value of ``1.0``) drives the combined score to
    ``1.0``, which is the desired behavior for a leaderboard ranking.
    """
    g_audet = 1.0 - audet
    g_apcer = 1.0 - apcer_at_bpcer

    denom = g_audet + g_apcer
    if denom <= 0.0:
        # Both components are at the worst possible value (1.0). The
        # harmonic mean of two zeros is undefined; the limiting score
        # is 1.0 (worst).
        return 1.0

    harmonic_mean = 2.0 * g_audet * g_apcer / denom
    return 1.0 - harmonic_mean


# ---------------------------------------------------------------------------
# Public scoring entry point (Kaggle-compatible signature)
# ---------------------------------------------------------------------------

def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    bpcer_target: float = DEFAULT_BPCER_TARGET,
) -> float:
    """Compute the official FREUID Challenge score. Lower is better.

    This is the function Kaggle's evaluation system invokes. It returns a
    single scalar in ``[0, 1]`` that combines AuDET and APCER @ BPCER
    via a harmonic-mean (F1-style) combination on the two metrics'
    "goodness" values (see module docstring for the full definition).

    Parameters
    ----------
    solution
        DataFrame with two columns: ``row_id_column_name`` and a binary
        label column (``0`` = bona-fide, ``1`` = attack).
    submission
        DataFrame with two columns: ``row_id_column_name`` and a real
        valued attack score (higher = more confident the document is
        fraudulent).
    row_id_column_name
        Name of the row id column. Kaggle uses this to align the two
        dataframes before invoking ``score``.
    bpcer_target
        BPCER operating point at which APCER is evaluated. Defaults to
        ``0.01`` (1%), as fixed by the FREUID Challenge rules.

    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"

    Perfect classifier - every attack scored higher than every bona-fide.
    Both AuDET and APCER@BPCER are 0, so the combined score is 0.

    >>> sol = pd.DataFrame({"id": range(4), "label": [0, 0, 1, 1]})
    >>> sub = pd.DataFrame({"id": range(4), "score": [0.1, 0.2, 0.8, 0.9]})
    >>> round(score(sol, sub, row_id_column_name), 6)
    0.0

    Worst-case classifier - every attack scored lower than every bona-fide.
    Both AuDET and APCER@BPCER are 1, so the combined score is 1.

    >>> sol = pd.DataFrame({"id": range(4), "label": [0, 0, 1, 1]})
    >>> sub = pd.DataFrame({"id": range(4), "score": [0.9, 0.8, 0.2, 0.1]})
    >>> round(score(sol, sub, row_id_column_name), 6)
    1.0

    Constant predictions degenerate the DET curve to the diagonal.
    AuDET = 0.5; APCER@1%BPCER = 1 (we cannot catch any attack at a 1%
    false-alarm budget when scores are constant). The harmonic-mean
    combination penalizes the operating-point failure heavily:
    ``1 - HM(0.5, 0) = 1``.

    >>> sol = pd.DataFrame({"id": range(6), "label": [0, 0, 0, 1, 1, 1]})
    >>> sub = pd.DataFrame({"id": range(6), "score": [0.5] * 6})
    >>> round(score(sol, sub, row_id_column_name), 6)
    1.0

    A larger, well-calibrated example: 100 bona-fide vs 100 attacks,
    perfectly separable. The combined score is 0.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> bona = rng.uniform(0.0, 0.4, size=100)
    >>> atk = rng.uniform(0.6, 1.0, size=100)
    >>> sol = pd.DataFrame({"id": range(200),
    ...                     "label": [0] * 100 + [1] * 100})
    >>> sub = pd.DataFrame({"id": range(200),
    ...                     "score": list(bona) + list(atk)})
    >>> round(score(sol, sub, row_id_column_name), 6)
    0.0

    Mixed example - one attack ranks below one bona-fide, exercising the
    DET-F1 combination on intermediate values.

    >>> sol = pd.DataFrame({"id": range(4), "label": [0, 0, 1, 1]})
    >>> sub = pd.DataFrame({"id": range(4), "score": [0.1, 0.6, 0.4, 0.9]})
    >>> # AuDET = 0.25, APCER @ 1% BPCER = 0.5 (only the score=0.9 attack
    >>> # is caught at any BPCER <= 1%). DET-F1 = 1 - 2*0.75*0.5/1.25 = 0.4
    >>> round(score(sol, sub, row_id_column_name), 6)
    0.4
    """
    if not np.isfinite(bpcer_target) or bpcer_target < 0.0 or bpcer_target > 1.0:
        raise ParticipantVisibleError(
            f"bpcer_target must be a finite value in [0, 1]; got {bpcer_target!r}."
        )

    y_true, y_score = _validate_and_extract(solution.copy(), submission.copy(), row_id_column_name)

    bpcer, apcer = _det_curve(y_true, y_score)
    audet = _audet_from_curve(bpcer, apcer)
    apcer_at_target = _apcer_at_bpcer_from_curve(bpcer, apcer, bpcer_target)

    combined = _combine_det_f1(audet, apcer_at_target)

    if not np.isfinite(combined):
        # Defensive: should never trigger given the validations above.
        raise ParticipantVisibleError(
            "FREUID score evaluation produced a non-finite value."
        )

    return float(combined)

# END VENDORED CODE
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Non-Kaggle convenience wrapper (NOT part of the vendored file)
# ---------------------------------------------------------------------------
#
# This repo's analysis scripts work with plain numpy arrays / pandas Series
# of (y_true, y_score), not Kaggle's (solution_df, submission_df,
# row_id_column_name) calling convention. This wrapper adapts to that
# convention by building the two single-column DataFrames ``score()``
# expects and calling the vendored function unmodified -- it does not
# reimplement or shortcut any of the vendored math.

from numpy.typing import ArrayLike


def official_freuid_score(
    y_true: ArrayLike,
    y_score: ArrayLike,
    bpcer_target: float = DEFAULT_BPCER_TARGET,
) -> dict[str, float]:
    """Compute AuDET, APCER@bpcer_target, and the combined FREUID score.

    Thin numpy-array convenience wrapper around the vendored ``score()``
    (and its internal ``_det_curve``/``_audet_from_curve``/
    ``_apcer_at_bpcer_from_curve``/``_combine_det_f1`` helpers), which is
    itself the actual Kaggle-invoked entry point. Returns all three values
    in one call since the DET curve only needs to be built once.

    Parameters
    ----------
    y_true
        Array-like of 0 (bona-fide) / 1 (attack) labels.
    y_score
        Array-like of real-valued attack scores (higher = more fraud-like).
    bpcer_target
        BPCER operating point for the APCER component. Defaults to the
        FREUID Challenge's fixed ``0.01`` (1%).

    Returns
    -------
    dict with keys ``"audet"``, ``"apcer_at_bpcer"``, ``"freuid"``.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score)
    n = y_true_arr.shape[0]

    solution = pd.DataFrame({"id": np.arange(n), "label": y_true_arr})
    submission = pd.DataFrame({"id": np.arange(n), "score": y_score_arr})

    y_true_v, y_score_v = _validate_and_extract(solution, submission, "id")
    bpcer, apcer = _det_curve(y_true_v, y_score_v)
    audet_val = _audet_from_curve(bpcer, apcer)
    apcer_val = _apcer_at_bpcer_from_curve(bpcer, apcer, bpcer_target)
    freuid_val = _combine_det_f1(audet_val, apcer_val)

    return {
        "audet": audet_val,
        "apcer_at_bpcer": apcer_val,
        "freuid": freuid_val,
    }
