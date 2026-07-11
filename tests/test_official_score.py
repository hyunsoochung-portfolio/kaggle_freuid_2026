"""Tests for the vendored official FREUID scorer (src/freuid/official_score.py).

Bit-exactness with Kaggle is the entire point of vendoring -- these tests check
against the vendored docstring's own exact expected values, not against a
reimplementation, plus one property test tying the AuDET component back to
sklearn's ROC AUC.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from freuid.official_score import official_freuid_score, score


ROW_ID = "id"


def test_docstring_example_perfect_classifier():
    sol = pd.DataFrame({"id": range(4), "label": [0, 0, 1, 1]})
    sub = pd.DataFrame({"id": range(4), "score": [0.1, 0.2, 0.8, 0.9]})
    assert round(score(sol, sub, ROW_ID), 6) == 0.0


def test_docstring_example_worst_case_classifier():
    sol = pd.DataFrame({"id": range(4), "label": [0, 0, 1, 1]})
    sub = pd.DataFrame({"id": range(4), "score": [0.9, 0.8, 0.2, 0.1]})
    assert round(score(sol, sub, ROW_ID), 6) == 1.0


def test_docstring_example_constant_predictions():
    sol = pd.DataFrame({"id": range(6), "label": [0, 0, 0, 1, 1, 1]})
    sub = pd.DataFrame({"id": range(6), "score": [0.5] * 6})
    assert round(score(sol, sub, ROW_ID), 6) == 1.0


def test_docstring_example_large_separable():
    rng = np.random.default_rng(0)
    bona = rng.uniform(0.0, 0.4, size=100)
    atk = rng.uniform(0.6, 1.0, size=100)
    sol = pd.DataFrame({"id": range(200), "label": [0] * 100 + [1] * 100})
    sub = pd.DataFrame({"id": range(200), "score": list(bona) + list(atk)})
    assert round(score(sol, sub, ROW_ID), 6) == 0.0


def test_docstring_example_mixed_intermediate():
    sol = pd.DataFrame({"id": range(4), "label": [0, 0, 1, 1]})
    sub = pd.DataFrame({"id": range(4), "score": [0.1, 0.6, 0.4, 0.9]})
    assert round(score(sol, sub, ROW_ID), 6) == 0.4


@pytest.mark.parametrize("seed", range(20))
def test_audet_component_matches_sklearn_roc_auc(seed):
    """On random data, the vendored AuDET component == 1 - sklearn roc_auc_score.

    AuDET is defined (module docstring) as the area under the DET curve on a
    linear axis, which is mathematically 1 - AUROC. This checks that identity
    holds to float precision across random label/score draws, including cases
    with tied scores (which exercise ``_det_curve``'s tie-collapsing).
    """
    rng = np.random.default_rng(seed)
    n = 300
    y_true = rng.integers(0, 2, size=n)
    # ensure both classes present
    if y_true.sum() == 0 or y_true.sum() == n:
        y_true[0] = 0
        y_true[1] = 1

    # half the draws use coarsely-rounded scores to force score ties
    if seed % 2 == 0:
        y_score = rng.integers(0, 10, size=n).astype(float)
    else:
        y_score = rng.random(n)

    result = official_freuid_score(y_true, y_score)
    expected_audet = 1.0 - roc_auc_score(y_true, y_score)
    assert result["audet"] == pytest.approx(expected_audet, abs=1e-12)


def test_official_freuid_score_matches_kaggle_entry_point():
    """The numpy wrapper must agree exactly with the Kaggle-signature score()."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    y_score = rng.random(n)

    sol = pd.DataFrame({"id": range(n), "label": y_true})
    sub = pd.DataFrame({"id": range(n), "score": y_score})
    expected = score(sol, sub, ROW_ID)

    result = official_freuid_score(y_true, y_score)
    assert result["freuid"] == pytest.approx(expected, abs=1e-12)
