"""Regression tests for input validation in metrics and preprocessing."""

import numpy as np
import pytest

from src.metrics.evaluation import roc_auc_score
from src.utils.preprocessing import drop_missing_rows


def test_drop_missing_rows_rejects_infinite_values() -> None:
    X = np.array([[1.0, 2.0], [3.0, np.inf]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="infinite"):
        drop_missing_rows(X, y)


def test_drop_missing_rows_still_removes_nan_rows() -> None:
    X = np.array([[1.0, 2.0], [3.0, np.nan]])
    y = np.array([0, 1])

    clean_X, clean_y = drop_missing_rows(X, y)

    np.testing.assert_array_equal(clean_X, np.array([[1.0, 2.0]]))
    np.testing.assert_array_equal(clean_y, np.array([0]))


@pytest.mark.parametrize(
    "scores",
    [np.array(0.5), np.zeros((2, 1, 1))],
)
def test_roc_auc_rejects_invalid_score_dimensions(scores: np.ndarray) -> None:
    with pytest.raises(ValueError, match="1-D score array or 2-D probability matrix"):
        roc_auc_score(np.array([0, 1]), scores)


def test_roc_auc_rejects_non_finite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        roc_auc_score(np.array([0, 1]), np.array([0.1, np.nan]))
