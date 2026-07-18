"""Regression tests for Random Forest configuration validation."""

import numpy as np
import pytest

from src.bagging.random_forest import RandomForestClassifier


@pytest.mark.parametrize("n_estimators", [0, -1])
def test_rejects_non_positive_estimator_count(n_estimators: int) -> None:
    with pytest.raises(ValueError, match="n_estimators must be >= 1"):
        RandomForestClassifier(n_estimators=n_estimators)


@pytest.mark.parametrize("n_jobs", [0, -1])
def test_rejects_non_positive_worker_count(n_jobs: int) -> None:
    with pytest.raises(ValueError, match="n_jobs must be >= 1"):
        RandomForestClassifier(n_jobs=n_jobs)


def test_oob_score_requires_bootstrap_sampling() -> None:
    with pytest.raises(ValueError, match="oob_score requires bootstrap=True"):
        RandomForestClassifier(bootstrap=False, oob_score=True)


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (np.empty((0, 2)), np.empty(0), "at least one sample and one feature"),
        (np.empty((2, 0)), np.array([0, 1]), "at least one sample and one feature"),
        (np.ones((2, 1)), np.array([[0], [1]]), "y must be a 1D array"),
        (np.array([[1.0], [np.inf]]), np.array([0, 1]), "finite values"),
    ],
)
def test_fit_rejects_invalid_training_data(
    X: np.ndarray, y: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RandomForestClassifier().fit(X, y)
