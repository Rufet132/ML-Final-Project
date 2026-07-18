"""Regression tests for Random Forest configuration validation."""

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
