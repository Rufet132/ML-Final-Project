"""Tests for src.metrics.evaluation against sklearn reference values."""

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from src.metrics.evaluation import (
    accuracy_score,
    classification_summary,
    f1_macro,
    roc_auc_score,
)

RANDOM_SEED = 42


@pytest.fixture()
def binary_case() -> tuple:
    rng = np.random.default_rng(RANDOM_SEED)
    y_true = rng.integers(0, 2, size=200)
    scores = np.clip(y_true * 0.4 + rng.normal(0.3, 0.25, size=200), 0.0, 1.0)
    y_pred = (scores >= 0.5).astype(int)
    return y_true, y_pred, scores


@pytest.fixture()
def multiclass_case() -> tuple:
    rng = np.random.default_rng(RANDOM_SEED)
    y_true = rng.integers(0, 4, size=300)
    logits = rng.normal(size=(300, 4))
    logits[np.arange(300), y_true] += 1.5
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_pred = np.argmax(proba, axis=1)
    return y_true, y_pred, proba


class TestAccuracy:
    def test_matches_sklearn(self, binary_case) -> None:
        y_true, y_pred, _ = binary_case
        assert accuracy_score(y_true, y_pred) == pytest.approx(
            sk_metrics.accuracy_score(y_true, y_pred)
        )

    def test_perfect_and_zero(self) -> None:
        y = np.array([0, 1, 2])
        assert accuracy_score(y, y) == 1.0
        assert accuracy_score(y, y[::-1]) == pytest.approx(1.0 / 3.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            accuracy_score(np.array([0, 1]), np.array([0]))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            accuracy_score(np.array([]), np.array([]))


class TestF1Macro:
    def test_matches_sklearn_binary(self, binary_case) -> None:
        y_true, y_pred, _ = binary_case
        assert f1_macro(y_true, y_pred) == pytest.approx(
            sk_metrics.f1_score(y_true, y_pred, average="macro")
        )

    def test_matches_sklearn_multiclass(self, multiclass_case) -> None:
        y_true, y_pred, _ = multiclass_case
        assert f1_macro(y_true, y_pred) == pytest.approx(
            sk_metrics.f1_score(y_true, y_pred, average="macro")
        )

    def test_missing_class_counts_as_zero(self) -> None:
        # Class 2 is never predicted; sklearn's zero_division=0 behaviour.
        y_true = np.array([0, 0, 1, 2])
        y_pred = np.array([0, 0, 1, 1])
        assert f1_macro(y_true, y_pred) == pytest.approx(
            sk_metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
        )


class TestRocAuc:
    def test_matches_sklearn_binary(self, binary_case) -> None:
        y_true, _, scores = binary_case
        assert roc_auc_score(y_true, scores) == pytest.approx(
            sk_metrics.roc_auc_score(y_true, scores)
        )

    def test_matches_sklearn_with_ties(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        scores = np.array([0.2, 0.5, 0.5, 0.9, 0.5, 0.2, 0.9, 0.1])
        assert roc_auc_score(y_true, scores) == pytest.approx(
            sk_metrics.roc_auc_score(y_true, scores)
        )

    def test_matches_sklearn_macro_ovr(self, multiclass_case) -> None:
        y_true, _, proba = multiclass_case
        assert roc_auc_score(y_true, proba) == pytest.approx(
            sk_metrics.roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        )

    def test_perfect_separation_is_one(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        assert roc_auc_score(y_true, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0

    def test_single_class_raises(self) -> None:
        with pytest.raises(ValueError, match="fewer than 2"):
            roc_auc_score(np.zeros(4), np.linspace(0, 1, 4))

    def test_multiclass_with_1d_scores_raises(self) -> None:
        with pytest.raises(ValueError, match="probability matrix"):
            roc_auc_score(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))

    def test_wrong_column_count_raises(self) -> None:
        with pytest.raises(ValueError, match="one column per class"):
            roc_auc_score(np.array([0, 1, 2]), np.ones((3, 2)))


class TestSummary:
    def test_binary_summary_keys_and_values(self, binary_case) -> None:
        y_true, y_pred, scores = binary_case
        proba = np.column_stack([1.0 - scores, scores])
        summary = classification_summary(y_true, y_pred, proba)
        assert summary["accuracy"] == pytest.approx(
            sk_metrics.accuracy_score(y_true, y_pred))
        assert summary["roc_auc"] == pytest.approx(
            sk_metrics.roc_auc_score(y_true, scores))

    def test_summary_without_proba_has_no_auc(self, binary_case) -> None:
        y_true, y_pred, _ = binary_case
        summary = classification_summary(y_true, y_pred)
        assert set(summary) == {"accuracy", "f1_macro"}
