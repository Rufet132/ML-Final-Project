"""Tests for the bonus GradientBoostingClassifier (log-loss GBM)."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.ensemble import GradientBoostingClassifier as SkGBM
from sklearn.model_selection import train_test_split

from src.boosting.gradient_boosting import GradientBoostingClassifier, _RegressionTree

RANDOM_SEED = 42

#: Tolerated accuracy gap against sklearn's GBM with equal parameters.
SKLEARN_TOLERANCE: float = 0.03


@pytest.fixture(scope="module")
def cancer_split() -> tuple:
    data = load_breast_cancer()
    return train_test_split(data.data, data.target, test_size=0.3,
                            random_state=RANDOM_SEED, stratify=data.target)


class TestRegressionTree:
    def test_recovers_piecewise_constant_target(self) -> None:
        X = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
        target = np.where(X[:, 0] < 0.5, -1.0, 2.0)
        tree = _RegressionTree(max_depth=2, min_samples_split=2).fit(X, target)
        np.testing.assert_allclose(tree.predict(X), target)

    def test_depth_zero_like_behavior_via_no_split(self) -> None:
        X = np.ones((5, 2))  # constant features: no valid split exists
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tree = _RegressionTree(max_depth=3, min_samples_split=2).fit(X, target)
        assert tree.n_leaves == 1
        np.testing.assert_allclose(tree.predict(X), target.mean())


class TestFitPredict:
    def test_separates_moons(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.2, random_state=RANDOM_SEED)
        model = GradientBoostingClassifier(n_estimators=60).fit(X, y)
        accuracy = float(np.mean(model.predict(X) == y))
        assert accuracy >= 0.95

    def test_proba_shape_and_rows_sum_to_one(self, cancer_split) -> None:
        X_train, X_test, y_train, _ = cancer_split
        model = GradientBoostingClassifier(n_estimators=30).fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-12)
        assert np.all((proba >= 0.0) & (proba <= 1.0))

    def test_predictions_map_to_original_labels(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        X = rng.normal(size=(120, 3))
        y = np.where(X[:, 0] > 0, "yes", "no")
        model = GradientBoostingClassifier(n_estimators=20).fit(X, y)
        assert set(np.unique(model.predict(X))).issubset({"no", "yes"})

    def test_deterministic(self, cancer_split) -> None:
        X_train, X_test, y_train, _ = cancer_split
        a = GradientBoostingClassifier(n_estimators=15).fit(X_train, y_train)
        b = GradientBoostingClassifier(n_estimators=15).fit(X_train, y_train)
        np.testing.assert_allclose(a.predict_proba(X_test), b.predict_proba(X_test))

    def test_staged_predict_last_stage_equals_predict(self, cancer_split) -> None:
        X_train, X_test, y_train, _ = cancer_split
        model = GradientBoostingClassifier(n_estimators=12).fit(X_train, y_train)
        stages = list(model.staged_predict(X_test))
        assert len(stages) == 12
        np.testing.assert_array_equal(stages[-1], model.predict(X_test))

    def test_more_rounds_do_not_hurt_train_error(self, cancer_split) -> None:
        X_train, _, y_train, _ = cancer_split
        model = GradientBoostingClassifier(n_estimators=40).fit(X_train, y_train)
        errors = [float(np.mean(stage != y_train))
                  for stage in model.staged_predict(X_train)]
        assert errors[-1] <= errors[0]


class TestSklearnParity:
    def test_accuracy_close_to_sklearn(self, cancer_split) -> None:
        X_train, X_test, y_train, y_test = cancer_split
        params = dict(n_estimators=50, learning_rate=0.1, max_depth=3)
        ours = GradientBoostingClassifier(**params).fit(X_train, y_train)
        theirs = SkGBM(random_state=RANDOM_SEED, **params).fit(X_train, y_train)
        acc_ours = float(np.mean(ours.predict(X_test) == y_test))
        acc_theirs = float(np.mean(theirs.predict(X_test) == y_test))
        assert abs(acc_ours - acc_theirs) <= SKLEARN_TOLERANCE, (
            f"accuracy gap too large: ours={acc_ours:.4f}, sklearn={acc_theirs:.4f}"
        )


class TestValidation:
    def test_invalid_constructor_args_raise(self) -> None:
        with pytest.raises(ValueError, match="n_estimators"):
            GradientBoostingClassifier(n_estimators=0)
        with pytest.raises(ValueError, match="learning_rate"):
            GradientBoostingClassifier(learning_rate=0.0)
        with pytest.raises(ValueError, match="max_depth"):
            GradientBoostingClassifier(max_depth=0)

    def test_multiclass_raises(self) -> None:
        X = np.random.default_rng(0).normal(size=(30, 2))
        y = np.arange(30) % 3
        with pytest.raises(ValueError, match="2 classes"):
            GradientBoostingClassifier(n_estimators=2).fit(X, y)

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            GradientBoostingClassifier().predict(np.zeros((2, 2)))

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="matching"):
            GradientBoostingClassifier(n_estimators=2).fit(
                np.zeros((3, 2)), np.zeros(4))
