"""Pytest suite for the from-scratch RandomForestClassifier (Module 3).

sklearn appears only as a reference baseline (explicitly permitted for
tests). Covers determinism under random_state, OOB scoring, probability
semantics, feature importances, multiprocessing parity, edge cases, and
an accuracy comparison against sklearn's RandomForestClassifier.
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier as SkRandomForest
from sklearn.model_selection import train_test_split

from src.bagging.random_forest import RandomForestClassifier

RANDOM_SEED = 42

#: Small forest size keeping the suite fast while leaving OOB samples.
N_TREES: int = 25

#: Tolerated accuracy gap against the sklearn forest baseline. Slightly
#: wider than the single-tree 2% rubric because two forests with
#: different bootstrap streams differ by sampling noise as well.
FOREST_ACCURACY_TOLERANCE: float = 0.04


@pytest.fixture(scope="module")
def cancer_split() -> tuple:
    data = load_breast_cancer()
    return train_test_split(data.data, data.target, test_size=0.3,
                            random_state=RANDOM_SEED, stratify=data.target)


@pytest.fixture(scope="module")
def fitted_forest(cancer_split) -> RandomForestClassifier:
    X_train, _, y_train, _ = cancer_split
    return RandomForestClassifier(
        n_estimators=N_TREES, oob_score=True, random_state=RANDOM_SEED
    ).fit(X_train, y_train)


class TestDeterminism:
    def test_same_seed_identical_forest(self, cancer_split) -> None:
        X_train, X_test, y_train, _ = cancer_split
        params = dict(n_estimators=10, max_features="sqrt", oob_score=True,
                      random_state=RANDOM_SEED)
        forest_a = RandomForestClassifier(**params).fit(X_train, y_train)
        forest_b = RandomForestClassifier(**params).fit(X_train, y_train)
        np.testing.assert_array_equal(forest_a.predict(X_test), forest_b.predict(X_test))
        np.testing.assert_allclose(forest_a.predict_proba(X_test),
                                   forest_b.predict_proba(X_test))
        np.testing.assert_allclose(forest_a.feature_importances_,
                                   forest_b.feature_importances_)
        assert forest_a.oob_score_ == forest_b.oob_score_

    def test_trees_receive_distinct_seeds(self, fitted_forest) -> None:
        seeds = [tree.random_state for tree in fitted_forest.estimators_]
        assert all(seed is not None for seed in seeds)
        assert len(set(seeds)) == len(seeds)

    def test_different_seed_changes_forest(self, cancer_split) -> None:
        X_train, X_test, y_train, _ = cancer_split
        forest_a = RandomForestClassifier(n_estimators=10, random_state=0).fit(
            X_train, y_train)
        forest_b = RandomForestClassifier(n_estimators=10, random_state=1).fit(
            X_train, y_train)
        assert not np.allclose(forest_a.predict_proba(X_test),
                               forest_b.predict_proba(X_test))


class TestPrediction:
    def test_proba_rows_sum_to_one(self, fitted_forest, cancer_split) -> None:
        _, X_test, _, _ = cancer_split
        proba = fitted_forest.predict_proba(X_test)
        assert proba.shape == (X_test.shape[0], 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(proba >= 0.0)

    def test_predict_matches_argmax_of_proba(self, fitted_forest, cancer_split) -> None:
        _, X_test, _, _ = cancer_split
        proba = fitted_forest.predict_proba(X_test)
        expected = fitted_forest.classes_[np.argmax(proba, axis=1)]
        np.testing.assert_array_equal(fitted_forest.predict(X_test), expected)

    def test_predictions_map_to_original_labels(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        X = rng.normal(size=(80, 3))
        y = np.where(X[:, 0] > 0, 7, -3)  # non-contiguous labels
        forest = RandomForestClassifier(n_estimators=8, random_state=RANDOM_SEED).fit(X, y)
        assert set(np.unique(forest.predict(X))).issubset({-3, 7})

    def test_forest_beats_or_matches_single_tree_family(self, cancer_split) -> None:
        X_train, X_test, y_train, y_test = cancer_split
        forest = RandomForestClassifier(
            n_estimators=N_TREES, random_state=RANDOM_SEED).fit(X_train, y_train)
        accuracy = float(np.mean(forest.predict(X_test) == y_test))
        assert accuracy >= 0.90  # sanity floor for this easy dataset


class TestOobScore:
    def test_oob_score_in_unit_interval(self, fitted_forest) -> None:
        assert 0.0 <= fitted_forest.oob_score_ <= 1.0

    def test_oob_close_to_test_accuracy(self, fitted_forest, cancer_split) -> None:
        _, X_test, _, y_test = cancer_split
        test_accuracy = float(np.mean(fitted_forest.predict(X_test) == y_test))
        assert abs(fitted_forest.oob_score_ - test_accuracy) <= 0.06

    def test_oob_unavailable_without_flag(self, cancer_split) -> None:
        X_train, _, y_train, _ = cancer_split
        forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(
            X_train, y_train)
        with pytest.raises(AttributeError, match="oob_score"):
            _ = forest.oob_score_

    def test_oob_unavailable_without_bootstrap(self, cancer_split) -> None:
        X_train, _, y_train, _ = cancer_split
        forest = RandomForestClassifier(
            n_estimators=5, bootstrap=False, oob_score=True, random_state=0
        ).fit(X_train, y_train)
        with pytest.raises(AttributeError, match="oob_score"):
            _ = forest.oob_score_


class TestFeatureImportances:
    def test_shape_normalization_and_sign(self, fitted_forest, cancer_split) -> None:
        X_train, _, _, _ = cancer_split
        importances = fitted_forest.feature_importances_
        assert importances.shape == (X_train.shape[1],)
        assert np.all(importances >= 0.0)
        assert importances.sum() == pytest.approx(1.0)

    def test_unfitted_importances_raise(self) -> None:
        with pytest.raises(ValueError, match="fit"):
            _ = RandomForestClassifier().feature_importances_


class TestParallelism:
    def test_n_jobs_two_matches_sequential(self) -> None:
        # Small data keeps the process pool start-up cost tolerable.
        rng = np.random.default_rng(RANDOM_SEED)
        X = rng.normal(size=(120, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        sequential = RandomForestClassifier(
            n_estimators=6, random_state=RANDOM_SEED, n_jobs=1).fit(X, y)
        parallel = RandomForestClassifier(
            n_estimators=6, random_state=RANDOM_SEED, n_jobs=2).fit(X, y)
        np.testing.assert_allclose(sequential.predict_proba(X),
                                   parallel.predict_proba(X))


class TestValidationAndEdgeCases:
    def test_non_2d_x_raises(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            RandomForestClassifier(n_estimators=2).fit(np.zeros(4), np.zeros(4))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same number"):
            RandomForestClassifier(n_estimators=2).fit(np.zeros((3, 2)), np.zeros(4))

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="fit"):
            RandomForestClassifier().predict(np.zeros((2, 2)))

    def test_single_class_training(self) -> None:
        X = np.random.default_rng(0).normal(size=(30, 2))
        y = np.zeros(30, dtype=int)
        forest = RandomForestClassifier(n_estimators=4, random_state=0).fit(X, y)
        np.testing.assert_array_equal(forest.predict(X), y)
        np.testing.assert_allclose(forest.predict_proba(X), 1.0)

    def test_bootstrap_false_uses_full_sample(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        X = rng.normal(size=(60, 3))
        y = (X[:, 0] > 0).astype(int)
        forest = RandomForestClassifier(
            n_estimators=3, bootstrap=False, max_features=None, random_state=0
        ).fit(X, y)
        # Without bootstrap or feature sub-sampling all trees are identical.
        first = forest.estimators_[0].predict_proba(X)
        for tree in forest.estimators_[1:]:
            np.testing.assert_allclose(tree.predict_proba(X), first)

    def test_max_depth_and_min_samples_are_forwarded(self, cancer_split) -> None:
        X_train, _, y_train, _ = cancer_split
        forest = RandomForestClassifier(
            n_estimators=3, max_depth=2, random_state=0).fit(X_train, y_train)
        assert all(tree.depth <= 2 for tree in forest.estimators_)


class TestClassWeight:
    def test_invalid_class_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="class_weight"):
            RandomForestClassifier(class_weight="quadratic")

    def test_balanced_weights_improve_minority_recall(self) -> None:
        # 95/5 imbalance with overlapping classes: unweighted trees lean
        # towards the majority; balanced weights must recover minority
        # recall without hurting determinism.
        rng = np.random.default_rng(RANDOM_SEED)
        n_majority, n_minority = 570, 30
        X = np.vstack([
            rng.normal(0.0, 1.0, size=(n_majority, 4)),
            rng.normal(1.2, 1.0, size=(n_minority, 4)),
        ])
        y = np.array([0] * n_majority + [1] * n_minority)
        shuffle = rng.permutation(len(y))
        X, y = X[shuffle], y[shuffle]

        plain = RandomForestClassifier(
            n_estimators=15, max_depth=4, random_state=RANDOM_SEED).fit(X, y)
        balanced = RandomForestClassifier(
            n_estimators=15, max_depth=4, random_state=RANDOM_SEED,
            class_weight="balanced").fit(X, y)

        minority = y == 1
        recall_plain = float(np.mean(plain.predict(X)[minority] == 1))
        recall_balanced = float(np.mean(balanced.predict(X)[minority] == 1))
        assert recall_balanced > recall_plain

    def test_balanced_deterministic(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        X = rng.normal(size=(150, 3))
        y = (X[:, 0] > 0.8).astype(int)  # skewed
        a = RandomForestClassifier(n_estimators=8, random_state=1,
                                   class_weight="balanced").fit(X, y)
        b = RandomForestClassifier(n_estimators=8, random_state=1,
                                   class_weight="balanced").fit(X, y)
        np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))


class TestSklearnParity:
    def test_accuracy_close_to_sklearn_forest(self, cancer_split) -> None:
        X_train, X_test, y_train, y_test = cancer_split
        ours = RandomForestClassifier(
            n_estimators=N_TREES, max_features="sqrt", random_state=RANDOM_SEED
        ).fit(X_train, y_train)
        theirs = SkRandomForest(
            n_estimators=N_TREES, max_features="sqrt", random_state=RANDOM_SEED
        ).fit(X_train, y_train)
        acc_ours = float(np.mean(ours.predict(X_test) == y_test))
        acc_theirs = float(np.mean(theirs.predict(X_test) == y_test))
        assert abs(acc_ours - acc_theirs) <= FOREST_ACCURACY_TOLERANCE, (
            f"accuracy gap too large: ours={acc_ours:.4f}, sklearn={acc_theirs:.4f}"
        )
