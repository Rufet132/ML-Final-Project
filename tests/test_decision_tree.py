"""Pytest suite for the from-scratch :class:`DecisionTree` (Module 1).

scikit-learn is imported here *only* as a reference baseline, which is
explicitly permitted for tests; the implementation under test uses NumPy
alone. The suite covers:

* a manually verifiable split on a tiny 2D two-Gaussians dataset,
* edge cases (single feature, constant labels, ``min_samples_split=1``,
  ``max_depth=0`` majority-vote leaf and ``max_depth=1`` decision stump),
* sample-weight semantics for AdaBoost compatibility,
* determinism under ``random_state`` with feature sub-sampling,
* a sanity comparison against ``sklearn.tree.DecisionTreeClassifier`` on
  the breast-cancer dataset (accuracy agreement within 2%).
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.trees.decision_tree import (
    CRITERION_ENTROPY,
    CRITERION_GINI,
    DecisionTree,
    Node,
    REPR_MAX_DEPTH,
)

# --------------------------------------------------------------------------- #
# Test-suite constants (no magic numbers inline).
# --------------------------------------------------------------------------- #

#: Seed shared by every randomized component of the suite.
RANDOM_SEED: int = 42

#: Depth cap used in the sklearn head-to-head comparison.
BENCHMARK_MAX_DEPTH: int = 5

#: Maximum tolerated absolute accuracy gap versus sklearn (the 2% rubric).
ACCURACY_TOLERANCE: float = 0.02

#: Fraction of the benchmark dataset held out for evaluation.
TEST_FRACTION: float = 0.3

#: Absolute tolerance for probability-sum and threshold checks.
FLOAT_ATOL: float = 1e-9

#: Expected root threshold of the manually constructed dataset:
#: midpoint between the largest class-0 value (3.0) and the smallest
#: class-1 value (10.0) on feature 0.
EXPECTED_ROOT_THRESHOLD: float = (3.0 + 10.0) / 2.0

#: Minimum fraction of importance mass sklearn and our tree must agree on
#: for the benchmark's most important feature set.
IMPORTANCE_TOP_K: int = 5


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def separable_2d() -> tuple:
    """Tiny 2D dataset with one informative and one constant feature.

    Feature 0 perfectly separates the classes between the values 3.0 and
    10.0, so the unique optimal root split is
    ``feature_0 <= (3.0 + 10.0) / 2 = 6.5``. Feature 1 is constant and can
    never be selected.
    """
    X = np.array(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
            [10.0, 5.0],
            [11.0, 5.0],
            [12.0, 5.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture(scope="module")
def breast_cancer_split() -> tuple:
    """Deterministic train/test split of the breast-cancer dataset."""
    data = load_breast_cancer()
    return train_test_split(
        data.data,
        data.target,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=data.target,
    )


# --------------------------------------------------------------------------- #
# 1. Manually verifiable split on a tiny dataset
# --------------------------------------------------------------------------- #


class TestKnownSplit:
    """The root split of a hand-checkable dataset must be exact."""

    def test_root_split_feature_and_threshold(self, separable_2d) -> None:
        X, y = separable_2d
        tree = DecisionTree(criterion=CRITERION_GINI).fit(X, y)
        assert tree.root_.feature_index == 0
        assert tree.root_.threshold == pytest.approx(
            EXPECTED_ROOT_THRESHOLD, abs=FLOAT_ATOL
        )

    def test_perfect_fit_shape_and_structure(self, separable_2d) -> None:
        X, y = separable_2d
        tree = DecisionTree().fit(X, y)
        np.testing.assert_array_equal(tree.predict(X), y)
        assert tree.depth == 1
        assert tree.n_leaves == 2

    def test_constant_feature_gets_zero_importance(self, separable_2d) -> None:
        X, y = separable_2d
        tree = DecisionTree().fit(X, y)
        importances = tree.feature_importances()
        assert importances.shape == (X.shape[1],)
        assert importances[0] == pytest.approx(1.0)
        assert importances[1] == pytest.approx(0.0)

    def test_predict_proba_rows_are_distributions(self, separable_2d) -> None:
        X, y = separable_2d
        tree = DecisionTree().fit(X, y)
        proba = tree.predict_proba(X)
        assert proba.shape == (X.shape[0], np.unique(y).shape[0])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=FLOAT_ATOL)
        # Pure leaves must yield one-hot probabilities.
        np.testing.assert_allclose(proba.max(axis=1), 1.0, atol=FLOAT_ATOL)

    def test_entropy_criterion_finds_same_split(self, separable_2d) -> None:
        X, y = separable_2d
        tree = DecisionTree(criterion=CRITERION_ENTROPY).fit(X, y)
        assert tree.root_.feature_index == 0
        assert tree.root_.threshold == pytest.approx(
            EXPECTED_ROOT_THRESHOLD, abs=FLOAT_ATOL
        )
        np.testing.assert_array_equal(tree.predict(X), y)

    def test_repr_renders_indented_tree_for_shallow_trees(
        self, separable_2d
    ) -> None:
        X, y = separable_2d
        tree = DecisionTree().fit(X, y)
        text = repr(tree)
        assert tree.depth <= REPR_MAX_DEPTH
        assert "feature_0" in text
        assert f"{EXPECTED_ROOT_THRESHOLD:.4f}" in text
        assert "samples=6" in text
        assert "leaf" in text


# --------------------------------------------------------------------------- #
# 2. Edge cases
# --------------------------------------------------------------------------- #


class TestEdgeCases:
    """Stopping criteria and degenerate configurations."""

    def test_single_feature_dataset(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTree().fit(X, y)
        np.testing.assert_array_equal(tree.predict(X), y)
        assert tree.root_.feature_index == 0
        assert tree.root_.threshold == pytest.approx((1.0 + 2.0) / 2.0)

    def test_all_labels_identical_yields_single_leaf(self) -> None:
        X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        y = np.array([7, 7, 7])
        tree = DecisionTree().fit(X, y)
        assert tree.depth == 0
        assert tree.n_leaves == 1
        np.testing.assert_array_equal(tree.predict(X), y)
        np.testing.assert_allclose(tree.predict_proba(X), 1.0)
        # No split happened, so no importance mass was distributed.
        np.testing.assert_allclose(tree.feature_importances(), 0.0)

    def test_identical_feature_vectors_yield_leaf(self) -> None:
        X = np.ones((4, 3))
        y = np.array([0, 1, 0, 1])
        tree = DecisionTree().fit(X, y)
        assert tree.depth == 0
        assert tree.n_leaves == 1

    def test_min_samples_split_one_grows_pure_tree(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        X = rng.normal(size=(40, 3))
        y = rng.integers(0, 2, size=40)
        tree = DecisionTree(min_samples_split=1).fit(X, y)
        # With continuous features and no depth cap, training data must be
        # memorized perfectly (every leaf pure or numerically unsplittable).
        np.testing.assert_array_equal(tree.predict(X), y)

    def test_max_depth_zero_is_majority_vote_leaf(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 0, 1, 1])
        tree = DecisionTree(max_depth=0).fit(X, y)
        assert tree.depth == 0
        assert tree.n_leaves == 1
        np.testing.assert_array_equal(tree.predict(X), np.zeros(5, dtype=int))
        expected_distribution = np.array([3.0 / 5.0, 2.0 / 5.0])
        np.testing.assert_allclose(
            tree.root_.value, expected_distribution, atol=FLOAT_ATOL
        )

    def test_max_depth_one_is_decision_stump(self, separable_2d) -> None:
        X, y = separable_2d
        tree = DecisionTree(max_depth=1).fit(X, y)
        assert tree.depth == 1
        assert tree.n_leaves == 2
        np.testing.assert_array_equal(tree.predict(X), y)

    def test_zero_gain_split_is_rejected(self) -> None:
        # XOR: every axis-aligned split leaves child impurities equal to the
        # parent impurity, so the strict delta-I > 0 rule forces a leaf.
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([0, 1, 1, 0])
        tree = DecisionTree().fit(X, y)
        assert tree.depth == 0
        assert tree.n_leaves == 1


# --------------------------------------------------------------------------- #
# 3. Sample weights (AdaBoost compatibility)
# --------------------------------------------------------------------------- #


class TestSampleWeights:
    """Weighted class probabilities must drive impurity and leaf values."""

    def test_weights_flip_majority_class(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 0, 1])
        heavy_minority = np.array([1.0, 1.0, 1.0, 10.0])
        stump = DecisionTree(max_depth=0).fit(
            X, y, sample_weight=heavy_minority
        )
        # The single weighted class-1 sample dominates the root leaf.
        np.testing.assert_array_equal(
            stump.predict(X), np.ones(4, dtype=int)
        )

    def test_integer_weights_match_sample_repetition(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([0, 0, 1, 1, 0, 0])
        repeats = np.array([1, 1, 3, 3, 1, 1])

        weighted = DecisionTree().fit(X, y, sample_weight=repeats.astype(float))
        repeated = DecisionTree().fit(
            np.repeat(X, repeats, axis=0), np.repeat(y, repeats)
        )

        grid = np.linspace(-1.0, 6.0, 50).reshape(-1, 1)
        np.testing.assert_array_equal(
            weighted.predict(grid), repeated.predict(grid)
        )

    def test_zero_weight_samples_do_not_influence_leaf(self) -> None:
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 1, 1])
        weights = np.array([1.0, 0.0, 0.0])
        stump = DecisionTree(max_depth=0).fit(X, y, sample_weight=weights)
        np.testing.assert_allclose(
            stump.root_.value, np.array([1.0, 0.0]), atol=FLOAT_ATOL
        )


# --------------------------------------------------------------------------- #
# 4. Determinism and feature sub-sampling
# --------------------------------------------------------------------------- #


class TestDeterminism:
    """Fixed random_state must produce bit-identical trees."""

    def test_same_seed_same_tree(self, breast_cancer_split) -> None:
        X_train, _, y_train, _ = breast_cancer_split
        params = dict(
            max_depth=BENCHMARK_MAX_DEPTH,
            max_features="sqrt",
            random_state=RANDOM_SEED,
        )
        tree_a = DecisionTree(**params).fit(X_train, y_train)
        tree_b = DecisionTree(**params).fit(X_train, y_train)
        assert tree_a.root_.feature_index == tree_b.root_.feature_index
        assert tree_a.root_.threshold == tree_b.root_.threshold
        np.testing.assert_array_equal(
            tree_a.predict(X_train), tree_b.predict(X_train)
        )
        np.testing.assert_allclose(
            tree_a.feature_importances(), tree_b.feature_importances()
        )

    def test_max_features_variants_run_and_predict(
        self, breast_cancer_split
    ) -> None:
        X_train, X_test, y_train, _ = breast_cancer_split
        for max_features in ("sqrt", "log2", 3):
            tree = DecisionTree(
                max_depth=BENCHMARK_MAX_DEPTH,
                max_features=max_features,
                random_state=RANDOM_SEED,
            ).fit(X_train, y_train)
            proba = tree.predict_proba(X_test)
            np.testing.assert_allclose(
                proba.sum(axis=1), 1.0, atol=FLOAT_ATOL
            )


# --------------------------------------------------------------------------- #
# 5. Input validation and API contract
# --------------------------------------------------------------------------- #


class TestValidation:
    """Constructor and runtime validation errors."""

    def test_invalid_criterion_raises(self) -> None:
        with pytest.raises(ValueError, match="criterion"):
            DecisionTree(criterion="misclassification")

    def test_invalid_max_features_string_raises(self) -> None:
        with pytest.raises(ValueError, match="max_features"):
            DecisionTree(max_features="cbrt")

    def test_invalid_min_samples_split_raises(self) -> None:
        with pytest.raises(ValueError, match="min_samples_split"):
            DecisionTree(min_samples_split=0)

    def test_invalid_max_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="max_depth"):
            DecisionTree(max_depth=-1)

    def test_invalid_integer_max_features_raises(self) -> None:
        with pytest.raises(ValueError, match="max_features"):
            DecisionTree(max_features=0)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="inconsistent"):
            DecisionTree().fit(np.zeros((3, 2)), np.zeros(4))

    def test_non_2d_x_raises_at_fit(self) -> None:
        with pytest.raises(ValueError, match="2-dimensional"):
            DecisionTree().fit(np.zeros(3), np.zeros(3))

    def test_non_1d_y_raises_at_fit(self) -> None:
        with pytest.raises(ValueError, match="1-dimensional"):
            DecisionTree().fit(np.zeros((3, 2)), np.zeros((3, 1)))

    def test_empty_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            DecisionTree().fit(np.zeros((0, 2)), np.zeros(0))

    def test_wrong_sample_weight_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="sample_weight"):
            DecisionTree().fit(
                np.zeros((3, 1)),
                np.array([0, 1, 0]),
                sample_weight=np.ones(2),
            )

    def test_zero_sum_sample_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="positive sum"):
            DecisionTree().fit(
                np.zeros((2, 1)),
                np.array([0, 1]),
                sample_weight=np.zeros(2),
            )

    def test_non_2d_x_raises_at_predict(self, separable_2d) -> None:
        X, y = separable_2d
        tree = DecisionTree().fit(X, y)
        with pytest.raises(ValueError, match="2-dimensional"):
            tree.predict(np.zeros(2))

    def test_negative_sample_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            DecisionTree().fit(
                np.zeros((2, 1)),
                np.array([0, 1]),
                sample_weight=np.array([1.0, -1.0]),
            )

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="not fitted"):
            DecisionTree().predict(np.zeros((1, 1)))

    def test_predict_with_wrong_feature_count_raises(
        self, separable_2d
    ) -> None:
        X, y = separable_2d
        tree = DecisionTree().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            tree.predict(np.zeros((2, 5)))

    def test_fit_returns_self(self, separable_2d) -> None:
        X, y = separable_2d
        tree = DecisionTree()
        assert tree.fit(X, y) is tree

    def test_unfitted_repr_is_single_line_summary(self) -> None:
        text = repr(DecisionTree(max_depth=BENCHMARK_MAX_DEPTH))
        assert "<unfitted>" in text
        assert "\n" not in text

    def test_deep_tree_repr_is_compact_summary(
        self, breast_cancer_split
    ) -> None:
        X_train, _, y_train, _ = breast_cancer_split
        tree = DecisionTree().fit(X_train, y_train)
        assert tree.depth > REPR_MAX_DEPTH
        text = repr(tree)
        assert "<fitted:" in text
        assert f"depth={tree.depth}" in text
        assert "\n" not in text

    def test_node_defaults_describe_a_leaf(self) -> None:
        node = Node()
        assert node.is_leaf
        assert node.feature_index is None
        assert node.threshold is None
        assert node.samples == 0


# --------------------------------------------------------------------------- #
# 6. Head-to-head sanity check against scikit-learn
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fitted_pair(breast_cancer_split) -> tuple:
    X_train, X_test, y_train, y_test = breast_cancer_split
    ours = DecisionTree(
        max_depth=BENCHMARK_MAX_DEPTH,
        criterion=CRITERION_GINI,
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)
    theirs = DecisionTreeClassifier(
        max_depth=BENCHMARK_MAX_DEPTH,
        criterion=CRITERION_GINI,
        random_state=RANDOM_SEED,
    ).fit(X_train, y_train)
    return ours, theirs, X_test, y_test


class TestSklearnParity:
    """Accuracy, depth, and importances must track sklearn within the rubric."""

    def test_test_accuracy_within_two_percent(self, fitted_pair) -> None:
        ours, theirs, X_test, y_test = fitted_pair
        acc_ours = float(np.mean(ours.predict(X_test) == y_test))
        acc_theirs = float(np.mean(theirs.predict(X_test) == y_test))
        assert abs(acc_ours - acc_theirs) <= ACCURACY_TOLERANCE, (
            f"accuracy gap too large: ours={acc_ours:.4f}, "
            f"sklearn={acc_theirs:.4f}"
        )

    def test_tree_depth_matches_sklearn(self, fitted_pair) -> None:
        ours, theirs, _, _ = fitted_pair
        assert ours.depth == theirs.get_depth()

    def test_feature_importances_track_sklearn(self, fitted_pair) -> None:
        ours, theirs, _, _ = fitted_pair
        mine = ours.feature_importances()
        reference = theirs.feature_importances_
        assert mine.shape == reference.shape
        assert mine.sum() == pytest.approx(1.0)
        assert np.all(mine >= 0.0)
        # The dominant feature of our tree must rank among sklearn's
        # top-K features (greedy tie-breaking may differ further down).
        sklearn_top = set(
            np.argsort(reference)[::-1][:IMPORTANCE_TOP_K].tolist()
        )
        assert int(np.argmax(mine)) in sklearn_top

    def test_entropy_benchmark_within_two_percent(
        self, breast_cancer_split
    ) -> None:
        X_train, X_test, y_train, y_test = breast_cancer_split
        ours = DecisionTree(
            max_depth=BENCHMARK_MAX_DEPTH, criterion=CRITERION_ENTROPY
        ).fit(X_train, y_train)
        theirs = DecisionTreeClassifier(
            max_depth=BENCHMARK_MAX_DEPTH,
            criterion="entropy",
            random_state=RANDOM_SEED,
        ).fit(X_train, y_train)
        acc_ours = float(np.mean(ours.predict(X_test) == y_test))
        acc_theirs = float(np.mean(theirs.predict(X_test) == y_test))
        assert abs(acc_ours - acc_theirs) <= ACCURACY_TOLERANCE
