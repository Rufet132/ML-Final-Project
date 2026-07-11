"""Tests for Module 2: DecisionStump / AdaBoostClassifier.
 
Run with: pytest tests/test_adaboost.py -v
"""
 
import numpy as np
import pytest
 
from src.trees.decision_tree import DecisionTree
from src.trees.adaboost import DecisionStump, AdaBoostClassifier
 
 
# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------
@pytest.fixture
def separable_binary_data():
    """Linearly separable 2D data, labels {0, 1}."""
    rng = np.random.RandomState(0)
    X0 = rng.normal(loc=[-2, -2], scale=0.5, size=(50, 2))
    X1 = rng.normal(loc=[2, 2], scale=0.5, size=(50, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y
 
 
@pytest.fixture
def separable_pm1_data(separable_binary_data):
    """Same data, labels {-1, +1}."""
    X, y01 = separable_binary_data
    y = np.where(y01 == 0, -1, 1)
    return X, y
 
 
@pytest.fixture
def moons_like_data():
    """Harder, non-linearly-separable data for a more realistic sanity check."""
    from sklearn.datasets import make_moons
 
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    return X, y

#----------------------------------------------------------------------
# Weighted impurity / weighted stump fitting
# ----------------------------------------------------------------------
class TestWeightedImpurity:
    def test_uniform_weights_match_unweighted(self, separable_binary_data):
        X, y = separable_binary_data
        tree_unweighted = DecisionTree(max_depth=3, random_state=1).fit(X, y)
        tree_weighted = DecisionTree(max_depth=3, random_state=1).fit(
            X, y, sample_weight=np.ones(len(y))
        )
        np.testing.assert_array_equal(
            tree_unweighted.predict(X), tree_weighted.predict(X)
        )
 
    def test_upweighting_a_class_biases_split(self):
        # Two samples per class; heavily up-weight one class so a stump
        # trained on weighted Gini should favor separating it out first.
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        weight = np.array([1.0, 1.0, 1.0, 100.0])  # last sample dominates
 
        stump = DecisionStump(random_state=0)
        stump.fit(X, y, sample_weight=weight)
        # The dominant sample (class 1, x=3.0) must be classified correctly.
        assert stump.predict(np.array([[3.0]]))[0] == 1
 
    def test_weighted_gini_formula_matches_manual_computation(self):
        tree = DecisionTree(criterion="gini")
        counts = np.array([[3.0, 1.0]])  # p = [0.75, 0.25]
        totals = np.array([4.0])
        expected = 1 - (0.75 ** 2 + 0.25 ** 2)
        assert tree._impurity(counts, totals)[0] == pytest.approx(expected)
 
    def test_weighted_entropy_formula_matches_manual_computation(self):
        tree = DecisionTree(criterion="entropy")
        counts = np.array([[2.0, 2.0]])  # p = [0.5, 0.5]
        totals = np.array([4.0])
        expected = -(0.5 * np.log2(0.5) + 0.5 * np.log2(0.5))
        assert tree._impurity(counts, totals)[0] == pytest.approx(expected, abs=1e-6)