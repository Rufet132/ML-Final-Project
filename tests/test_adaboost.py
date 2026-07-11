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