from __future__ import annotations

import numpy as np

from src.bagging import RandomForestClassifier
from src.trees import DecisionTree


def _toy_dataset() -> tuple[np.ndarray, np.ndarray]:
	X = np.array(
		[
			[0.0, 0.0],
			[0.0, 1.0],
			[1.0, 0.0],
			[1.0, 1.0],
			[0.1, 0.2],
			[0.9, 0.8],
		],
		dtype=float,
	)
	y = np.array([0, 1, 1, 0, 0, 1])
	return X, y


def test_decision_tree_predicts_and_exposes_structure() -> None:
	X, y = _toy_dataset()

	tree = DecisionTree(max_depth=2, random_state=0).fit(X, y)

	np.testing.assert_array_equal(tree.predict(X), y)
	probabilities = tree.predict_proba(X)
	assert probabilities.shape == (len(X), 2)
	np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
	assert tree.depth <= 2
	assert tree.n_leaves >= 2
	np.testing.assert_allclose(tree.feature_importances().sum(), 1.0)


def test_random_forest_is_deterministic_and_reports_oob_score() -> None:
	X, y = _toy_dataset()

	forest_one = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0, oob_score=True).fit(X, y)
	forest_two = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0, oob_score=True).fit(X, y)

	np.testing.assert_array_equal(forest_one.predict(X), forest_two.predict(X))
	np.testing.assert_allclose(forest_one.predict_proba(X), forest_two.predict_proba(X))
	np.testing.assert_allclose(forest_one.feature_importances_.sum(), 1.0)
	assert 0.0 <= forest_one.oob_score_ <= 1.0
