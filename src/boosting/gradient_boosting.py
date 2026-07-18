"""Gradient Boosting classifier with logistic (log-loss) boosting.

Bonus module. Binary gradient boosting following Friedman (2001):
the model maintains an additive score F(x) (log-odds), each round fits a
small CART *regression* tree to the negative gradient of the log-loss
(the residuals ``y - sigmoid(F)``), and each leaf applies a Newton step

    gamma_leaf = sum(residuals) / sum(p * (1 - p))

so a round is a damped Newton update of the log-odds. Contrast with
AdaBoost (see ``adaboost.py``): AdaBoost reweights *samples* and adds
hard votes of classifier stumps, while GBM keeps samples unweighted and
adds real-valued corrections fitted to the *loss gradient* — same
additive-model family, different loss and different way of focusing on
mistakes.

The regression tree here is intentionally minimal (MSE splits over
continuous features, depth-limited); the classification DecisionTree in
``src/trees`` cannot be reused because boosting needs real-valued
targets, not class labels.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np

#: Numerical floor for the Newton-step denominator sum(p * (1 - p)).
_HESSIAN_EPSILON: float = 1e-10

#: Clip for initial class prior, avoiding log(0) on degenerate inputs.
_PRIOR_CLIP: float = 1e-10


class _RegressionNode:
    """One node of the internal regression tree."""

    __slots__ = ("feature_index", "threshold", "left", "right", "value")

    def __init__(self) -> None:
        self.feature_index: Optional[int] = None
        self.threshold: float = 0.0
        self.left: Optional["_RegressionNode"] = None
        self.right: Optional["_RegressionNode"] = None
        self.value: float = 0.0

    @property
    def is_leaf(self) -> bool:
        """Return True when this node has no children."""
        return self.left is None


class _RegressionTree:
    """Minimal CART regression tree (MSE splits, midpoint thresholds).

    Follows the same split-search scheme as the project's decision tree:
    per feature the column is sorted once and prefix sums of the targets
    give every candidate split's MSE reduction in one vectorized pass.
    Leaf values start as target means and are overwritten by the GBM's
    Newton step through :meth:`set_leaf_values`.
    """

    def __init__(self, max_depth: int, min_samples_split: int) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root_: Optional[_RegressionNode] = None
        self._leaves: List[_RegressionNode] = []

    def fit(self, X: np.ndarray, target: np.ndarray) -> "_RegressionTree":
        """Grow the tree on continuous targets; returns self."""
        self._leaves = []
        self.root_ = self._grow(X, target, np.arange(X.shape[0]), depth=0)
        return self

    def _grow(self, X: np.ndarray, target: np.ndarray,
              indices: np.ndarray, depth: int) -> _RegressionNode:
        """Recursively split ``indices``; leaves remember their samples."""
        node = _RegressionNode()
        node.value = float(target[indices].mean())

        if (depth >= self.max_depth
                or indices.shape[0] < self.min_samples_split):
            self._register_leaf(node, indices)
            return node

        split = self._best_split(X, target, indices)
        if split is None:
            self._register_leaf(node, indices)
            return node

        feature_index, threshold, left_index, right_index = split
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._grow(X, target, left_index, depth + 1)
        node.right = self._grow(X, target, right_index, depth + 1)
        return node

    def _register_leaf(self, node: _RegressionNode, indices: np.ndarray) -> None:
        """Track leaf order and the training samples that reached it."""
        node_id = len(self._leaves)
        self._leaves.append(node)
        if not hasattr(self, "_leaf_samples"):
            self._leaf_samples: List[np.ndarray] = []
        self._leaf_samples.append(indices)
        del node_id  # leaf identity is its list position

    def _best_split(
        self, X: np.ndarray, target: np.ndarray, indices: np.ndarray
    ) -> Optional[Tuple[int, float, np.ndarray, np.ndarray]]:
        """Best MSE-reducing split, or None when no split helps.

        Maximizing SSE reduction is equivalent to maximizing
        ``S_L^2/n_L + S_R^2/n_R`` where ``S`` are child target sums, so
        only prefix sums are needed per candidate threshold.
        """
        y_node = target[indices]
        n_samples = indices.shape[0]
        total_sum = float(y_node.sum())
        baseline = total_sum * total_sum / n_samples

        best_gain = 0.0
        best: Optional[Tuple[int, float, np.ndarray, np.ndarray]] = None

        for feature_index in range(X.shape[1]):
            values = X[indices, feature_index]
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            boundary = np.nonzero(sorted_values[:-1] < sorted_values[1:])[0]
            if boundary.shape[0] == 0:
                continue

            prefix = np.cumsum(y_node[order])
            left_sum = prefix[boundary]
            left_count = boundary + 1
            right_sum = total_sum - left_sum
            right_count = n_samples - left_count

            scores = (left_sum**2 / left_count + right_sum**2 / right_count
                      - baseline)
            best_local = int(np.argmax(scores))
            if scores[best_local] > best_gain:
                position = boundary[best_local]
                best_gain = float(scores[best_local])
                threshold = float(
                    (sorted_values[position] + sorted_values[position + 1]) / 2.0
                )
                best = (feature_index, threshold,
                        indices[order[: position + 1]],
                        indices[order[position + 1:]])
        return best

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return the leaf index (position in ``self._leaves``) per sample."""
        assert self.root_ is not None
        leaf_ids = {id(leaf): i for i, leaf in enumerate(self._leaves)}
        result = np.empty(X.shape[0], dtype=np.intp)
        for row in range(X.shape[0]):
            node: _RegressionNode = self.root_
            while not node.is_leaf:
                assert node.left is not None and node.right is not None
                node = (node.left if X[row, node.feature_index] <= node.threshold
                        else node.right)
            result[row] = leaf_ids[id(node)]
        return result

    def set_leaf_values(self, values: np.ndarray) -> None:
        """Overwrite leaf outputs (the GBM Newton step) in leaf order."""
        for leaf, value in zip(self._leaves, values):
            leaf.value = float(value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Leaf output per sample."""
        leaf_values = np.array([leaf.value for leaf in self._leaves])
        return leaf_values[self.apply(X)]

    @property
    def leaf_samples(self) -> List[np.ndarray]:
        """Training-sample indices that reached each leaf, in leaf order."""
        return self._leaf_samples

    @property
    def n_leaves(self) -> int:
        """Number of leaves."""
        return len(self._leaves)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


class GradientBoostingClassifier:
    """Binary gradient boosting with log-loss (bonus model).

    Parameters
    ----------
    n_estimators:
        Number of boosting rounds (regression trees).
    learning_rate:
        Shrinkage applied to every leaf's Newton step. Smaller values
        need more rounds but generalize better (Friedman 2001).
    max_depth:
        Depth of each regression tree; shallow trees (default 3) keep
        each round a weak learner, mirroring AdaBoost's stumps.
    min_samples_split:
        Minimum node size in the regression trees.
    random_state:
        Accepted for interface symmetry with the other ensembles; the
        basic GBM here is fully deterministic (no subsampling), so the
        seed has no effect and is stored only for reporting.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.classes_: Optional[np.ndarray] = None
        self._trees: List[_RegressionTree] = []
        self._initial_score: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        """Fit the boosting chain on binary labels.

        Args:
            X: Feature matrix ``(n_samples, n_features)``.
            y: Binary labels (any two distinct values).

        Returns:
            The fitted estimator (``self``).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching n_samples")

        self.classes_ = np.unique(y)
        if self.classes_.shape[0] != 2:
            raise ValueError("GradientBoostingClassifier supports exactly 2 classes")
        y01 = (y == self.classes_[1]).astype(np.float64)

        # F_0 = log-odds of the positive class (the loss-minimizing constant).
        prior = float(np.clip(y01.mean(), _PRIOR_CLIP, 1.0 - _PRIOR_CLIP))
        self._initial_score = float(np.log(prior / (1.0 - prior)))
        scores = np.full(X.shape[0], self._initial_score)

        self._trees = []
        for _ in range(self.n_estimators):
            probabilities = _sigmoid(scores)
            residuals = y01 - probabilities  # negative log-loss gradient

            tree = _RegressionTree(self.max_depth, self.min_samples_split)
            tree.fit(X, residuals)

            # Newton step per leaf: sum(residual) / sum(p * (1 - p)).
            hessian = probabilities * (1.0 - probabilities)
            leaf_values = np.empty(tree.n_leaves)
            for leaf_index, samples in enumerate(tree.leaf_samples):
                denominator = float(hessian[samples].sum())
                numerator = float(residuals[samples].sum())
                leaf_values[leaf_index] = numerator / max(denominator,
                                                          _HESSIAN_EPSILON)
            tree.set_leaf_values(leaf_values)

            scores += self.learning_rate * tree.predict(X)
            self._trees.append(tree)

        return self

    def _decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Accumulated log-odds F(x) for every sample."""
        if self.classes_ is None or not self._trees:
            raise RuntimeError("GradientBoostingClassifier is not fitted yet")
        X = np.asarray(X, dtype=np.float64)
        scores = np.full(X.shape[0], self._initial_score)
        for tree in self._trees:
            scores += self.learning_rate * tree.predict(X)
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities ``(n_samples, 2)`` in ``classes_`` order."""
        positive = _sigmoid(self._decision_scores(X))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most probable class label per sample."""
        positive = _sigmoid(self._decision_scores(X)) >= 0.5
        assert self.classes_ is not None  # guaranteed by _decision_scores
        return self.classes_[positive.astype(int)]

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after each boosting round (for curves)."""
        if self.classes_ is None or not self._trees:
            raise RuntimeError("GradientBoostingClassifier is not fitted yet")
        X = np.asarray(X, dtype=np.float64)
        scores = np.full(X.shape[0], self._initial_score)
        for tree in self._trees:
            scores += self.learning_rate * tree.predict(X)
            yield self.classes_[(_sigmoid(scores) >= 0.5).astype(int)]
