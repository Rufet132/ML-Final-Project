from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Node:
	feature_index: Optional[int] = None
	threshold: Optional[float] = None
	left: Optional["Node"] = None
	right: Optional["Node"] = None
	value: Optional[np.ndarray] = None
	samples: int = 0
	impurity: float = 0.0

	@property
	def is_leaf(self) -> bool:
		return self.left is None and self.right is None


class DecisionTree:
	def __init__(
		self,
		max_depth: int | None = None,
		min_samples_split: int = 2,
		criterion: str = "gini",
		max_features: int | str | None = None,
		random_state: int | None = None,
	) -> None:
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.criterion = criterion
		self.max_features = max_features
		self.random_state = random_state

		self.n_features_in_: int | None = None
		self.n_classes_: int | None = None
		self.classes_: np.ndarray | None = None
		self.root_: Node | None = None
		self._feature_importances: np.ndarray | None = None
		self._rng = np.random.default_rng(random_state)

	def fit(
		self,
		X: np.ndarray,
		y: np.ndarray,
		sample_weight: np.ndarray | None = None,
	) -> "DecisionTree":
		X = np.asarray(X, dtype=float)
		y = np.asarray(y)

		if X.ndim != 2:
			raise ValueError("X must be a 2D array")
		if len(X) != len(y):
			raise ValueError("X and y must have the same number of samples")

		self.n_features_in_ = X.shape[1]
		self.classes_, y_encoded = np.unique(y, return_inverse=True)
		self.n_classes_ = len(self.classes_)

		if sample_weight is None:
			sample_weight = np.ones(len(y), dtype=float)
		else:
			sample_weight = np.asarray(sample_weight, dtype=float)
			if sample_weight.shape != (len(y),):
				raise ValueError("sample_weight must have shape (n_samples,)")

		self._rng = np.random.default_rng(self.random_state)
		self._feature_importances = np.zeros(self.n_features_in_, dtype=float)
		self.root_ = self._build_tree(X, y_encoded, sample_weight, depth=0)

		total_reduction = float(self._feature_importances.sum())
		if total_reduction > 0:
			self._feature_importances /= total_reduction
		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		proba = self.predict_proba(X)
		class_indices = np.argmax(proba, axis=1)
		return self.classes_[class_indices]

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		if self.root_ is None or self.classes_ is None:
			raise ValueError("DecisionTree must be fit before calling predict_proba")

		X = np.asarray(X, dtype=float)
		predictions = np.zeros((len(X), self.n_classes_), dtype=float)
		for i, row in enumerate(X):
			node = self.root_
			while node is not None and not node.is_leaf:
				assert node.feature_index is not None
				assert node.threshold is not None
				node = node.left if row[node.feature_index] <= node.threshold else node.right
			counts = node.value if node is not None and node.value is not None else np.ones(self.n_classes_)
			total = counts.sum()
			predictions[i] = counts / total if total > 0 else np.full(self.n_classes_, 1.0 / self.n_classes_)
		return predictions

	@property
	def depth(self) -> int:
		def _depth(node: Node | None) -> int:
			if node is None or node.is_leaf:
				return 0
			return 1 + max(_depth(node.left), _depth(node.right))

		return _depth(self.root_)

	@property
	def n_leaves(self) -> int:
		def _count(node: Node | None) -> int:
			if node is None:
				return 0
			if node.is_leaf:
				return 1
			return _count(node.left) + _count(node.right)

		return _count(self.root_)

	def feature_importances(self) -> np.ndarray:
		if self._feature_importances is None:
			raise ValueError("DecisionTree must be fit before feature_importances")
		return self._feature_importances.copy()

	def __repr__(self) -> str:
		if self.root_ is None:
			return "DecisionTree(unfitted)"
		if self.depth > 4:
			return f"DecisionTree(depth={self.depth}, n_leaves={self.n_leaves})"

		lines: list[str] = []

		def _render(node: Node, indent: int) -> None:
			prefix = "  " * indent
			counts = np.array2string(node.value, precision=3, separator=", ") if node.value is not None else "[]"
			if node.is_leaf:
				lines.append(f"{prefix}Leaf(samples={node.samples}, impurity={node.impurity:.4f}, value={counts})")
				return
			lines.append(
				f"{prefix}X[{node.feature_index}] <= {node.threshold:.6f} "
				f"(samples={node.samples}, impurity={node.impurity:.4f}, value={counts})"
			)
			assert node.left is not None and node.right is not None
			_render(node.left, indent + 1)
			_render(node.right, indent + 1)

		_render(self.root_, 0)
		return "DecisionTree(\n" + "\n".join(lines) + "\n)"

	def _build_tree(
		self,
		X: np.ndarray,
		y: np.ndarray,
		sample_weight: np.ndarray,
		depth: int,
	) -> Node:
		counts = np.bincount(y, weights=sample_weight, minlength=self.n_classes_)
		node_impurity = self._impurity_from_counts(counts)
		node = Node(value=counts, samples=len(y), impurity=node_impurity)

		if self._should_stop(X, y, depth):
			return node

		feature_indices = self._select_features(X.shape[1])
		best_split = self._best_split(X, y, sample_weight, feature_indices, node_impurity)
		if best_split is None:
			return node

		feature_index, threshold, gain = best_split
		left_mask = X[:, feature_index] <= threshold
		right_mask = ~left_mask
		if not left_mask.any() or not right_mask.any():
			return node

		self._feature_importances[feature_index] += gain

		node.feature_index = feature_index
		node.threshold = threshold
		node.left = self._build_tree(X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1)
		node.right = self._build_tree(X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1)
		return node

	def _should_stop(self, X: np.ndarray, y: np.ndarray, depth: int) -> bool:
		if self.max_depth is not None and depth >= self.max_depth:
			return True
		if len(y) < self.min_samples_split:
			return True
		if np.unique(y).size == 1:
			return True
		if X.shape[0] > 1 and np.all(np.all(X == X[0], axis=1)):
			return True
		return False

	def _select_features(self, n_features: int) -> np.ndarray:
		if self.max_features is None:
			k = n_features
		elif isinstance(self.max_features, str):
			if self.max_features == "sqrt":
				k = max(1, int(np.sqrt(n_features)))
			elif self.max_features == "log2":
				k = max(1, int(np.log2(n_features)))
			else:
				raise ValueError("max_features must be int, 'sqrt', 'log2', or None")
		else:
			k = max(1, min(int(self.max_features), n_features))

		if k >= n_features:
			return np.arange(n_features)
		return self._rng.choice(n_features, size=k, replace=False)

	def _best_split(
		self,
		X: np.ndarray,
		y: np.ndarray,
		sample_weight: np.ndarray,
		feature_indices: np.ndarray,
		parent_impurity: float,
	) -> tuple[int, float, float] | None:
		best_gain = 0.0
		best_feature = None
		best_threshold = None

		total_weight = sample_weight.sum()
		if total_weight <= 0:
			return None

		for feature_index in feature_indices:
			values = X[:, feature_index]
			order = np.argsort(values, kind="mergesort")
			sorted_values = values[order]
			sorted_labels = y[order]
			sorted_weights = sample_weight[order]

			if np.all(sorted_values == sorted_values[0]):
				continue

			weighted_onehot = np.zeros((len(y), self.n_classes_), dtype=float)
			weighted_onehot[np.arange(len(y)), sorted_labels] = sorted_weights
			cumulative = np.cumsum(weighted_onehot, axis=0)
			total_counts = cumulative[-1]

			for split_index in range(len(y) - 1):
				if sorted_values[split_index] == sorted_values[split_index + 1]:
					continue

				left_counts = cumulative[split_index]
				right_counts = total_counts - left_counts
				left_weight = left_counts.sum()
				right_weight = right_counts.sum()
				if left_weight <= 0 or right_weight <= 0:
					continue

				left_impurity = self._impurity_from_counts(left_counts)
				right_impurity = self._impurity_from_counts(right_counts)
				gain = parent_impurity - (left_weight / total_weight) * left_impurity - (right_weight / total_weight) * right_impurity

				if gain > best_gain:
					best_gain = gain
					best_feature = int(feature_index)
					best_threshold = float((sorted_values[split_index] + sorted_values[split_index + 1]) / 2.0)

		if best_feature is None or best_threshold is None or best_gain <= 0:
			return None
		return best_feature, best_threshold, best_gain

	def _impurity_from_counts(self, counts: np.ndarray) -> float:
		total = counts.sum()
		if total <= 0:
			return 0.0

		probabilities = counts / total
		if self.criterion == "gini":
			return 1.0 - float(np.sum(probabilities**2))
		if self.criterion == "entropy":
			eps = 1e-12
			return -float(np.sum(probabilities * np.log2(probabilities + eps)))
		raise ValueError("criterion must be 'gini' or 'entropy'")
