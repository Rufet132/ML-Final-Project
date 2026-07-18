from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

import numpy as np

from src.trees.decision_tree import DecisionTree


@dataclass
class _TreeResult:
	tree: DecisionTree
	oob_mask: np.ndarray


def _fit_single_tree(
	args: tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any], int, int]
) -> _TreeResult:
	X, y, sample_weight, tree_params, tree_seed, n_samples = args
	rng = np.random.default_rng(tree_seed)

	bootstrap = tree_params.pop("bootstrap")
	if bootstrap:
		sample_indices = rng.integers(0, n_samples, size=n_samples)
		counts = np.bincount(sample_indices, minlength=n_samples)
		oob_mask = counts == 0
		X_train = X[sample_indices]
		y_train = y[sample_indices]
		weight_train = sample_weight[sample_indices] if sample_weight is not None else None
	else:
		oob_mask = np.zeros(n_samples, dtype=bool)
		X_train = X
		y_train = y
		weight_train = sample_weight

	tree = DecisionTree(**tree_params)
	tree.fit(X_train, y_train, sample_weight=weight_train)
	return _TreeResult(tree=tree, oob_mask=oob_mask)


class RandomForestClassifier:
	def __init__(
		self,
		n_estimators: int = 100,
		max_depth: int | None = None,
		max_features: int | str = "sqrt",
		min_samples_split: int = 2,
		bootstrap: bool = True,
		oob_score: bool = False,
		n_jobs: int = 1,
		random_state: int | None = None,
		class_weight: str | None = None,
	) -> None:
		if class_weight not in (None, "balanced"):
			raise ValueError("class_weight must be None or 'balanced'")
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.max_features = max_features
		self.min_samples_split = min_samples_split
		self.bootstrap = bootstrap
		self.oob_score = oob_score
		self.n_jobs = n_jobs
		self.random_state = random_state
		self.class_weight = class_weight

		self.estimators_: list[DecisionTree] = []
		self.classes_: np.ndarray | None = None
		self._class_to_index: dict[Any, int] = {}
		self.n_features_in_: int | None = None
		self._feature_importances: np.ndarray | None = None
		self._oob_score: float | None = None
		self._oob_masks: list[np.ndarray] = []
		self._y_train: np.ndarray | None = None

	def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
		X = np.asarray(X, dtype=float)
		y = np.asarray(y)

		if X.ndim != 2:
			raise ValueError("X must be a 2D array")
		if len(X) != len(y):
			raise ValueError("X and y must have the same number of samples")

		self.n_features_in_ = X.shape[1]
		self.classes_ = np.unique(y)
		self._class_to_index = {label: index for index, label in enumerate(self.classes_)}
		self._y_train = y.copy()
		self.estimators_ = []
		self._oob_masks = []
		self._feature_importances = np.zeros(self.n_features_in_, dtype=float)
		self._oob_score = None

		rng = np.random.default_rng(self.random_state)
		tree_seeds = rng.integers(0, np.iinfo(np.int32).max, size=self.n_estimators)

		tree_params = {
			"max_depth": self.max_depth,
			"min_samples_split": self.min_samples_split,
			"criterion": "gini",
			"max_features": self.max_features,
			"random_state": None,
		}
		sample_weight = None
		if self.class_weight == "balanced":
			classes, class_counts = np.unique(y, return_counts=True)
			class_weights = len(y) / (len(classes) * class_counts)
			weight_by_class = dict(zip(classes, class_weights))
			sample_weight = np.array([weight_by_class[label] for label in y], dtype=float)

		jobs = [
			(X, y, sample_weight,
			 {**tree_params, "bootstrap": self.bootstrap, "random_state": int(tree_seed)},
			 int(tree_seed), len(X))
			for tree_seed in tree_seeds
		]

		if self.n_jobs > 1:
			with Pool(processes=self.n_jobs) as pool:
				results = pool.map(_fit_single_tree, jobs)
		else:
			results = [_fit_single_tree(job) for job in jobs]

		for result in results:
			self.estimators_.append(result.tree)
			self._oob_masks.append(result.oob_mask)
			self._feature_importances += result.tree.feature_importances()

		if len(self.estimators_) > 0:
			self._feature_importances /= len(self.estimators_)

		if self.oob_score and self.bootstrap:
			self._oob_score = self._compute_oob_score(X)

		return self

	def predict(self, X: np.ndarray) -> np.ndarray:
		proba = self.predict_proba(X)
		class_indices = np.argmax(proba, axis=1)
		return self.classes_[class_indices]

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		if self.classes_ is None or not self.estimators_:
			raise ValueError("RandomForestClassifier must be fit before predict_proba")

		X = np.asarray(X, dtype=float)
		probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)
		for tree in self.estimators_:
			tree_proba = tree.predict_proba(X)
			aligned = np.zeros_like(probabilities)
			for class_index, class_label in enumerate(tree.classes_):
				target_index = self._class_to_index[class_label]
				aligned[:, target_index] = tree_proba[:, class_index]
			probabilities += aligned
		probabilities /= len(self.estimators_)
		return probabilities

	@property
	def oob_score_(self) -> float:
		if self._oob_score is None:
			raise AttributeError("oob_score_ is only available after fit when oob_score=True")
		return self._oob_score

	@property
	def feature_importances_(self) -> np.ndarray:
		if self._feature_importances is None:
			raise ValueError("RandomForestClassifier must be fit before feature_importances_")
		return self._feature_importances.copy()

	def _compute_oob_score(self, X: np.ndarray) -> float:
		assert self.classes_ is not None
		votes = np.zeros((len(X), len(self.classes_)), dtype=float)
		vote_counts = np.zeros(len(X), dtype=int)

		for tree, oob_mask in zip(self.estimators_, self._oob_masks):
			if not np.any(oob_mask):
				continue
			predictions = tree.predict(X[oob_mask])
			for local_index, sample_index in enumerate(np.flatnonzero(oob_mask)):
				class_index = self._class_to_index[predictions[local_index]]
				votes[sample_index, class_index] += 1
				vote_counts[sample_index] += 1

		valid = vote_counts > 0
		if not np.any(valid):
			return float("nan")

		oob_predictions = self.classes_[np.argmax(votes[valid], axis=1)]
		y_valid = self._y_train[valid] if self._y_train is not None else None
		if y_valid is None:
			raise ValueError("Training labels are not available for OOB scoring")
		return float(np.mean(oob_predictions == y_valid))
