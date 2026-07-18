"""Random Forest classifier built on the team's from-scratch DecisionTree.

Implements bootstrap aggregation (bagging) with per-node feature
sub-sampling, out-of-bag (OOB) scoring, optional multiprocessing across
trees, and averaged impurity-based feature importances. Reproducibility
is guaranteed by deriving one child seed per tree from the forest's
``random_state``; the same seed drives both that tree's bootstrap sample
and its internal feature sub-sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

import numpy as np

from src.trees.decision_tree import DecisionTree


@dataclass
class _TreeResult:
    """Fitted tree plus the OOB mask of the bootstrap sample it was trained on."""

    tree: DecisionTree
    oob_mask: np.ndarray


def _fit_single_tree(
    args: tuple[np.ndarray, np.ndarray, "np.ndarray | None", dict[str, Any], int, int]
) -> _TreeResult:
    """Fit one forest member; module-level so multiprocessing can pickle it.

    Args:
        args: Tuple of ``(X, y, sample_weight, tree_params, tree_seed,
            n_samples)``. ``tree_params`` carries the DecisionTree
            constructor arguments (including this tree's own
            ``random_state``) plus a ``bootstrap`` flag consumed here.
            ``tree_seed`` additionally seeds the bootstrap row sampling,
            so a tree's whole training procedure is reproducible from
            that one integer. ``sample_weight`` (or ``None``) follows the
            bootstrap draw so class weighting survives resampling.

    Returns:
        The fitted tree together with a boolean OOB mask (True for
        samples that never appeared in the bootstrap draw; all-False
        when bootstrapping is disabled).
    """
    X, y, sample_weight, tree_params, tree_seed, n_samples = args
    rng = np.random.default_rng(tree_seed)

    bootstrap = tree_params.pop("bootstrap")
    if bootstrap:
        sample_indices = rng.integers(0, n_samples, size=n_samples)
        counts = np.bincount(sample_indices, minlength=n_samples)
        oob_mask = counts == 0
        X_train = X[sample_indices]
        y_train = y[sample_indices]
        w_train = sample_weight[sample_indices] if sample_weight is not None else None
    else:
        oob_mask = np.zeros(n_samples, dtype=bool)
        X_train = X
        y_train = y
        w_train = sample_weight

    tree = DecisionTree(**tree_params)
    tree.fit(X_train, y_train, sample_weight=w_train)
    return _TreeResult(tree=tree, oob_mask=oob_mask)


class RandomForestClassifier:
    """Bagging ensemble of DecisionTrees with feature sub-sampling.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Depth cap forwarded to every tree (``None`` = unbounded).
    max_features:
        Features considered at each split of each tree: an int,
        ``"sqrt"``, ``"log2"``, or ``None`` (all features).
    min_samples_split:
        Minimum node size forwarded to every tree.
    bootstrap:
        When True each tree trains on an N-sized sample drawn with
        replacement; when False every tree sees the full training set.
    oob_score:
        When True (and ``bootstrap`` is True), compute the out-of-bag
        accuracy after fitting and expose it as ``oob_score_``.
    n_jobs:
        Number of worker processes for tree training (1 = sequential).
    random_state:
        Seed for reproducibility; fixes both bootstrap draws and each
        tree's feature sub-sampling.
    class_weight:
        ``"balanced"`` gives every sample the weight
        ``n_samples / (n_classes * count(class))`` so minority classes
        carry the same total weight as the majority; the weighted
        impurity of the underlying trees then treats them fairly. This
        is the imbalance treatment used for the severely skewed
        Covertype dataset. ``None`` (default) keeps uniform weights.
    """

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
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")
        if oob_score and not bootstrap:
            raise ValueError("oob_score requires bootstrap=True")
        if class_weight not in (None, "balanced"):
            raise ValueError('class_weight must be None or "balanced"')
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
        """Train ``n_estimators`` trees on bootstrap samples of ``(X, y)``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Class labels of shape ``(n_samples,)``.

        Returns:
            The fitted estimator (``self``).
        """
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

        sample_weight: np.ndarray | None = None
        if self.class_weight == "balanced":
            _, class_counts = np.unique(y, return_counts=True)
            weight_per_class = len(y) / (len(class_counts) * class_counts.astype(float))
            encoded = np.searchsorted(self.classes_, y)
            sample_weight = weight_per_class[encoded]

        tree_params = {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "criterion": "gini",
            "max_features": self.max_features,
        }

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
        """Return the majority-vote class label for each sample in ``X``."""
        proba = self.predict_proba(X)
        assert self.classes_ is not None  # predict_proba already validated fit
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average the class-probability vectors of all trees.

        Each tree may have seen only a subset of the classes in its
        bootstrap sample, so per-tree probabilities are re-aligned to the
        forest-level ``classes_`` order before averaging.

        Returns:
            Array of shape ``(n_samples, n_classes)`` whose rows sum to 1.
        """
        if self.classes_ is None or not self.estimators_:
            raise ValueError("RandomForestClassifier must be fit before predict_proba")

        X = np.asarray(X, dtype=float)
        probabilities = np.zeros((len(X), len(self.classes_)), dtype=float)
        for tree in self.estimators_:
            tree_proba = tree.predict_proba(X)
            aligned = np.zeros_like(probabilities)
            assert tree.classes_ is not None  # every stored tree is fitted
            for class_index, class_label in enumerate(tree.classes_):
                target_index = self._class_to_index[class_label]
                aligned[:, target_index] = tree_proba[:, class_index]
            probabilities += aligned
        probabilities /= len(self.estimators_)
        return probabilities

    @property
    def oob_score_(self) -> float:
        """Out-of-bag accuracy (available after fit with ``oob_score=True``)."""
        if self._oob_score is None:
            raise AttributeError("oob_score_ is only available after fit when oob_score=True")
        return self._oob_score

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean impurity-based importances across all trees (sums to 1)."""
        if self._feature_importances is None:
            raise ValueError("RandomForestClassifier must be fit before feature_importances_")
        return self._feature_importances.copy()

    def _compute_oob_score(self, X: np.ndarray) -> float:
        """Accuracy of majority votes restricted to each sample's OOB trees.

        For every sample, only trees whose bootstrap draw excluded that
        sample cast a vote; samples that were in-bag for every tree are
        left out of the score entirely (NaN is returned if no sample has
        any OOB vote, which can only happen for very small forests).
        """
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
