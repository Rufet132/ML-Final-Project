"""CART-style Decision Tree Classifier implemented from scratch with NumPy.

This module implements a binary decision tree classifier for *continuous*
features supporting:

* Gini impurity and Shannon entropy splitting criteria.
* Sample weights (``sample_weight``), making the tree directly reusable as a
  weak learner inside AdaBoost.
* An O(N log N) per-feature split search: each feature column is sorted once
  and weighted class counts are accumulated cumulatively, so every candidate
  threshold (midpoints between consecutive distinct sorted values) is
  evaluated in a single vectorized pass instead of an O(N^2) exhaustive scan.
* Per-node random feature sub-sampling (``max_features``) for Random Forest
  compatibility, fully deterministic under a fixed ``random_state``.

"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------- #
# Module-level constants (no magic numbers inline).
# --------------------------------------------------------------------------- #

#: Additive constant inside log2 to avoid evaluating log2(0) in the entropy.
EPSILON: float = 1e-7

#: Identifier of the Gini impurity criterion.
CRITERION_GINI: str = "gini"

#: Identifier of the Shannon entropy criterion.
CRITERION_ENTROPY: str = "entropy"

#: The set of supported splitting criteria.
VALID_CRITERIA: frozenset = frozenset({CRITERION_GINI, CRITERION_ENTROPY})

#: Identifier for max_features = floor(sqrt(n_features)).
MAX_FEATURES_SQRT: str = "sqrt"

#: Identifier for max_features = floor(log2(n_features)).
MAX_FEATURES_LOG2: str = "log2"

#: The set of supported string values for ``max_features``.
VALID_MAX_FEATURES_STRINGS: frozenset = frozenset(
    {MAX_FEATURES_SQRT, MAX_FEATURES_LOG2}
)

#: A split is only accepted if its impurity reduction strictly exceeds this.
MIN_IMPURITY_DECREASE: float = 0.0

#: Maximum tree depth for which ``__repr__`` renders the full indented tree.
REPR_MAX_DEPTH: int = 4

#: Indentation unit used when pretty-printing the tree structure.
REPR_INDENT: str = "|   "

#: Branch marker used when pretty-printing the tree structure.
REPR_BRANCH: str = "|--- "


class Node:
    """A single node of a binary decision tree.

    Internal (split) nodes carry a ``feature_index``/``threshold`` pair and
    two children; leaf nodes carry ``left is None and right is None`` and are
    described entirely by their class distribution ``value``.

    Parameters
    ----------
    feature_index:
        Index of the feature this node splits on (``None`` for leaves).
    threshold:
        Split threshold; samples with ``x[feature_index] <= threshold`` are
        routed to the left child (``None`` for leaves).
    left:
        Left child node (``None`` for leaves).
    right:
        Right child node (``None`` for leaves).
    value:
        Weighted class probability distribution at this node, an array of
        shape ``[n_classes]`` that sums to 1.
    samples:
        Number of training samples that reached this node.
    """

    __slots__ = ("feature_index", "threshold", "left", "right", "value", "samples")

    def __init__(
        self,
        feature_index: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[np.ndarray] = None,
        samples: int = 0,
    ) -> None:
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = samples

    @property
    def is_leaf(self) -> bool:
        """Return ``True`` when this node has no children."""
        return self.left is None and self.right is None


class DecisionTree:
    """CART-style binary decision tree classifier for continuous features.

    Parameters
    ----------
    max_depth:
        Maximum depth of the tree. A tree consisting of a single leaf (the
        root) has depth 0. ``None`` means the depth is unbounded.
    min_samples_split:
        Minimum number of samples a node must contain to be considered for
        splitting. Must be at least 1.
    criterion:
        Impurity criterion, either ``"gini"`` or ``"entropy"``.
    max_features:
        Number of features randomly sampled (without replacement) at *each*
        node when searching for the best split. Accepts a positive integer,
        ``"sqrt"`` (``floor(sqrt(n_features))``), ``"log2"``
        (``floor(log2(n_features))``), or ``None`` (use all features).
    random_state:
        Seed for the random generator driving feature sub-sampling. Fitting
        is fully deterministic when this is an integer.

    Attributes
    ----------
    root_:
        Root :class:`Node` of the fitted tree.
    classes_:
        Sorted array of unique class labels seen during :meth:`fit`.
    n_features_:
        Number of features seen during :meth:`fit`.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        max_features: Optional[Union[int, str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if criterion not in VALID_CRITERIA:
            raise ValueError(
                f"criterion must be one of {sorted(VALID_CRITERIA)}, "
                f"got {criterion!r}."
            )
        if max_depth is not None and max_depth < 0:
            raise ValueError(f"max_depth must be >= 0 or None, got {max_depth}.")
        if min_samples_split < 1:
            raise ValueError(
                f"min_samples_split must be >= 1, got {min_samples_split}."
            )
        if isinstance(max_features, str) and max_features not in VALID_MAX_FEATURES_STRINGS:
            raise ValueError(
                f"max_features string must be one of "
                f"{sorted(VALID_MAX_FEATURES_STRINGS)}, got {max_features!r}."
            )
        if isinstance(max_features, (int, np.integer)) and max_features < 1:
            raise ValueError(f"max_features must be >= 1, got {max_features}.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Optional[Node] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self._n_classes: int = 0
        self._rng: Optional[np.random.Generator] = None
        self._importances: Optional[np.ndarray] = None
        self._total_weight: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "DecisionTree":
        """Build the decision tree from the training set ``(X, y)``.

        Parameters
        ----------
        X:
            Feature matrix of shape ``[n_samples, n_features]`` containing
            continuous values.
        y:
            Class labels of shape ``[n_samples]``.
        sample_weight:
            Optional non-negative per-sample weights of shape
            ``[n_samples]``. When ``None``, uniform weights ``1 / n_samples``
            are used. Weighted class probabilities
            ``p_c = sum_{i: y_i = c} w_i / sum_i w_i`` drive both the
            impurity computations and the leaf distributions, which makes
            this tree directly usable as an AdaBoost weak learner.

        Returns
        -------
        DecisionTree
            The fitted estimator (``self``), enabling method chaining.
        """
        X, y, w = self._validate_fit_inputs(X, y, sample_weight)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self._n_classes = self.classes_.shape[0]
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self._importances = np.zeros(self.n_features_, dtype=np.float64)
        self._total_weight = float(w.sum())

        self.root_ = self._grow(X, y_encoded, w, depth=0)

        importance_total = self._importances.sum()
        if importance_total > 0.0:
            self._importances /= importance_total
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for ``X``.

        Parameters
        ----------
        X:
            Feature matrix of shape ``[n_samples, n_features]``.

        Returns
        -------
        np.ndarray
            Predicted class labels of shape ``[n_samples]``, drawn from
            ``classes_``. Ties in the leaf distribution resolve to the class
            with the lowest label (NumPy ``argmax`` semantics), which keeps
            prediction deterministic.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for ``X``.

        Each sample is routed from the root to a leaf and receives that
        leaf's weighted class distribution. Routing is performed batch-wise
        (index arrays flow down the tree), so the cost is
        O(n_samples * depth) with vectorized comparisons at every node.

        Parameters
        ----------
        X:
            Feature matrix of shape ``[n_samples, n_features]``.

        Returns
        -------
        np.ndarray
            Array of shape ``[n_samples, n_classes]`` whose rows sum to 1.
        """
        self._check_fitted()
        X = self._validate_predict_inputs(X)
        proba = np.empty((X.shape[0], self._n_classes), dtype=np.float64)
        self._route(self.root_, X, np.arange(X.shape[0]), proba)
        return proba

    @property
    def depth(self) -> int:
        """Depth of the fitted tree (a single-leaf tree has depth 0)."""
        self._check_fitted()
        return self._subtree_depth(self.root_)

    @property
    def n_leaves(self) -> int:
        """Number of leaves in the fitted tree."""
        self._check_fitted()
        return self._subtree_leaves(self.root_)

    def feature_importances(self) -> np.ndarray:
        """Return normalized impurity-based feature importances.

        Each split contributes ``(W_node / W_total) * delta_impurity`` to the
        importance of its feature, where weights ``W`` are the (sample-weight
        based) node weights. The returned array sums to 1 whenever at least
        one split was made, and is all zeros for a single-leaf tree.

        Returns
        -------
        np.ndarray
            Importance array of shape ``[n_features]``.
        """
        self._check_fitted()
        return self._importances.copy()

    def __repr__(self) -> str:
        """Return a readable representation of the tree.

        For fitted trees of depth <= ``REPR_MAX_DEPTH`` an indented text tree
        is rendered, showing at every node the split feature and threshold
        (internal nodes), the node impurity under the configured criterion,
        the number of training samples, and the class distribution. Deeper or
        unfitted trees render a compact one-line summary.
        """
        header = (
            f"DecisionTree(criterion={self.criterion!r}, "
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"max_features={self.max_features!r}, "
            f"random_state={self.random_state})"
        )
        if self.root_ is None:
            return header + " <unfitted>"
        tree_depth = self.depth
        if tree_depth > REPR_MAX_DEPTH:
            return (
                header
                + f" <fitted: depth={tree_depth}, n_leaves={self.n_leaves}>"
            )
        lines: List[str] = [header]
        self._render(self.root_, level=0, lines=lines)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Tree construction
    # ------------------------------------------------------------------ #

    def _grow(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int
    ) -> Node:
        """Recursively grow the subtree for the samples ``(X, y, w)``.

        The recursion turns the current node into a leaf when any stopping
        criterion holds:

        1. ``max_depth`` has been reached.
        2. The node holds fewer than ``min_samples_split`` samples.
        3. The node is pure (a single class label remains).
        4. All samples share an identical feature vector, so no numerical
           split exists (detected as the absence of any valid threshold).
        5. No candidate split achieves a strictly positive impurity
           reduction.
        """
        node = Node(value=self._distribution(y, w), samples=y.shape[0])

        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if y.shape[0] < self.min_samples_split:
            return node
        if np.all(y == y[0]):
            return node

        split = self._best_split(X, y, w)
        if split is None:
            return node

        feature_index, threshold, gain, left_index, right_index = split

        node_weight = float(w.sum())
        self._importances[feature_index] += (
            node_weight / self._total_weight
        ) * gain

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._grow(
            X[left_index], y[left_index], w[left_index], depth + 1
        )
        node.right = self._grow(
            X[right_index], y[right_index], w[right_index], depth + 1
        )
        return node

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> Optional[Tuple[int, float, float, np.ndarray, np.ndarray]]:
        """Search the best split over a (possibly sub-sampled) feature set.

        For every candidate feature the column is sorted once
        (O(N log N)) and weighted per-class counts are accumulated with a
        single cumulative sum, so all candidate thresholds — midpoints
        between consecutive *distinct* sorted values — are scored in one
        vectorized pass. Children are partitioned by sorted position, which
        guarantees both sides are non-empty independent of floating-point
        rounding of the midpoint.

        Returns
        -------
        Optional[tuple]
            ``(feature_index, threshold, gain, left_index, right_index)`` for
            the best split with strictly positive impurity reduction, or
            ``None`` when no such split exists.
        """
        n_samples = y.shape[0]
        total_counts = np.bincount(y, weights=w, minlength=self._n_classes)
        total_weight = total_counts.sum()
        parent_impurity = self._impurity(
            total_counts[np.newaxis, :], np.asarray([total_weight])
        )[0]

        best_gain = MIN_IMPURITY_DECREASE
        best: Optional[Tuple[int, float, float, np.ndarray, np.ndarray]] = None

        for feature_index in self._select_features():
            order = np.argsort(X[:, feature_index], kind="stable")
            sorted_values = X[order, feature_index]

            # Candidate boundaries sit between consecutive distinct values.
            boundary = np.nonzero(sorted_values[:-1] < sorted_values[1:])[0]
            if boundary.shape[0] == 0:
                continue  # Feature is constant within this node.

            # Cumulative weighted class counts along the sorted order:
            # cumulative[i, c] = sum of weights of samples 0..i with class c.
            weighted_one_hot = np.zeros(
                (n_samples, self._n_classes), dtype=np.float64
            )
            weighted_one_hot[np.arange(n_samples), y[order]] = w[order]
            cumulative = np.cumsum(weighted_one_hot, axis=0)

            left_counts = cumulative[boundary]
            right_counts = total_counts[np.newaxis, :] - left_counts
            left_weight = left_counts.sum(axis=1)
            right_weight = total_weight - left_weight

            children_impurity = (
                left_weight * self._impurity(left_counts, left_weight)
                + right_weight * self._impurity(right_counts, right_weight)
            ) / max(total_weight, EPSILON)
            gains = parent_impurity - children_impurity

            best_local = int(np.argmax(gains))
            if gains[best_local] > best_gain:
                position = boundary[best_local]
                best_gain = float(gains[best_local])
                threshold = float(
                    (sorted_values[position] + sorted_values[position + 1])
                    / 2.0
                )
                best = (
                    int(feature_index),
                    threshold,
                    best_gain,
                    order[: position + 1],
                    order[position + 1 :],
                )
        return best

    def _select_features(self) -> np.ndarray:
        """Return the feature indices examined at the current node.

        When ``max_features`` restricts the search, a fresh subset is drawn
        without replacement at every node from the seeded generator; indices
        are sorted so ties between equally good splits always resolve to the
        lowest feature index.
        """
        n_features = self.n_features_
        if self.max_features is None:
            return np.arange(n_features)
        if self.max_features == MAX_FEATURES_SQRT:
            k = int(np.floor(np.sqrt(n_features)))
        elif self.max_features == MAX_FEATURES_LOG2:
            k = int(np.floor(np.log2(n_features)))
        else:
            k = int(self.max_features)
        k = min(max(k, 1), n_features)
        return np.sort(self._rng.choice(n_features, size=k, replace=False))

    # ------------------------------------------------------------------ #
    # Impurity computations
    # ------------------------------------------------------------------ #

    def _impurity(self, counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
        """Compute node impurity from weighted class counts, vectorized.

        Parameters
        ----------
        counts:
            Weighted class counts of shape ``[n_nodes, n_classes]``.
        totals:
            Total weights per node of shape ``[n_nodes]``.

        Returns
        -------
        np.ndarray
            Impurity per node of shape ``[n_nodes]`` under the configured
            criterion: Gini ``1 - sum(p_c^2)`` or entropy
            ``-sum(p_c * log2(p_c + EPSILON))``.
        """
        safe_totals = np.maximum(np.asarray(totals, dtype=np.float64), EPSILON)
        probabilities = counts / safe_totals[:, np.newaxis]
        if self.criterion == CRITERION_GINI:
            return 1.0 - np.sum(probabilities**2, axis=1)
        return -np.sum(
            probabilities * np.log2(probabilities + EPSILON), axis=1
        )

    def _distribution(self, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Weighted class probability distribution for a node's samples."""
        counts = np.bincount(y, weights=w, minlength=self._n_classes)
        return counts / max(counts.sum(), EPSILON)

    # ------------------------------------------------------------------ #
    # Prediction helpers
    # ------------------------------------------------------------------ #

    def _route(
        self,
        node: Node,
        X: np.ndarray,
        indices: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """Route the samples ``indices`` through ``node``, filling ``out``."""
        if indices.shape[0] == 0:
            return
        if node.is_leaf:
            out[indices] = node.value
            return
        goes_left = X[indices, node.feature_index] <= node.threshold
        self._route(node.left, X, indices[goes_left], out)
        self._route(node.right, X, indices[~goes_left], out)

    # ------------------------------------------------------------------ #
    # Structure inspection helpers
    # ------------------------------------------------------------------ #

    def _subtree_depth(self, node: Node) -> int:
        """Depth of the subtree rooted at ``node`` (leaf => 0)."""
        if node.is_leaf:
            return 0
        return 1 + max(
            self._subtree_depth(node.left), self._subtree_depth(node.right)
        )

    def _subtree_leaves(self, node: Node) -> int:
        """Number of leaves in the subtree rooted at ``node``."""
        if node.is_leaf:
            return 1
        return self._subtree_leaves(node.left) + self._subtree_leaves(node.right)

    def _render(self, node: Node, level: int, lines: List[str]) -> None:
        """Append the indented text rendering of ``node`` to ``lines``."""
        impurity = self._impurity(
            node.value[np.newaxis, :], np.asarray([1.0])
        )[0]
        distribution = np.array2string(node.value, precision=3)
        prefix = REPR_INDENT * level + REPR_BRANCH
        if node.is_leaf:
            lines.append(
                f"{prefix}leaf | {self.criterion}={impurity:.4f} | "
                f"samples={node.samples} | dist={distribution}"
            )
            return
        lines.append(
            f"{prefix}feature_{node.feature_index} <= {node.threshold:.4f} | "
            f"{self.criterion}={impurity:.4f} | samples={node.samples} | "
            f"dist={distribution}"
        )
        self._render(node.left, level + 1, lines)
        self._render(node.right, level + 1, lines)

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        """Raise if the estimator has not been fitted yet."""
        if self.root_ is None:
            raise RuntimeError(
                "This DecisionTree instance is not fitted yet; "
                "call fit(X, y) before using this method."
            )

    def _validate_fit_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and normalize the arrays passed to :meth:`fit`."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} != "
                f"{y.shape[0]}."
            )
        if X.shape[0] == 0:
            raise ValueError("Cannot fit a tree on an empty dataset.")

        if sample_weight is None:
            w = np.full(X.shape[0], 1.0 / X.shape[0], dtype=np.float64)
        else:
            w = np.asarray(sample_weight, dtype=np.float64)
            if w.shape != (X.shape[0],):
                raise ValueError(
                    f"sample_weight must have shape ({X.shape[0]},), got "
                    f"{w.shape}."
                )
            if np.any(w < 0.0):
                raise ValueError("sample_weight entries must be non-negative.")
            if w.sum() <= 0.0:
                raise ValueError("sample_weight must have a positive sum.")
        return X, y, w

    def _validate_predict_inputs(self, X: np.ndarray) -> np.ndarray:
        """Validate the feature matrix passed to prediction methods."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}.")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this tree was fitted with "
                f"{self.n_features_} features."
            )
        return X
