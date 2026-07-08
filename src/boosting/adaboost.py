"""AdaBoost (discrete SAMME) with decision stumps as weak learners.

Implements the algorithm from Freund & Schapire (1997), generalised to
K >= 2 classes via the SAMME weight update (Zhu et al., 2009):

    alpha_m = ln((1 - eps_m) / eps_m) + ln(K - 1)

which reduces to the classic alpha_m = ln((1-eps_m)/eps_m) for K = 2.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

from src.trees.decision_tree import DecisionTree


class DecisionStump(DecisionTree):
    """Convenience subclass: a depth-1 decision tree (a single binary split)."""

    def __init__(self, criterion: str = "gini", random_state: Optional[int] = None) -> None:
        super().__init__(max_depth=1, criterion=criterion, random_state=random_state)


class AdaBoostClassifier:
    """Discrete SAMME AdaBoost using ``DecisionStump`` weak learners.

    Attributes:
        n_estimators: Maximum number of boosting rounds requested.
        learning_rate: Shrinkage applied to each estimator's alpha
            (``effective_alpha = learning_rate * alpha``), analogous to
            sklearn's ``AdaBoostClassifier``. Smaller values require more
            estimators but tend to generalise better.
        criterion: Impurity criterion passed to each stump ("gini"/"entropy").
        random_state: Seed for reproducibility. Round ``m`` seeds its stump
            with ``random_state + m`` (only when random_state is not None),
            so re-running fit() with the same seed reproduces the exact
            same sequence of stumps and weight trajectories.
    """

    # Error below which we clip to avoid division by zero / log(0) in alpha.
    _EPSILON_CLIP = 1e-10

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.random_state = random_state

        self._estimators: List[DecisionStump] = []
        self._alphas: List[float] = []
        self._errors: List[float] = []
        self._classes: Optional[np.ndarray] = None  # original label values, sorted

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        """Fit the ensemble.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Class labels, shape (n_samples,). Any encoding is accepted
                (e.g. {0,1}, {-1,+1}, or arbitrary K classes) — labels are
                internally mapped to {0, ..., K-1} and mapped back on output.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching n_samples")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset")

        self._classes, y_encoded = np.unique(y, return_inverse=True)
        n_classes = len(self._classes)
        if n_classes < 2:
            raise ValueError("AdaBoostClassifier requires at least 2 classes")
        n_samples = X.shape[0]

        self._estimators = []
        self._alphas = []
        self._errors = []

        # Step 1: initialize uniform sample weights w_i = 1/N.
        sample_weight = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)

        for m in range(self.n_estimators):
            stump_seed = (
                self.random_state + m if self.random_state is not None else None
            )
            stump = DecisionStump(criterion=self.criterion, random_state=stump_seed)

            # Step 2: train a weighted stump. Weighted impurity (see
            # DecisionTree._impurity) makes the split search "listen"
            # harder to samples the ensemble is currently getting wrong.
            stump.fit(X, y_encoded, sample_weight=sample_weight)
            predictions = stump.predict(X)
            incorrect = predictions != y_encoded

            # Step 3: weighted error.
            err = np.sum(sample_weight * incorrect) / np.sum(sample_weight)

            # Step 4: edge cases.
            # eps == 0 -> the stump is "too good": ln(0) would blow up alpha
            # to infinity. Clipping to a tiny epsilon keeps alpha large but
            # finite, so this near-perfect stump still gets an (enormous
            # but valid) vote instead of crashing training.
            if err <= 0:
                err = self._EPSILON_CLIP
            if err >= 0.5:
                # eps >= 0.5 means the weak learner is no better than (or
                # worse than) random guessing on K=2, i.e. it has stopped
                # contributing signal. SAMME's alpha formula would assign
                # it zero or negative weight, and continuing to reweight
                # samples based on a non-informative learner just injects
                # noise. We stop early and return the ensemble built so
                # far, matching sklearn's AdaBoostClassifier behaviour.
                # (Alternative, spec-permitted behaviour: raise
                # ValueError instead of a silent early stop — not used
                # here so that staged_predict / experiments over
                # n_estimators degrade gracefully rather than crashing.)
                break

            # Step 5 (SAMME): alpha_m = ln((1-eps)/eps) + ln(K-1).
            # For K=2, ln(K-1) = ln(1) = 0, recovering the classic binary
            # AdaBoost formula. The ln(K-1) term is what keeps a "better
            # than random" (1/K accuracy) learner's alpha positive for
            # K > 2, since binary-style eps < 0.5 alone isn't the right
            # threshold once there are more than 2 classes to guess among.
            alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(n_classes - 1))

            # Step 6: reweight — up-weight misclassified samples by
            # exp(alpha), leave correctly-classified samples unchanged,
            # then renormalize so weights remain a valid distribution.
            # This is what forces the *next* stump to focus on the
            # samples the ensemble-so-far still gets wrong.
            sample_weight = sample_weight * np.exp(alpha * incorrect)
            sample_weight /= sample_weight.sum()

            self._estimators.append(stump)
            self._alphas.append(alpha)
            self._errors.append(err)

        if not self._estimators:
            raise RuntimeError(
                "No estimators were trained (first stump already had error >= 0.5)."
            )

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted majority vote: argmax_k sum_m alpha_m * 1[h_m(x) = k]."""
        self._check_is_fitted()
        votes = self._weighted_votes(X)
        winning_encoded = np.argmax(votes, axis=1)
        return self._classes[winning_encoded]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax over per-class alpha-sums.

        Discrete SAMME does not natively produce calibrated probabilities
        (it only ever casts a hard vote per estimator). We convert the
        weighted-vote totals into a probability distribution via softmax:
        classes with a larger accumulated alpha get exponentially more
        probability mass, and softmax guarantees a valid distribution
        (non-negative, sums to 1) even when raw alpha-sums are negative
        or wildly different in scale.
        """
        self._check_is_fitted()
        votes = self._weighted_votes(X)
        shifted = votes - votes.max(axis=1, keepdims=True)  # numerical stability
        exp_votes = np.exp(shifted)
        return exp_votes / exp_votes.sum(axis=1, keepdims=True)

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after each boosting round (1..M so far)."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self._classes)
        votes = np.zeros((X.shape[0], n_classes), dtype=np.float64)

        for stump, alpha in zip(self._estimators, self._alphas):
            predictions = stump.predict(X)
            votes[np.arange(X.shape[0]), predictions] += alpha
            winning_encoded = np.argmax(votes, axis=1)
            yield self._classes[winning_encoded]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def estimator_weights(self) -> np.ndarray:
        return np.array(self._alphas)

    @property
    def estimator_errors(self) -> np.ndarray:
        return np.array(self._errors)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _weighted_votes(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self._classes)
        votes = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for stump, alpha in zip(self._estimators, self._alphas):
            predictions = stump.predict(X)
            votes[np.arange(X.shape[0]), predictions] += alpha
        return votes

    def _check_is_fitted(self) -> None:
        if not self._estimators or self._classes is None:
            raise RuntimeError("AdaBoostClassifier is not fitted yet. Call fit() first.")