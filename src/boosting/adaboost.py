from __future__ import annotations

from typing import Iterator, List, Optional
import numpy as np

from src.trees.decision_tree import DecisionTree


class DecisionStump(DecisionTree):
    """A depth-1 decision tree (single binary split) used as a weak learner."""

    def __init__(self, criterion: str = "gini", random_state: Optional[int] = None) -> None:
        super().__init__(max_depth=1, criterion=criterion, random_state=random_state)


class AdaBoostClassifier:
    """Discrete SAMME AdaBoost implementation using decision stumps.
    
    Implements multi-class boosting as described in Zhu et al. (2009).
    """

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
        self._classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
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

        sample_weight = np.full(n_samples, 1.0 / n_samples, dtype=np.float64)

        for m in range(self.n_estimators):
            seed = self.random_state + m if self.random_state is not None else None
            stump = DecisionStump(criterion=self.criterion, random_state=seed)

            stump.fit(X, y_encoded, sample_weight=sample_weight)
            predictions = stump.predict(X)
            incorrect = predictions != y_encoded

            err = np.sum(sample_weight * incorrect) / np.sum(sample_weight)

            # Avoid division by zero if a stump achieves perfect classification
            if err <= 0:
                err = self._EPSILON_CLIP
            
            # Stop early if the weak learner is no better than random guessing
            if err >= 0.5:
                break

            # Discrete SAMME weight update
            alpha = self.learning_rate * (np.log((1.0 - err) / err) + np.log(n_classes - 1))

            # Up-weight misclassified samples and renormalize
            sample_weight *= np.exp(alpha * incorrect)
            sample_weight /= sample_weight.sum()

            self._estimators.append(stump)
            self._alphas.append(alpha)
            self._errors.append(err)

        if not self._estimators:
            raise RuntimeError(
                "No estimators were trained (first stump already had error >= 0.5)."
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        votes = self._weighted_votes(X)
        winning_encoded = np.argmax(votes, axis=1)
        return self._classes[winning_encoded]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate class probabilities using a softmax over weighted votes."""
        self._check_is_fitted()
        votes = self._weighted_votes(X)
        
        # Shift by max for numerical stability before exponentiation
        shifted = votes - votes.max(axis=1, keepdims=True)
        exp_votes = np.exp(shifted)
        return exp_votes / exp_votes.sum(axis=1, keepdims=True)

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after each boosting round."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(self._classes)
        votes = np.zeros((X.shape[0], n_classes), dtype=np.float64)

        for stump, alpha in zip(self._estimators, self._alphas):
            predictions = stump.predict(X)
            votes[np.arange(X.shape[0]), predictions] += alpha
            winning_encoded = np.argmax(votes, axis=1)
            yield self._classes[winning_encoded]

    @property
    def estimator_weights(self) -> np.ndarray:
        return np.array(self._alphas)

    @property
    def estimator_errors(self) -> np.ndarray:
        return np.array(self._errors)

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