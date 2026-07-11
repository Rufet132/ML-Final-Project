"""AdaBoost with decision stumps as weak learners.

Supports two variants via the ``algorithm`` parameter:

* ``"SAMME"`` (default): discrete AdaBoost, generalised to K >= 2 classes
  via the SAMME weight update (Zhu et al., 2009):

      alpha_m = ln((1 - eps_m) / eps_m) + ln(K - 1)

  which reduces to the classic Freund & Schapire (1997) binary formula
  alpha_m = ln((1-eps_m)/eps_m) when K = 2. Each weak learner contributes
  a single hard vote, scaled by its scalar alpha_m.

* ``"SAMME.R"`` (bonus, real-valued SAMME, Zhu et al., 2009, Sec. 4):
  instead of a hard class vote, each round contributes a *vector* of
  class scores derived from the weak learner's predicted probabilities.
  This uses more information per round (confidence, not just the argmax
  class), typically converging in fewer rounds and with smoother staged
  accuracy curves than discrete SAMME.

Built on top of the team's ``DecisionTree`` (see ``decision_tree.py``),
which already supports ``sample_weight`` and weighted Gini/entropy
natively -- no modifications to that module were needed for this file.
``predict_proba`` on the stump (required for SAMME.R) is likewise
provided by the existing DecisionTree implementation.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

from src.trees.decision_tree import DecisionTree

#: Valid values for AdaBoostClassifier's ``algorithm`` parameter.
_VALID_ALGORITHMS = frozenset({"SAMME", "SAMME.R"})


class DecisionStump(DecisionTree):
    """Convenience subclass: a depth-1 decision tree (a single binary split)."""

    def __init__(self, criterion: str = "gini", random_state: Optional[int] = None) -> None:
        super().__init__(max_depth=1, criterion=criterion, random_state=random_state)


class AdaBoostClassifier:
    """AdaBoost using ``DecisionStump`` weak learners (SAMME or SAMME.R).

    Attributes:
        n_estimators: Maximum number of boosting rounds requested.
        learning_rate: Shrinkage applied to each round's contribution
            (scalar alpha for SAMME, the real-valued score vector for
            SAMME.R), analogous to sklearn's ``AdaBoostClassifier``.
            Smaller values require more estimators but tend to
            generalise better.
        criterion: Impurity criterion passed to each stump ("gini"/"entropy").
        algorithm: ``"SAMME"`` (discrete, hard votes) or ``"SAMME.R"``
            (real-valued, probability-weighted votes). See module
            docstring for the difference.
        random_state: Seed for reproducibility. Round ``m`` seeds its stump
            with ``random_state + m`` (only when random_state is not None),
            so re-running fit() with the same seed reproduces the exact
            same sequence of stumps and weight trajectories.
    """

    # Error below which we clip to avoid division by zero / log(0) in alpha.
    _EPSILON_CLIP = 1e-10
    # Probability floor for SAMME.R's log(p) terms, avoiding log(0) for
    # classes a stump assigns zero probability to.
    _PROBA_CLIP = 1e-5

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        criterion: str = "gini",
        algorithm: str = "SAMME",
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if algorithm not in _VALID_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {sorted(_VALID_ALGORITHMS)}, got {algorithm!r}")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.algorithm = algorithm
        self.random_state = random_state

        self._estimators: List[DecisionStump] = []
        self._alphas: List[float] = []
        self._errors: List[float] = []
        self._classes: Optional[np.ndarray] = None  # sorted original label values

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaBoostClassifier":
        """Fit the ensemble.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Class labels, shape (n_samples,). Any encoding is accepted
                (e.g. {0,1}, {-1,+1}, or arbitrary K classes). Each stump's
                own ``fit`` re-derives ``classes_`` from this same full
                label set every round (only the weights change round to
                round), so every stump's ``classes_`` matches
                ``self._classes`` exactly -- that invariant is what lets
                stump outputs be mapped straight into the vote matrix via
                ``np.searchsorted`` below.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have matching n_samples")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset")

        self._classes = np.unique(y)
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

            # Step 2: train a weighted stump. DecisionTree's weighted Gini
            # / entropy (see decision_tree.py: _impurity, _distribution)
            # makes the split search "listen" harder to samples the
            # ensemble is currently getting wrong.
            stump.fit(X, y, sample_weight=sample_weight)

            if self.algorithm == "SAMME":
                stop = self._samme_round(stump, X, y, sample_weight, self._classes, n_classes)
            else:
                stop = self._samme_r_round(stump, X, y, sample_weight, self._classes, n_classes)

            if stop:
                break

            self._estimators.append(stump)

        if not self._estimators:
            raise RuntimeError(
                "No estimators were trained (first stump already had error >= 0.5)."
            )

        return self

    def _samme_round(
        self,
        stump: DecisionStump,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        classes: np.ndarray,
        n_classes: int,
    ) -> bool:
        """One discrete-SAMME boosting round. Mutates ``sample_weight`` in
        place via the caller's reference (numpy arrays reassigned through
        the shared closure below); returns True if training should stop.
        """
        predictions = stump.predict(X)  # original label space
        incorrect = predictions != y

        # Step 3: weighted error.
        err = np.sum(sample_weight * incorrect) / np.sum(sample_weight)

        # Step 4: edge cases.
        # eps == 0 -> the stump is "too good": ln(0) would blow up alpha
        # to infinity. Clipping to a tiny epsilon keeps alpha large but
        # finite, so this near-perfect stump still gets an (enormous but
        # valid) vote instead of crashing training.
        if err <= 0:
            err = self._EPSILON_CLIP
        if err >= 0.5:
            # eps >= 0.5 means the weak learner is no better than (or
            # worse than) random guessing, i.e. it has stopped
            # contributing signal. SAMME's alpha formula would assign it
            # zero or negative weight, and continuing to reweight samples
            # based on a non-informative learner just injects noise. We
            # stop early and return the ensemble built so far, matching
            # sklearn's AdaBoostClassifier behaviour. (Alternative,
            # spec-permitted behaviour: raise ValueError instead of a
            # silent early stop -- not used here so that staged_predict /
            # experiments over n_estimators degrade gracefully rather
            # than crashing.)
            return True

        # Step 5 (SAMME): alpha_m = ln((1-eps)/eps) + ln(K-1).
        # For K=2, ln(K-1) = ln(1) = 0, recovering the classic binary
        # AdaBoost formula. The ln(K-1) term is what keeps a "better than
        # random" (1/K accuracy) learner's alpha positive for K > 2,
        # since binary-style eps < 0.5 alone isn't the right threshold
        # once there are more than 2 classes to guess among.
        alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(n_classes - 1))

        # Step 6: reweight -- up-weight misclassified samples by
        # exp(alpha), leave correctly-classified samples unchanged, then
        # renormalize so weights remain a valid distribution. This is
        # what forces the *next* stump to focus on the samples the
        # ensemble-so-far still gets wrong.
        sample_weight *= np.exp(alpha * incorrect)
        sample_weight /= sample_weight.sum()

        self._alphas.append(alpha)
        self._errors.append(err)
        return False

    def _samme_r_round(
        self,
        stump: DecisionStump,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        classes: np.ndarray,
        n_classes: int,
    ) -> bool:
        """One SAMME.R (real-valued) boosting round.

        Unlike discrete SAMME, no scalar alpha is stored -- each round
        instead contributes a *vector* h_m(x) in R^K built from the
        stump's class probabilities (see ``_samme_r_contribution``).
        ``self._alphas`` still gets one entry per round purely so
        ``estimator_weights`` stays a valid array of length
        len(estimators) for interface compatibility; the value stored is
        ``learning_rate`` (a constant), not a per-round-varying weight --
        SAMME.R has no scalar equivalent of SAMME's alpha_m, since the
        "weight" of a round is baked into the magnitude of h_m(x) itself.
        """
        proba = stump.predict_proba(X)
        predictions = classes[np.argmax(proba, axis=1)]
        incorrect = predictions != y

        # err/estimator_errors are diagnostic only for SAMME.R (there is
        # no eps-based formula for alpha here), but we still use the 0.5
        # threshold as an early-stopping heuristic: a stump doing no
        # better than the majority class contributes ~0 signal to h_m(x)
        # and further rounds on a saturated/degenerate weight
        # distribution tend to just add noise.
        err = np.sum(sample_weight * incorrect) / np.sum(sample_weight)
        if err <= 0:
            err = self._EPSILON_CLIP
        if err >= 0.5:
            return True

        log_proba = np.log(np.clip(proba, self._PROBA_CLIP, 1.0))

        # y_coded[i, k] = 1 if k is sample i's true class, else -1/(K-1).
        # This is the SAMME.R "symmetric" class coding (Zhu et al. 2009,
        # eq. 9): it makes the K per-class terms sum to zero, matching
        # h_m(x) below (which is also zero-sum across classes by
        # construction), so the true class' log-probability being high
        # relative to the others is what actually drives the weight down.
        true_class_idx = np.searchsorted(classes, y)
        y_coded = np.full((X.shape[0], n_classes), -1.0 / (n_classes - 1))
        y_coded[np.arange(X.shape[0]), true_class_idx] = 1.0

        # Weight update (Zhu et al. 2009, Algorithm 2, step 2c):
        # w_i <- w_i * exp(-((K-1)/K) * y_i^T log p(x_i))
        # A sample gets down-weighted exactly when the stump assigned
        # high probability to the true class relative to the others;
        # it gets up-weighted when the stump was confidently wrong.
        exponent = -((n_classes - 1) / n_classes) * np.sum(y_coded * log_proba, axis=1)
        sample_weight *= np.exp(exponent)
        sample_weight /= sample_weight.sum()

        self._alphas.append(self.learning_rate)  # placeholder; see docstring
        self._errors.append(err)
        return False

    def _samme_r_contribution(self, stump: DecisionStump, X: np.ndarray) -> np.ndarray:
        """h_m(x) in R^K for one SAMME.R round (Zhu et al. 2009, eq. 8):

            h_m(x)_k = (K-1) * (log p_k(x) - (1/K) * sum_j log p_j(x))

        Centering each row (subtracting the mean log-probability) is what
        makes h_m zero-sum across classes -- a class only accumulates
        positive score for being *more likely than average*, not just
        likely in absolute terms, which keeps rounds comparable to each
        other regardless of how confident any single stump is overall.
        """
        proba = stump.predict_proba(X)
        n_classes = proba.shape[1]
        log_proba = np.log(np.clip(proba, self._PROBA_CLIP, 1.0))
        return self.learning_rate * (n_classes - 1) * (
            log_proba - log_proba.mean(axis=1, keepdims=True)
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted vote across all stored stumps.

        SAMME: argmax_k sum_m alpha_m * 1[h_m(x) = k].
        SAMME.R: argmax_k sum_m h_m(x)_k (see ``_samme_r_contribution``).
        """
        classes = self._check_is_fitted()
        votes = self._weighted_votes(X, classes)
        winning_encoded = np.argmax(votes, axis=1)
        return classes[winning_encoded]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax over per-class accumulated scores.

        For SAMME, raw per-class totals are alpha-sums (a hard-vote count
        weighted by confidence); for SAMME.R they are already
        log-odds-like real values. Neither is a calibrated probability on
        its own, so both go through the same softmax normalization:
        classes with a larger accumulated score get exponentially more
        probability mass, and softmax guarantees a valid distribution
        (non-negative, sums to 1) regardless of the raw scale.
        """
        classes = self._check_is_fitted()
        votes = self._weighted_votes(X, classes)
        shifted = votes - votes.max(axis=1, keepdims=True)  # numerical stability
        exp_votes = np.exp(shifted)
        return exp_votes / exp_votes.sum(axis=1, keepdims=True)

    def staged_predict(self, X: np.ndarray) -> Iterator[np.ndarray]:
        """Yield predictions after each boosting round (1..M so far)."""
        classes = self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(classes)
        votes = np.zeros((X.shape[0], n_classes), dtype=np.float64)

        for stump, alpha in zip(self._estimators, self._alphas):
            votes += self._round_contribution(stump, X, classes, alpha)
            winning_encoded = np.argmax(votes, axis=1)
            yield classes[winning_encoded]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def estimator_weights(self) -> np.ndarray:
        """alpha_m per round for SAMME. For SAMME.R this is a constant
        placeholder (``learning_rate``) kept only for interface
        compatibility -- see ``_samme_r_round`` docstring.
        """
        return np.array(self._alphas)

    @property
    def estimator_errors(self) -> np.ndarray:
        return np.array(self._errors)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _round_contribution(
        self, stump: DecisionStump, X: np.ndarray, classes: np.ndarray, alpha: float
    ) -> np.ndarray:
        """(n_samples, n_classes) contribution of one round, algorithm-aware."""
        if self.algorithm == "SAMME":
            n_samples = X.shape[0]
            contribution = np.zeros((n_samples, len(classes)), dtype=np.float64)
            class_indices = np.searchsorted(classes, stump.predict(X))
            contribution[np.arange(n_samples), class_indices] = alpha
            return contribution
        return self._samme_r_contribution(stump, X)

    def _weighted_votes(self, X: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Accumulate per-round contributions into a single (n, K) score matrix.

        ``classes`` is sorted (np.unique guarantees this), and every stump
        was fit on the same full label set each round, so ``stump.classes_``
        always equals ``classes`` -- the invariant that lets a stump's raw
        outputs be mapped directly to vote-matrix columns.
        """
        X = np.asarray(X, dtype=np.float64)
        n_classes = len(classes)
        votes = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        for stump, alpha in zip(self._estimators, self._alphas):
            votes += self._round_contribution(stump, X, classes, alpha)
        return votes

    def _check_is_fitted(self) -> np.ndarray:
        """Return ``self._classes``, raising if the ensemble isn't fitted.

        Returning the (non-Optional) array here -- instead of just
        asserting and letting callers re-read ``self._classes`` -- is what
        lets mypy narrow away the ``Optional`` at every call site.
        """
        if not self._estimators or self._classes is None:
            raise RuntimeError("AdaBoostClassifier is not fitted yet. Call fit() first.")
        return self._classes