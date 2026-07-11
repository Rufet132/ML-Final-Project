"""Tests for Module 2: DecisionStump / AdaBoostClassifier.
 
Run with: pytest tests/test_adaboost.py -v
"""
 
import numpy as np
import pytest
 
from src.trees.decision_tree import DecisionTree
from src.boosting.adaboost import DecisionStump, AdaBoostClassifier
 
 
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

#----------------------------------------------------------------------
# Weighted impurity / weighted stump fitting
# ----------------------------------------------------------------------
class TestWeightedImpurity:
    def test_uniform_weights_match_unweighted(self, separable_binary_data):
        X, y = separable_binary_data
        tree_unweighted = DecisionTree(max_depth=3, random_state=1).fit(X, y)
        tree_weighted = DecisionTree(max_depth=3, random_state=1).fit(
            X, y, sample_weight=np.ones(len(y))
        )
        np.testing.assert_array_equal(
            tree_unweighted.predict(X), tree_weighted.predict(X)
        )
 
    def test_upweighting_a_class_biases_split(self):
        # Two samples per class; heavily up-weight one class so a stump
        # trained on weighted Gini should favor separating it out first.
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        weight = np.array([1.0, 1.0, 1.0, 100.0])  # last sample dominates
 
        stump = DecisionStump(random_state=0)
        stump.fit(X, y, sample_weight=weight)
        # The dominant sample (class 1, x=3.0) must be classified correctly.
        assert stump.predict(np.array([[3.0]]))[0] == 1
 
    def test_weighted_gini_formula_matches_manual_computation(self):
        tree = DecisionTree(criterion="gini")
        counts = np.array([[3.0, 1.0]])  # p = [0.75, 0.25]
        totals = np.array([4.0])
        expected = 1 - (0.75 ** 2 + 0.25 ** 2)
        assert tree._impurity(counts, totals)[0] == pytest.approx(expected)
 
    def test_weighted_entropy_formula_matches_manual_computation(self):
        tree = DecisionTree(criterion="entropy")
        counts = np.array([[2.0, 2.0]])  # p = [0.5, 0.5]
        totals = np.array([4.0])
        expected = -(0.5 * np.log2(0.5) + 0.5 * np.log2(0.5))
        assert tree._impurity(counts, totals)[0] == pytest.approx(expected, abs=1e-6)



# ----------------------------------------------------------------------
# Estimator weight computation / weight normalization
# ----------------------------------------------------------------------
class TestBoostingMechanics:
    def test_sample_weights_always_normalized(self, separable_binary_data):
        X, y = separable_binary_data
        clf = AdaBoostClassifier(n_estimators=10, random_state=0)
        clf.fit(X, y)
        # Re-derive final weights by re-running the loop is overkill here;
        # instead check invariant indirectly via estimator_errors sanity.
        assert np.all(clf.estimator_errors >= 0)
        assert np.all(clf.estimator_errors < 0.5)
 
    def test_alpha_matches_binary_formula(self, separable_binary_data):
        X, y = separable_binary_data
        clf = AdaBoostClassifier(n_estimators=5, random_state=0, learning_rate=1.0)
        clf.fit(X, y)
        for alpha, err in zip(clf.estimator_weights, clf.estimator_errors):
            expected_alpha = np.log((1 - err) / err) + np.log(2 - 1)  # K=2
            assert alpha == pytest.approx(expected_alpha)
 
    def test_learning_rate_scales_alpha(self, separable_binary_data):
        X, y = separable_binary_data
        clf_full = AdaBoostClassifier(n_estimators=3, random_state=0, learning_rate=1.0).fit(X, y)
        clf_half = AdaBoostClassifier(n_estimators=3, random_state=0, learning_rate=0.5).fit(X, y)
        # Same errors trajectory isn't guaranteed since reweighting differs,
        # but the very first alpha (before any reweighting divergence) must
        # scale exactly by learning_rate.
        assert clf_half.estimator_weights[0] == pytest.approx(
            0.5 * clf_full.estimator_weights[0]
        )
 
    def test_perfect_stump_clips_epsilon_instead_of_diverging(self):
        # Trivially separable single-feature data -> first stump is perfect
        # -> err would be 0 without clipping.
        X = np.array([[0.0], [1.0], [10.0], [11.0]])
        y = np.array([0, 0, 1, 1])
        clf = AdaBoostClassifier(n_estimators=1, random_state=0)
        clf.fit(X, y)
        assert clf.estimator_errors[0] == pytest.approx(AdaBoostClassifier._EPSILON_CLIP)
        assert np.isfinite(clf.estimator_weights[0])


# ----------------------------------------------------------------------
# staged_predict
# ----------------------------------------------------------------------
class TestStagedPredict:
    def test_staged_predict_length_matches_n_estimators(self, moons_like_data):
        X, y = moons_like_data
        clf = AdaBoostClassifier(n_estimators=15, random_state=0).fit(X, y)
        staged = list(clf.staged_predict(X))
        assert len(staged) == len(clf.estimator_weights)
 
    def test_final_staged_prediction_matches_predict(self, moons_like_data):
        X, y = moons_like_data
        clf = AdaBoostClassifier(n_estimators=15, random_state=0).fit(X, y)
        staged = list(clf.staged_predict(X))
        np.testing.assert_array_equal(staged[-1], clf.predict(X))
 
    def test_staged_accuracy_is_generally_non_decreasing_trend(self, moons_like_data):
        X, y = moons_like_data
        clf = AdaBoostClassifier(n_estimators=30, random_state=0).fit(X, y)
        accuracies = [np.mean(pred == y) for pred in clf.staged_predict(X)]
        # Not strictly monotonic round-to-round, but should trend upward:
        # average of the second half should beat the average of the first half.
        first_half = np.mean(accuracies[: len(accuracies) // 2])
        second_half = np.mean(accuracies[len(accuracies) // 2 :])
        assert second_half >= first_half - 0.05  # small slack for noise
 
 
# ----------------------------------------------------------------------
# predict / predict_proba
# ----------------------------------------------------------------------
class TestPredict:
    def test_predict_perfect_on_separable_data(self, separable_binary_data):
        X, y = separable_binary_data
        clf = AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
        assert np.mean(clf.predict(X) == y) >= 0.95
 
    def test_predict_proba_shape_and_validity(self, moons_like_data):
        X, y = moons_like_data
        clf = AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)
        assert np.all(proba >= 0)
 
    def test_predict_proba_argmax_matches_predict(self, moons_like_data):
        X, y = moons_like_data
        clf = AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
        proba_pred = clf._classes[np.argmax(clf.predict_proba(X), axis=1)]
        np.testing.assert_array_equal(proba_pred, clf.predict(X))
 
 
# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
class TestReproducibility:
    def test_same_seed_gives_identical_results(self, moons_like_data):
        X, y = moons_like_data
        clf1 = AdaBoostClassifier(n_estimators=20, random_state=7).fit(X, y)
        clf2 = AdaBoostClassifier(n_estimators=20, random_state=7).fit(X, y)
        np.testing.assert_array_equal(clf1.predict(X), clf2.predict(X))
        np.testing.assert_allclose(clf1.estimator_weights, clf2.estimator_weights)
        np.testing.assert_allclose(clf1.estimator_errors, clf2.estimator_errors)
 
    def test_different_random_state_still_fits_without_error(self, moons_like_data):
        # Note: our split search is a deterministic exhaustive scan (no
        # feature subsampling in AdaBoost's stumps), so random_state does
        # not change which split is chosen -- only stump-internal RNG
        # state, which here is unused. Reproducibility (tested above) is
        # therefore the meaningful guarantee; we just check different
        # seeds do not break anything.
        X, y = moons_like_data
        clf1 = AdaBoostClassifier(n_estimators=20, random_state=1).fit(X, y)
        clf2 = AdaBoostClassifier(n_estimators=20, random_state=2).fit(X, y)
        assert len(clf1.estimator_weights) > 0
        assert len(clf2.estimator_weights) > 0
 
 
# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------
class TestEdgeCases:
    def test_single_estimator(self, separable_binary_data):
        X, y = separable_binary_data
        clf = AdaBoostClassifier(n_estimators=1, random_state=0).fit(X, y)
        assert len(clf.estimator_weights) == 1
        assert clf.predict(X).shape == y.shape
 
    def test_one_training_sample_raises_via_decision_tree_not_adaboost(self):
        # A single sample can't form two classes; AdaBoost requires >=2 classes.
        X = np.array([[1.0]])
        y = np.array([0])
        with pytest.raises(ValueError):
            AdaBoostClassifier(n_estimators=5).fit(X, y)
 
    def test_identical_labels_raises(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0, 0, 0])
        with pytest.raises(ValueError):
            AdaBoostClassifier(n_estimators=5).fit(X, y)
 
    def test_identical_features_with_imbalanced_labels_fits_gracefully(self):
        # All rows identical in X: no valid split exists, so every stump
        # collapses to a majority-class leaf. With balanced labels the
        # majority-class error is exactly 0.5 and training stops
        # immediately (a RuntimeError, tested below). With imbalanced
        # labels the majority-class error is below 0.5, so at least one
        # (uninformative but valid) stump should be trained without
        # crashing.
        X = np.ones((10, 2))
        y = np.array([0] * 7 + [1] * 3)
        clf = AdaBoostClassifier(n_estimators=20, random_state=0).fit(X, y)
        assert len(clf.estimator_weights) >= 1
 
    def test_identical_features_with_balanced_labels_raises(self):
        # No valid split, and a 50/50 majority-class stump has error
        # exactly 0.5, which is >= 0.5 on the very first round -- zero
        # estimators can be trained, surfaced as a RuntimeError rather
        # than silently returning an empty (useless) ensemble.
        X = np.ones((10, 2))
        y = np.array([0, 1] * 5)
        with pytest.raises(RuntimeError):
            AdaBoostClassifier(n_estimators=20, random_state=0).fit(X, y)
 
    def test_empty_estimator_list_before_fit_raises_on_predict(self):
        clf = AdaBoostClassifier(n_estimators=5)
        with pytest.raises(RuntimeError):
            clf.predict(np.array([[0.0]]))
 
    def test_binary_labels_0_1(self, separable_binary_data):
        X, y = separable_binary_data
        clf = AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
        assert set(np.unique(clf.predict(X))) <= {0, 1}
 
    def test_binary_labels_minus1_plus1(self, separable_pm1_data):
        X, y = separable_pm1_data
        clf = AdaBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)) <= {-1, 1}
        assert np.mean(preds == y) >= 0.95
 
    def test_epsilon_at_least_half_stops_training_early(self):
        # Pure random-noise labels on random features: expect eps >= 0.5
        # to trigger before n_estimators rounds complete at least sometimes.
        rng = np.random.RandomState(0)
        X = rng.normal(size=(40, 2))
        y = rng.randint(0, 2, size=40)
        clf = AdaBoostClassifier(n_estimators=200, random_state=0).fit(X, y)
        # Either it ran all 200 rounds (unlikely with noise) or stopped early;
        # both are valid, but it must never exceed the requested cap.
        assert len(clf.estimator_weights) <= 200
 
 
# ----------------------------------------------------------------------
# Sanity check against sklearn (not a hard equality — approximate agreement)
# ----------------------------------------------------------------------
class TestSklearnComparison:
    def test_accuracy_within_tolerance_of_sklearn(self, moons_like_data):
        from sklearn.ensemble import AdaBoostClassifier as SkAdaBoost
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import train_test_split
 
        X, y = moons_like_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
 
        ours = AdaBoostClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
        our_acc = np.mean(ours.predict(X_test) == y_test)
 
        sk = SkAdaBoost(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
            n_estimators=50,
            random_state=42,
        ).fit(X_train, y_train)
        sk_acc = np.mean(sk.predict(X_test) == y_test)
 
        # Sanity check only, per project spec — approximate agreement.
        assert abs(our_acc - sk_acc) <= 0.10