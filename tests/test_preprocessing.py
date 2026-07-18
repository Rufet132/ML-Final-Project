"""Tests for src.utils.preprocessing."""

import numpy as np
import pytest

from src.utils.preprocessing import (
    StandardScaler,
    drop_missing_rows,
    one_hot_encode,
    random_oversample,
    stratified_subsample,
)

RANDOM_SEED = 42


class TestStandardScaler:
    def test_train_statistics_only(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        X_train = rng.normal(5.0, 3.0, size=(200, 4))
        X_test = rng.normal(5.0, 3.0, size=(50, 4))
        scaler = StandardScaler().fit(X_train)

        transformed_train = scaler.transform(X_train)
        np.testing.assert_allclose(transformed_train.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(transformed_train.std(axis=0), 1.0, atol=1e-12)

        # The test transform must reuse train statistics, not its own.
        expected = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)
        np.testing.assert_allclose(scaler.transform(X_test), expected)

    def test_constant_column_maps_to_zero(self) -> None:
        X = np.column_stack([np.arange(5.0), np.full(5, 7.0)])
        transformed = StandardScaler().fit_transform(X)
        np.testing.assert_allclose(transformed[:, 1], 0.0)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="fitted"):
            StandardScaler().transform(np.ones((2, 2)))

    def test_feature_count_mismatch_raises(self) -> None:
        scaler = StandardScaler().fit(np.ones((3, 2)))
        with pytest.raises(ValueError, match="different number of features"):
            scaler.transform(np.ones((3, 3)))


class TestDropMissingRows:
    def test_drops_only_nan_rows(self) -> None:
        X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
        y = np.array([0, 1, 2])
        X_clean, y_clean = drop_missing_rows(X, y)
        np.testing.assert_array_equal(y_clean, [0, 2])
        assert not np.isnan(X_clean).any()

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same number"):
            drop_missing_rows(np.ones((3, 2)), np.ones(2))


class TestOneHotEncode:
    def test_indicator_columns_and_names(self) -> None:
        encoded, names = one_hot_encode(
            {"color": np.array(["red", "blue", "red"])}
        )
        assert names == ["color=blue", "color=red"]
        np.testing.assert_array_equal(encoded, [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(encoded.sum(axis=1), 1.0)

    def test_multiple_columns_stack(self) -> None:
        encoded, names = one_hot_encode({
            "a": np.array(["x", "y"]),
            "b": np.array([1, 1]),
        })
        assert encoded.shape == (2, 3)
        assert "b=1" in names

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            one_hot_encode({"a": np.array([1, 2]), "b": np.array([1])})

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one column"):
            one_hot_encode({})


class TestRandomOversample:
    def test_balances_class_counts(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        X = rng.normal(size=(105, 3))
        y = np.array([0] * 100 + [1] * 5)
        X_balanced, y_balanced = random_oversample(X, y, random_state=RANDOM_SEED)
        _, counts = np.unique(y_balanced, return_counts=True)
        np.testing.assert_array_equal(counts, [100, 100])
        # Every oversampled row must be a copy of an original minority row.
        minority_rows = {tuple(row) for row in X[y == 1]}
        assert all(tuple(row) in minority_rows for row in X_balanced[y_balanced == 1])

    def test_deterministic_with_seed(self) -> None:
        X = np.arange(20.0).reshape(10, 2)
        y = np.array([0] * 8 + [1] * 2)
        first = random_oversample(X, y, random_state=RANDOM_SEED)
        second = random_oversample(X, y, random_state=RANDOM_SEED)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class TestStratifiedSubsample:
    def test_preserves_minority_presence_and_ratio(self) -> None:
        rng = np.random.default_rng(RANDOM_SEED)
        X = rng.normal(size=(10_000, 2))
        y = np.array([0] * 9900 + [1] * 100)  # 1% minority
        X_small, y_small = stratified_subsample(X, y, 2000, random_state=RANDOM_SEED)
        assert y_small.shape[0] == pytest.approx(2000, abs=5)
        minority_fraction = float(np.mean(y_small == 1))
        assert 0.005 <= minority_fraction <= 0.02  # ratio survives the subsample

    def test_returns_data_unchanged_when_small(self) -> None:
        X = np.ones((5, 2))
        y = np.arange(5)
        X_out, y_out = stratified_subsample(X, y, 10, random_state=0)
        assert X_out.shape == X.shape and y_out.shape == y.shape

    def test_invalid_size_raises(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            stratified_subsample(np.ones((3, 1)), np.zeros(3), 0)
