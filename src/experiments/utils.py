"""Shared infrastructure for the experiment scripts.

Provides the three study datasets required by the project brief, split /
scaling helpers, and figure-directory management. Dataset profile:

* ``breast_cancer`` — binary, 569 samples, 30 features (the high-
  dimensional ``>20`` feature requirement).
* ``adult`` — binary and naturally imbalanced (~24% positive), >=5000
  sample subset of the 48,842-row UCI Adult Income data; categorical
  columns are one-hot encoded because the trees split on continuous
  values only. Rows with missing values ("?") are dropped (<8% of data).
* ``covertype`` — 7-class UCI Covertype, >=5000 sample stratified subset
  that preserves the natural class ratios, including the severely rare
  class 4 (~0.5% of samples, satisfying the "minority <= 1%" brief
  requirement). Treated with balanced class weights in the experiments.
* ``mnist_38`` — the brief's suggested MNIST 2-class subset: digits 3
  vs 8 (a classically confusable pair), >=5000 samples, 784 pixel
  features — the second high-dimensional dataset of the study.

Memory note: Covertype is fetched once (~75 MB compressed download,
~250 MB in memory while subsampling) and immediately reduced to the
requested subset, so experiments themselves never hold the full 581k-row
matrix. Subsets keep every experiment tractable for the from-scratch
(pure NumPy) tree implementation; sizes are constants below and shrink
further in ``--fast`` mode.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split

from src.utils.preprocessing import StandardScaler, one_hot_encode, stratified_subsample

#: Global seed used by every experiment, per the project brief.
RANDOM_SEED: int = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"

#: Subset sizes for the two large datasets (brief: subsets >= 5000).
FULL_SUBSET_SIZE: int = 6000
FAST_SUBSET_SIZE: int = 1500

#: Fraction held out for testing in single-split experiments.
TEST_FRACTION: float = 0.2

#: Column names of the UCI Adult file (it ships without a header row).
_ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
_ADULT_NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain",
                  "capital-loss", "hours-per-week"]
_ADULT_CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
                      "relationship", "race", "sex", "native-country"]


@dataclass
class Dataset:
    """One prepared study dataset plus the metadata experiments rely on."""

    name: str
    X: np.ndarray
    y: np.ndarray
    #: True when the dataset needs the documented imbalance treatment.
    imbalanced: bool = False
    #: Depth cap for ensemble members on this dataset. ``None`` means
    #: unpruned; large datasets get a cap purely as a computational
    #: budget (applied identically to our models and sklearn baselines,
    #: so comparisons stay fair).
    ensemble_max_depth: int | None = None
    feature_names: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        """Number of distinct labels."""
        return int(np.unique(self.y).shape[0])


def load_datasets(fast: bool = False, names: list[str] | None = None) -> list[Dataset]:
    """Load and prepare the study datasets.

    Args:
        fast: Use the reduced subset sizes for smoke runs.
        names: Optional subset of dataset names to load (default: all).

    Returns:
        List of prepared :class:`Dataset` objects.
    """
    subset_size = FAST_SUBSET_SIZE if fast else FULL_SUBSET_SIZE
    loaders = {
        "breast_cancer": lambda: _load_breast_cancer(),
        "adult": lambda: _load_adult(subset_size),
        "covertype": lambda: _load_covertype(subset_size),
        "mnist_38": lambda: _load_mnist(subset_size),
    }
    selected = names or list(loaders)
    unknown = set(selected) - set(loaders)
    if unknown:
        raise ValueError(f"unknown dataset name(s): {sorted(unknown)}")
    return [loaders[name]() for name in selected]


def _load_breast_cancer() -> Dataset:
    """Breast Cancer Wisconsin from sklearn (binary, 30 features)."""
    data = load_breast_cancer()
    return Dataset(name="breast_cancer",
                   X=np.asarray(data.data, dtype=float),
                   y=np.asarray(data.target),
                   feature_names=list(data.feature_names))


def _load_adult(subset_size: int) -> Dataset:
    """UCI Adult Income; local ``data/adult.data`` first, OpenML fallback.

    Missing values are dropped (rows containing "?"), categoricals are
    one-hot encoded, and a stratified subset preserves the natural
    ~3:1 class imbalance.
    """
    local_file = DATA_DIR / "adult.data"
    if local_file.exists():
        frame = pd.read_csv(local_file, header=None, names=_ADULT_COLUMNS,
                            skipinitialspace=True, na_values="?")
        target = frame.pop("income").astype(str).str.strip()
        y_all = (target.str.startswith(">50K")).to_numpy(dtype=int)
    else:  # pragma: no cover - network path exercised only on fresh machines
        bundle = fetch_openml("adult", version=2, as_frame=True,
                              data_home=str(DATA_DIR / "sklearn-cache"))
        frame = bundle.frame.rename(columns={"education-num": "education-num"})
        frame = frame.replace("?", np.nan)
        y_all = (bundle.target.astype(str).str.strip() == ">50K").to_numpy(dtype=int)
        frame = frame.drop(columns=[bundle.target.name], errors="ignore")

    keep = ~frame.isna().any(axis=1)
    frame, y_all = frame.loc[keep], y_all[np.asarray(keep)]

    numeric = frame[[c for c in _ADULT_NUMERIC if c in frame.columns]].to_numpy(dtype=float)
    categorical_columns = {c: frame[c].astype(str).to_numpy()
                           for c in _ADULT_CATEGORICAL if c in frame.columns}
    encoded, encoded_names = one_hot_encode(categorical_columns)
    X_all = np.hstack([numeric, encoded])
    names = [c for c in _ADULT_NUMERIC if c in frame.columns] + encoded_names

    X, y = stratified_subsample(X_all, y_all, subset_size, random_state=RANDOM_SEED)
    return Dataset(name="adult", X=X, y=y, imbalanced=False,
                   ensemble_max_depth=12, feature_names=names)


def _load_covertype(subset_size: int) -> Dataset:
    """UCI Covertype stratified subset preserving the <=1% minority class."""
    bundle = fetch_covtype(data_home=str(DATA_DIR / "sklearn-cache"))
    X_all = np.asarray(bundle.data, dtype=float)
    y_all = np.asarray(bundle.target, dtype=int)
    X, y = stratified_subsample(X_all, y_all, subset_size, random_state=RANDOM_SEED)
    del X_all, y_all  # free the ~250 MB full matrix immediately
    return Dataset(name="covertype", X=X, y=y, imbalanced=True,
                   ensemble_max_depth=12)


def _load_mnist(subset_size: int) -> Dataset:
    """MNIST digits 3 vs 8 (binary, 784 features, brief's high-dim option).

    The full 70k-image matrix exists only transiently (~440 MB) while
    the two digit classes are selected and subsampled; the subset kept
    for the experiments is a few MB. Label 1 means "digit 8".
    """
    bundle = fetch_openml("mnist_784", version=1, as_frame=False,
                          data_home=str(DATA_DIR / "sklearn-cache"))
    y_full = np.asarray(bundle.target, dtype=int)
    mask = (y_full == 3) | (y_full == 8)
    X_pair = np.asarray(bundle.data[mask], dtype=float)
    y_pair = (y_full[mask] == 8).astype(int)
    del bundle, y_full, mask  # release the full 70k x 784 matrix
    X, y = stratified_subsample(X_pair, y_pair, subset_size, random_state=RANDOM_SEED)
    return Dataset(name="mnist_38", X=X, y=y, ensemble_max_depth=12)


def prepare_split(dataset: Dataset, test_size: float = TEST_FRACTION,
                  random_state: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray,
                                                            np.ndarray, np.ndarray]:
    """Stratified train/test split with train-fitted standardization.

    Returns:
        ``(X_train, X_test, y_train, y_test)`` with both feature blocks
        transformed by a scaler fitted on the training split only.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=test_size,
        random_state=random_state, stratify=dataset.y)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def figure_dir(experiment: str) -> Path:
    """Ensure and return ``figures/<experiment>/``."""
    path = FIGURES_DIR / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results_csv(rows: list[dict], path: Path) -> None:
    """Write experiment result rows to CSV (one dict per row)."""
    pd.DataFrame(rows).to_csv(path, index=False)


def default_n_jobs() -> int:
    """Worker processes for forest training.

    Defaults to ``min(4, cpu_count)``; the ``ML_N_JOBS`` environment
    variable overrides it (set ``ML_N_JOBS=1`` to force sequential
    training, e.g. when diagnosing multiprocessing issues on Windows).
    """
    value = os.environ.get("ML_N_JOBS")
    if value is not None:
        return max(1, int(value))
    return max(1, min(4, os.cpu_count() or 1))


class Timer:
    """Context manager reporting elapsed wall-clock seconds."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        print(f"    [{self.label}] finished in {self.elapsed:.1f}s")
