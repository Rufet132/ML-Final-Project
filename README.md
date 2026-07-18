# Ensemble Methods: Boosting vs. Bagging

Final project for the Machine Learning course (AI Academy, National AI
Center). Decision Tree, AdaBoost (SAMME + SAMME.R), Random Forest, PCA,
K-Means, and DBSCAN are all implemented **from scratch with NumPy**;
scikit-learn appears only as a test/experiment baseline, for dataset
loading, and for the permitted helper utilities (splits, ARI).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows (source .venv/bin/activate on Unix)
pip install -r requirements.txt
```

Python 3.11+ is required (CI runs 3.12).

## Datasets

| Dataset | Task | Size used | Why it is in the study |
|---|---|---|---|
| Breast Cancer Wisconsin | binary | 569 x 30 | high-dimensional (>20 features), bundled with sklearn |
| Adult Income (UCI) | binary | 6000-row stratified subset of 48,842 | naturally imbalanced (~24% positive), mixed categorical features (one-hot encoded) |
| Covertype (UCI) | 7-class | 6000-row stratified subset | severe class imbalance — the rarest class is ~0.5% (satisfies the "minority <= 1%" requirement); treated with balanced class weights |
| MNIST 3 vs 8 (OpenML) | binary | 6000-row stratified subset | the brief's suggested 2-class MNIST subset: 784 pixel features, second high-dimensional dataset |

Adult and Covertype are fetched automatically on first use and cached
under `data/` (internet needed once; `bash download_data.sh` can
pre-download the raw UCI files instead). Subsets are stratified so the
natural class ratios — including the <=1% Covertype minority — are
preserved, and they keep the pure-NumPy models tractable. Missing
values in Adult (`?` entries, <8% of rows) are dropped; features are
standardized with a scaler fitted on training data only.

## Reproducing every experiment

```bash
python src/experiments/run_all.py             # full study (report figures)
python src/experiments/run_all.py --fast      # minutes-long smoke run
python src/experiments/run_all.py --only baseline unsupervised
```

All seven studies from the brief run with seed 42 and write figures +
CSV tables under `figures/<experiment>/`:

1. `baseline` — our tree & stump vs `sklearn.tree.DecisionTreeClassifier` (2% parity check)
2. `adaboost_scaling` — staged train/test error over 1..200 rounds
3. `rf_scaling` — accuracy vs n_estimators (with OOB) and vs max_depth
4. `head_to_head` — 5-fold CV of tree / AdaBoost / RF / sklearn-RF (accuracy, macro F1, AUC)
5. `noise_robustness` — 5/10/20% label noise, clean-test degradation
6. `bias_variance` — Breiman-style 0-1 decomposition over 100 bootstrap replicates
7. `gbm_comparison` — bonus: staged GBM (log-loss) vs AdaBoost SAMME/SAMME.R
8. `unsupervised` — PCA scree, K-Means elbow + ARI, DBSCAN k-distance + ARI, plus a bonus t-SNE vs PCA embedding figure

**Runtime note:** the models are pure-NumPy, so the full study is
CPU-heavy — expect several hours (Random Forest sweeps dominate; tree
training uses up to 4 processes). Use `--fast` to validate the pipeline
first. Memory stays under ~1 GB because the large datasets are cut to
subsets immediately after download.

## Tests, coverage, type checks

```bash
python -m pytest tests/ --cov=src         # coverage gate: 60% (CI enforces)
python -m mypy src/ --ignore-missing-imports
flake8 src/ tests/ --select=E9,F63,F7,F82
```

Coverage is configured in `.coveragerc` to measure the library modules
(models, metrics, preprocessing, unsupervised pipeline); the experiment
driver scripts are reproduced via `run_all.py` rather than unit-tested.
GitHub Actions runs lint + mypy + tests with the 60% coverage gate on
every push and pull request (`.github/workflows/ci.yml`).

## Notebooks

* `notebooks/exploration.ipynb` — slider-driven decision boundaries
  (tree vs AdaBoost vs RF), staged AdaBoost error/weights with a label
  noise slider, and a live PCA + K-Means + DBSCAN view of breast cancer.
  Falls back to static figures when ipywidgets is unavailable; every
  figure's latest state is saved under `figures/notebook/`.
* `notebooks/data_overview.ipynb` — the four datasets, their class
  distributions, MNIST samples, and the preprocessing decisions.

## Bonuses implemented

* SAMME.R real-valued AdaBoost (`algorithm="SAMME.R"`)
* Gradient Boosting with log-loss (`src/boosting/gradient_boosting.py`)
  + staged comparison against AdaBoost (`gbm_comparison` experiment)
* t-SNE vs PCA embedding comparison (sklearn t-SNE, as permitted)
* GitHub Actions CI (lint + mypy + tests with 60% coverage gate)
* Interactive figure notebook (see above)

## Repository layout

```
src/
  trees/decision_tree.py      # Module 1 — CART decision tree (sample_weight support)
  boosting/adaboost.py        # Module 2 — SAMME + SAMME.R over decision stumps
  boosting/gradient_boosting.py  # bonus — log-loss GBM on regression trees
  bagging/random_forest.py    # Module 3 — bagging, OOB, n_jobs, class_weight
  unsupervised/               # Module 4 — PCA, KMeans, DBSCAN
  metrics/evaluation.py       # accuracy, macro F1, ROC-AUC (from scratch)
  utils/preprocessing.py      # scaler, one-hot, oversampling, subsampling
  experiments/                # run_all.py + the seven study scripts
tests/                        # pytest suites for every module
notebooks/exploration.ipynb   # interactive figures (bonus)
report/  slides/  contribution/
```
