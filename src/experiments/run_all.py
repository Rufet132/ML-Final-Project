"""One-command reproduction of every experiment in the project.

Runs all seven studies from the brief (baseline, AdaBoost scaling,
Random Forest scaling, head-to-head CV, noise robustness, bias-variance
decomposition, and the unsupervised PCA/K-Means/DBSCAN pipeline) with
the global seed 42, writing figures and CSV tables under ``figures/``.

Usage (from the repository root)::

    python src/experiments/run_all.py            # full study
    python src/experiments/run_all.py --fast     # reduced smoke run
    python src/experiments/run_all.py --only baseline rf_scaling

Runtime expectations: the models are from-scratch NumPy implementations,
so the full study is CPU-heavy — expect several hours on a laptop
(Random Forest sweeps dominate; tree training parallelizes over up to 4
processes). ``--fast`` finishes in minutes and is meant for smoke
checks, not for report figures. Memory stays modest (< 1 GB): the large
UCI datasets are cut to stratified subsets right after download.
"""

from pathlib import Path
import argparse
import sys
import time

import matplotlib

matplotlib.use("Agg")  # figures are written to disk; no GUI needed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.preprocessing import StandardScaler as SkStandardScaler  # noqa: E402

from src.experiments import (  # noqa: E402
    adaboost_scaling,
    baseline,
    bias_variance,
    gbm_comparison,
    head_to_head,
    noise_robustness,
    rf_scaling,
)
from src.experiments.unsupervised_analysis import (  # noqa: E402
    run_tsne_comparison,
    run_unsupervised_pipeline,
)
from src.experiments.utils import RANDOM_SEED, figure_dir, load_datasets  # noqa: E402

#: Experiment registry in brief order (name -> callable taking datasets, fast).
SUPERVISED_EXPERIMENTS = {
    "baseline": baseline.run,
    "adaboost_scaling": adaboost_scaling.run,
    "rf_scaling": rf_scaling.run,
    "head_to_head": head_to_head.run,
    "noise_robustness": noise_robustness.run,
    "bias_variance": bias_variance.run,
    "gbm_comparison": gbm_comparison.run,  # bonus: GBM vs AdaBoost
}

EXPERIMENT_NAMES = list(SUPERVISED_EXPERIMENTS) + ["unsupervised"]


def run_unsupervised(datasets, fast: bool) -> None:
    """Experiment 7: PCA/K-Means/DBSCAN pipeline on every dataset.

    Features are standardized on the full dataset here (there is no
    train/test split in unsupervised analysis).
    """
    del fast  # subset sizes already come from the dataset layer
    output = figure_dir("unsupervised")
    results = []
    for dataset in datasets:
        X = SkStandardScaler().fit_transform(dataset.X)
        result = run_unsupervised_pipeline(
            X, dataset.y, dataset.name, random_state=RANDOM_SEED,
            figures_dir=output)
        # Bonus: non-linear t-SNE embedding next to the PCA projection.
        run_tsne_comparison(X, dataset.y, result["X_pca"], dataset.name,
                            random_state=RANDOM_SEED, figures_dir=output)
        results.append(result)
    print("\n  Unsupervised summary")
    print("  dataset          PCA90   K-Means ARI   DBSCAN ARI   noise")
    for result in results:
        print(f"  {result['dataset']:<16} {result['components_90']:>5}   "
              f"{result['kmeans_ari']:>11.4f}   {result['dbscan_ari']:>10.4f}   "
              f"{result['dbscan_noise_fraction']:>6.1%}")


def main() -> None:
    """Parse the CLI and run the selected experiments in brief order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="reduced sizes for a quick smoke run")
    parser.add_argument("--only", nargs="+", choices=EXPERIMENT_NAMES,
                        metavar="EXPERIMENT",
                        help=f"subset of experiments ({', '.join(EXPERIMENT_NAMES)})")
    args = parser.parse_args()
    selected = args.only or EXPERIMENT_NAMES

    mode = "fast" if args.fast else "full"
    print(f"Running {len(selected)} experiment(s) in {mode} mode "
          f"(seed={RANDOM_SEED}); figures -> {figure_dir('').resolve()}")
    datasets = load_datasets(fast=args.fast)
    for dataset in datasets:
        print(f"  dataset {dataset.name}: {dataset.X.shape[0]} samples, "
              f"{dataset.X.shape[1]} features, {dataset.n_classes} classes")

    total_start = time.perf_counter()
    for name in EXPERIMENT_NAMES:
        if name not in selected:
            continue
        print(f"\n=== {name} ===")
        start = time.perf_counter()
        if name == "unsupervised":
            run_unsupervised(datasets, args.fast)
        else:
            SUPERVISED_EXPERIMENTS[name](datasets, fast=args.fast)
        print(f"=== {name} done in {time.perf_counter() - start:.1f}s ===")

    print(f"\nAll selected experiments finished in "
          f"{(time.perf_counter() - total_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
