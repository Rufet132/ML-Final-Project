#!/usr/bin/env bash
# download_data.sh – fetch datasets for the ML final project
# Run from the repository root:   bash download_data.sh
set -euo pipefail

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "==> 1. Breast Cancer Wisconsin (Diagnostic)"
wget -nc -P "$DATA_DIR" \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
echo "    Saved to $DATA_DIR/wdbc.data"

echo "==> 2. Adult Income"
wget -nc -P "$DATA_DIR" \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
wget -nc -P "$DATA_DIR" \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
echo "    Saved adult.data and adult.test"

echo "==> 3. Covertype"
wget -nc -P "$DATA_DIR" \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
gunzip -k "$DATA_DIR/covtype.data.gz" 2>/dev/null || true   # keep .gz if unzip fails
echo "    Saved covtype.data.gz (and extracted covtype.data if gunzip succeeded)"

echo "==> 4. MNIST (2‑class subset)"
echo "    MNIST is best obtained via sklearn in Python:"
echo "      from sklearn.datasets import fetch_openml"
echo "      X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)"
echo "    No static file is downloaded here; see run_all.py for dataset loading."

echo "All datasets ready in ./$DATA_DIR/"
