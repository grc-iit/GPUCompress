# Reproducibility Guide

This document describes how to reproduce the GPUCompress neural network training
pipeline from scratch: synthetic data generation, hyperparameter sweep, and final
model evaluation with figures.

All commands are run from the repository root (`/workspaces/GPUCompress`).

---

## Prerequisites

Build GPUCompress and the synthetic benchmark:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Install Python dependencies:

```bash
pip install torch torchvision pyyaml xgboost shap matplotlib seaborn scikit-learn pandas numpy
```

---

## Step 1 — Generate Synthetic Training Data (128 K entries)

The synthetic benchmark runs every compression configuration (8 algorithms ×
2 shuffle modes × 4 quantization levels = 64 configs) against randomly generated
float32 chunks and records timing, ratio, and quality metrics for each
(chunk, config) pair.

128 000 rows = **2 000 chunks × 64 configs**:

```bash
build/synthetic_benchmark \
    --num-chunks 2000 \
    --output results/synthetic_results.csv \
    --seed 42
```

The output CSV has one row per (chunk × config) trial with columns:
`chunk_id`, `algorithm`, `shuffle_bytes`, `quantized`, `error_bound`,
`entropy_bits`, `mad`, `second_derivative`, `comp_time_ms`, `decomp_time_ms`,
`compression_ratio`, `psnr_db`, `ssim`, `rmse`, `max_abs_error`, `success`, …

To generate a larger dataset (e.g. 640 K rows = 10 000 chunks):

```bash
build/synthetic_benchmark --num-chunks 10000 --output results/synthetic_10k.csv
```

---

## Step 2 — Hyperparameter Sweep

The sweep runs 5-fold cross-validation over a grid of hidden layer width,
depth, learning rate, and batch size.  It reports mean validation loss and R²
for every combination and saves a ranked CSV summary.

```bash
python3 neural_net/training/hyperparam_study.py \
    --csv results/synthetic_results.csv \
    --folds 5 \
    --epochs 150 \
    --patience 15
```

Output written to `results/hyperparam_study.csv`.

Each row contains:
`hidden_dim`, `num_hidden_layers`, `lr`, `batch_size`,
`val_loss_mean`, `val_loss_std`, `r2_mean`, `r2_std`,
and per-output R² / MAPE columns
(`r2_comp_time`, `r2_decomp_time`, `r2_ratio`, `r2_psnr`, …).

The best configuration is printed to stdout at the end of the run, e.g.:

```
Best hyperparameters:
  hidden_dim           = 128
  num_hidden_layers    = 2
  lr                   = 0.001
  batch_size           = 512
  val_loss_mean        = 0.012345 ± 0.000678
  r2_mean              = 0.9712 ± 0.0031
```

Use the best `hidden_dim` and `num_hidden_layers` values in the next step.

---

## Step 3 — Cross-Validation on the Best Model (Neural Network)

Run 5-fold CV with the architecture selected in Step 2 to get final
per-output MAE, R², and MAPE numbers for the paper:

```bash
python3 neural_net/training/cross_validate.py \
    --csv results/synthetic_results.csv \
    --folds 5 \
    --hidden-dim 128 \
    --num-hidden-layers 2 \
    --lr 0.001 \
    --epochs 300
```

Per-fold and summary statistics are printed to stdout.

---

---

## Summary of Outputs

| Step | Command | Output |
|---|---|---|
| 1 | `synthetic_benchmark` | `results/synthetic_results.csv` |
| 2 | `hyperparam_study.py` | `results/hyperparam_study.csv` |
| 3 | `cross_validate.py` | stdout (MAE / R² / MAPE per fold) |
