# Evaluation Commands

All commands assume you are in the GPUCompress root directory.

```bash
export LD_LIBRARY_PATH=build:$LD_LIBRARY_PATH
```

## Build

```bash
cmake --build build/ --target eval_simulation -j$(nproc)
```

## Basic Run (Frozen Weights, No Reinforcement)

```bash
./build/eval_simulation \
  -d eval/data_gs \
  -w neural_net/weights/model.nnwt \
  -o results_frozen.csv \
  -e /tmp/exp_frozen.csv
```

## Run with Online Reinforcement

```bash
./build/eval_simulation \
  -d eval/data_gs \
  -w neural_net/weights/model.nnwt \
  -o results_reinforce.csv \
  -e /tmp/exp_reinforce.csv \
  --reinforce --reinforce-threshold 0.20
```

## Run with Custom Learning Rate

```bash
./build/eval_simulation \
  -d eval/data_gs \
  -w neural_net/weights/model.nnwt \
  -o results.csv \
  -e /tmp/exp.csv \
  --reinforce --reinforce-threshold 0.20 --reinforce-lr 0.001
```

## Run with Verbose Trace (Shows SGD Details Per File)

```bash
./build/eval_simulation \
  -d eval/data_gs \
  -w neural_net/weights/model.nnwt \
  -o results.csv \
  -e /tmp/exp.csv \
  --reinforce --reinforce-threshold 0.20 --verbose
```

## Limit Number of Files

```bash
./build/eval_simulation \
  -d eval/data_gs \
  -w neural_net/weights/model.nnwt \
  -o results.csv \
  -e /tmp/exp.csv \
  --reinforce --reinforce-threshold 0.20 -m 50
```

## Reinforcement Only (No Exploration)

```bash
./build/eval_simulation \
  -d eval/data_gs \
  -w neural_net/weights/model.nnwt \
  -o results.csv \
  -e /tmp/exp.csv \
  --threshold 0.0 \
  --reinforce --reinforce-threshold 0.20
```

## Lossy Compression (With Error Bound)

```bash
./build/eval_simulation \
  -d eval/data_gs \
  -w neural_net/weights/model.nnwt \
  -o results_lossy.csv \
  -e /tmp/exp_lossy.csv \
  --error-bound 0.01 \
  --reinforce --reinforce-threshold 0.20
```

## Quick Single-File Test

```bash
./build/quick_test eval/data_gs/float32_gs_maze_A_tr000_t0500.bin neural_net/weights/model.nnwt
```

## Generate Plot Only (From Existing CSV)

```bash
python3 eval/plot_mape.py results.csv eval/mape_reinforcement.png
```

## CLI Reference

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--data-dir` | `-d` | (required) | Directory containing .bin files |
| `--weights` | `-w` | (required) | NN weights file (.nnwt) |
| `--experience` | `-e` | experience.csv | Experience buffer CSV output |
| `--output` | `-o` | results.csv | Results CSV output |
| `--error-bound` | `-b` | 0.0 (lossless) | Quantization error bound |
| `--threshold` | `-t` | 0.20 | Exploration threshold (0.0 = no exploration) |
| `--max-files` | `-m` | 0 (all) | Max files to process |
| `--reinforce` | | off | Enable online SGD reinforcement |
| `--reinforce-lr` | | 0.0001 | SGD learning rate |
| `--reinforce-threshold` | | 0.60 | MAPE threshold to trigger SGD |
| `--verbose` | `-v` | off | Detailed per-file reinforcement trace |
| `--help` | `-h` | | Show help |

## Output

- **Console**: Per-file line with ratio, MAPE, ct_mape, rolling averages, *SGD* markers
- **CSV**: Full per-file metrics (entropy, MAD, derivative, ratio, predicted values, MAPE, etc.)
- **Plot**: `eval/mape_reinforcement.png` — rolling MAPE + reinforcement rate (auto-generated)
