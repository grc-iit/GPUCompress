#!/bin/bash
# ============================================================
# Full WarpX Paper Evaluation
#
# Mirrors run_vpic_full.sh for WarpX LWFA compression benchmarks.
#
# 1. Threshold sweep (4x4 grid, 4MB chunks, 192MB dataset)
# 2. Full benchmark — all phases (fixed + NN), balanced + ratio,
#    lossy (eb=0.01), 32MB chunks, ~192MB dataset, 25 timesteps
#
# Usage:
#   bash benchmarks/Paper_Evaluations/4/run_warpx_full.sh
# ============================================================
set -e

cd /home/cc/GPUCompress

WARPX_BIN="./warpx_hyperparameter_study"
WEIGHTS="neural_net/weights/model.nnwt"
RESULTS_BASE="benchmarks/Paper_Evaluations/4/results"

if [ ! -x "$WARPX_BIN" ]; then
    echo "Building warpx_hyperparameter_study..."
    cmake --build . --target warpx_hyperparameter_study -j $(nproc) 2>&1 | tail -3
fi

echo "============================================================"
echo "Full WarpX Paper Evaluation"
echo "============================================================"

# ── Step 1: Threshold Sweep ──
echo ""
echo ">>> Step 1: Threshold Sweep (4MB chunks, 192MB dataset, 50 timesteps)"
CHUNK_MB=4 \
TIMESTEPS=50 \
WARPX_DATA_MB=192 \
WARPX_ERROR_BOUND=0.01 \
WARMUP_STEPS=100 \
SIM_INTERVAL=10 \
    bash benchmarks/Paper_Evaluations/4/4.2.1_eval_warpx_threshold_sweep.sh

# Run plotting if available
if [ -f benchmarks/Paper_Evaluations/4/4.2.1_plot_threshold_sweep.py ]; then
    echo ""
    echo "  Generating threshold sweep heatmaps..."
    python3 benchmarks/Paper_Evaluations/4/4.2.1_plot_threshold_sweep.py \
        "$RESULTS_BASE/warpx_threshold_sweep_balanced_eb0.01_lr0.2" 2>/dev/null || true
fi

# ── Step 2: Full Benchmark ──
# 192 MB per timestep, 32MB chunks = 6 chunks per write
# 25 timesteps with warmup=100 and sim_interval=10 for data evolution
echo ""
echo ">>> Step 2: Full Benchmark (all phases, 32MB chunks, 192MB dataset, 25 timesteps)"

run_full_benchmark() {
    local POLICY=$1
    local TAG=$2

    echo ""
    echo "  [$TAG] Policy=$POLICY, lossy (eb=0.01)"

    local OUT_DIR="$RESULTS_BASE/warpx_full_${POLICY}_chunk32mb_ts25_lossy0.01"
    mkdir -p "$OUT_DIR"

    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    WARPX_TIMESTEPS=25 \
    WARPX_WARMUP_STEPS=100 \
    WARPX_SIM_INTERVAL=10 \
    WARPX_POLICIES=$POLICY \
    WARPX_DATA_MB=192 \
    WARPX_CHUNK_MB=32 \
    WARPX_ERROR_BOUND=0.01 \
    WARPX_LR=0.2 \
    WARPX_MAPE_THRESHOLD=0.20 \
    WARPX_EXPLORE_THRESH=0.20 \
    WARPX_EXPLORE_K=4 \
    WARPX_VERIFY=0 \
    WARPX_RESULTS_DIR="$OUT_DIR" \
    "$WARPX_BIN" 2>&1 | tee "$OUT_DIR/warpx.log"

    echo ""
    echo "  Results: $OUT_DIR/"
    if [ -f "$OUT_DIR/benchmark_warpx_deck.csv" ]; then
        echo "  Aggregate CSV summary:"
        head -1 "$OUT_DIR/benchmark_warpx_deck.csv"
        cat "$OUT_DIR/benchmark_warpx_deck.csv" | tail -n +2 | \
            awk -F, '{ printf "    %-20s ratio=%-6s write=%-8s sgd=%-4s expl=%-4s mape=%-6s\n", $3, $11, $5, $15, $16, $27 }'
    fi
}

# 2a: Balanced policy
run_full_benchmark "balanced" "2a"

# 2b: Ratio policy
run_full_benchmark "ratio" "2b"

echo ""
echo "============================================================"
echo "Full WarpX Paper Evaluation Complete"
echo ""
echo "Results:"
echo "  Sweep:    $RESULTS_BASE/warpx_threshold_sweep_balanced_eb0.01_lr0.2/"
echo "  Balanced: $RESULTS_BASE/warpx_full_balanced_chunk32mb_ts25_lossy0.01/"
echo "  Ratio:    $RESULTS_BASE/warpx_full_ratio_chunk32mb_ts25_lossy0.01/"
echo "============================================================"
