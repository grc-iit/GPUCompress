#!/bin/bash
# ============================================================
# Combined Benchmark Runner — VPIC + Gray-Scott + SDRBench
#
# VPIC: runs the binary directly (single warmup, all phases per timestep,
#       per-NN-phase GPU weight isolation — no redundant simulation reruns).
# Gray-Scott: uses run_all.sh (eval script).
# SDRBench: uses run_all_sdr.sh (all 3 datasets).
#
# Usage:
#   bash benchmarks/vpic-kokkos/runBEnchCombo.sh
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Shared configuration ──
TIMESTEPS=50
SGD_LR=0.15
SGD_MAPE=0.1
EXPLORE_K=4
EXPLORE_THRESH=0.2

# ── VPIC configuration ──
VPIC_BIN="$SCRIPT_DIR/vpic_benchmark_deck.Linux"
VPIC_DECK="$SCRIPT_DIR/vpic_benchmark_deck.cxx"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
VPIC_LD_PATH="/tmp/hdf5-install/lib:$GPU_DIR/build:/tmp/lib"

# VPIC physics: faster reconnection for data variety
VPIC_PHYSICS="VPIC_MI_ME=5 VPIC_WPE_WCE=2 VPIC_TI_TE=3"

# NX for ~1GB field data: (NX+2)^3 * 64 bytes ≈ 1 GB → NX=254
VPIC_NX=254
VPIC_WARMUP=500
VPIC_SIM_INTERVAL=10

# Phases to exclude (keep: no-comp, lz4, zstd, nn, nn-rl, nn-rl+exp50)
VPIC_EXCLUDE="snappy,deflate,gdeflate,ans,cascaded,bitcomp"

run_vpic() {
    local CHUNK_MB=$1
    local LABEL="NX${VPIC_NX}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}"
    local RESULTS_DIR="$SCRIPT_DIR/results/eval_${LABEL}"
    mkdir -p "$RESULTS_DIR"

    echo ""
    echo "============================================================"
    echo "  VPIC: $LABEL (single invocation, all phases per timestep)"
    echo "============================================================"
    echo "  Field data : ~1 GB  |  Chunks: ${CHUNK_MB} MB"
    echo "  Warmup: ${VPIC_WARMUP} steps  |  Timesteps: ${TIMESTEPS}"
    echo "  Sim interval: ${VPIC_SIM_INTERVAL} steps between writes"
    echo "  Results: $RESULTS_DIR"
    echo ""

    LD_LIBRARY_PATH="$VPIC_LD_PATH" \
    GPUCOMPRESS_DETAILED_TIMING=1 \
    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    VPIC_NX=$VPIC_NX \
    VPIC_NPPC=2 \
    VPIC_MI_ME=5 VPIC_WPE_WCE=2 VPIC_TI_TE=3 \
    VPIC_WARMUP_STEPS=$VPIC_WARMUP \
    VPIC_TIMESTEPS=$TIMESTEPS \
    VPIC_SIM_INTERVAL=$VPIC_SIM_INTERVAL \
    VPIC_CHUNK_MB=$CHUNK_MB \
    VPIC_VERIFY=0 \
    VPIC_EXCLUDE="$VPIC_EXCLUDE" \
    VPIC_RESULTS_DIR="$RESULTS_DIR" \
    VPIC_W0=1.0 VPIC_W1=1.0 VPIC_W2=1.0 \
    VPIC_LR=$SGD_LR \
    VPIC_MAPE_THRESHOLD=$SGD_MAPE \
    VPIC_EXPLORE_K=$EXPLORE_K \
    VPIC_EXPLORE_THRESH=$EXPLORE_THRESH \
    "$VPIC_BIN" "$VPIC_DECK" \
    > "$RESULTS_DIR/vpic_benchmark.log" 2>&1

    echo "  Done. Log: $RESULTS_DIR/vpic_benchmark.log"
}

echo "============================================================"
echo "  GPUCompress Combined Benchmark Suite"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ── VPIC: 3 chunk sizes, single warmup each ──
for CHUNK in 4 16 64; do
    echo ">>> VPIC (CHUNK_MB=$CHUNK) at $(date)"
    run_vpic $CHUNK
    echo ">>> VPIC (CHUNK_MB=$CHUNK) finished at $(date)"
    echo ""
done

# ── Gray-Scott: 3 chunk sizes via run_all.sh ──
for CHUNK in 4 16 64; do
    echo ">>> Gray-Scott (CHUNK_MB=$CHUNK) at $(date)"
    BENCHMARKS=grayscott DATA_MB=1024 CHUNK_MB=$CHUNK TIMESTEPS=$TIMESTEPS \
    SGD_LR=$SGD_LR SGD_MAPE=$SGD_MAPE \
    EXPLORE_K=$EXPLORE_K EXPLORE_THRESH=$EXPLORE_THRESH \
    VERIFY=0 bash "$GPU_DIR/benchmarks/run_all.sh"
    echo ">>> Gray-Scott (CHUNK_MB=$CHUNK) finished at $(date)"
    echo ""
done

# ── SDRBench: all 3 datasets, 3 chunk sizes ──
for CHUNK in 4 16 64; do
    echo ">>> SDRBench all datasets (CHUNK_MB=$CHUNK) at $(date)"
    CHUNK_MB=$CHUNK VERIFY=0 \
    PHASES="no-comp,lz4,zstd,nn,nn-rl,nn-rl+exp50" \
    POLICIES="balanced" \
    SGD_LR=$SGD_LR SGD_MAPE=$SGD_MAPE \
    EXPLORE_K=$EXPLORE_K EXPLORE_THRESH=$EXPLORE_THRESH \
    bash "$GPU_DIR/benchmarks/sdrbench/run_all_sdr.sh"
    echo ">>> SDRBench (CHUNK_MB=$CHUNK) finished at $(date)"
    echo ""
done

echo ""
echo "============================================================"
echo "  All benchmarks complete at $(date)"
echo "============================================================"
