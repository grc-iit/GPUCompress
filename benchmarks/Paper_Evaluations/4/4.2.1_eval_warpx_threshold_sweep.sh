#!/bin/bash
# ============================================================
# WarpX Threshold Sweep: SGD MAPE (X1) x Exploration (X2) grid
#
# Mirrors 4.2.1_eval_vpic_threshold_sweep.sh for WarpX LWFA data.
# Runs nn-rl+exp phase only, sweeping X1 and X2=X1+delta.
#
# Environment overrides:
#   CHUNK_MB          Chunk size (default 4)
#   TIMESTEPS         Number of write cycles (default 50)
#   WARPX_DATA_MB     Data per write in MB (default 192)
#   WARPX_ERROR_BOUND Lossy error bound (default 0.01, 0 for lossless)
#   WARMUP_STEPS      Simulation warmup (default 100)
#   SIM_INTERVAL      Physics steps between writes (default 10)
#   EXPLORE_K         Exploration alternatives (default 4)
#   SGD_LR            Learning rate (default 0.2)
#   POLICY            Cost policy: balanced|ratio|speed (default balanced)
#   DRY_RUN           1 = print commands without running
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

WARPX_BIN="$REPO_DIR/warpx_hyperparameter_study"
WEIGHTS="$REPO_DIR/neural_net/weights/model.nnwt"

if [ ! -x "$WARPX_BIN" ]; then
    echo "ERROR: $WARPX_BIN not found. Build with: cmake --build . --target warpx_hyperparameter_study"
    exit 1
fi

# ── Defaults ──
CHUNK_MB=${CHUNK_MB:-4}
TIMESTEPS=${TIMESTEPS:-50}
WARPX_DATA_MB=${WARPX_DATA_MB:-192}
WARPX_ERROR_BOUND=${WARPX_ERROR_BOUND:-0.01}
WARMUP_STEPS=${WARMUP_STEPS:-100}
SIM_INTERVAL=${SIM_INTERVAL:-10}
EXPLORE_K=${EXPLORE_K:-4}
SGD_LR=${SGD_LR:-0.2}
POLICY=${POLICY:-balanced}
DRY_RUN=${DRY_RUN:-0}

# Policy weights
case "$POLICY" in
    balanced) W0=1.0; W1=1.0; W2=1.0 ;;
    ratio)    W0=0.0; W1=0.0; W2=1.0 ;;
    speed)    W0=1.0; W1=1.0; W2=0.0 ;;
    *) echo "Unknown policy: $POLICY"; exit 1 ;;
esac

# Error bound tag for directory name
if [ "$(echo "$WARPX_ERROR_BOUND > 0" | bc -l)" = "1" ]; then
    EB_TAG="eb${WARPX_ERROR_BOUND}"
else
    EB_TAG="lossless"
fi

# Results directory
RESULTS_BASE="$SCRIPT_DIR/results/warpx_threshold_sweep_${POLICY}_${EB_TAG}_lr${SGD_LR}"
mkdir -p "$RESULTS_BASE"

echo "============================================================"
echo "WarpX Threshold Sweep"
echo "  Policy:      $POLICY (w0=$W0, w1=$W1, w2=$W2)"
echo "  Data:        ${WARPX_DATA_MB} MB, chunk ${CHUNK_MB} MB"
echo "  Timesteps:   $TIMESTEPS (warmup=$WARMUP_STEPS, interval=$SIM_INTERVAL)"
echo "  Error bound: $WARPX_ERROR_BOUND ($EB_TAG)"
echo "  Explore K:   $EXPLORE_K, LR: $SGD_LR"
echo "  Results:     $RESULTS_BASE"
echo "============================================================"
echo ""

# ── Sweep grid: X1 (SGD MAPE) x delta (exploration = X1 + delta) ──
X1_VALUES=(0.05 0.10 0.20 0.30)
DELTA_VALUES=(0.05 0.10 0.20 0.30)

TOTAL=$(( ${#X1_VALUES[@]} * ${#DELTA_VALUES[@]} ))
RUN=0

for x1 in "${X1_VALUES[@]}"; do
    for delta in "${DELTA_VALUES[@]}"; do
        x2=$(echo "$x1 + $delta" | bc -l)
        RUN=$((RUN + 1))

        OUT_DIR="$RESULTS_BASE/x1_${x1}_delta_${delta}"
        if [ -f "$OUT_DIR/benchmark_warpx_deck.csv" ]; then
            echo "[$RUN/$TOTAL] SKIP x1=$x1 delta=$delta (results exist)"
            continue
        fi

        mkdir -p "$OUT_DIR"
        echo "[$RUN/$TOTAL] x1=$x1 x2=$x2 (delta=$delta) → $OUT_DIR"

        if [ "$DRY_RUN" = "1" ]; then
            echo "  DRY_RUN: would run warpx_hyperparameter_study"
            continue
        fi

        GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
        WARPX_TIMESTEPS=$TIMESTEPS \
        WARPX_WARMUP_STEPS=$WARMUP_STEPS \
        WARPX_SIM_INTERVAL=$SIM_INTERVAL \
        WARPX_POLICIES=$POLICY \
        WARPX_DATA_MB=$WARPX_DATA_MB \
        WARPX_CHUNK_MB=$CHUNK_MB \
        WARPX_ERROR_BOUND=$WARPX_ERROR_BOUND \
        WARPX_LR=$SGD_LR \
        WARPX_MAPE_THRESHOLD=$x1 \
        WARPX_EXPLORE_THRESH=$x2 \
        WARPX_EXPLORE_K=$EXPLORE_K \
        WARPX_VERIFY=0 \
        WARPX_PHASE=nn-rl+exp \
        WARPX_RESULTS_DIR="$OUT_DIR" \
        "$WARPX_BIN" > "$OUT_DIR/warpx.log" 2>&1

        # Show summary
        if [ -f "$OUT_DIR/benchmark_warpx_deck.csv" ]; then
            tail -1 "$OUT_DIR/benchmark_warpx_deck.csv" | head -1
        fi
        echo ""
    done
done

echo "============================================================"
echo "Threshold sweep complete: $TOTAL configurations"
echo "Results: $RESULTS_BASE/"
echo "============================================================"
