#!/bin/bash
# ============================================================
# Run All Benchmarks (Gray-Scott + VPIC)
#
# Single entry point for both benchmarks with shared configuration.
#
# Examples:
#
#   # Default: 512MB data, 16MB chunks, 50 timesteps, all phases, all policies
#   bash benchmarks/run_all.sh
#
#   # 1GB dataset, 32MB chunks, 100 timesteps
#   DATA_MB=1024 CHUNK_MB=32 TIMESTEPS=100 bash benchmarks/run_all.sh
#
#   # Only Gray-Scott, ratio policy, LR=0.1
#   BENCHMARKS=grayscott POLICIES=ratio SGD_LR=0.1 bash benchmarks/run_all.sh
#
#   # VPIC only, 1GB, 16MB chunks, 100 timesteps, all phases + all policies
#   BENCHMARKS=vpic DATA_MB=1024 CHUNK_MB=16 TIMESTEPS=100 \
#     SGD_LR=0.1 SGD_MAPE=0.1 EXPLORE_K=4 EXPLORE_THRESH=0.2 \
#     bash benchmarks/run_all.sh
#
#   # Both benchmarks, aggressive SGD (LR=0.5, MAPE=0.20), more exploration
#   SGD_LR=0.5 SGD_MAPE=0.20 EXPLORE_K=8 EXPLORE_THRESH=0.10 \
#     DATA_MB=512 CHUNK_MB=16 TIMESTEPS=50 bash benchmarks/run_all.sh
#
#   # Only NN phases, speed policy, small quick test
#   PHASES="nn,nn-rl,nn-rl+exp50" POLICIES=speed \
#     DATA_MB=16 CHUNK_MB=4 TIMESTEPS=5 bash benchmarks/run_all.sh
#
#   # Override grid sizes directly (ignore DATA_MB)
#   GS_L=640 VPIC_NX=254 CHUNK_MB=64 TIMESTEPS=100 bash benchmarks/run_all.sh
#
# ── Environment Variables ──────────────────────────────────
#
#   Variable        Default     Description
#   --------------- ----------- -----------------------------------------
#   BENCHMARKS      grayscott,vpic,sdrbench  Which to run (comma-separated)
#   SDR_DATASET     nyx              SDRBench dataset (hurricane_isabel, nyx, cesm_atm)
#   DATA_MB         512         Target dataset size in MB (auto-computes grid)
#   CHUNK_MB        16          Chunk size in MB
#   TIMESTEPS       50          Number of write/read/verify cycles
#
#   SGD_LR          0.2         SGD learning rate (higher = faster convergence)
#   SGD_MAPE        0.10        MAPE threshold for SGD updates (0.10 = 10%)
#   EXPLORE_K       4           Number of alternative configs to try
#   EXPLORE_THRESH  0.20        Cost error % that triggers exploration (0.20 = 20%)
#   VERIFY          1           Read-back + bitwise verify (0=skip, 1=verify)
#
#   POLICIES        balanced,ratio,speed   Cost model policies
#   PHASES          (all 12)    Compression phases to run
#   GS_L            (auto)      Gray-Scott grid size (L^3 * 4 bytes = dataset)
#   GS_STEPS        500         PDE simulation steps per timestep
#   VPIC_NX         (auto)      VPIC grid size (NX^3 * 24 bytes = dataset)
#   DEBUG_NN        0           NN debug output (0=off, 1=on)
#
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Shared defaults ──
BENCHMARKS=${BENCHMARKS:-"grayscott,vpic,sdrbench"}
DATA_MB=${DATA_MB:-512}
CHUNK_MB=${CHUNK_MB:-16}
TIMESTEPS=${TIMESTEPS:-50}
SGD_LR=${SGD_LR:-0.2}
SGD_MAPE=${SGD_MAPE:-0.10}
EXPLORE_K=${EXPLORE_K:-4}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.20}
VERIFY=${VERIFY:-1}
POLICIES=${POLICIES:-"balanced,ratio,speed"}
PHASES=${PHASES:-"no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"}
DEBUG_NN=${DEBUG_NN:-0}

# ── GS-specific defaults ──
GS_STEPS=${GS_STEPS:-500}

# ── Auto-compute grid sizes from DATA_MB if not explicitly set ──
# Gray-Scott: L^3 * 4 bytes = DATA_MB * 1024^2
# VPIC: NX^3 * 6 fields * 4 bytes = DATA_MB * 1024^2
if [ -z "$GS_L" ]; then
    # L = cube_root(DATA_MB * 1024 * 1024 / 4)
    GS_L=$(python3 -c "import math; print(int(round(math.pow($DATA_MB * 1024 * 1024 / 4, 1/3))))")
fi
if [ -z "$VPIC_NX" ]; then
    # VPIC writes (NX+2)^3 * 16 fields * 4 bytes = (NX+2)^3 * 64
    # NX = cube_root(DATA_MB * 1024 * 1024 / 64) - 2
    VPIC_NX=$(python3 -c "import math; print(int(round(math.pow($DATA_MB * 1024 * 1024 / 64, 1/3))) - 2)")
fi

GS_DATA=$(( GS_L * GS_L * GS_L * 4 / 1024 / 1024 ))
VPIC_DATA=$(( (VPIC_NX+2) * (VPIC_NX+2) * (VPIC_NX+2) * 64 / 1024 / 1024 ))

echo "============================================================"
echo "  GPUCompress Benchmark Suite"
echo "============================================================"
echo "  Benchmarks  : ${BENCHMARKS}"
echo "  Target data : ${DATA_MB} MB"
echo "  Chunk size  : ${CHUNK_MB} MB"
echo "  Timesteps   : ${TIMESTEPS}"
echo "  Policies    : ${POLICIES}"
echo "  Phases      : ${PHASES}"
echo ""
echo "  NN hyperparameters:"
echo "    SGD_LR=${SGD_LR}  SGD_MAPE=${SGD_MAPE}"
echo "    EXPLORE_K=${EXPLORE_K}  EXPLORE_THRESH=${EXPLORE_THRESH}"
echo ""
echo "  Gray-Scott  : L=${GS_L} (~${GS_DATA} MB), steps=${GS_STEPS}"
echo "  VPIC        : NX=${VPIC_NX} (~${VPIC_DATA} MB)"
echo "============================================================"
echo ""

IFS=',' read -ra BENCH_LIST <<< "$BENCHMARKS"

for bench in "${BENCH_LIST[@]}"; do
    case "$bench" in
        grayscott)
            echo ""
            echo ">>> Running Gray-Scott benchmark..."
            echo ""
            L=$GS_L \
            CHUNK_MB=$CHUNK_MB \
            TIMESTEPS=$TIMESTEPS \
            STEPS=$GS_STEPS \
            PHASES=$PHASES \
            POLICIES=$POLICIES \
            SGD_LR=$SGD_LR \
            SGD_MAPE=$SGD_MAPE \
            EXPLORE_K=$EXPLORE_K \
            EXPLORE_THRESH=$EXPLORE_THRESH \
            VERIFY=$VERIFY \
            DEBUG_NN=$DEBUG_NN \
            bash "$SCRIPT_DIR/grayscott/run_gs_pm_eval.sh"
            ;;
        vpic)
            echo ""
            echo ">>> Running VPIC benchmark (single invocation, GPU weight isolation)..."
            echo ""

            VPIC_BIN="$SCRIPT_DIR/vpic-kokkos/vpic_benchmark_deck.Linux"
            VPIC_DECK="$SCRIPT_DIR/vpic-kokkos/vpic_benchmark_deck.cxx"
            VPIC_WEIGHTS="$SCRIPT_DIR/../neural_net/weights/model.nnwt"
            VPIC_LD_PATH="/tmp/hdf5-install/lib:$SCRIPT_DIR/../build:/tmp/lib"

            # VPIC physics: faster reconnection for data variety
            VPIC_MI_ME=${VPIC_MI_ME:-5}
            VPIC_WPE_WCE=${VPIC_WPE_WCE:-2}
            VPIC_TI_TE=${VPIC_TI_TE:-3}
            VPIC_NPPC=${VPIC_NPPC:-2}
            VPIC_WARMUP=${VPIC_WARMUP_STEPS:-1000}
            VPIC_SIM_INT=${VPIC_SIM_INTERVAL:-50}

            # Convert PHASES to VPIC_EXCLUDE (invert the selection)
            # VPIC binary runs all 12 phases by default; VPIC_EXCLUDE skips unwanted ones
            ALL_VPIC_PHASES="no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"
            VPIC_EXCLUDE_LIST=""
            IFS=',' read -ra _ALL <<< "$ALL_VPIC_PHASES"
            for ap in "${_ALL[@]}"; do
                if ! echo ",$PHASES," | grep -q ",$ap,"; then
                    VPIC_EXCLUDE_LIST="${VPIC_EXCLUDE_LIST:+$VPIC_EXCLUDE_LIST,}$ap"
                fi
            done

            # Run per policy (different cost model weights)
            IFS=',' read -ra _POLICIES <<< "$POLICIES"
            for pol in "${_POLICIES[@]}"; do
                case "$pol" in
                    balanced) W0=1.0; W1=1.0; W2=1.0; LABEL="balanced_w1-1-1" ;;
                    ratio)    W0=0.0; W1=0.0; W2=1.0; LABEL="ratio_only_w0-0-1" ;;
                    speed)    W0=1.0; W1=1.0; W2=0.0; LABEL="speed_only_w1-1-0" ;;
                    *)        W0=1.0; W1=1.0; W2=1.0; LABEL="$pol" ;;
                esac

                VPIC_RESULTS="$SCRIPT_DIR/vpic-kokkos/results/eval_NX${VPIC_NX}_chunk${CHUNK_MB}mb_ts${TIMESTEPS}/$LABEL"
                mkdir -p "$VPIC_RESULTS"

                echo "  VPIC: NX=$VPIC_NX (~${VPIC_DATA} MB), chunk=${CHUNK_MB}MB, ts=${TIMESTEPS}, policy=$LABEL"
                echo "    warmup=$VPIC_WARMUP, sim_interval=$VPIC_SIM_INT, physics: mi_me=$VPIC_MI_ME wpe_wce=$VPIC_WPE_WCE Ti_Te=$VPIC_TI_TE"

                LD_LIBRARY_PATH="$VPIC_LD_PATH" \
                GPUCOMPRESS_DETAILED_TIMING=1 \
                GPUCOMPRESS_WEIGHTS="$VPIC_WEIGHTS" \
                GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
                VPIC_NX=$VPIC_NX \
                VPIC_NPPC=$VPIC_NPPC \
                VPIC_MI_ME=$VPIC_MI_ME \
                VPIC_WPE_WCE=$VPIC_WPE_WCE \
                VPIC_TI_TE=$VPIC_TI_TE \
                VPIC_WARMUP_STEPS=$VPIC_WARMUP \
                VPIC_TIMESTEPS=$TIMESTEPS \
                VPIC_SIM_INTERVAL=$VPIC_SIM_INT \
                VPIC_CHUNK_MB=$CHUNK_MB \
                VPIC_VERIFY=$VERIFY \
                VPIC_EXCLUDE="$VPIC_EXCLUDE_LIST" \
                VPIC_RESULTS_DIR="$VPIC_RESULTS" \
                VPIC_W0=$W0 VPIC_W1=$W1 VPIC_W2=$W2 \
                VPIC_LR=$SGD_LR \
                VPIC_MAPE_THRESHOLD=$SGD_MAPE \
                VPIC_EXPLORE_K=$EXPLORE_K \
                VPIC_EXPLORE_THRESH=$EXPLORE_THRESH \
                "$VPIC_BIN" "$VPIC_DECK" \
                > "$VPIC_RESULTS/vpic_benchmark.log" 2>&1

                echo "    Done. Log: $VPIC_RESULTS/vpic_benchmark.log"

                # Generate plots
                VPIC_DIR="$VPIC_RESULTS" \
                python3 "$SCRIPT_DIR/plots/generate_dataset_figures.py" \
                    --dataset vpic --policy "$LABEL" 2>&1 | grep -E "Generated"
            done
            ;;
        sdrbench)
            echo ""
            echo ">>> Running SDRBench benchmark..."
            echo ""
            DATASET=${SDR_DATASET:-nyx} \
            CHUNK_MB=$CHUNK_MB \
            PHASES=$PHASES \
            POLICIES=$POLICIES \
            SGD_LR=$SGD_LR \
            SGD_MAPE=$SGD_MAPE \
            EXPLORE_K=$EXPLORE_K \
            EXPLORE_THRESH=$EXPLORE_THRESH \
            VERIFY=$VERIFY \
            DEBUG_NN=$DEBUG_NN \
            bash "$SCRIPT_DIR/sdrbench/run_sdr_pm_eval.sh"
            ;;
        *)
            echo "ERROR: Unknown benchmark '$bench'. Use: grayscott, vpic, sdrbench"
            ;;
    esac
done

echo ""
echo "============================================================"
echo "  All benchmarks complete."
echo "============================================================"
