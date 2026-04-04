#!/bin/bash
# ============================================================
# Unified Benchmark Runner
#
# Runs all workloads: VPIC, SDRBench, AI Training
# Each with 4MB + 16MB chunks × lossless + lossy × LR sweep
#
# Directory naming: includes chunk, mode, LR, and run ID
# so multiple runs never overwrite each other.
#
# Usage:
#   nohup bash benchmarks/run_all_benchmarks.sh > all_benchmarks.log 2>&1 &
#
#   # Only VPIC
#   WORKLOADS=vpic bash benchmarks/run_all_benchmarks.sh
#
#   # Only SDRBench, single LR
#   WORKLOADS=sdr SGD_LR_LIST=0.2 bash benchmarks/run_all_benchmarks.sh
#
#   # Quick smoke test
#   WORKLOADS=vpic VPIC_NX=64 TIMESTEPS=1 SGD_LR_LIST=0.2 bash benchmarks/run_all_benchmarks.sh
#
#   # AI only, ResNet-18, 2 epochs
#   WORKLOADS=ai AI_VIT_MODEL=resnet18 AI_EPOCHS=2 bash benchmarks/run_all_benchmarks.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export LD_LIBRARY_PATH=/tmp/hdf5-install/lib:${PROJECT_DIR}/build:${LD_LIBRARY_PATH:-}

# ── Global defaults ──
WORKLOADS=${WORKLOADS:-"vpic,sdr,ai"}
ERROR_BOUND=${ERROR_BOUND:-0.01}
POLICIES=${POLICIES:-"balanced,ratio"}
VERIFY=${VERIFY:-1}
SGD_LR_LIST=${SGD_LR_LIST:-"0.2,0.3,0.4"}
SGD_MAPE=${SGD_MAPE:-0.10}
EXPLORE_K=${EXPLORE_K:-4}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.20}

# ── VPIC defaults ──
VPIC_NX=${VPIC_NX:-320}
TIMESTEPS=${TIMESTEPS:-50}

# ── SDR defaults ──
SDR_DATASETS=${SDR_DATASETS:-"nyx,hurricane_isabel,cesm_atm"}

# ── AI defaults ──
AI_EPOCHS=${AI_EPOCHS:-10}
AI_VIT_MODEL=${AI_VIT_MODEL:-vit_b_16}

TOTAL_START=$(date +%s)
RUN_ID=$(date +%Y%m%d_%H%M%S)

# Chunk × mode configs
CONFIGS=(
    "4:0.0:4mb_lossless"
    "16:0.0:16mb_lossless"
    "4:${ERROR_BOUND}:4mb_lossy_eb${ERROR_BOUND}"
    "16:${ERROR_BOUND}:16mb_lossy_eb${ERROR_BOUND}"
)

IFS=',' read -ra WORKLOAD_LIST <<< "$WORKLOADS"
IFS=',' read -ra LR_LIST <<< "$SGD_LR_LIST"

# Count total runs
N_CONFIGS=${#CONFIGS[@]}
N_LRS=${#LR_LIST[@]}
N_WORKLOADS=${#WORKLOAD_LIST[@]}

echo "============================================================"
echo "  Unified Benchmark Runner"
echo "  Started: $(date)"
echo "  Run ID:  ${RUN_ID}"
echo "============================================================"
echo ""
echo "  Workloads:      ${WORKLOADS}"
echo "  Chunk sizes:    4 MB, 16 MB"
echo "  Modes:          lossless, lossy (eb=${ERROR_BOUND})"
echo "  Policies:       ${POLICIES}"
echo "  SGD LR sweep:   ${SGD_LR_LIST} (${N_LRS} values)"
echo "  SGD MAPE:       ${SGD_MAPE}"
echo "  Explore K:      ${EXPLORE_K}"
echo "  Explore Thresh: ${EXPLORE_THRESH}"
echo "  Run ID:         ${RUN_ID}"
echo ""
echo "  Per workload: ${N_CONFIGS} configs × ${N_LRS} LRs = $(( N_CONFIGS * N_LRS )) runs"
echo ""

# ============================================================
# VPIC
# ============================================================
run_vpic() {
    echo ""
    echo "============================================================"
    echo "  VPIC Benchmark (NX=${VPIC_NX}, ${TIMESTEPS} timesteps)"
    echo "  LR sweep: ${SGD_LR_LIST}"
    echo "============================================================"

    local W_START=$(date +%s)

    for LR in "${LR_LIST[@]}"; do
        for cfg in "${CONFIGS[@]}"; do
            IFS=: read -r CHUNK EB LABEL <<< "$cfg"

            # Unique suffix: _4mb_lossless_lr0.2_20260403_053900
            local SUFFIX="_${LABEL}_lr${LR}_${RUN_ID}"

            echo ""
            echo "  ── VPIC ${LABEL} lr=${LR} ──"

            BENCHMARKS=vpic \
            VPIC_NX=$VPIC_NX \
            CHUNK_MB=$CHUNK \
            TIMESTEPS=$TIMESTEPS \
            POLICIES=$POLICIES \
            VERIFY=$VERIFY \
            SGD_LR=$LR \
            SGD_MAPE=$SGD_MAPE \
            EXPLORE_K=$EXPLORE_K \
            EXPLORE_THRESH=$EXPLORE_THRESH \
            VPIC_ERROR_BOUND=$EB \
            VPIC_EVAL_SUFFIX="$SUFFIX" \
            bash "$SCRIPT_DIR/benchmark.sh"
        done
    done

    local W_END=$(date +%s)
    echo "  VPIC complete: $(( (W_END - W_START) / 60 )) minutes"
}

# ============================================================
# SDRBench
# ============================================================
run_sdr() {
    echo ""
    echo "============================================================"
    echo "  SDRBench (${SDR_DATASETS})"
    echo "  LR sweep: ${SGD_LR_LIST}"
    echo "============================================================"

    local W_START=$(date +%s)

    for LR in "${LR_LIST[@]}"; do
        for cfg in "${CONFIGS[@]}"; do
            IFS=: read -r CHUNK EB LABEL <<< "$cfg"

            # Unique suffix: _4mb_lossless_lr0.2_20260403_053900
            local SUFFIX="_${LABEL}_lr${LR}_${RUN_ID}"

            echo ""
            echo "  ── SDR ${LABEL} lr=${LR} ──"

            SDR_DATASETS="$SDR_DATASETS" \
            CHUNK_MB=$CHUNK \
            ERROR_BOUND=$EB \
            VERIFY=$VERIFY \
            POLICIES=$POLICIES \
            SGD_LR=$LR \
            SGD_MAPE=$SGD_MAPE \
            EXPLORE_K=$EXPLORE_K \
            EXPLORE_THRESH=$EXPLORE_THRESH \
            VPIC_EVAL_SUFFIX="$SUFFIX" \
            bash "$SCRIPT_DIR/sdrbench/run_sdr.sh"
        done
    done

    local W_END=$(date +%s)
    echo "  SDRBench complete: $(( (W_END - W_START) / 60 )) minutes"
}

# ============================================================
# AI Training (ViT-Base + GPT-2, inline benchmark)
# ============================================================
run_ai() {
    echo ""
    echo "============================================================"
    echo "  AI Training Benchmark"
    echo "  Models: ${AI_VIT_MODEL}, GPT-2"
    echo "  Epochs: ${AI_EPOCHS}"
    echo "  SGD LR sweep: ${SGD_LR_LIST}"
    echo "============================================================"

    local W_START=$(date +%s)
    local CKPT_STR=$(seq -s, 1 "$AI_EPOCHS")

    # Build benchmark-configs string for all 4 chunk×mode combos
    build_ai_configs() {
        local BASE="$1"
        echo "4:0.0:${BASE}_4mb_lossless,16:0.0:${BASE}_16mb_lossless,4:${ERROR_BOUND}:${BASE}_4mb_lossy_eb${ERROR_BOUND},16:${ERROR_BOUND}:${BASE}_16mb_lossy_eb${ERROR_BOUND}"
    }

    for LR in "${LR_LIST[@]}"; do
        echo ""
        echo "  ── SGD_LR=${LR} ──"

        # ViT-Base (or ResNet-18)
        local VIT_LABEL
        if [ "$AI_VIT_MODEL" = "resnet18" ]; then
            VIT_LABEL="resnet18"
        else
            VIT_LABEL="vit_b"
        fi
        local VIT_BASE="${PROJECT_DIR}/data/ai_training/${VIT_LABEL}_lr${LR}_${RUN_ID}"
        local VIT_CONFIGS=$(build_ai_configs "$VIT_BASE")

        echo "    ViT: ${AI_VIT_MODEL} (lr=${LR})"
        python3 "${PROJECT_DIR}/scripts/train_and_export_checkpoints.py" \
            --model "${AI_VIT_MODEL}" \
            --epochs "${AI_EPOCHS}" \
            --checkpoint-epochs "${CKPT_STR}" \
            --hdf5-direct \
            --benchmark-configs "${VIT_CONFIGS}" \
            --sgd-lr "${LR}" \
            --sgd-mape "${SGD_MAPE}" \
            --explore-k "${EXPLORE_K}" \
            --explore-thresh "${EXPLORE_THRESH}" \
            --outdir "${VIT_BASE}_default" \
            2>&1 | tail -3

        # GPT-2
        local GPT2_BASE="${PROJECT_DIR}/data/ai_training/gpt2_lr${LR}_${RUN_ID}"
        local GPT2_CONFIGS=$(build_ai_configs "$GPT2_BASE")

        echo "    GPT-2 (lr=${LR})"
        python3 "${PROJECT_DIR}/scripts/train_gpt2_checkpoints.py" \
            --epochs "${AI_EPOCHS}" \
            --checkpoint-epochs "${CKPT_STR}" \
            --hdf5-direct \
            --benchmark-configs "${GPT2_CONFIGS}" \
            --sgd-lr "${LR}" \
            --sgd-mape "${SGD_MAPE}" \
            --explore-k "${EXPLORE_K}" \
            --explore-thresh "${EXPLORE_THRESH}" \
            --outdir "${GPT2_BASE}_default" \
            2>&1 | tail -3
    done

    local W_END=$(date +%s)
    echo "  AI complete: $(( (W_END - W_START) / 60 )) minutes"
}

# ============================================================
# Run selected workloads
# ============================================================
for workload in "${WORKLOAD_LIST[@]}"; do
    case "$workload" in
        vpic) run_vpic ;;
        sdr)  run_sdr ;;
        ai)   run_ai ;;
        *)    echo "Unknown workload: $workload (use: vpic, sdr, ai)" ;;
    esac
done

# ============================================================
# Summary
# ============================================================
TOTAL_END=$(date +%s)
TOTAL_MIN=$(( (TOTAL_END - TOTAL_START) / 60 ))
TOTAL_HRS=$(( TOTAL_MIN / 60 ))
REMAINDER_MIN=$(( TOTAL_MIN % 60 ))

echo ""
echo "============================================================"
echo "  All Benchmarks Complete"
echo "  Finished: $(date)"
echo "  Total time: ${TOTAL_MIN} minutes (${TOTAL_HRS}h ${REMAINDER_MIN}m)"
echo "  Run ID: ${RUN_ID}"
echo "============================================================"
echo ""
echo "  Directory naming convention:"
echo "    VPIC: eval_NX{nx}_chunk{mb}mb_ts{ts}_{label}_lr{lr}_{run_id}/"
echo "    SDR:  eval_{dataset}_chunk{mb}mb_{label}_lr{lr}_{run_id}/"
echo "    AI:   {model}_lr{lr}_{run_id}_{chunk}mb_{mode}/"
echo "============================================================"
