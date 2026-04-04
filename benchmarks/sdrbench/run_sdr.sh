#!/bin/bash
# ============================================================
# SDRBench Benchmark Driver
#
# Runs compression phases across one or more SDRBench datasets,
# merges per-phase CSVs, and generates plots.
#
# Usage:
#   # Single dataset (default: nyx)
#   bash benchmarks/sdrbench/run_sdr.sh
#
#   # Multiple datasets
#   DATASET=nyx,hurricane_isabel,cesm_atm bash benchmarks/sdrbench/run_sdr.sh
#
#   # Custom chunk size + fewer phases
#   CHUNK_MB=64 PHASES="no-comp,lz4,zstd,nn,nn-rl,nn-rl+exp50" \
#     bash benchmarks/sdrbench/run_sdr.sh
#
#   # Re-generate plots only (no benchmark runs)
#   PLOT_ONLY=1 bash benchmarks/sdrbench/run_sdr.sh
#
# Environment Variables:
#   DATASET         nyx                     Comma-separated dataset names
#   CHUNK_MB        8                       Chunk size in MB
#   PHASES          (all 12)                Compression phases
#   POLICIES        balanced,ratio,speed    Cost model policies
#   SGD_LR          0.2                     SGD learning rate
#   SGD_MAPE        0.10                    MAPE threshold
#   EXPLORE_K       4                       Exploration alternatives
#   EXPLORE_THRESH  0.20                    Exploration error threshold
#   VERIFY          1                       Read-back verification (0=skip)
#   ERROR_BOUND     0.0                     Lossy error bound (0=lossless)
#   DEBUG_NN        0                       NN debug output
#   PLOT_ONLY       0                       Skip benchmarks, re-plot only
#   NO_RANKING      0                       Skip Kendall tau ranking profiler
#   MPI_NP          1                       MPI ranks
#   GPUS_PER_NODE   1                       GPUs per node
#   LAUNCHER        auto                    MPI launcher (auto/srun/mpirun)
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── Configuration ──
DATASET=${DATASET:-"nyx"}
CHUNK_MB=${CHUNK_MB:-8}
PHASES=${PHASES:-"no-comp,lz4,snappy,deflate,gdeflate,zstd,ans,cascaded,bitcomp,nn,nn-rl,nn-rl+exp50"}
POLICIES=${POLICIES:-"balanced,ratio,speed"}
SGD_LR=${SGD_LR:-0.2}
SGD_MAPE=${SGD_MAPE:-0.10}
EXPLORE_K=${EXPLORE_K:-4}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.20}
VERIFY=${VERIFY:-1}
ERROR_BOUND=${ERROR_BOUND:-0.0}
DEBUG_NN=${DEBUG_NN:-0}
PLOT_ONLY=${PLOT_ONLY:-0}
MPI_NP=${MPI_NP:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
LAUNCHER=${LAUNCHER:-auto}

# ── Paths ──
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"
SDR_BIN="$GPU_DIR/build/generic_benchmark"
SDR_DATA_DIR="$GPU_DIR/data/sdrbench"

# ── Dataset configs ──
declare -A DS_SUBDIR DS_DIMS DS_EXT
DS_SUBDIR[hurricane_isabel]="hurricane_isabel/100x500x500"
DS_DIMS[hurricane_isabel]="100,500,500"
DS_EXT[hurricane_isabel]=".bin.f32"
DS_SUBDIR[nyx]="nyx/SDRBENCH-EXASKY-NYX-512x512x512"
DS_DIMS[nyx]="512,512,512"
DS_EXT[nyx]=".f32"
DS_SUBDIR[cesm_atm]="cesm_atm/SDRBENCH-CESM-ATM-cleared-1800x3600"
DS_DIMS[cesm_atm]="1800,3600"
DS_EXT[cesm_atm]=".dat"
DS_SUBDIR[vit_b_cifar10]="vit_b_cifar10"
DS_DIMS[vit_b_cifar10]="2,42903173"
DS_EXT[vit_b_cifar10]=".f32"

# ── Policy configs ──
declare -A POLICY_W0 POLICY_W1 POLICY_W2 POLICY_LABELS
POLICY_W0[balanced]=1.0;  POLICY_W1[balanced]=1.0;  POLICY_W2[balanced]=1.0
POLICY_W0[ratio]=0.0;     POLICY_W1[ratio]=0.0;     POLICY_W2[ratio]=1.0
POLICY_W0[speed]=1.0;     POLICY_W1[speed]=1.0;     POLICY_W2[speed]=0.0
POLICY_LABELS[balanced]="balanced_w1-1-1"
POLICY_LABELS[ratio]="ratio_only_w0-0-1"
POLICY_LABELS[speed]="speed_only_w1-1-0"

# ── MPI launcher ──
mpi_launch() {
    if [ "$MPI_NP" -le 1 ]; then
        "$@"
        return
    fi
    local use_launcher="$LAUNCHER"
    if [ "$use_launcher" = "auto" ]; then
        if [ -n "$SLURM_JOB_ID" ]; then use_launcher="srun"; else use_launcher="mpirun"; fi
    fi
    case "$use_launcher" in
        srun)
            srun --ntasks="$MPI_NP" --gpus-per-task=1 --kill-on-bad-exit=1 "$@"
            ;;
        mpirun)
            mpirun -np "$MPI_NP" \
                bash -c 'export CUDA_VISIBLE_DEVICES=$((${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${MPI_LOCALRANKID:-${SLURM_LOCALID:-0}}}} % '"$GPUS_PER_NODE"')); exec "$@"' \
                _ "$@"
            ;;
        *) echo "ERROR: Unknown LAUNCHER=$use_launcher" >&2; return 1 ;;
    esac
}

# ── Generate plots for one dataset eval directory ──
generate_plots() {
    local ds="$1" eval_dir="$2"
    echo "Generating figures for $ds..."
    for policy in "${POLICY_LIST[@]}"; do
        local label="${POLICY_LABELS[$policy]}"
        local policy_dir="$eval_dir/$label"
        local fig_dir="$policy_dir/figures"
        mkdir -p "$fig_dir"

        SDR_DIR="$eval_dir" SDR_CHUNK="$CHUNK_MB" \
        python3 "$GPU_DIR/benchmarks/plots/generate_dataset_figures.py" \
            --dataset "$ds" --policy "$label" 2>&1 | grep -E "Saved|Generated|Error"

        local plot_src="$GPU_DIR/benchmarks/results/per_dataset/${ds}/${EVAL_NAME}/${label}"
        if [ -d "$plot_src" ]; then
            mv "$plot_src"/*.png "$fig_dir/" 2>/dev/null
            echo "  Figures → $fig_dir/ ($(ls "$fig_dir"/*.png 2>/dev/null | wc -l) plots)"
        else
            local plot_src2="$GPU_DIR/benchmarks/results/per_dataset/${ds}/${label}"
            if [ -d "$plot_src2" ]; then
                mv "$plot_src2"/*.png "$fig_dir/" 2>/dev/null
                echo "  Figures → $fig_dir/ ($(ls "$fig_dir"/*.png 2>/dev/null | wc -l) plots)"
            fi
        fi
    done
}

# ── Run one dataset ──
run_dataset() {
    local ds="$1"
    local data_dir="${DATA_DIR:-$SDR_DATA_DIR/${DS_SUBDIR[$ds]}}"
    local dims="${DIMS:-${DS_DIMS[$ds]}}"
    local ext="${EXT:-${DS_EXT[$ds]}}"

    # Eval directory
    local verify_tag="" lossy_tag=""
    [ "${VERIFY}" = "0" ] && verify_tag="_noverify"
    [ "$ERROR_BOUND" != "0.0" ] && [ "$ERROR_BOUND" != "0" ] && lossy_tag="_lossy"
    EVAL_NAME="eval_${ds}_chunk${CHUNK_MB}mb${verify_tag}${lossy_tag}${VPIC_EVAL_SUFFIX:-}"
    local eval_dir="$SCRIPT_DIR/results/$EVAL_NAME"

    # Plot-only mode
    if [ "$PLOT_ONLY" = "1" ]; then
        generate_plots "$ds" "$eval_dir"
        return
    fi

    # Verify binary and data
    if [ ! -f "$SDR_BIN" ]; then
        echo "ERROR: generic_benchmark not found: $SDR_BIN"
        echo "Build it: cmake --build build --target generic_benchmark -j\$(nproc)"
        return 1
    fi
    if [ ! -d "$data_dir" ]; then
        echo "ERROR: Data directory not found: $data_dir"
        return 1
    fi

    mkdir -p "$eval_dir"
    cat > "$eval_dir/params.txt" <<PARAMS_EOF
# SDRBench Benchmark Parameters
# Generated: $(date -Iseconds)
DATASET=$ds
DATA_DIR=$data_dir
DIMS=$dims
EXT=$ext
CHUNK_MB=$CHUNK_MB
PHASES=$PHASES
POLICIES=$POLICIES
SGD_LR=$SGD_LR
SGD_MAPE=$SGD_MAPE
EXPLORE_K=$EXPLORE_K
EXPLORE_THRESH=$EXPLORE_THRESH
VERIFY=$VERIFY
ERROR_BOUND=$ERROR_BOUND
DEBUG_NN=$DEBUG_NN
WEIGHTS=$WEIGHTS
PARAMS_EOF

    echo "============================================================"
    echo "  SDRBench: ${ds}"
    echo "============================================================"
    echo "  Data dir   : ${data_dir}"
    echo "  Dims       : ${dims}"
    echo "  Ext        : ${ext}"
    echo "  Chunks     : ${CHUNK_MB} MB"
    echo "  Phases     : ${PHASES}"
    echo "  Policies   : ${POLICIES}"
    echo "  Verify     : ${VERIFY}"
    echo "  Output     : ${eval_dir}"
    echo "============================================================"
    echo ""

    # Separate NN phases from fixed phases
    local nn_phases=() fixed_phases=()
    for phase in "${PHASE_LIST[@]}"; do
        case "$phase" in
            nn|nn-rl|nn-rl+exp50) nn_phases+=("$phase") ;;
            *) fixed_phases+=("$phase") ;;
        esac
    done

    local total=$(( ${#fixed_phases[@]} + ${#nn_phases[@]} * ${#POLICY_LIST[@]} ))
    local run_num=0

    # Common args
    local eb_arg=""
    [ "$ERROR_BOUND" != "0.0" ] && [ "$ERROR_BOUND" != "0" ] && eb_arg="--error-bound $ERROR_BOUND"
    local common_args="--data-dir $data_dir --dims $dims --ext $ext --chunk-mb $CHUNK_MB $eb_arg"
    local verify_arg=""
    [ "$VERIFY" = "0" ] && verify_arg="--no-verify"

    # Run fixed phases once
    local fixed_dir="$eval_dir/fixed_phases"
    if [ ${#fixed_phases[@]} -gt 0 ]; then
        mkdir -p "$fixed_dir"
        echo "  Fixed phases (policy-invariant)"
        echo "  ────────────────────────────────"

        for phase in "${fixed_phases[@]}"; do
            run_num=$((run_num + 1))
            local phase_dir="$fixed_dir/phase_${phase}"
            mkdir -p "$phase_dir"

            echo "  [$run_num/$total] $phase"

            local t0=$(date +%s)
            GPUCOMPRESS_DETAILED_TIMING=1 \
            GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
            mpi_launch "$SDR_BIN" "$WEIGHTS" \
                $common_args \
                --phase "$phase" \
                --w0 1.0 --w1 1.0 --w2 1.0 \
                --name "$ds" \
                $verify_arg \
                --out-dir "$phase_dir" \
                > "$phase_dir/sdr_benchmark.log" 2>&1

            echo "    Done ($(($(date +%s) - t0))s)"
        done
        echo ""
    fi

    # Run NN phases per policy
    for policy in "${POLICY_LIST[@]}"; do
        local label="${POLICY_LABELS[$policy]}"
        local w0="${POLICY_W0[$policy]}"
        local w1="${POLICY_W1[$policy]}"
        local w2="${POLICY_W2[$policy]}"
        local policy_dir="$eval_dir/$label"
        mkdir -p "$policy_dir"

        if [ ${#nn_phases[@]} -eq 0 ]; then continue; fi

        echo "  Policy: $label (w0=$w0 w1=$w1 w2=$w2)"
        echo "  ────────────────────────────────────────"

        for phase in "${nn_phases[@]}"; do
            run_num=$((run_num + 1))
            local phase_dir="$policy_dir/phase_${phase}"
            mkdir -p "$phase_dir"

            echo "  [$run_num/$total] $phase"

            local t0=$(date +%s)
            GPUCOMPRESS_DETAILED_TIMING=1 \
            GPUCOMPRESS_DEBUG_NN=$DEBUG_NN \
            mpi_launch "$SDR_BIN" "$WEIGHTS" \
                $common_args \
                --phase "$phase" \
                --w0 $w0 --w1 $w1 --w2 $w2 \
                --lr $SGD_LR --mape $SGD_MAPE \
                --explore-k $EXPLORE_K --explore-thresh $EXPLORE_THRESH \
                --name "$ds" \
                $verify_arg \
                --out-dir "$phase_dir" \
                > "$phase_dir/sdr_benchmark.log" 2>&1

            echo "    Done ($(($(date +%s) - t0))s)"
        done

        # Merge per-phase CSVs
        local merge_dir="$policy_dir/merged_csv"
        mkdir -p "$merge_dir"
        echo ""
        echo "  Merging CSVs → $merge_dir"

        for csv_name in benchmark_${ds}.csv benchmark_${ds}_timesteps.csv benchmark_${ds}_timestep_chunks.csv benchmark_${ds}_ranking.csv benchmark_${ds}_ranking_costs.csv; do
            local merged="$merge_dir/$csv_name"
            local first=1
            for phase in "${PHASE_LIST[@]}"; do
                local src
                case "$phase" in
                    nn|nn-rl|nn-rl+exp50) src="$policy_dir/phase_${phase}/$csv_name" ;;
                    *) src="$fixed_dir/phase_${phase}/$csv_name" ;;
                esac
                if [ -f "$src" ]; then
                    if [ $first -eq 1 ]; then
                        cp "$src" "$merged"
                        first=0
                    else
                        tail -n+2 "$src" >> "$merged"
                    fi
                fi
            done
            if [ -f "$merged" ]; then
                local rows=$(( $(wc -l < "$merged") - 1 ))
                echo "    $csv_name: $rows rows"
                ln -sf "merged_csv/$csv_name" "$policy_dir/$csv_name"
            fi
        done
        echo ""
    done

    # Generate plots
    generate_plots "$ds" "$eval_dir"

    echo ""
    echo "  Results: $eval_dir"
    echo ""
}

# ── Main ──
IFS=',' read -ra DS_LIST  <<< "$DATASET"
IFS=',' read -ra PHASE_LIST <<< "$PHASES"
IFS=',' read -ra POLICY_LIST <<< "$POLICIES"

echo "============================================================"
echo "  SDRBench Benchmark"
echo "============================================================"
echo "  Datasets  : ${DATASET}"
echo "  Chunk size: ${CHUNK_MB} MB"
echo "  Phases    : ${PHASES}"
echo "  Policies  : ${POLICIES}"
if [ "$PLOT_ONLY" = "1" ]; then
    echo "  Mode      : PLOT ONLY (no benchmark runs)"
fi
echo "============================================================"
echo ""

for ds in "${DS_LIST[@]}"; do
    run_dataset "$ds"
done

echo "============================================================"
echo "  All datasets complete."
echo "============================================================"
