#!/bin/bash
# ============================================================
# SC26 — Sequential cross-workload sweep for Section 7.1
#
# Runs each of the 5 workloads (VPIC, NYX, WarpX, LAMMPS, AI/ViT-B/16)
# under BOTH balanced and ratio policies, with the equalized
# hyperparameters fixed across every run:
#
#   PHASE          = nn-rl+exp50         (online SGD + always-on exploration)
#   ERROR_BOUND    = 0.01                (LOSSY, relative bound 1%)
#   SGD_LR         = 0.2
#   SGD_MAPE       = 0.10                (10% — SGD fires above this)
#   EXPLORE_K      = 4                   (alternative configs per chunk)
#   EXPLORE_THRESH = 0.25                (25% — exploration fires above this)
#   CHUNK_MB       = 2
#
# Output layout:
#   SC26/
#     vpic_balanced/   vpic_ratio/
#     nyx_balanced/    nyx_ratio/
#     warpx_balanced/  warpx_ratio/
#     lammps_balanced/ lammps_ratio/
#     ai_balanced/     ai_ratio/
#     timing.csv       (one row per run with wall-clock seconds)
#
# Each workload subdir contains the standard 7.1 layout: cross_workload_*.png
# figures, cross_workload_*.csv combined CSVs, run_metadata.txt, and a
# per-workload subdir with the raw ranking + per-chunk CSVs.
#
# Usage:
#   bash evaluations/figure8_cross_workload_regret/run_all.sh
#
# Optional env overrides:
#   SC26_DIR=/some/path  (default: $PROJECT_DIR/SC26)
#   SKIP_VPIC=1 SKIP_NYX=1 SKIP_WARPX=1 SKIP_LAMMPS=1 SKIP_AI=1
#       — skip a workload entirely (both policies)
#   ONLY_POLICY=balanced (or ratio)
#       — run only one policy half of the sweep
#   RESUME=1
#       — skip runs whose results dir already contains a complete figure
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_one_workload.sh"

[ -f "$RUNNER" ] || { echo "ERROR: runner not found: $RUNNER" >&2; exit 1; }

# ── Output dir ──────────────────────────────────────────────
SC26_DIR="${SC26_DIR:-$PROJECT_DIR/SC26}"
mkdir -p "$SC26_DIR"
TIMING_CSV="$SC26_DIR/timing.csv"
SWEEP_LOG="$SC26_DIR/sweep.log"

# Header (only if file doesn't exist or is empty)
if [ ! -s "$TIMING_CSV" ]; then
    echo "workload,policy,wall_seconds,start_iso,end_iso,exit_code,results_dir,run_log" > "$TIMING_CSV"
fi

# ── Equalized hyperparameters (the SC26 contract) ──────────
export PHASE="nn-rl+exp50"
export ERROR_BOUND="0.01"
export SGD_LR="0.2"
export SGD_MAPE="0.10"
export EXPLORE_K="4"
export EXPLORE_THRESH="0.25"
export CHUNK_MB="${CHUNK_MB:-2}"

# ── Workload binaries (paths the runner needs) ─────────────
export VPIC_BIN="${VPIC_BIN:-$PROJECT_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux}"
export NYX_BIN="${NYX_BIN:-/home/cc/sims/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests}"
export WARPX_BIN="${WARPX_BIN:-/home/cc/sims/warpx/build-gpucompress/bin/warpx.3d}"
export WARPX_INPUTS="${WARPX_INPUTS:-/home/cc/sims/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}"
export LMP_BIN="${LMP_BIN:-/home/cc/sims/lammps/build/lmp}"

# ── Workload-specific simulation parameters ────────────────
# VPIC: slow-reconnection Harris-sheet regime, sim_interval and warmup
# aligned with deltaRunVPICParameters.md.
#   Grid:    NX=147  → (147+2)^3 * 64 B ≈ 200 MB/field
#   Dumps:   8 writes, warmup=500, sim_interval=190
#            (500 + 8 × 190 = 2020 total physics steps)
#   Physics: mi_me=25 (slow reconnection, ~20k steps for nonlinear phase)
#            wpe_wce=1, Ti_Te=5
#   Rationale: mi_me=25 + wpe_wce=1 + Ti_Te=5 is the combination commit
#   918fa14 found gave the cleanest MAPE convergence and ratio swing.
#   warmup=500 / sim_interval=190 is the Delta-doc pairing for
#   single-GPU evaluation (918fa14 used 200/500; switching to 500/190
#   keeps the per-dump physics evolution similar while reducing the
#   single-dump wall cost 2.6×).
#
#   nppc is intentionally left at the deck default (2) via the
#   explicit export below so the configuration matches 918fa14's
#   tested setup exactly.
export VPIC_NX="${VPIC_NX:-147}"
export VPIC_TIMESTEPS="${VPIC_TIMESTEPS:-8}"
export VPIC_NPPC="${VPIC_NPPC:-2}"
export VPIC_MI_ME="${VPIC_MI_ME:-25}"
export VPIC_WPE_WCE="${VPIC_WPE_WCE:-1}"
export VPIC_TI_TE="${VPIC_TI_TE:-5}"
export VPIC_WARMUP_STEPS="${VPIC_WARMUP_STEPS:-500}"
export VPIC_SIM_INTERVAL="${VPIC_SIM_INTERVAL:-190}"
# NYX: 88³ Sedov blast (~33 MB/dump), runs to physical stop_time
export NYX_NCELL="${NYX_NCELL:-88}"
export NYX_MAX_STEP="${NYX_MAX_STEP:-500}"
export NYX_PLOT_INT="${NYX_PLOT_INT:-10}"
# WarpX: paper-default LWFA grid
export WARPX_NCELL="${WARPX_NCELL:-128 128 384}"
export WARPX_MAX_STEP="${WARPX_MAX_STEP:-200}"
export WARPX_DIAG_INTERVAL="${WARPX_DIAG_INTERVAL:-25}"
# LAMMPS: SC26 diversity-tuned multi-regime deck (LMP_DIVERSE=1).
# 80³ box → 2.05M atoms, 12 dumps × 400 steps = 2 full 6-stage cycles.
# T_COLD/T_HOT/TDAMP and LMP_DIVERSE inherit from the runner defaults.
export LMP_ATOMS="${LMP_ATOMS:-80}"
export LMP_TIMESTEPS="${LMP_TIMESTEPS:-12}"
export LMP_SIM_INTERVAL="${LMP_SIM_INTERVAL:-400}"
# AI: ViT-B/16 fine-tuning, 20 epochs
export AI_MODEL="${AI_MODEL:-vit_b_16}"
export AI_EPOCHS="${AI_EPOCHS:-20}"
export AI_CHECKPOINTS="${AI_CHECKPOINTS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"

# ── Skip/resume controls ────────────────────────────────────
SKIP_VPIC="${SKIP_VPIC:-0}"
SKIP_NYX="${SKIP_NYX:-0}"
SKIP_WARPX="${SKIP_WARPX:-0}"
SKIP_LAMMPS="${SKIP_LAMMPS:-0}"
SKIP_AI="${SKIP_AI:-0}"
ONLY_POLICY="${ONLY_POLICY:-}"   # blank = run both balanced+ratio
RESUME="${RESUME:-0}"

# ── Banner ──────────────────────────────────────────────────
{
    echo "============================================================"
    echo "SC26 Sequential Sweep — Section 7.1"
    echo "============================================================"
    echo "Started:        $(date -Iseconds)"
    echo "Output dir:     $SC26_DIR"
    echo "Timing CSV:     $TIMING_CSV"
    echo "Runner:         $RUNNER"
    echo ""
    echo "Equalized hyperparameters (LOSSY, balanced + ratio sweep):"
    echo "  PHASE          = $PHASE"
    echo "  ERROR_BOUND    = $ERROR_BOUND  (LOSSY 1% relative)"
    echo "  SGD_LR         = $SGD_LR"
    echo "  SGD_MAPE       = $SGD_MAPE  (10% — SGD threshold)"
    echo "  EXPLORE_K      = $EXPLORE_K"
    echo "  EXPLORE_THRESH = $EXPLORE_THRESH  (25%)"
    echo "  CHUNK_MB       = $CHUNK_MB"
    echo ""
    echo "Workload simulation parameters:"
    echo "  VPIC   : NX=$VPIC_NX timesteps=$VPIC_TIMESTEPS nppc=$VPIC_NPPC mi_me=$VPIC_MI_ME wpe_wce=$VPIC_WPE_WCE Ti_Te=$VPIC_TI_TE warmup=$VPIC_WARMUP_STEPS sim_int=$VPIC_SIM_INTERVAL"
    echo "  NYX    : NCELL=$NYX_NCELL max_step=$NYX_MAX_STEP plot_int=$NYX_PLOT_INT"
    echo "  WarpX  : ncell=$WARPX_NCELL max_step=$WARPX_MAX_STEP diag_int=$WARPX_DIAG_INTERVAL"
    echo "  LAMMPS : atoms=${LMP_ATOMS}^3 timesteps=$LMP_TIMESTEPS sim_int=$LMP_SIM_INTERVAL"
    echo "  AI     : model=$AI_MODEL epochs=$AI_EPOCHS"
    echo ""
    echo "Skip flags: VPIC=$SKIP_VPIC NYX=$SKIP_NYX WarpX=$SKIP_WARPX LAMMPS=$SKIP_LAMMPS AI=$SKIP_AI"
    if [ -n "$ONLY_POLICY" ]; then
        echo "Policy filter: ONLY_POLICY=$ONLY_POLICY"
    fi
    if [ "$RESUME" = "1" ]; then
        echo "Resume mode: existing run dirs will be skipped if cross_workload_regret.png exists"
    fi
    echo "============================================================"
} | tee -a "$SWEEP_LOG"

# ── Helper: run one (workload, policy) combo ───────────────
run_one() {
    local workload="$1"
    local policy="$2"

    local rname="${workload}_${policy}"
    local rdir="$SC26_DIR/$rname"

    # Resume check
    if [ "$RESUME" = "1" ] && [ -f "$rdir/cross_workload_regret.png" ]; then
        echo "  [SKIP $rname] already complete (resume mode)" | tee -a "$SWEEP_LOG"
        return 0
    fi

    mkdir -p "$rdir"
    local run_log="$rdir/sc26_run.log"

    # Build the per-workload skip flags (run only THIS workload)
    local skip_v=1 skip_n=1 skip_w=1 skip_l=1 skip_a=1
    case "$workload" in
        vpic)   skip_v=0 ;;
        nyx)    skip_n=0 ;;
        warpx)  skip_w=0 ;;
        lammps) skip_l=0 ;;
        ai)     skip_a=0 ;;
    esac

    local start_iso end_iso start_s end_s elapsed rc
    start_iso=$(date -Iseconds)
    start_s=$(date +%s)

    {
        echo ""
        echo "============================================================"
        echo "[${workload^^} / $policy] start  $start_iso"
        echo "  Output: $rdir"
        echo "============================================================"
    } | tee -a "$SWEEP_LOG"

    # Invoke the runner with all the right env vars.
    # POLICY, RESULTS_DIR, RUN_NAME, and the per-workload SKIP_* override
    # whatever the runner has as defaults.
    POLICY="$policy" \
    RESULTS_DIR="$rdir" \
    RUN_NAME="$rname" \
    SKIP_VPIC="$skip_v" \
    SKIP_NYX="$skip_n" \
    SKIP_WARPX="$skip_w" \
    SKIP_LAMMPS="$skip_l" \
    SKIP_AI="$skip_a" \
    bash "$RUNNER" > "$run_log" 2>&1
    rc=$?

    end_iso=$(date -Iseconds)
    end_s=$(date +%s)
    elapsed=$((end_s - start_s))

    {
        echo "[${workload^^} / $policy] done   $end_iso  wall=${elapsed}s  exit=$rc"
        if [ $rc -ne 0 ]; then
            echo "  WARNING: non-zero exit. Last 10 lines of $run_log:"
            tail -10 "$run_log" | sed 's/^/    /'
        fi
    } | tee -a "$SWEEP_LOG"

    # Append timing row (CSV-quote any commas in fields just in case)
    echo "$workload,$policy,$elapsed,$start_iso,$end_iso,$rc,$rdir,$run_log" >> "$TIMING_CSV"

    # Aggregate cross_workload_* artifacts into the top-level csv/ and
    # figures/ subdirs, prefixed by workload+policy so files don't collide.
    mkdir -p "$SC26_DIR/figures" "$SC26_DIR/csv"
    for png in "$rdir"/cross_workload_*.png; do
        [ -f "$png" ] || continue
        local base="$(basename "$png")"           # e.g. cross_workload_cost_mape.png
        local kind="${base#cross_workload_}"      # cost_mape.png
        cp -f "$png" "$SC26_DIR/figures/${rname}_${kind}"
    done
    for csv in "$rdir"/cross_workload_*.csv; do
        [ -f "$csv" ] || continue
        local base="$(basename "$csv")"
        local kind="${base#cross_workload_}"
        cp -f "$csv" "$SC26_DIR/csv/${rname}_${kind}"
    done

    return $rc
}

# ── Sweep order ─────────────────────────────────────────────
# Workloads in order from fastest to slowest. AI last because it can
# take 45-75 minutes per policy.
WORKLOADS=(
    vpic
    nyx
    warpx
    lammps
    ai
)

# Python binary for the AI workload. The ViT/resnet training script depends
# on torch+torchvision, which are installed in the project venv at
# /home/cc/gpuc-venv rather than system python. Override via AI_PYTHON if
# your venv lives elsewhere.
export AI_PYTHON="${AI_PYTHON:-/home/cc/gpuc-venv/bin/python3}"

# Determine which policies to run
if [ -n "$ONLY_POLICY" ]; then
    POLICIES=("$ONLY_POLICY")
else
    POLICIES=(balanced ratio)
fi

# ── Main loop ───────────────────────────────────────────────
SWEEP_START=$(date +%s)

for workload in "${WORKLOADS[@]}"; do
    case "$workload" in
        vpic)   wl_skip=$SKIP_VPIC   ;;
        nyx)    wl_skip=$SKIP_NYX    ;;
        warpx)  wl_skip=$SKIP_WARPX  ;;
        lammps) wl_skip=$SKIP_LAMMPS ;;
        ai)     wl_skip=$SKIP_AI     ;;
    esac

    if [ "$wl_skip" = "1" ]; then
        echo "" | tee -a "$SWEEP_LOG"
        echo "[${workload^^}] SKIPPED (SKIP_${workload^^}=1)" | tee -a "$SWEEP_LOG"
        continue
    fi

    for policy in "${POLICIES[@]}"; do
        run_one "$workload" "$policy" || true
    done
done

SWEEP_END=$(date +%s)
TOTAL=$((SWEEP_END - SWEEP_START))

# ── Combined cross-workload figures (Figures 1, 2, 3) ───────
COMBINED_PLOTTER="$SCRIPT_DIR/plot.py"
if [ -f "$COMBINED_PLOTTER" ]; then
    {
        echo ""
        echo "============================================================"
        echo "Building combined cross-workload figures"
        echo "  Plotter: $COMBINED_PLOTTER"
        echo "  Policies: ${POLICIES[*]}"
        echo "============================================================"
    } | tee -a "$SWEEP_LOG"
    python3 "$COMBINED_PLOTTER" --sc26-dir "$SC26_DIR" --policies "${POLICIES[@]}" \
        2>&1 | tee -a "$SWEEP_LOG" || \
        echo "WARNING: combined plotter failed (see $SWEEP_LOG)" | tee -a "$SWEEP_LOG"
fi

# ── Final summary ──────────────────────────────────────────
{
    echo ""
    echo "============================================================"
    echo "SC26 Sweep COMPLETE"
    echo "============================================================"
    echo "Total wall time: ${TOTAL}s ($(printf '%dh %dm %ds' $((TOTAL/3600)) $(((TOTAL%3600)/60)) $((TOTAL%60))))"
    echo ""
    echo "Per-run timing (from $TIMING_CSV):"
    echo ""
    column -t -s, "$TIMING_CSV" 2>/dev/null || cat "$TIMING_CSV"
    echo ""
    echo "Results in: $SC26_DIR/{vpic,nyx,warpx,lammps,ai}_{balanced,ratio}/"
    echo "============================================================"
} | tee -a "$SWEEP_LOG"
