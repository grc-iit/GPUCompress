#!/bin/bash
# ============================================================
# Nyx Exploration Threshold Sweep
#
# Mirrors 4.2.1_eval_{vpic,warpx,lammps}_threshold_sweep.sh for the
# Nyx Sedov blast wave workload. Because re-running Nyx per cell
# is expensive (each simulation is minutes), we use the
# dump-once-then-replay pattern from benchmarks/nyx/run_nyx_benchmark.sh:
#
#   Phase 1 (once):     Run Nyx Sedov with NYX_DUMP_FIELDS=1 to
#                       materialize raw .f32 field files under
#                       SWEEP_DIR/flat_fields/.
#   Phase 2 (per cell): Run examples/generic_benchmark with
#                       --phase nn-rl+exp50, --mape $x1,
#                       --explore-thresh $x2 on the flat dir. The
#                       SGD learner sees the full evolving Sedov
#                       trajectory (density, momentum, energy,
#                       species) once per cell — fresh warm-up
#                       for each (X1, X2) so the online learner
#                       converges to whatever the thresholds allow.
#
# This matches how the warpx/vpic sweeps measure threshold
# sensitivity: same input data per cell, different online-learning
# hyperparameters. The plotter (4.2.1_plot_threshold_sweep.py)
# consumes benchmark_nyx_sedov.csv / benchmark_nyx_sedov_ranking.csv
# written directly by generic_benchmark — no aggregator needed.
#
# X1 (SGD MAPE):    {0.05, 0.10, 0.20, 0.30}
# delta (X2-X1):    {0.05, 0.10, 0.20, 0.30}
# Total: 16 replay runs (+1 Nyx dump)
#
# Environment overrides:
#   NYX_BIN              (auto)   Nyx HydroTests or MiniSB binary
#   GENERIC_BIN          (auto)   examples/generic_benchmark
#   NYX_NCELL            128      Grid size per dimension
#   NYX_MAX_STEP         200      Nyx simulation steps
#   NYX_PLOT_INT         10       Steps between dumps
#   CHUNK_MB             4        HDF5 chunk size in MiB
#   ERROR_BOUND          0.0      Lossless by default
#   EXPLORE_K            4        Exploration alternatives K
#   SGD_LR               0.2      SGD learning rate
#   POLICY               balanced Cost policy (balanced|ratio|speed)
#   DRY_RUN              0        Print commands without running
#   REUSE_DUMPS          0        Skip Phase 1 if flat_fields/ exists
# ============================================================
set -eo pipefail

command -v bc >/dev/null 2>&1 || { echo "ERROR: bc not found"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Fixed parameters ──
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPU_DIR/neural_net/weights/model.nnwt}"
GENERIC_BIN="${GENERIC_BIN:-$GPU_DIR/build/generic_benchmark}"

NYX_BIN="${NYX_BIN:-}"
if [ -z "$NYX_BIN" ]; then
    for cand in \
        "$HOME/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests" \
        "$HOME/Nyx/build-gpucompress/Exec/MiniSB/nyx_MiniSB" \
        "$HOME/Nyx/build/Exec/HydroTests/nyx_HydroTests" \
        "$HOME/Nyx/build/Exec/MiniSB/nyx_MiniSB"; do
        if [ -x "$cand" ]; then NYX_BIN="$cand"; break; fi
    done
fi

NYX_NCELL=${NYX_NCELL:-128}
NYX_MAX_STEP=${NYX_MAX_STEP:-200}
NYX_PLOT_INT=${NYX_PLOT_INT:-10}
CHUNK_MB=${CHUNK_MB:-4}
ERROR_BOUND=${ERROR_BOUND:-0.0}
EXPLORE_K=${EXPLORE_K:-4}
SGD_LR=${SGD_LR:-0.2}
POLICY=${POLICY:-balanced}
DRY_RUN=${DRY_RUN:-0}
REUSE_DUMPS=${REUSE_DUMPS:-0}

# Policy weights
case "$POLICY" in
    balanced) W0=1.0; W1=1.0; W2=1.0 ;;
    ratio)    W0=0.0; W1=0.0; W2=1.0 ;;
    speed)    W0=1.0; W1=1.0; W2=0.0 ;;
    *) echo "ERROR: unknown policy '$POLICY' (use balanced, ratio, speed)"; exit 1 ;;
esac

# Error bound tag
if [ "$ERROR_BOUND" = "0" ] || [ "$ERROR_BOUND" = "0.0" ]; then
    EB_TAG="lossless"
else
    EB_TAG="eb${ERROR_BOUND}"
fi

SWEEP_DIR="$PARENT_DIR/results/nyx_threshold_sweep_${POLICY}_${EB_TAG}_lr${SGD_LR}"
RAW_DIR="$SWEEP_DIR/raw_fields"
FLAT_DIR="$SWEEP_DIR/flat_fields"

# ── Validate ──
[ -x "$NYX_BIN" ]     || { echo "ERROR: Nyx binary not found at $NYX_BIN (set NYX_BIN)"; exit 1; }
[ -x "$GENERIC_BIN" ] || { echo "ERROR: generic_benchmark not found at $GENERIC_BIN"; exit 1; }
[ -f "$WEIGHTS" ]     || { echo "ERROR: NN weights not found at $WEIGHTS"; exit 1; }

mkdir -p "$SWEEP_DIR"

export LD_LIBRARY_PATH="$GPU_DIR/build:$GPU_DIR/examples:/tmp/hdf5-install/lib:/tmp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "============================================================"
echo "Nyx Threshold Sweep (Sedov Blast)"
echo "  Nyx bin:     $NYX_BIN"
echo "  Generic:     $GENERIC_BIN"
echo "  Grid:        ${NYX_NCELL}^3 ($NYX_MAX_STEP steps, plot every $NYX_PLOT_INT)"
echo "  Policy:      $POLICY (w0=$W0 w1=$W1 w2=$W2)"
echo "  Chunk:       ${CHUNK_MB} MB"
echo "  Error bound: $ERROR_BOUND ($EB_TAG)"
echo "  SGD LR:      $SGD_LR    Explore K: $EXPLORE_K"
echo "  Output:      $SWEEP_DIR"
echo "============================================================"
echo ""

# ============================================================
# Phase 1: Dump Nyx Sedov fields once (reused across all cells)
# ============================================================
if [ "$REUSE_DUMPS" = "1" ] && [ -d "$FLAT_DIR" ] && \
   [ "$(ls "$FLAT_DIR"/*.f32 2>/dev/null | wc -l)" -gt 0 ]; then
    N_FLAT=$(ls "$FLAT_DIR"/*.f32 2>/dev/null | wc -l)
    echo ">>> Phase 1: Reusing $N_FLAT existing .f32 fields in $FLAT_DIR"
else
    echo ">>> Phase 1: Dumping Nyx Sedov raw fields (once, shared across all cells)"
    mkdir -p "$RAW_DIR"

    INPUT_FILE="$SWEEP_DIR/inputs.sedov"
    cat > "$INPUT_FILE" <<EOF
# Sedov blast wave — evolving shock structure
amr.n_cell         = $NYX_NCELL $NYX_NCELL $NYX_NCELL
amr.max_level      = 0
amr.max_grid_size  = $(( NYX_NCELL > 128 ? 128 : NYX_NCELL ))

nyx.do_hydro       = 1
nyx.initial_z      = 0.0
nyx.final_z        = 0.0
nyx.do_santa_barbara = 0
nyx.ppm_type       = 1

amr.plot_int       = $NYX_PLOT_INT
amr.max_step       = $NYX_MAX_STEP
amr.check_int      = 0

geometry.prob_lo    = 0.0 0.0 0.0
geometry.prob_hi    = 1.0 1.0 1.0
geometry.is_periodic = 1 1 1

nyx.use_gpucompress       = 1
nyx.gpucompress_weights   = $WEIGHTS
nyx.gpucompress_algorithm = auto
nyx.gpucompress_policy    = $POLICY
nyx.gpucompress_verify    = 0
nyx.gpucompress_chunk_mb  = $CHUNK_MB
EOF

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN: $NYX_BIN $INPUT_FILE  (NYX_DUMP_FIELDS=1 → $RAW_DIR)"
    else
        (
            cd "$SWEEP_DIR"
            NYX_DUMP_FIELDS=1 NYX_DUMP_DIR="$RAW_DIR" \
            "$NYX_BIN" "$INPUT_FILE" > nyx_dump.log 2>&1
        ) || echo "  WARNING: Nyx exited non-zero (check $SWEEP_DIR/nyx_dump.log)"

        N_DIRS=$(ls -d "$RAW_DIR"/plt* 2>/dev/null | wc -l)
        echo "  Dumped: $N_DIRS timestep directories"
        if [ "$N_DIRS" -eq 0 ]; then
            echo "ERROR: No field directories dumped. See $SWEEP_DIR/nyx_dump.log"
            exit 1
        fi

        # Flatten all .f32 files into a single dir so generic_benchmark
        # walks them in timestep order (each file is one "field").
        mkdir -p "$FLAT_DIR"
        for ts_dir in "$RAW_DIR"/plt*; do
            ts_name=$(basename "$ts_dir")
            for f in "$ts_dir"/*.f32; do
                [ -f "$f" ] || continue
                ln -sf "$f" "$FLAT_DIR/${ts_name}_$(basename "$f")"
            done
        done
    fi
fi

# Determine per-field dimensions from first .f32 file (skip in DRY_RUN
# since Phase 1 didn't actually dump anything)
if [ "$DRY_RUN" != "1" ]; then
    FIRST_FILE=""
    if [ -d "$FLAT_DIR" ]; then
        FIRST_FILE=$(find "$FLAT_DIR" -name "*.f32" -type f 2>/dev/null | head -1 || true)
    fi
    [ -n "$FIRST_FILE" ] || { echo "ERROR: No .f32 files in $FLAT_DIR"; exit 1; }
    FIRST_SIZE=$(stat -c%s "$FIRST_FILE")
    N_FLOATS=$((FIRST_SIZE / 4))
    DIMS="${N_FLOATS},1"
    N_FLAT=$(ls "$FLAT_DIR"/*.f32 2>/dev/null | wc -l)
    echo "  Per-component: $N_FLOATS floats, dims=$DIMS, $N_FLAT total fields"
else
    DIMS="<dry-run>"
fi
echo ""

# ============================================================
# Phase 2: Per-cell threshold sweep via generic_benchmark
# ============================================================
# 7x7 = 49 cells
X1_VALUES=(0.05 0.10 0.20 0.30 0.50 1.00 10.00)
DELTA_VALUES=(0.05 0.10 0.20 0.30 0.50 1.00 10.00)

PAIRS=()
for x1 in "${X1_VALUES[@]}"; do
    for delta in "${DELTA_VALUES[@]}"; do
        PAIRS+=("${x1},${delta}")
    done
done
TOTAL=${#PAIRS[@]}
for (( i=TOTAL-1; i>0; i-- )); do
    j=$(( RANDOM % (i+1) ))
    tmp="${PAIRS[$i]}"; PAIRS[$i]="${PAIRS[$j]}"; PAIRS[$j]="$tmp"
done

RUN=0
for pair in "${PAIRS[@]}"; do
    x1="${pair%%,*}"
    delta="${pair##*,}"
    RUN=$((RUN + 1))
    x2=$(echo "$x1 + $delta" | bc -l)

    OUT_DIR="$SWEEP_DIR/x1_${x1}_delta_${delta}"

    if [ -f "$OUT_DIR/benchmark_nyx_sedov.csv" ]; then
        echo "[$RUN/$TOTAL] X1=$x1 delta=$delta — already done, skipping."
        continue
    fi

    echo ""
    echo "[$RUN/$TOTAL] X1=$x1 delta=$delta (X2=$x2)"
    mkdir -p "$OUT_DIR"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN: generic_benchmark --phase nn-rl+exp50 --mape $x1 --explore-thresh $x2"
        continue
    fi

    # Run generic_benchmark only for the nn-rl+exp50 phase — the sweep
    # measures online-learning threshold sensitivity, so the fixed-algo
    # baselines would just waste time per cell.
    EXTRA_ARGS=()
    if [ "$ERROR_BOUND" != "0" ] && [ "$ERROR_BOUND" != "0.0" ]; then
        EXTRA_ARGS+=(--error-bound "$ERROR_BOUND")
    fi

    "$GENERIC_BIN" "$WEIGHTS" \
        --data-dir "$FLAT_DIR" \
        --dims "$DIMS" \
        --ext .f32 \
        --chunk-mb "$CHUNK_MB" \
        --out-dir "$OUT_DIR" \
        --name "nyx_sedov" \
        --phase nn-rl+exp50 \
        --w0 "$W0" --w1 "$W1" --w2 "$W2" \
        --lr "$SGD_LR" \
        --mape "$x1" \
        --explore-k "$EXPLORE_K" \
        --explore-thresh "$x2" \
        --no-verify \
        "${EXTRA_ARGS[@]}" \
        > "$OUT_DIR/benchmark.log" 2>&1 \
        || echo "  WARNING: generic_benchmark exited non-zero (check benchmark.log)"

    if [ ! -f "$OUT_DIR/benchmark_nyx_sedov.csv" ]; then
        echo "  ERROR: benchmark_nyx_sedov.csv missing — see $OUT_DIR/benchmark.log"
        continue
    fi

    # Quick summary line (ratio + write bw from the aggregate row)
    python3 - "$OUT_DIR" "$x1" "$x2" <<'PYEOF'
import csv, os, sys
run_dir, x1, x2 = sys.argv[1:4]
p = os.path.join(run_dir, "benchmark_nyx_sedov.csv")
with open(p) as f:
    rows = list(csv.DictReader(f))
row = next((r for r in rows if r.get("phase","").startswith("nn-rl+exp50")), rows[-1] if rows else None)
if row:
    print(f"  ratio={float(row.get('ratio',0)):.2f}x  "
          f"write={float(row.get('write_mibps',0)):.0f}MiB/s  "
          f"read={float(row.get('read_mibps',0)):.0f}MiB/s  "
          f"mape_ratio={float(row.get('mape_ratio_pct',0)):.1f}%  "
          f"sgd={row.get('sgd_fires','?')}  expl={row.get('explorations','?')}")
PYEOF
done

echo ""
echo "============================================================"
echo "Nyx threshold sweep complete: $TOTAL configurations"
echo "Results: $SWEEP_DIR/"
echo ""
echo "Generate heatmaps:"
echo "  python3 $SCRIPT_DIR/4.2.1_plot_threshold_sweep.py $SWEEP_DIR"
echo "============================================================"

# ── Generate plots if matplotlib available ──
PLOT_SCRIPT="$SCRIPT_DIR/4.2.1_plot_threshold_sweep.py"
if [ -f "$PLOT_SCRIPT" ] && python3 -c "import matplotlib" 2>/dev/null; then
    echo ""
    echo ">>> Generating threshold sweep plots..."
    python3 "$PLOT_SCRIPT" "$SWEEP_DIR" || echo "WARNING: plotter failed"
fi
