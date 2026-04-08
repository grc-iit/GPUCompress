#!/bin/bash
# ============================================================
# Nyx Exploration Threshold Sweep — IN-SITU per-cell mode
#
# Mirrors eval_{vpic,warpx,lammps}.sh: re-runs the live nyx_HydroTests
# Sedov blast simulation once per (X1, X2-X1) cell so each cell trains
# the online learner on a fresh evolving Sedov trajectory through the
# GPUCompress AMReX bridge (examples/nyx_amrex_bridge.hpp). The bridge
# runs the NN-guided algorithm selector + Kendall-tau ranking profiler
# inside the running simulator and honors the per-cell sgd_mape /
# explore_thresh values via the Nyx inputs file.
#
# X1 (SGD MAPE):    {0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 10.00}
# delta (X2-X1):    {0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 10.00}
# Total: 7 x 7 = 49 live Nyx runs
#
# Per-cell output (under
#   PARENT_DIR/results/nyx_threshold_sweep_<...>/x1_<x1>_delta_<delta>/):
#   benchmark_nyx_timestep_chunks.csv  (per-chunk MAPE + sgd + explore)
#   benchmark_nyx_ranking.csv          (per-milestone top1_regret)
#   benchmark_nyx_ranking_costs.csv
#   benchmark_nyx.csv                  (synthesized aggregate)
#   nyx_sim.log                        (full nyx stdout)
#
# nyx_work/ subdirs (containing AMReX plotfiles + the per-cell inputs
# file) are deleted after each cell's CSV aggregator finishes — they
# would otherwise blow the user /u quota over 49 cells.
#
# Pair with plot.py to render the heatmaps for the paper's
# "Effect of online learning thresholds" figure.
#
# Environment overrides:
#   NYX_BIN              (auto)   nyx_HydroTests binary
#   NYX_NCELL            88       Grid size per dim (88^3 * 6 vars * 4 B ≈ 16 MiB/dump)
#   NYX_MAX_STEP         150      Nyx simulation steps (15 dumps at plot_int=10)
#   NYX_PLOT_INT         10       Steps between dumps → 15 dumps per run
#   CHUNK_MB             8        HDF5 chunk size in MiB
#   ERROR_BOUND          0.01     Lossy 1% by default (set to 0.0 for lossless)
#   EXPLORE_K            4        Exploration alternatives K
#   SGD_LR               0.2      SGD learning rate
#   POLICY               balanced Cost policy (balanced|ratio|speed)
#   DRY_RUN              0        Print commands without running
#
# WARNING: switching from dump-once-replay to in-situ adds ~5x to the
# total wall time (49 live Nyx runs vs 49 generic_benchmark replays).
# Default NYX_NCELL is 88 (not 176) so each cell finishes in ~3-5 min.
# At NYX_NCELL=176 each cell would take ~30+ min and the full sweep
# would not fit in a 4h wall.
# ============================================================
set -o pipefail
# NOTE: do NOT use `set -e` — quota/IO failures on a single cell would
# abort the entire 49-cell sweep. The for-loop already validates each cell
# via the benchmark_nyx.csv sentinel and skips broken cells.

command -v bc >/dev/null 2>&1 || { echo "ERROR: bc not found"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARENT_DIR="$GPU_DIR/benchmarks/Paper_Evaluations/4"

# ── Fixed parameters ──
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPU_DIR/neural_net/weights/model.nnwt}"

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

NYX_NCELL=${NYX_NCELL:-88}        # 88^3 cells * 6 state vars * 4 B ≈ 16 MiB/dump
NYX_MAX_STEP=${NYX_MAX_STEP:-150}  # 15 dumps at plot_int=10
NYX_PLOT_INT=${NYX_PLOT_INT:-10}
CHUNK_MB=${CHUNK_MB:-8}

# AMReX requires NYX_NCELL % max_grid_size == 0. Pick the largest divisor of
# NYX_NCELL that is ≤ 128 so non-power-of-2 grids (e.g. 88 = 2³·11 → max 88)
# decompose cleanly without hitting "not divisible" assertions.
nyx_max_grid_size_divisor() {
    local n=$1 cap=128 best=1 d
    for (( d=1; d<=cap && d<=n; d++ )); do
        if (( n % d == 0 )); then best=$d; fi
    done
    echo "$best"
}
NYX_MAX_GRID_SIZE=${NYX_MAX_GRID_SIZE:-$(nyx_max_grid_size_divisor "$NYX_NCELL")}
ERROR_BOUND=${ERROR_BOUND:-0.01}  # lossy 1% by default (matches VPIC/WarpX)
EXPLORE_K=${EXPLORE_K:-4}
SGD_LR=${SGD_LR:-0.2}
POLICY=${POLICY:-balanced}
DRY_RUN=${DRY_RUN:-0}

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

# ── Validate ──
[ -x "$NYX_BIN" ] || { echo "ERROR: Nyx binary not found at '$NYX_BIN' (set NYX_BIN)"; exit 1; }
[ -f "$WEIGHTS" ] || { echo "ERROR: NN weights not found at $WEIGHTS"; exit 1; }

# Locate the canonical Sedov inputs file (provides ~20 required ParmParse
# parameters: init_shrink, cfl, dt_cutoff, BC flags, comoving constants,
# prob.* problem parameters, etc.). We append our overrides to this base.
NYX_EXEC_DIR="$(dirname "$NYX_BIN")"
BASE_INPUTS="$NYX_EXEC_DIR/inputs.3d.sph.sedov"
[ -f "$BASE_INPUTS" ] || BASE_INPUTS="/u/imuradli/Nyx/Exec/HydroTests/inputs.3d.sph.sedov"
[ -f "$BASE_INPUTS" ] || { echo "ERROR: canonical Nyx Sedov inputs not found"; exit 1; }

mkdir -p "$SWEEP_DIR"

export LD_LIBRARY_PATH="$GPU_DIR/build:$GPU_DIR/examples:/tmp/hdf5-install/lib:/tmp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "============================================================"
echo "Nyx Threshold Sweep — IN-SITU per cell"
echo "  Nyx bin:     $NYX_BIN"
echo "  Base inputs: $BASE_INPUTS"
echo "  Grid:        ${NYX_NCELL}^3 (max_grid_size=${NYX_MAX_GRID_SIZE})"
echo "  Steps:       max_step=$NYX_MAX_STEP plot_int=$NYX_PLOT_INT (≈15 dumps)"
echo "  Policy:      $POLICY (w0=$W0 w1=$W1 w2=$W2)"
echo "  Chunk:       ${CHUNK_MB} MB"
echo "  Error bound: $ERROR_BOUND ($EB_TAG)"
echo "  SGD LR:      $SGD_LR    Explore K: $EXPLORE_K"
echo "  Output:      $SWEEP_DIR"
echo "============================================================"
echo ""

# ── Threshold values — 7x7 = 49 cells ──
X1_VALUES=(0.05 0.10 0.20 0.30 0.50 1.00 10.00)
DELTA_VALUES=(0.05 0.10 0.20 0.30 0.50 1.00 10.00)

# ── Build randomized parameter pairs (Fisher-Yates) ──
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

    if [ -f "$OUT_DIR/benchmark_nyx.csv" ]; then
        echo "[$RUN/$TOTAL] X1=$x1 delta=$delta — already done, skipping."
        continue
    fi

    echo ""
    echo "[$RUN/$TOTAL] X1=$x1 delta=$delta (X2=$x2)"
    mkdir -p "$OUT_DIR"
    WORK_DIR="$OUT_DIR/nyx_work"
    mkdir -p "$WORK_DIR"

    # ── Generate inputs file: canonical Sedov defaults + per-cell overrides ──
    INPUT_FILE="$WORK_DIR/inputs.sedov"
    cp "$BASE_INPUTS" "$INPUT_FILE"
    cat >> "$INPUT_FILE" <<EOF

# ─── GPUCompress threshold-sweep overrides (later values win) ───
amr.n_cell         = $NYX_NCELL $NYX_NCELL $NYX_NCELL
amr.max_grid_size  = $NYX_MAX_GRID_SIZE
amr.max_level      = 0
amr.plot_int       = $NYX_PLOT_INT
amr.check_int      = 0
max_step           = $NYX_MAX_STEP
stop_time          = 1.0e30
# (canonical inputs.3d.sph.sedov uses is_periodic = 0 0 0 with Outflow BCs;
#  don't override is_periodic — would conflict with the BC flags and abort)

nyx.use_gpucompress         = 1
nyx.gpucompress_weights     = $WEIGHTS
nyx.gpucompress_algorithm   = auto
nyx.gpucompress_policy      = $POLICY
nyx.gpucompress_verify      = 0
nyx.gpucompress_error_bound = $ERROR_BOUND
nyx.gpucompress_chunk_mb    = $CHUNK_MB
nyx.sgd_lr         = $SGD_LR
nyx.sgd_mape       = $x1
nyx.explore_k      = $EXPLORE_K
nyx.explore_thresh = $x2
EOF

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN: $NYX_BIN inputs.sedov  (sgd_mape=$x1 explore_thresh=$x2)"
        rm -rf "$WORK_DIR"
        continue
    fi

    # ── Run live Nyx (NYX_LOG_DIR is where the bridge writes per-chunk CSVs) ──
    (
        cd "$WORK_DIR"
        NYX_LOG_DIR="$OUT_DIR" \
        NYX_PHASE="nn-rl+exp50" \
        "$NYX_BIN" inputs.sedov > "$OUT_DIR/nyx_sim.log" 2>&1
    ) || echo "  WARNING: nyx exited non-zero (teardown crash is harmless if chunks CSV is present)"

    if [ ! -s "$OUT_DIR/benchmark_nyx_timestep_chunks.csv" ]; then
        echo "  ERROR: chunks CSV missing or empty — see $OUT_DIR/nyx_sim.log"
        # Even on failure, drop the bulky nyx_work/ dir so a crashed cell
        # doesn't leak ~2 GiB to /u quota until the whole sweep ends.
        rm -rf "$WORK_DIR"
        continue
    fi

    # ── Synthesize benchmark_nyx.csv aggregate from the per-chunk CSV.
    #
    # Mirrors the LAMMPS aggregator: derives avg ratio, mape_*_pct from
    # nonzero rows, kernel-only write/read MiB/s from total bytes /
    # Σ actual_*_ms_raw, plus sgd/explore counts. The Nyx bridge does emit
    # both decomp and psnr columns when ERROR_BOUND > 0, so mape_decomp_pct
    # and mape_psnr_pct will populate (unlike LAMMPS which is lossless-pass).
    python3 - "$OUT_DIR" "$x1" "$x2" "$CHUNK_MB" <<'PYEOF'
import os, sys, csv, math

run_dir, x1, x2, chunk_mb = sys.argv[1:5]
chunk_mb = float(chunk_mb)

chunk_csv = os.path.join(run_dir, "benchmark_nyx_timestep_chunks.csv")
n_chunks = 0
sum_ratio, n_ratio = 0.0, 0
sum_mr, n_mr = 0.0, 0
sum_mc, n_mc = 0.0, 0
sum_md, n_md = 0.0, 0
sum_mp, n_mp = 0.0, 0
sum_comp_ms_raw = 0.0
sum_decomp_ms_raw = 0.0
sgd_fires = 0
explorations = 0

with open(chunk_csv) as f:
    for r in csv.DictReader(f):
        try:
            ar  = float(r.get("actual_ratio", 0))
            mr  = float(r.get("mape_ratio",  0))
            mc  = float(r.get("mape_comp",   0))
            md  = float(r.get("mape_decomp", 0))
            ac  = float(r.get("actual_comp_ms_raw", 0))
            ad  = float(r.get("actual_decomp_ms_raw", 0))
            mp_raw = r.get("mape_psnr", "nan")
            try:
                mp = float(mp_raw)
            except ValueError:
                mp = float("nan")

            if ar > 0:
                sum_ratio += ar; n_ratio += 1
            if mr > 0:
                sum_mr += mr; n_mr += 1
            if mc > 0:
                sum_mc += mc; n_mc += 1
            if md > 0:
                sum_md += md; n_md += 1
            if math.isfinite(mp) and mp > 0:
                sum_mp += mp; n_mp += 1
            if ac > 0:
                sum_comp_ms_raw += ac
            if ad > 0:
                sum_decomp_ms_raw += ad
            sgd_fires    += int(float(r.get("sgd_fired", 0)))
            explorations += int(float(r.get("exploration_triggered", 0)))
            n_chunks += 1
        except (ValueError, TypeError):
            pass

avg_ratio = (sum_ratio / n_ratio) if n_ratio else 0.0
avg_mr    = (sum_mr    / n_mr)    if n_mr    else 0.0
avg_mc    = (sum_mc    / n_mc)    if n_mc    else 0.0
avg_md    = (sum_md    / n_md)    if n_md    else 0.0
avg_mp    = (sum_mp    / n_mp)    if n_mp    else 0.0

total_mib = n_chunks * chunk_mb
write_mibps = (total_mib / (sum_comp_ms_raw / 1000.0)) if sum_comp_ms_raw > 0 else 0.0
read_mibps  = (total_mib / (sum_decomp_ms_raw / 1000.0)) if sum_decomp_ms_raw > 0 else 0.0

out_csv = os.path.join(run_dir, "benchmark_nyx.csv")
fields = [
    "rank", "source", "phase", "n_runs",
    "write_ms", "write_ms_std", "read_ms", "read_ms_std",
    "file_mib", "orig_mib", "ratio", "write_mibps", "read_mibps",
    "mismatches", "sgd_fires", "explorations", "n_chunks",
    "mape_ratio_pct", "mape_comp_pct", "mape_decomp_pct", "mape_psnr_pct",
    "x1_sgd_mape", "x2_explore_thresh",
]
row = {f: 0 for f in fields}
row["rank"] = 0
row["source"] = "nyx"
row["phase"] = "nn-rl+exp50"
row["n_runs"] = 1
row["file_mib"] = total_mib / max(avg_ratio, 1.0) if avg_ratio > 0 else 0.0
row["orig_mib"] = total_mib
row["ratio"] = avg_ratio
row["write_mibps"] = write_mibps
row["read_mibps"]  = read_mibps
row["sgd_fires"] = sgd_fires
row["explorations"] = explorations
row["n_chunks"] = n_chunks
row["mape_ratio_pct"]  = avg_mr
row["mape_comp_pct"]   = avg_mc
row["mape_decomp_pct"] = avg_md
row["mape_psnr_pct"]   = avg_mp
row["x1_sgd_mape"] = float(x1)
row["x2_explore_thresh"] = float(x2)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow(row)

print(f"  ratio={avg_ratio:.2f}x  kcomp={write_mibps:.0f}MiB/s  kdecomp={read_mibps:.0f}MiB/s  "
      f"mape_ratio={avg_mr:.1f}%  sgd={sgd_fires}  expl={explorations}  chunks={n_chunks}")
PYEOF

    # ── Per-cell cleanup: drop the bulky nyx_work/ dir (AMReX plotfiles
    #    and the per-cell inputs file) now that all CSVs have been
    #    extracted. CSVs (benchmark_nyx*, ranking*, timestep_chunks,
    #    nyx_sim.log) are preserved.
    rm -rf "$WORK_DIR" 2>/dev/null
done

echo ""
echo "============================================================"
echo "Nyx threshold sweep complete: $TOTAL configurations"
echo "Results: $SWEEP_DIR/"
echo ""
echo "Generate heatmaps:"
echo "  python3 $SCRIPT_DIR/plot.py $SWEEP_DIR"
echo "============================================================"

# ── Generate plots if matplotlib available ──
PLOT_SCRIPT="$SCRIPT_DIR/plot.py"
if [ -f "$PLOT_SCRIPT" ] && python3 -c "import matplotlib" 2>/dev/null; then
    echo ""
    echo ">>> Generating threshold sweep plots..."
    python3 "$PLOT_SCRIPT" "$SWEEP_DIR" || echo "WARNING: plotter failed"
fi
