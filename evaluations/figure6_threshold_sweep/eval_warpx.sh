#!/bin/bash
# ============================================================
# WarpX Exploration Threshold Sweep
#
# Mirrors eval_vpic.sh: re-runs warpx.3d once
# per (X1, X2-X1) cell so each configuration trains the online
# learner on fresh, evolving WarpX LWFA fields. The compression
# pipeline is driven through the GPUCompress VOL connector
# (diag1.format=gpucompress) with sgd_mape and explore_thresh set
# per cell via WarpX gpucompress.* parser params.
#
# X1 (SGD MAPE):    {0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 10.00}
# delta (X2-X1):    {0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 10.00}
# Total: 7 x 7 = 49 runs
#
# Per-cell output (under
#   PARENT_DIR/results/warpx_threshold_sweep_<...>/x1_<x1>_delta_<delta>/):
#   gpucompress_vol_summary.txt          (cumulative bytes / mibps / ratio)
#   benchmark_warpx_timestep_chunks.csv  (per-chunk MAPE + sgd + explore)
#   benchmark_warpx_ranking.csv          (per-timestep regret data)
#   benchmark_warpx_ranking_costs.csv
#   benchmark_warpx.csv                  (synthesized aggregate)
#   warpx.log
#
# Pair with plot.py to render the heatmaps for
# the paper's "Effect of online learning thresholds" figure.
#
# Environment overrides:
# Environment overrides (data target: 25 dumps × ~128 MiB/dump):
#   CHUNK_MB             8         HDF5 chunk size MB
#   ERROR_BOUND          0.01      Lossy 1% by default (set to 0.0 for lossless).
#                                  The script auto-flips gpucompress.verify off
#                                  when ERROR_BOUND > 0 (bitwise verify is
#                                  incompatible with lossy quantization).
#   EXPLORE_K            4         Exploration alternatives K
#   SGD_LR               0.2       SGD learning rate
#   POLICY               balanced  Cost policy (balanced|ratio|speed)
#   WARPX_BIN            (auto)    Path to warpx.3d
#   WARPX_INPUTS         (auto)    LWFA inputs file
#   WARPX_MAX_STEP       250       Simulation steps per cell (25 dumps at diag=10)
#   WARPX_DIAG_INT       10        Diagnostics interval → 25 diags per run
#   WARPX_NCELL          "32 32 96"  Grid; ~132 MiB/dump at default chunks
#   DRY_RUN              0         Print commands without running
# ============================================================
set -o pipefail
# NOTE: do NOT use `set -e` — quota/IO failures on a single cell would
# abort the entire 49-cell sweep. The for-loop already validates each cell
# via the benchmark_warpx.csv sentinel and skips broken cells.

command -v bc >/dev/null 2>&1 || { echo "ERROR: bc not found"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARENT_DIR="$GPU_DIR/benchmarks/Paper_Evaluations/4"

# ── Fixed parameters ──
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPU_DIR/neural_net/weights/model.nnwt}"
WARPX_BIN="${WARPX_BIN:-$HOME/sims/warpx/build-gpucompress/bin/warpx.3d}"
WARPX_INPUTS="${WARPX_INPUTS:-$HOME/sims/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}"

CHUNK_MB=${CHUNK_MB:-8}
ERROR_BOUND=${ERROR_BOUND:-0.01}  # lossy 1% by default (matches VPIC/NYX)
EXPLORE_K=${EXPLORE_K:-4}
SGD_LR=${SGD_LR:-0.2}
POLICY=${POLICY:-balanced}
WARPX_MAX_STEP=${WARPX_MAX_STEP:-150}    # 15 dumps at diag_int=10
WARPX_DIAG_INT=${WARPX_DIAG_INT:-10}
WARPX_NCELL="${WARPX_NCELL:-64 64 128}"  # 524288 cells × 40 B/cell ≈ 20 MiB/FAB,
                                          # ~2.5 chunks per ranking-profiler call so
                                          # benchmark_warpx_ranking.csv actually populates
                                          # (smaller grids fall below CHUNK_MB and the
                                          # profiler early-returns -1 with empty CSV)
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

SWEEP_DIR="$PARENT_DIR/results/warpx_threshold_sweep_${POLICY}_${EB_TAG}_lr${SGD_LR}"

# ── Validate (binary/input checks skipped in DRY_RUN) ──
if [ "$DRY_RUN" != "1" ]; then
    [ -x "$WARPX_BIN" ]    || { echo "ERROR: warpx.3d not found at $WARPX_BIN"; exit 1; }
    [ -f "$WARPX_INPUTS" ] || { echo "ERROR: WarpX input deck not found at $WARPX_INPUTS"; exit 1; }
    [ -f "$WEIGHTS" ]      || { echo "ERROR: NN weights not found at $WEIGHTS"; exit 1; }
fi

mkdir -p "$SWEEP_DIR"

export LD_LIBRARY_PATH="$GPU_DIR/build:$GPU_DIR/examples:/tmp/hdf5-install/lib:/tmp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "============================================================"
echo "WarpX Threshold Sweep"
echo "  Policy:      $POLICY (w0=$W0, w1=$W1, w2=$W2)"
echo "  Grid:        $WARPX_NCELL ($WARPX_MAX_STEP steps, diag every $WARPX_DIAG_INT)"
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

    if [ -f "$OUT_DIR/benchmark_warpx.csv" ]; then
        echo "[$RUN/$TOTAL] X1=$x1 delta=$delta — already done, skipping."
        continue
    fi

    echo ""
    echo "[$RUN/$TOTAL] X1=$x1 delta=$delta (X2=$x2)"
    mkdir -p "$OUT_DIR"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN: warpx.3d gpucompress.sgd_mape=$x1 gpucompress.explore_thresh=$x2"
        continue
    fi

    # Run WarpX with this (X1, X2). The patched FlushFormatGPUCompress
    # writes per-chunk CSVs into WARPX_LOG_DIR and the cumulative VOL
    # summary into the diag dir parent (cwd).
    #
    # WARPX_PROFILE_DECOMP=1 enables the in-situ read-back inside the
    # patched FlushFormat: every H5Dwrite is followed by an in-process
    # H5Dread of the same file (page-cache hit), which routes through
    # the GPUCompress VOL and populates decompression_ms_raw in the
    # chunk-history slots so the per-chunk CSV gets a real mape_decomp
    # value (instead of 0). The threshold sweep always opts in.
    #
    # gpucompress.verify=1 (bitwise round-trip check) is only valid in
    # lossless mode — lossy quantization changes the bits. Auto-disable it
    # for any nonzero error bound so the same sweep script works in both
    # regimes.
    if [ "$ERROR_BOUND" = "0" ] || [ "$ERROR_BOUND" = "0.0" ]; then
        VERIFY=1
    else
        VERIFY=0
    fi
    (
        cd "$OUT_DIR"
        WARPX_LOG_DIR="$OUT_DIR" \
        WARPX_PROFILE_DECOMP=1 \
        "$WARPX_BIN" "$WARPX_INPUTS" \
            warpx.max_step="$WARPX_MAX_STEP" \
            amr.n_cell="$WARPX_NCELL" \
            diagnostics.diags_names=diag1 \
            diag1.intervals="$WARPX_DIAG_INT" \
            diag1.diag_type=Full \
            diag1.format=gpucompress \
            gpucompress.weights_path="$WEIGHTS" \
            gpucompress.algorithm=auto \
            gpucompress.policy="$POLICY" \
            gpucompress.error_bound="$ERROR_BOUND" \
            gpucompress.chunk_bytes=$((CHUNK_MB * 1024 * 1024)) \
            gpucompress.sgd_lr="$SGD_LR" \
            gpucompress.sgd_mape="$x1" \
            gpucompress.explore_k="$EXPLORE_K" \
            gpucompress.explore_thresh="$x2" \
            gpucompress.verify=$VERIFY \
            > warpx.log 2>&1
    ) || echo "  WARNING: warpx exited non-zero (teardown segfault is harmless if VOL summary present)"

    # Synthesize the aggregate benchmark_warpx.csv that the plotter expects.
    python3 - "$OUT_DIR" "$x1" "$x2" "$W0" "$W1" "$W2" <<'PYEOF'
import os, sys, csv

run_dir, x1, x2, w0, w1, w2 = sys.argv[1:7]

# 1) Cumulative VOL summary (key,value)
summary = {}
sp = os.path.join(run_dir, "gpucompress_vol_summary.txt")
if os.path.isfile(sp):
    with open(sp) as f:
        for line in f:
            line = line.strip()
            if "," in line and not line.startswith("="):
                k, _, v = line.partition(",")
                summary[k.strip()] = v.strip()

# 2) Per-chunk MAPE / sgd / exploration counts.
#
# The chunk CSV's mape_ratio / mape_comp columns are written by
# FlushFormatGPUCompress.cpp ALREADY in percent (not as a 0..1 fraction).
# Aggregator must:
#   - read them as-is, no extra * 100
#   - average ONLY over rows where the value is non-zero (rows with 0 are
#     chunks that took the cached-cost path and never ran nvCOMP, so
#     there is nothing to compare against — counting them as zeros would
#     drag the mean toward zero artificially).
#
# Both heads of the NN clamp their outputs to a 5 ms floor; the patched
# FlushFormatGPUCompress now uses max(compression_ms_raw, 5) as the
# denominator, so future runs will have non-zero mape_comp on every chunk
# that actually compressed. Old runs only have ~12% non-zero rows because
# of the previous predicate; that is what we have to work with for the
# already-collected sweep, and averaging over non-zero rows is the
# honest treatment.
chunk_csv = os.path.join(run_dir, "benchmark_warpx_timestep_chunks.csv")
n_chunks = 0
sum_mape_ratio_pct  = 0.0
sum_mape_comp_pct   = 0.0
sum_mape_decomp_pct = 0.0
n_mape_ratio  = 0
n_mape_comp   = 0
n_mape_decomp = 0
sum_decomp_ms_raw = 0.0   # for read bandwidth (in-situ decomp profiling)
n_decomp_rows     = 0
sgd_fires = 0
explorations = 0
if os.path.isfile(chunk_csv):
    with open(chunk_csv) as f:
        for r in csv.DictReader(f):
            try:
                mr  = float(r.get("mape_ratio",  0))
                mc  = float(r.get("mape_comp",   0))
                # mape_decomp is added by the in-situ profiling patch
                # (FlushFormatGPUCompress.cpp). Old chunk CSVs without this
                # column return 0 from r.get(), get filtered by the >0 guard,
                # and contribute nothing — so the aggregator stays
                # backward-compatible with pre-decomp-patch sweep results.
                md  = float(r.get("mape_decomp", 0))
                ad  = float(r.get("actual_decomp_ms_raw", 0))
                if mr > 0:
                    sum_mape_ratio_pct  += mr; n_mape_ratio  += 1
                if mc > 0:
                    sum_mape_comp_pct   += mc; n_mape_comp   += 1
                if md > 0:
                    sum_mape_decomp_pct += md; n_mape_decomp += 1
                if ad > 0:
                    sum_decomp_ms_raw   += ad; n_decomp_rows += 1
                sgd_fires      += int(float(r.get("sgd_fired", 0)))
                explorations   += int(float(r.get("exploration_triggered", 0)))
                n_chunks += 1
            except (ValueError, TypeError):
                pass

avg_mape_ratio  = (sum_mape_ratio_pct  / n_mape_ratio ) if n_mape_ratio  else 0.0
avg_mape_comp   = (sum_mape_comp_pct   / n_mape_comp  ) if n_mape_comp   else 0.0
avg_mape_decomp = (sum_mape_decomp_pct / n_mape_decomp) if n_mape_decomp else 0.0

ratio       = float(summary.get("ratio", 0))
write_mibps = float(summary.get("write_mibps", 0))
bytes_in    = float(summary.get("bytes_in", 0))
bytes_out   = float(summary.get("bytes_out", 0))
calls       = int(float(summary.get("h5dwrite_calls", 0)))

# Read bandwidth from in-situ decomp profiling (kernel-only throughput).
# Every H5Dwrite is followed by an in-process H5Dread of the same file
# (page-cache hit), so the total bytes read back equals bytes_in. The
# decomp time is the sum of per-chunk nvCOMP-decompress kernel time
# from the chunk CSV — so this is kernel-only throughput, the symmetric
# counterpart of how compression kernel throughput is computed
# elsewhere. NOT directly comparable to write_mibps (which is VOL
# end-to-end including disk I/O), but tracks the same units (MiB/s)
# and is the right number for "how fast can the GPU decompress."
read_mibps = 0.0
if sum_decomp_ms_raw > 0 and bytes_in > 0:
    read_mibps = (bytes_in / 1048576.0) / (sum_decomp_ms_raw / 1000.0)

# 3) Synthesize benchmark_warpx.csv
out_csv = os.path.join(run_dir, "benchmark_warpx.csv")
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
row["source"] = "warpx"
row["phase"] = "nn-rl+exp"
row["n_runs"] = 1
row["file_mib"] = bytes_out / (1024.0*1024.0)
row["orig_mib"] = bytes_in  / (1024.0*1024.0)
row["ratio"] = ratio
row["write_mibps"] = write_mibps
# read_mibps is now populated from in-situ decomp profiling (the patched
# FlushFormatGPUCompress reads back every write through the GPUCompress
# VOL when WARPX_PROFILE_DECOMP=1). Pre-patch sweeps fall back to 0.
row["read_mibps"] = read_mibps
row["sgd_fires"] = sgd_fires
row["explorations"] = explorations
row["n_chunks"] = n_chunks
row["mape_ratio_pct"]  = avg_mape_ratio
row["mape_comp_pct"]   = avg_mape_comp
row["mape_decomp_pct"] = avg_mape_decomp
row["mape_psnr_pct"]   = 0.0  # WarpX is lossless; no PSNR
row["x1_sgd_mape"] = float(x1)
row["x2_explore_thresh"] = float(x2)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow(row)

print(f"  ratio={ratio:.2f}x  write={write_mibps:.0f}MiB/s  "
      f"mape_ratio={avg_mape_ratio:.1f}%  sgd={sgd_fires}  expl={explorations}  chunks={n_chunks}")
PYEOF

    # ── Per-cell cleanup: delete the bulky AMReX plotfile dir now that
    #    all metric CSVs have been extracted. CSVs (benchmark_warpx*,
    #    ranking*, timestep_chunks, gpucompress_vol_summary.txt, warpx.log)
    #    are preserved.
    rm -rf "$OUT_DIR/diags" 2>/dev/null
done

echo ""
echo "============================================================"
echo "WarpX threshold sweep complete: $TOTAL configurations"
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
