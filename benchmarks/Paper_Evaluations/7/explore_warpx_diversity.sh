#!/bin/bash
# ============================================================
# WarpX Parameter Diversity Explorer
#
# Systematically tries 4 grid/step/interval configurations to
# find which produces the MOST diverse compression characteristics
# across consecutive diagnostic writes.
#
# "Diversity" is measured as the coefficient of variation (CV =
# stddev/mean) of the per-write mean actual_ratio, plus the
# max-to-min spread of per-write mean ratios.  Higher CV means
# the physics is actively transitioning the data between
# compressible and incompressible states — which exercises the
# NN's online adaptation most aggressively.
#
# Physics insight: the LWFA moving window continuously sweeps
# fresh zero-initialized plasma into the domain.  At t=0 the
# entire domain is near-zero (compressible).  As the laser
# pulse enters and drives the wake, field energy fills the
# domain and chunks become incompressible.  Configurations that
# capture this transition window — enough steps to drive the
# physics but fine enough diagnostic spacing to see each stage —
# will show the largest ratio spread.
#
# Configurations explored:
#   C1: 32x32x128, 60 steps, diag every 10  (baseline, quick)
#   C2: 32x32x256, 80 steps, diag every 10  (production grid, capture wake)
#   C3: 32x32x256, 60 steps, diag every  6  (finer time resolution)
#   C4: 64x64x256, 60 steps, diag every 10  (wider domain, more zero-fill)
#
# Usage:
#   cd /home/cc/GPUCompress
#   bash benchmarks/Paper_Evaluations/7/explore_warpx_diversity.sh
#
# Output:
#   benchmarks/Paper_Evaluations/7/results/diversity_explore_<timestamp>/
#     config_N/warpx_sim.log        — full WarpX stdout
#     config_N/timestep_chunks.csv  — per-chunk compression data
#     diversity_summary.txt         — ranked summary printed to stdout
#
# Environment overrides:
#   WARPX_BIN     Path to warpx.3d binary
#   WARPX_INPUTS  Path to base inputs file
#   WEIGHTS       Path to .nnwt weights file
#   RESULTS_DIR   Override output directory
# ============================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

WARPX_BIN="${WARPX_BIN:-/home/cc/src/warpx/build-gpucompress/bin/warpx.3d}"
WARPX_INPUTS="${WARPX_INPUTS:-/home/cc/src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d}"
WEIGHTS="${WEIGHTS:-$PROJECT_DIR/neural_net/weights/model.nnwt}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/diversity_explore_$TIMESTAMP}"

export LD_LIBRARY_PATH="/tmp/hdf5-install/lib:$PROJECT_DIR/build:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Helpers ──────────────────────────────────────────────────
note()  { printf "[explore] %s\n" "$*"; }
ruler() { printf "%-60s\n" "" | tr ' ' '-'; }

# ── Sanity checks ────────────────────────────────────────────
if [ ! -x "$WARPX_BIN" ]; then
    echo "ERROR: WarpX binary not found: $WARPX_BIN"
    echo "  Set WARPX_BIN= to override."
    exit 1
fi
if [ ! -f "$WARPX_INPUTS" ]; then
    echo "ERROR: Base inputs not found: $WARPX_INPUTS"
    exit 1
fi
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: NN weights not found: $WEIGHTS"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ── Configuration table ──────────────────────────────────────
# Each entry: "label|ncell_x ncell_y ncell_z|max_step|diag_interval|rationale"
#
# Design reasoning per config:
#
#   C1 (32x32x128, 60 steps, int=10, 6 writes):
#     Half-length z domain.  The shorter domain means the laser
#     pulse reaches the plasma earlier, compressing the "zero
#     fill" phase.  Expect rapid transition: writes 0-1 near-zero
#     → writes 3-5 laser-filled.  Small grid keeps runtime <30s.
#
#   C2 (32x32x256, 80 steps, int=10, 8 writes):
#     Full production grid from existing benchmarks.  Extended to
#     80 steps so the wake is fully developed by write 7.  The
#     additional steps versus the existing 70-step runs let the
#     moving window scroll further, introducing more fresh plasma
#     and creating a second zero-fill wave behind the bubble.
#
#   C3 (32x32x256, 60 steps, int=6, 10 writes):
#     Same grid, fewer total steps, but diagnostic every 6 steps
#     instead of 10.  Finer temporal sampling captures the rapid
#     transition when the laser first enters the plasma (~steps
#     20-35) at higher resolution.  10 diagnostic points reveal
#     the gradient of compressibility change more precisely.
#
#   C4 (64x64x256, 60 steps, int=10, 6 writes):
#     Double transverse resolution.  A 64x64 transverse domain
#     has 4x more cells in x-y, so the laser occupies a smaller
#     FRACTION of each chunk by volume.  More zero-valued cells
#     outside the plasma channel → higher initial compressibility
#     AND steeper drop-off as the wake grows into the domain.
#     This amplifies the ratio contrast between early and late
#     writes.  Runtime ~45s expected (4x data per write).
#
CONFIGS=(
    "C1|32 32 128|60|10|Short-z domain: rapid zero→laser transition"
    "C2|32 32 256|80|10|Full grid extended: wake fully developed by write 7"
    "C3|32 32 256|60|6|Fine diag interval: captures transition gradient with 10 writes"
    "C4|64 64 256|60|10|Wide transverse: low laser fill fraction amplifies ratio contrast"
)

# ── Statistics helpers (pure awk, no Python) ─────────────────
# These are embedded as awk programs called after each run.

# Compute per-write mean ratio from the timestep_chunks CSV.
# CSV columns (0-indexed): rank,phase,timestep,chunk,predicted_ratio,actual_ratio,...
# We want column 5 (actual_ratio) grouped by column 2 (timestep).
AWK_PER_WRITE_STATS='
BEGIN { FS="," }
NR==1 { next }   # skip header
{
    ts=$3+0; ratio=$6
    sum[ts]   += ratio
    sumsq[ts] += ratio*ratio
    n[ts]++
    if (!(ts in min) || ratio < min[ts]) min[ts] = ratio
    if (!(ts in max) || ratio > max[ts]) max[ts] = ratio
}
END {
    printf "%-6s  %-10s  %-10s  %-10s  %-10s  %-8s\n",
           "write", "mean_ratio", "min_ratio", "max_ratio", "spread", "n_chunks"
    for (ts in sum) {
        m      = sum[ts] / n[ts]
        spread = max[ts] - min[ts]
        printf "%d %.4f %.4f %.4f %.4f %d\n",
               ts, m, min[ts], max[ts], spread, n[ts]
    }
}
'
# Note: output is piped through "sort -n" on first column before display

# Compute the diversity score: CV of per-write mean ratios.
# Returns: n_writes mean_of_means stddev cv min_mean max_mean spread_of_means
AWK_DIVERSITY_SCORE='
BEGIN { FS="," }
NR==1 { next }
{
    ts=$3; ratio=$6
    wsum[ts]  += ratio
    wn[ts]++
}
END {
    # First compute per-write means
    n_writes = 0
    grand_sum = 0; grand_sumsq = 0
    min_mean = 1e18; max_mean = -1e18
    for (ts in wsum) {
        m = wsum[ts] / wn[ts]
        write_means[n_writes++] = m
        grand_sum   += m
        grand_sumsq += m*m
        if (m < min_mean) min_mean = m
        if (m > max_mean) max_mean = m
    }
    if (n_writes < 2) {
        print "0 0 0 0 0 0 0"
        exit
    }
    grand_mean = grand_sum / n_writes
    var = (grand_sumsq / n_writes) - grand_mean*grand_mean
    if (var < 0) var = 0
    stddev = sqrt(var)
    cv = (grand_mean > 0) ? stddev / grand_mean : 0
    spread = max_mean - min_mean
    printf "%d %.4f %.4f %.4f %.4f %.4f %.4f\n",
           n_writes, grand_mean, stddev, cv, min_mean, max_mean, spread
}
'

# ── Per-config results accumulator ───────────────────────────
declare -a RESULT_LABELS
declare -a RESULT_SCORES
declare -a RESULT_SUMMARIES
declare -a RESULT_CSVPATHS
declare -a RESULT_WALLTIME

CFG_INDEX=0

ruler
note "WarpX Diversity Explorer"
note "Output: $RESULTS_DIR"
note "$(date)"
ruler

# ── Main loop ────────────────────────────────────────────────
for cfg_entry in "${CONFIGS[@]}"; do
    # Parse config entry
    IFS='|' read -r cfg_label ncell max_step diag_interval rationale <<< "$cfg_entry"

    CFG_INDEX=$((CFG_INDEX + 1))
    CFG_DIR="$RESULTS_DIR/${cfg_label}"
    WORK_DIR="$CFG_DIR/work"
    mkdir -p "$CFG_DIR" "$WORK_DIR"

    n_writes=$(( max_step / diag_interval ))

    echo ""
    ruler
    note "Config $cfg_label: ncell=[$ncell] max_step=$max_step diag_int=$diag_interval (~$n_writes writes)"
    note "Rationale: $rationale"
    ruler

    # ── Build inputs file ────────────────────────────────────
    # Copy base inputs, then append all overrides at the end so they
    # take precedence (WarpX uses last-definition-wins for duplicates).
    INPUT_FILE="$WORK_DIR/inputs"
    cp "$WARPX_INPUTS" "$INPUT_FILE"

    # Compute chunk bytes: 4 MB
    CHUNK_BYTES=$(( 4 * 1024 * 1024 ))

    cat >> "$INPUT_FILE" << INPUTEOF

# ── Diversity explorer overrides for $cfg_label ──
amr.n_cell = $ncell
amr.max_grid_size = 512
amr.blocking_factor = 32
max_step = $max_step

# Replace diag1 (openpmd) with gpucompress diagnostic
diagnostics.diags_names = gpuc_diag
gpuc_diag.intervals = $diag_interval
gpuc_diag.diag_type = Full
gpuc_diag.format = gpucompress
gpuc_diag.fields_to_plot = Ex Ey Ez Bx By Bz jx jy jz rho

# GPUCompress settings
gpucompress.weights_path = $WEIGHTS
gpucompress.algorithm = auto
gpucompress.policy = ratio
gpucompress.error_bound = 0.0
gpucompress.chunk_bytes = $CHUNK_BYTES
gpucompress.verify = 0
gpucompress.sgd_lr = 0.1
gpucompress.sgd_mape = 0.10
gpucompress.explore_k = 4
gpucompress.explore_thresh = 0.20
INPUTEOF

    note "  Inputs written to $INPUT_FILE"

    # ── Run WarpX ────────────────────────────────────────────
    LOG_FILE="$CFG_DIR/warpx_sim.log"
    TC_CSV="$CFG_DIR/timestep_chunks.csv"

    note "  Launching WarpX (output: $LOG_FILE) ..."
    T_START=$(date +%s)

    cd "$WORK_DIR"
    # WARPX_LOG_DIR tells FlushFormatGPUCompress where to write the
    # per-chunk CSV (benchmark_<name>_timestep_chunks.csv).
    WARPX_LOG_DIR="$CFG_DIR" \
        "$WARPX_BIN" inputs \
        > "$LOG_FILE" 2>&1 \
        || {
            note "  WARNING: WarpX exited non-zero — check $LOG_FILE"
        }
    cd "$PROJECT_DIR"

    T_END=$(date +%s)
    WALL=$(( T_END - T_START ))
    RESULT_WALLTIME[$CFG_INDEX]="$WALL"
    note "  WarpX finished in ${WALL}s"

    # ── Locate timestep_chunks CSV ───────────────────────────
    # FlushFormatGPUCompress writes: benchmark_<diag_name>_timestep_chunks.csv
    # The diag name is "gpuc_diag" so the file is:
    #   $WARPX_LOG_DIR/benchmark_gpuc_diag_timestep_chunks.csv
    FOUND_CSV=""
    for candidate in \
            "$CFG_DIR/benchmark_gpuc_diag_timestep_chunks.csv" \
            "$CFG_DIR/benchmark_warpx_timestep_chunks.csv" \
            "$CFG_DIR/timestep_chunks.csv"; do
        if [ -f "$candidate" ]; then
            FOUND_CSV="$candidate"
            break
        fi
    done

    # Also search the work directory (WarpX sometimes writes relative to cwd)
    if [ -z "$FOUND_CSV" ]; then
        for candidate in \
                "$WORK_DIR/benchmark_gpuc_diag_timestep_chunks.csv" \
                "$WORK_DIR/benchmark_warpx_timestep_chunks.csv"; do
            if [ -f "$candidate" ]; then
                cp "$candidate" "$CFG_DIR/timestep_chunks.csv"
                FOUND_CSV="$CFG_DIR/timestep_chunks.csv"
                break
            fi
        done
    fi

    if [ -z "$FOUND_CSV" ]; then
        note "  WARNING: No timestep_chunks CSV found — checking log for clues"
        grep -iE "csv|chunk|gpucompress|wrote|ERROR" "$LOG_FILE" | tail -20 || true
        RESULT_LABELS[$CFG_INDEX]="$cfg_label"
        RESULT_SCORES[$CFG_INDEX]="N/A"
        RESULT_SUMMARIES[$CFG_INDEX]="NO CSV PRODUCED"
        RESULT_CSVPATHS[$CFG_INDEX]="none"
        continue
    fi

    # Copy to canonical name for later reference
    cp "$FOUND_CSV" "$TC_CSV" 2>/dev/null || true
    N_ROWS=$(tail -n +2 "$TC_CSV" | wc -l)
    note "  CSV: $FOUND_CSV ($N_ROWS chunk rows)"

    if [ "$N_ROWS" -eq 0 ]; then
        note "  WARNING: CSV has no data rows"
        RESULT_LABELS[$CFG_INDEX]="$cfg_label"
        RESULT_SCORES[$CFG_INDEX]="N/A"
        RESULT_SUMMARIES[$CFG_INDEX]="CSV EMPTY"
        RESULT_CSVPATHS[$CFG_INDEX]="$TC_CSV"
        continue
    fi

    # ── Per-write statistics ─────────────────────────────────
    note "  Per-write ratio statistics:"
    echo ""
    # Header line printed separately; data sorted numerically by timestep
    printf "    %-6s  %-10s  %-10s  %-10s  %-10s  %-8s\n" \
           "write" "mean_ratio" "min_ratio" "max_ratio" "spread" "n_chunks"
    PER_WRITE_OUTPUT=$(awk "$AWK_PER_WRITE_STATS" "$TC_CSV" | sort -n)
    echo "$PER_WRITE_OUTPUT" | awk '{
        printf "    %-6d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-8d\n",
               $1, $2, $3, $4, $5, $6
    }'
    echo ""

    # Save per-write stats with header
    {
        printf "%-6s  %-10s  %-10s  %-10s  %-10s  %-8s\n" \
               "write" "mean_ratio" "min_ratio" "max_ratio" "spread" "n_chunks"
        echo "$PER_WRITE_OUTPUT" | awk '{
            printf "%-6d  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-8d\n",
                   $1, $2, $3, $4, $5, $6
        }'
    } > "$CFG_DIR/per_write_stats.txt"

    # ── Diversity score ──────────────────────────────────────
    DIVERSITY_RAW=$(awk "$AWK_DIVERSITY_SCORE" "$TC_CSV")
    read -r n_writes_actual mean_ratio stddev cv min_mean max_mean spread_means <<< "$DIVERSITY_RAW"

    note "  Diversity metrics:"
    note "    n_writes       = $n_writes_actual"
    note "    mean ratio     = $mean_ratio"
    note "    stddev ratio   = $stddev"
    note "    CV (std/mean)  = $cv   [higher = more diverse]"
    note "    min write mean = $min_mean"
    note "    max write mean = $max_mean"
    note "    spread (max-min of write means) = $spread_means"

    # The primary diversity score is:
    #   spread_means * n_writes
    # This rewards configs where:
    #   (a) the write-to-write variation is large (spread), AND
    #   (b) there are enough writes to observe the transition (n_writes)
    # CV alone would give high score to configs with only 2 writes where
    # one is 100.0 and one is 1.0, which is trivial.
    SCORE=$(awk "BEGIN { printf \"%.4f\", $spread_means * $n_writes_actual }")
    note "    SCORE (spread * n_writes) = $SCORE"

    RESULT_LABELS[$CFG_INDEX]="$cfg_label"
    RESULT_SCORES[$CFG_INDEX]="$SCORE"
    RESULT_SUMMARIES[$CFG_INDEX]="writes=$n_writes_actual mean=$mean_ratio spread=$spread_means cv=$cv wall=${WALL}s"
    RESULT_CSVPATHS[$CFG_INDEX]="$TC_CSV"
done

# ── Final ranking ─────────────────────────────────────────────
echo ""
ruler
note "DIVERSITY RANKING (score = spread_of_write_means * n_writes)"
ruler
echo ""
printf "  %-6s  %-10s  %-50s  %s\n" "Config" "Score" "Summary" "CSV"
printf "  %-6s  %-10s  %-50s  %s\n" "------" "----------" \
       "--------------------------------------------------" "---"

# Sort by score descending using awk
RANKING_INPUT=""
for i in "${!RESULT_LABELS[@]}"; do
    lbl="${RESULT_LABELS[$i]:-?}"
    scr="${RESULT_SCORES[$i]:-0}"
    sum="${RESULT_SUMMARIES[$i]:-}"
    csv="${RESULT_CSVPATHS[$i]:-}"
    RANKING_INPUT+="${scr}|${lbl}|${sum}|${csv}"$'\n'
done

SORTED=$(printf "%s" "$RANKING_INPUT" | sort -t'|' -k1 -rn)
RANK=1
BEST_LABEL=""
BEST_CSV=""
while IFS='|' read -r scr lbl sum csv; do
    [ -z "$lbl" ] && continue
    printf "  #%-5d %-10s %-50s  %s\n" "$RANK" "$lbl($scr)" "$sum" "$csv"
    if [ "$RANK" -eq 1 ]; then
        BEST_LABEL="$lbl"
        BEST_CSV="$csv"
    fi
    RANK=$((RANK + 1))
done <<< "$SORTED"

echo ""
ruler
note "WINNER: $BEST_LABEL"
ruler

if [ -n "$BEST_CSV" ] && [ -f "$BEST_CSV" ]; then
    echo ""
    note "Per-write ratio detail for $BEST_LABEL:"
    echo ""

    # Expanded stats: show per-component variation (body chunks 0-3 vs trailing)
    printf "  %-6s  %-10s  %-11s  %-11s  %-10s  %-10s\n" \
           "write" "mean_ratio" "body_mean" "tail_mean" "min" "max"
    printf "  %-6s  %-10s  %-11s  %-11s  %-10s  %-10s\n" \
           "------" "----------" "-----------" "-----------" "----------" "----------"
    awk -F, '
    NR==1 { next }
    {
        ts=$3+0; chunk=$4+0; ratio=$6
        # body = chunk index 0-3; tail = chunk index >= 4
        if (chunk < 4) {
            body_sum[ts] += ratio
            body_n[ts]++
        } else {
            tail_sum[ts] += ratio
            tail_n[ts]++
        }
        all_sum[ts] += ratio
        all_n[ts]++
        if (!(ts in all_min) || ratio < all_min[ts]) all_min[ts] = ratio
        if (!(ts in all_max) || ratio > all_max[ts]) all_max[ts] = ratio
    }
    END {
        for (ts in all_sum) {
            m  = all_sum[ts] / all_n[ts]
            bm = (body_n[ts] > 0) ? body_sum[ts] / body_n[ts] : 0
            tm = (tail_n[ts] > 0) ? tail_sum[ts] / tail_n[ts] : 0
            printf "%d %.4f %.4f %.4f %.4f %.4f\n",
                   ts, m, bm, tm, all_min[ts], all_max[ts]
        }
    }
    ' "$BEST_CSV" | sort -n | awk '{
        printf "  %-6d  %-10.4f  %-11.4f  %-11.4f  %-10.4f  %-10.4f\n",
               $1, $2, $3, $4, $5, $6
    }'

    echo ""
    note "Interpretation:"
    note "  body_mean = mean ratio for body chunks (chunk index 0-3)"
    note "  tail_mean = mean ratio for trailing chunks (chunk index >= 4)"
    note "  Large body_mean spread across writes → bulk field evolution"
    note "  Large tail_mean spread → partial-FAB boundary effects"
fi

echo ""

# ── Write summary file ────────────────────────────────────────
SUMMARY_FILE="$RESULTS_DIR/diversity_summary.txt"
{
    echo "WarpX Diversity Explorer Summary"
    echo "Run: $TIMESTAMP"
    echo ""
    echo "Configs tried:"
    for cfg in "${CONFIGS[@]}"; do
        IFS='|' read -r label ncell steps diag_int rationale <<< "$cfg"
        echo "  $label: ncell=[$ncell] max_step=$steps diag_int=$diag_int — $rationale"
    done
    echo ""
    echo "Scoring: score = spread(per-write mean ratio) * n_writes"
    echo "Higher score = more write-to-write compression variation = more NN challenge"
    echo ""
    echo "Results (sorted by score descending):"
    printf "%s" "$SORTED" | while IFS='|' read -r scr lbl sum csv; do
        [ -z "$lbl" ] && continue
        echo "  $lbl  score=$scr  $sum"
    done
    echo ""
    echo "Winner: $BEST_LABEL"
    echo "Winner CSV: $BEST_CSV"
} > "$SUMMARY_FILE"

note "Summary written to $SUMMARY_FILE"
echo ""
note "All results in: $RESULTS_DIR"
note "Done. $(date)"
ruler
