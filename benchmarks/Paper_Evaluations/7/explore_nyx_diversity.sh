#!/bin/bash
# ============================================================
# NYX Parameter Diversity Explorer
#
# Systematically tries 4 Sedov blast wave configurations to find
# which produces the MOST ratio variation across timestep dumps.
# Metric: stddev of per-timestep mean compression ratio.
#
# Higher stddev => shock evolves more through compression regimes
# between dumps => richer NN training signal => better paper story.
#
# Configurations explore two axes:
#   Grid size   : 64^3 (fast) vs 96^3 (more chunks, slower shock spread)
#   Dump spacing: tight (many early dumps, catch shock forming) vs
#                 coarse (fewer dumps, catch shock at different stages)
#
# Keeps each run to 30-60 s wall time by:
#   - Disabling verify (no round-trip I/O)
#   - Disabling checkpoints
#   - Silencing verbosity (nyx.v=0, amr.v=0, nyx.sum_interval=-1)
#   - max_step chosen so CFL ramp finishes and shock is visible
#
# Usage:
#   bash benchmarks/Paper_Evaluations/7/explore_nyx_diversity.sh
#
# Optional overrides:
#   NYX_BIN=<path>         Override binary path
#   WEIGHTS=<path>         Override model weights path
#   SKIP_CLEANUP=1         Keep per-config run directories
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Paths ──────────────────────────────────────────────────────
NYX_BIN="${NYX_BIN:-/home/cc/Nyx/build-gpucompress/Exec/HydroTests/nyx_HydroTests}"
NYX_INPUTS_REF="/home/cc/Nyx/build-gpucompress/Exec/HydroTests/inputs.3d.sph.sedov"
WEIGHTS="${WEIGHTS:-${GPUCOMPRESS_WEIGHTS:-$PROJECT_DIR/neural_net/weights/model.nnwt}}"
RESULTS_DIR="$SCRIPT_DIR/results/diversity_sweep_$(date +%Y%m%d_%H%M%S)"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"

export LD_LIBRARY_PATH="/tmp/hdf5-install/lib:$PROJECT_DIR/build:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GPUCOMPRESS_WEIGHTS="$WEIGHTS"

# ── Helpers ────────────────────────────────────────────────────
note() { printf "[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }
die()  { printf "ERROR: %s\n" "$*" >&2; exit 1; }
hr()   { printf "%s\n" "$(printf '%0.s─' {1..68})"; }

# ── Preflight ──────────────────────────────────────────────────
[ -x "$NYX_BIN"      ] || die "NYX binary not found: $NYX_BIN"
[ -f "$NYX_INPUTS_REF" ] || die "Sedov input file not found: $NYX_INPUTS_REF"
[ -f "$WEIGHTS"      ] || die "NN weights not found: $WEIGHTS"

mkdir -p "$RESULTS_DIR"

note "NYX diversity explorer"
note "Binary : $NYX_BIN"
note "Weights: $WEIGHTS"
note "Results: $RESULTS_DIR"
hr

# ── Configuration table ────────────────────────────────────────
# Format: "label:n_cell:max_grid_size:max_step:plot_int"
#
# Design rationale:
#   A) 64^3, plot every 25 steps for 200 steps  — 8 dumps, tight early dumps
#      catch shock forming and expanding rapidly through small domain.
#      CFL ramp (init_shrink=0.01, change_max=1.1) takes ~50 steps to reach
#      full speed; dumps at 25,50,75... sample all phases.
#
#   B) 64^3, plot every 50 steps for 200 steps  — 4 dumps, coarser spacing.
#      Each dump sees more shock travel. Larger ratio drop per interval.
#      Baseline from README (step 0→50→100→150→200 shows 369→343→250→188→141).
#
#   C) 96^3, plot every 40 steps for 240 steps  — 6 dumps, medium grid.
#      More cells per FAB => more 4 MiB chunks per dump => better per-dump
#      statistics. max_grid_size=96 keeps one FAB covering the whole domain.
#      Shock travels proportionally slower through larger domain.
#
#   D) 96^3, plot every 80 steps for 320 steps  — 4 dumps, coarser on 96^3.
#      Each dump separated by large shock advance; expect biggest ratio swing
#      per interval but fewer data points. Tests whether fewer dumps with
#      larger delta wins over more dumps with smaller delta.
#
# Timing estimates (from existing 128^3 run: ~2.7 s/plotfile at 104 MB):
#   64^3 = 12 MB/dump => ~0.3 s/dump; step wall time ~14 ms => very fast
#   96^3 = 41 MB/dump => ~1.0 s/dump; step wall time ~50 ms
#   Worst case: config D = 320 steps * 50 ms + 4 * 1.0 s ~ 20 s total
CONFIGS=(
    "A_64cell_plot25:64:64:200:25"
    "B_64cell_plot50:64:64:200:50"
    "C_96cell_plot40:96:96:240:40"
    "D_96cell_plot80:96:96:320:80"
)

# ── Per-config result storage ──────────────────────────────────
declare -A CFG_STDDEV
declare -A CFG_RANGE
declare -A CFG_RATIOS
declare -A CFG_NDUMPS
declare -A CFG_WALL_S

# ── CSV analysis function ──────────────────────────────────────
# Reads benchmark_nyx_timestep_chunks.csv and prints per-timestep stats,
# then computes stddev and range of the per-timestep mean ratios.
# Outputs: sets global LAST_STDDEV, LAST_RANGE, LAST_RATIO_LIST, LAST_N_DUMPS
analyze_csv() {
    local csv_path="$1"

    if [ ! -f "$csv_path" ]; then
        note "  WARNING: CSV not found: $csv_path"
        LAST_STDDEV="0"; LAST_RANGE="0"; LAST_RATIO_LIST="(none)"; LAST_N_DUMPS=0
        return
    fi

    # Python3 inline: compute per-timestep mean actual_ratio from the CSV.
    # Column indices (0-based): timestep=2, actual_ratio=7
    # Rows with actual_ratio <= 0 are skipped (failed/lossless-zero chunks).
    python3 - "$csv_path" <<'PYEOF'
import sys, csv, math
from collections import defaultdict

path = sys.argv[1]
ts_ratios = defaultdict(list)

with open(path) as f:
    reader = csv.reader(f)
    header = next(reader)
    # Locate columns by name so we're robust to minor schema changes
    try:
        ts_col  = header.index("timestep")
        rat_col = header.index("actual_ratio")
    except ValueError:
        # Fallback to fixed positions matching schema we read in the review
        ts_col, rat_col = 2, 7

    for row in reader:
        if len(row) <= rat_col:
            continue
        try:
            ts  = int(row[ts_col])
            rat = float(row[rat_col])
        except ValueError:
            continue
        if rat > 0:
            ts_ratios[ts].append(rat)

if not ts_ratios:
    print("NO_DATA 0 0 (no valid rows)")
    sys.exit(0)

means = []
for ts in sorted(ts_ratios.keys()):
    vals = ts_ratios[ts]
    m = sum(vals) / len(vals)
    means.append(m)
    print(f"  ts={ts:3d}  chunks={len(vals):4d}  mean_ratio={m:7.2f}"
          f"  min={min(vals):7.2f}  max={max(vals):7.2f}")

if len(means) < 2:
    stddev = 0.0
else:
    mu = sum(means) / len(means)
    stddev = math.sqrt(sum((x - mu) ** 2 for x in means) / len(means))

rng = max(means) - min(means)
# Emit a machine-readable summary line for the shell to parse
print(f"SUMMARY stddev={stddev:.4f} range={rng:.4f} n_dumps={len(means)} ratios={','.join(f'{m:.2f}' for m in means)}")
PYEOF
}

# ── Main loop ──────────────────────────────────────────────────
for cfg_spec in "${CONFIGS[@]}"; do
    IFS=':' read -r label n_cell max_grid max_step plot_int <<< "$cfg_spec"

    hr
    note "Config $label"
    note "  n_cell=$n_cell  max_grid_size=$max_grid  max_step=$max_step  plot_int=$plot_int"

    RUN_DIR="$RESULTS_DIR/run_$label"
    LOG_DIR="$RESULTS_DIR/logs_$label"
    mkdir -p "$RUN_DIR" "$LOG_DIR"

    # Write inputs file: start from reference, append overrides at the end.
    # Nyx's ParmParse takes the LAST definition of each parameter, so
    # appending overrides is safe even if the reference file also sets them.
    cp "$NYX_INPUTS_REF" "$RUN_DIR/inputs"
    cat >> "$RUN_DIR/inputs" <<NYXEOF

# ── explore_nyx_diversity.sh overrides ──
amr.n_cell           = $n_cell $n_cell $n_cell
amr.max_grid_size    = $max_grid
max_step             = $max_step
amr.plot_int         = $plot_int
amr.check_int        = 0
nyx.v                = 0
amr.v                = 0
nyx.sum_interval     = -1
nyx.use_gpucompress        = 1
nyx.gpucompress_weights    = $WEIGHTS
nyx.gpucompress_algorithm  = auto
nyx.gpucompress_policy     = ratio
nyx.gpucompress_verify     = 0
NYXEOF

    note "  Running NYX..."
    T_START=$(date +%s)

    cd "$RUN_DIR"
    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    NYX_LOG_DIR="$LOG_DIR" \
        "$NYX_BIN" inputs \
        > "$LOG_DIR/nyx_run.log" 2>&1 || {
            note "  WARNING: NYX exited non-zero — results may be partial"
        }
    cd "$PROJECT_DIR"

    T_END=$(date +%s)
    WALL=$(( T_END - T_START ))
    CFG_WALL_S[$label]=$WALL
    note "  Wall time: ${WALL}s"

    # Locate the timestep_chunks CSV written by NYX_LOG_DIR
    # The DiagLogger names it benchmark_nyx_timestep_chunks.csv
    TC_CSV="$LOG_DIR/benchmark_nyx_timestep_chunks.csv"

    note "  Analyzing ratio evolution..."
    ANALYSIS=$(analyze_csv "$TC_CSV")
    # Print per-timestep table lines (all but the SUMMARY line)
    while IFS= read -r line; do
        case "$line" in
            SUMMARY*|NO_DATA*) : ;;
            *) printf "%s\n" "$line" ;;
        esac
    done <<< "$ANALYSIS"

    # Extract SUMMARY fields
    SUMMARY_LINE=$(printf "%s\n" "$ANALYSIS" | grep -E '^(SUMMARY|NO_DATA)' || true)
    if [[ "$SUMMARY_LINE" =~ stddev=([0-9.]+) ]]; then
        SD="${BASH_REMATCH[1]}"
    else
        SD="0"
    fi
    if [[ "$SUMMARY_LINE" =~ range=([0-9.]+) ]]; then
        RG="${BASH_REMATCH[1]}"
    else
        RG="0"
    fi
    if [[ "$SUMMARY_LINE" =~ n_dumps=([0-9]+) ]]; then
        ND="${BASH_REMATCH[1]}"
    else
        ND="0"
    fi
    if [[ "$SUMMARY_LINE" =~ ratios=([0-9.,]+) ]]; then
        RL="${BASH_REMATCH[1]}"
    else
        RL="(none)"
    fi

    CFG_STDDEV[$label]="$SD"
    CFG_RANGE[$label]="$RG"
    CFG_RATIOS[$label]="$RL"
    CFG_NDUMPS[$label]="$ND"

    note "  Diversity: stddev=$SD  range=$RG  n_dumps=$ND"

    # Optionally clean up large plotfile directories to save disk
    if [ "$SKIP_CLEANUP" != "1" ]; then
        rm -rf "$RUN_DIR"/plt* "$RUN_DIR"/chk* 2>/dev/null || true
        note "  Cleaned plotfile dirs (set SKIP_CLEANUP=1 to retain)"
    fi
done

# ── Final ranking ──────────────────────────────────────────────
hr
printf "\n"
printf "  NYX DIVERSITY SUMMARY\n"
printf "  %-30s  %8s  %8s  %8s  %8s  %s\n" \
    "Config" "stddev" "range" "n_dumps" "wall_s" "per-ts means"
printf "  %-30s  %8s  %8s  %8s  %8s  %s\n" \
    "------" "------" "-----" "-------" "------" "------------"

BEST_LABEL=""
BEST_SD="-1"

for cfg_spec in "${CONFIGS[@]}"; do
    IFS=':' read -r label _ _ _ _ <<< "$cfg_spec"
    SD="${CFG_STDDEV[$label]:-0}"
    RG="${CFG_RANGE[$label]:-0}"
    ND="${CFG_NDUMPS[$label]:-0}"
    WS="${CFG_WALL_S[$label]:-0}"
    RL="${CFG_RATIOS[$label]:-(none)}"
    printf "  %-30s  %8.4f  %8.4f  %8s  %8s  %s\n" \
        "$label" "$SD" "$RG" "$ND" "${WS}s" "$RL"

    # Track best by stddev (using awk for float comparison; no bc needed)
    if [ "$(awk "BEGIN{print ($SD > $BEST_SD) ? 1 : 0}")" = "1" ]; then
        BEST_SD="$SD"
        BEST_LABEL="$label"
    fi
done

printf "\n"
printf "  WINNER (highest ratio stddev across timesteps): %s\n" "$BEST_LABEL"
printf "  Stddev: %s   Ratios: %s\n" \
    "${CFG_STDDEV[$BEST_LABEL]:-?}" \
    "${CFG_RATIOS[$BEST_LABEL]:-(none)}"
printf "\n"
printf "  Interpretation:\n"
printf "    Higher stddev = more variation in mean compressibility between dumps.\n"
printf "    This config maximises diverse NN training episodes across timesteps.\n"
printf "    Use it as NYX_NCELL / NYX_MAX_STEP / NYX_PLOT_INT for the paper run.\n"
printf "\n"
printf "  CSV logs : %s/logs_<config>/benchmark_nyx_timestep_chunks.csv\n" "$RESULTS_DIR"
printf "  Full logs: %s/logs_<config>/nyx_run.log\n" "$RESULTS_DIR"
hr
