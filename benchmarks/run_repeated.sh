#!/bin/bash
# ============================================================
# Run a benchmark N times independently for experimental error bars.
#
# Collects per-run aggregate CSVs, then computes mean ± std across runs.
# This gives true experimental uncertainty (GPU clock, OS scheduling,
# tmpfs variability) — required for SC publication.
#
# Usage:
#   bash benchmarks/run_repeated.sh [--runs N] [-- <benchmark args>]
#
# Examples:
#   # VPIC 5 runs
#   bash benchmarks/run_repeated.sh --runs 5 -- \
#     BENCHMARKS=vpic DATA_MB=256 CHUNK_MB=4 TIMESTEPS=10 VERIFY=0
#
#   # AI training 3 runs
#   bash benchmarks/run_repeated.sh --runs 3 -- \
#     BENCHMARKS=ai_training AI_MODEL=vit_b_16 CHUNK_MB=4 VERIFY=0
#
#   # Gray-Scott 5 runs
#   bash benchmarks/run_repeated.sh --runs 5 -- \
#     BENCHMARKS=grayscott DATA_MB=64 CHUNK_MB=4 TIMESTEPS=10 VERIFY=0
#
# Output:
#   benchmarks/repeated_results/<benchmark>_<timestamp>/
#     run_1/  run_2/  ...  run_N/    ← per-run results (copied from benchmark output)
#     aggregate_mean_std.csv          ← mean ± std across N runs
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
N_RUNS=5

# Parse arguments
BENCH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) N_RUNS="$2"; shift 2 ;;
        --) shift; BENCH_ARGS=("$@"); break ;;
        *) BENCH_ARGS+=("$1"); shift ;;
    esac
done

if [ ${#BENCH_ARGS[@]} -eq 0 ]; then
    echo "Usage: bash benchmarks/run_repeated.sh --runs N -- BENCHMARKS=vpic DATA_MB=256 ..."
    exit 1
fi

# Extract benchmark name for output directory
BENCH_NAME=""
for arg in "${BENCH_ARGS[@]}"; do
    case "$arg" in
        BENCHMARKS=*) BENCH_NAME="${arg#BENCHMARKS=}" ;;
    esac
done
BENCH_NAME="${BENCH_NAME:-unknown}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$SCRIPT_DIR/repeated_results/${BENCH_NAME}_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  Repeated Benchmark Runs for Error Bars"
echo "============================================================"
echo "  Benchmark : ${BENCH_NAME}"
echo "  Runs      : ${N_RUNS}"
echo "  Args      : ${BENCH_ARGS[*]}"
echo "  Output    : ${OUT_DIR}"
echo "============================================================"
echo ""

# ── Run N times ──
for run in $(seq 1 "$N_RUNS"); do
    echo ">>> Run $run/$N_RUNS ..."
    RUN_DIR="$OUT_DIR/run_${run}"
    mkdir -p "$RUN_DIR"

    # Create a timestamp marker before the run
    MARKER=$(mktemp)

    # Run the benchmark with a per-run suffix so each run gets its own
    # results directory (e.g., eval_NX159_chunk4mb_ts10_run1/).
    # VPIC uses VPIC_EVAL_SUFFIX, GS/SDR/AI use the eval dir directly.
    RUN_START=$(date +%s)
    # Append _runN to any existing VPIC_EVAL_SUFFIX (e.g., _lossless_run1)
    _EXISTING_SUFFIX=""
    for arg in "${BENCH_ARGS[@]}"; do
        case "$arg" in VPIC_EVAL_SUFFIX=*) _EXISTING_SUFFIX="${arg#VPIC_EVAL_SUFFIX=}" ;; esac
    done
    env "${BENCH_ARGS[@]}" VPIC_EVAL_SUFFIX="${_EXISTING_SUFFIX}_run${run}" \
        bash "$SCRIPT_DIR/benchmark.sh" \
        > "$RUN_DIR/benchmark.log" 2>&1 || true
    RUN_END=$(date +%s)
    ELAPSED=$((RUN_END - RUN_START))

    # Copy result CSVs that were modified AFTER the marker (this run's output only)
    for results_dir in \
        "$SCRIPT_DIR/grayscott/results" \
        "$SCRIPT_DIR/vpic-kokkos/results" \
        "$SCRIPT_DIR/sdrbench/results" \
        "$SCRIPT_DIR/ai_training/results"; do
        if [ -d "$results_dir" ]; then
            find "$results_dir" -name "benchmark_*.csv" \
                ! -name "*timesteps*" ! -name "*chunks*" ! -name "*ranking*" \
                -newer "$MARKER" \
                -exec cp {} "$RUN_DIR/" \; 2>/dev/null || true
        fi
    done
    rm -f "$MARKER"

    N_CSVS=$(ls "$RUN_DIR"/*.csv 2>/dev/null | wc -l)
    echo "    Done (${ELAPSED}s, ${N_CSVS} CSVs collected)"
    echo ""
done

# ── Aggregate: compute mean ± std across runs ──
echo "Computing mean ± std across $N_RUNS runs..."

python3 -c "
import csv, os, sys, math, glob

out_dir = '$OUT_DIR'
n_runs = $N_RUNS

# Find aggregate CSVs (the short ones with source,phase,n_runs,write_ms,...)
# Use run_1 as the template
run1_csvs = glob.glob(os.path.join(out_dir, 'run_1', 'benchmark_*.csv'))
# Filter to aggregate CSVs (not timesteps, not chunks, not ranking)
agg_csvs = [c for c in run1_csvs
            if '_timesteps' not in c and '_chunks' not in c
            and '_ranking' not in c and '_timestep_' not in c]

if not agg_csvs:
    print('No aggregate CSVs found in run_1/')
    sys.exit(0)

for agg_template in agg_csvs:
    basename = os.path.basename(agg_template)
    print(f'  Processing {basename}...')

    # Collect rows from all runs
    all_runs = []
    for run_i in range(1, n_runs + 1):
        csv_path = os.path.join(out_dir, f'run_{run_i}', basename)
        if not os.path.exists(csv_path):
            print(f'    WARNING: {csv_path} missing, skipping')
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
            all_runs.append(rows)

    if len(all_runs) < 2:
        print(f'    Need at least 2 runs, got {len(all_runs)}')
        continue

    # Group by phase across runs
    # Assume all runs have same phases in same order
    n_phases = len(all_runs[0])
    header = list(all_runs[0][0].keys())

    # Numeric columns to average
    numeric_cols = ['write_ms', 'read_ms', 'ratio', 'write_mibps', 'read_mibps',
                    'comp_ms', 'decomp_ms', 'nn_ms', 'stats_ms', 'preproc_ms',
                    'explore_ms', 'sgd_ms', 'comp_gbps', 'decomp_gbps',
                    'mape_ratio_pct', 'mape_comp_pct', 'mape_decomp_pct',
                    'mae_ratio', 'mae_comp_ms', 'mae_decomp_ms',
                    'vol_stage1_ms', 'vol_drain_ms', 'vol_io_drain_ms',
                    'vol_s2_busy_ms', 'vol_s3_busy_ms', 'file_mib']

    out_path = os.path.join(out_dir, f'mean_std_{basename}')
    with open(out_path, 'w', newline='') as f:
        # Write header with _mean and _std suffixes for key metrics
        out_header = []
        for h in header:
            out_header.append(h)
        out_header.append('write_ms_exp_std')
        out_header.append('read_ms_exp_std')
        out_header.append('n_exp_runs')
        writer = csv.DictWriter(f, fieldnames=out_header)
        writer.writeheader()

        for pi in range(n_phases):
            # Collect this phase's rows across runs
            phase_rows = []
            for run_rows in all_runs:
                if pi < len(run_rows):
                    phase_rows.append(run_rows[pi])

            if not phase_rows:
                continue

            # Compute mean for numeric columns
            out_row = dict(phase_rows[0])  # start with run_1 values for non-numeric
            for col in numeric_cols:
                if col in phase_rows[0]:
                    vals = []
                    for r in phase_rows:
                        try:
                            vals.append(float(r.get(col, 0)))
                        except (ValueError, TypeError):
                            pass
                    if vals:
                        out_row[col] = f'{sum(vals)/len(vals):.4f}'

            # Compute experimental std for write_ms and read_ms.
            # Write into BOTH the _exp_std column AND the standard write_ms_std/read_ms_std
            # columns so the plot script picks them up as error bars.
            for metric, std_col, main_std_col in [
                    ('write_ms', 'write_ms_exp_std', 'write_ms_std'),
                    ('read_ms', 'read_ms_exp_std', 'read_ms_std')]:
                vals = []
                for r in phase_rows:
                    try:
                        vals.append(float(r.get(metric, 0)))
                    except (ValueError, TypeError):
                        pass
                if len(vals) >= 2:
                    mean = sum(vals) / len(vals)
                    var = sum((v - mean)**2 for v in vals) / (len(vals) - 1)
                    std_val = f'{math.sqrt(var):.4f}'
                    out_row[std_col] = std_val
                    out_row[main_std_col] = std_val  # for plot script
                else:
                    out_row[std_col] = '0.0000'

            out_row['n_exp_runs'] = str(len(phase_rows))

            # Update the timestep std with run_1's value (data variance)
            # The new columns are experimental std
            writer.writerow(out_row)

    n_phases_out = n_phases
    print(f'    → {out_path} ({n_phases_out} phases, {len(all_runs)} runs)')

print()
print(f'Results in: {out_dir}/')
print(f'  run_1/ ... run_{n_runs}/     ← per-run CSVs')
print(f'  mean_std_*.csv               ← mean ± experimental std across {n_runs} runs')
" 2>&1

# ── Generate summary plot from averaged CSV with experimental error bars ──
echo "Generating summary plot from averaged results..."
for mean_csv in "$OUT_DIR"/mean_std_benchmark_*.csv; do
    [ -f "$mean_csv" ] || continue
    PLOT_DIR="$OUT_DIR/plots"
    mkdir -p "$PLOT_DIR"

    python3 -c "
import sys, os
sys.path.insert(0, os.path.join('$SCRIPT_DIR', 'plots'))
sys.path.insert(0, '$SCRIPT_DIR')
import visualize as viz

csv_path = '$mean_csv'
plot_dir = '$PLOT_DIR'
rows = viz.parse_csv(csv_path)
if rows:
    source = rows[0].get('source', 'benchmark')
    # Build meta text
    n_runs = rows[0].get('n_exp_runs', '?')
    meta = f'Averaged across {n_runs} independent runs | Policy: {rows[0].get(\"policy\", \"balanced\")}'
    out = os.path.join(plot_dir, '1_summary_averaged.png')
    viz.make_summary_figure(source.replace('_', ' ').title(), rows, out, meta)
    print(f'  Saved: {out}')
else:
    print(f'  No data in {csv_path}')
" 2>&1
done

echo ""
echo "============================================================"
echo "  Repeated runs complete: $OUT_DIR"
echo "============================================================"
