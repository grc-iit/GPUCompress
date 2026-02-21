#!/usr/bin/env bash
#
# GPUCompress Simulation Evaluation Pipeline
#
# End-to-end: download data → build → phase 1 eval → retrain → phase 2 eval → compare
#
# Usage:
#   bash eval/run_eval_pipeline.sh
#   bash eval/run_eval_pipeline.sh --timesteps 3   # quick test with fewer files
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
DATA_DIR="$SCRIPT_DIR/data"
BUILD_DIR="$PROJECT_DIR/build"
WEIGHTS="$PROJECT_DIR/model.nnwt"
TIMESTEPS=15
ERROR_BOUND=0.0
THRESHOLD=0.20

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timesteps) TIMESTEPS="$2"; shift 2 ;;
        --weights)   WEIGHTS="$2"; shift 2 ;;
        --error-bound) ERROR_BOUND="$2"; shift 2 ;;
        --threshold) THRESHOLD="$2"; shift 2 ;;
        --data-dir)  DATA_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "GPUCompress Simulation Evaluation Pipeline"
echo "========================================"
echo "  Project:     $PROJECT_DIR"
echo "  Data dir:    $DATA_DIR"
echo "  Weights:     $WEIGHTS"
echo "  Timesteps:   $TIMESTEPS"
echo "  Error bound: $ERROR_BOUND"
echo "  Threshold:   $THRESHOLD"
echo ""

# -------------------------------------------------------
# Step 1: Download simulation data
# -------------------------------------------------------
echo "=== Step 1: Download simulation data ==="
if [ -d "$DATA_DIR" ] && [ "$(ls -1 "$DATA_DIR"/*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  Data directory already has .bin files, skipping download"
    echo "  (delete $DATA_DIR to re-download)"
else
    python3 "$SCRIPT_DIR/download_well_data.py" \
        --output-dir "$DATA_DIR" \
        --timesteps "$TIMESTEPS" \
        --scenario 0
fi
echo ""

FILE_COUNT=$(ls -1 "$DATA_DIR"/*.bin 2>/dev/null | wc -l)
echo "  Files available: $FILE_COUNT"
echo ""

# -------------------------------------------------------
# Step 2: Build eval_simulation
# -------------------------------------------------------
echo "=== Step 2: Build eval_simulation ==="
cmake --build "$BUILD_DIR" --target eval_simulation 2>&1
echo "  Build complete"
echo ""

EVAL_BIN="$BUILD_DIR/eval_simulation"
if [ ! -x "$EVAL_BIN" ]; then
    echo "Error: $EVAL_BIN not found or not executable"
    exit 1
fi

# -------------------------------------------------------
# Step 3: Phase 1 — Evaluate with original model
# -------------------------------------------------------
echo "=== Step 3: Phase 1 — Evaluate with original model ==="
RESULTS_P1="$SCRIPT_DIR/results_phase1.csv"
EXP_P1="$SCRIPT_DIR/experience_phase1.csv"

# Remove stale experience file so we start fresh
rm -f "$EXP_P1"

"$EVAL_BIN" \
    --data-dir "$DATA_DIR" \
    --weights "$WEIGHTS" \
    --experience "$EXP_P1" \
    --output "$RESULTS_P1" \
    --error-bound "$ERROR_BOUND" \
    --threshold "$THRESHOLD"
echo ""

# -------------------------------------------------------
# Step 4: Retrain model with collected experience
# -------------------------------------------------------
echo "=== Step 4: Retrain model with collected experience ==="
RETRAINED_WEIGHTS="$SCRIPT_DIR/model_retrained.nnwt"

if [ ! -f "$EXP_P1" ]; then
    echo "  Warning: no experience file produced, skipping retrain"
    echo "  Phase 2 will use the same model as Phase 1"
    RETRAINED_WEIGHTS="$WEIGHTS"
else
    EXP_LINES=$(wc -l < "$EXP_P1")
    echo "  Experience samples: $((EXP_LINES - 1)) (excluding header)"

    if python3 "$PROJECT_DIR/neural_net/retrain.py" \
        --experience "$EXP_P1" \
        --output "$RETRAINED_WEIGHTS" 2>&1; then
        echo "  Retrained model: $RETRAINED_WEIGHTS"
    else
        echo "  Warning: retrain failed, Phase 2 will use original model"
        RETRAINED_WEIGHTS="$WEIGHTS"
    fi
fi
echo ""

# -------------------------------------------------------
# Step 5: Phase 2 — Evaluate with retrained model
# -------------------------------------------------------
echo "=== Step 5: Phase 2 — Evaluate with retrained model ==="
RESULTS_P2="$SCRIPT_DIR/results_phase2.csv"
EXP_P2="$SCRIPT_DIR/experience_phase2.csv"

rm -f "$EXP_P2"

"$EVAL_BIN" \
    --data-dir "$DATA_DIR" \
    --weights "$RETRAINED_WEIGHTS" \
    --experience "$EXP_P2" \
    --output "$RESULTS_P2" \
    --error-bound "$ERROR_BOUND" \
    --threshold "$THRESHOLD"
echo ""

# -------------------------------------------------------
# Step 6: Compare Phase 1 vs Phase 2
# -------------------------------------------------------
echo "=== Step 6: Comparison ==="
python3 -c "
import csv, sys

def load_results(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

p1 = load_results('$RESULTS_P1')
p2 = load_results('$RESULTS_P2')

def summarize(rows, label):
    ratios = [float(r['compression_ratio']) for r in rows if float(r['compression_ratio']) > 0]
    exp_deltas = [int(r['experience_delta']) for r in rows]
    explorations = sum(1 for d in exp_deltas if d > 1)
    algos = {}
    for r in rows:
        a = r['algorithm']
        algos[a] = algos.get(a, 0) + 1

    print(f'  {label}:')
    print(f'    Files:        {len(rows)}')
    print(f'    Mean ratio:   {sum(ratios)/len(ratios):.4f}' if ratios else '    Mean ratio:   N/A')
    print(f'    Explorations: {explorations} ({100*explorations/len(rows):.1f}%)' if rows else '')
    print(f'    Algorithms:   {dict(sorted(algos.items(), key=lambda x: -x[1]))}')
    return sum(ratios)/len(ratios) if ratios else 0, explorations

print()
r1, e1 = summarize(p1, 'Phase 1 (original model)')
print()
r2, e2 = summarize(p2, 'Phase 2 (retrained model)')
print()

if r1 > 0 and r2 > 0:
    ratio_delta = r2 - r1
    print(f'  Ratio improvement:      {ratio_delta:+.4f} ({100*ratio_delta/r1:+.1f}%)')
if len(p1) > 0 and len(p2) > 0:
    print(f'  Exploration reduction:  {e1} -> {e2} ({e2 - e1:+d})')
print()
"

echo "========================================"
echo "Pipeline complete"
echo "  Phase 1 results: $RESULTS_P1"
echo "  Phase 2 results: $RESULTS_P2"
echo "  Experience P1:   $EXP_P1"
echo "  Experience P2:   $EXP_P2"
echo "========================================"
