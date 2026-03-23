#!/bin/bash
# ============================================================
# Generate all benchmark figures.
#
# Usage:
#   # After running eval scripts (standard paths):
#   bash benchmarks/plots/generate_all.sh
#
#   # With custom data directories:
#   GS_DIR=benchmarks/grayscott/results/verify_final \
#   SDR_DIR=benchmarks/sdrbench/results/verify_all \
#   bash benchmarks/plots/generate_all.sh
# ============================================================
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$GPU_DIR"

echo "============================================================"
echo "  GPUCompress Benchmark Figure Generation"
echo "============================================================"
echo ""

# ── Per-dataset figures ──
echo "── Per-Dataset Figures ──"
python3 "$SCRIPT_DIR/generate_dataset_figures.py" --all
echo ""

# ── Cross-dataset figures ──
echo "── Cross-Dataset Figures ──"
python3 "$SCRIPT_DIR/generate_cross_dataset_figures.py"
echo ""

# ── Summary ──
echo "============================================================"
echo "  Output directories:"
echo "    Per-dataset:   benchmarks/results/per_dataset/"
echo "    Cross-dataset: benchmarks/results/cross_dataset/"
echo "    Paper figures: benchmarks/results/paper_figures/"
echo ""

N_PER=$(find benchmarks/results/per_dataset -name "*.png" 2>/dev/null | wc -l)
N_CROSS=$(find benchmarks/results/cross_dataset -name "*.png" 2>/dev/null | wc -l)
echo "  Per-dataset PNGs:   $N_PER"
echo "  Cross-dataset PNGs: $N_CROSS"
echo "  Total:              $((N_PER + N_CROSS))"
echo "============================================================"
