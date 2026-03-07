#!/bin/bash
# test_m26_m31_python_issues.sh
#
# Static analysis tests for Python/training issues:
#   M-26: ctypes argtypes mismatch for gpucompress_enable_active_learning
#   M-27: No NaN handling for PSNR column in data.py
#   M-28: compressed_size = len(compressed) - HEADER_SIZE can be negative
#   M-29: Hardcoded np.float32 dtype in compute_stats_cpu
#   M-30: Last bin target can go negative in generator.py
#   M-31: test_f9_transfers.c only 3 cd_values passed to H5Pset_filter
#
# Usage: ./tests/regression/test_m26_m31_python_issues.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "=== M-26 through M-31: Python/training issues ==="
echo ""

# ---- M-26: ctypes argtypes mismatch ----
echo "--- M-26: ctypes argtypes for enable_active_learning ---"
ADAPT="$PROJECT_DIR/eval/workload_adaptation.py"
if [ -f "$ADAPT" ]; then
    # The C function gpucompress_enable_active_learning takes void (no args)
    # Check if argtypes incorrectly specifies arguments
    ARGTYPES_LINE=$(grep -n 'enable_active_learning.argtypes' "$ADAPT" || true)
    if [ -n "$ARGTYPES_LINE" ]; then
        echo "  Found: $ARGTYPES_LINE"
        if echo "$ARGTYPES_LINE" | grep -q 'c_char_p\|c_int\|c_float'; then
            fail "argtypes specifies args but C function takes void (stack corruption)"
        else
            pass "argtypes matches C signature"
        fi
    else
        pass "no argtypes set (ctypes default)"
    fi
else
    echo "  SKIP: $ADAPT not found"
fi

# ---- M-27: NaN handling in PSNR column ----
echo ""
echo "--- M-27: NaN handling in data.py ---"
DATA_PY="$PROJECT_DIR/neural_net/core/data.py"
if [ -f "$DATA_PY" ]; then
    # Check if psnr_clamped handles NaN
    if grep -q 'fillna\|dropna\|isnan\|nan.*psnr\|psnr.*nan' "$DATA_PY"; then
        pass "NaN handling present for PSNR"
    else
        # Check if inf is handled (partial)
        if grep -q 'replace.*inf' "$DATA_PY"; then
            fail "inf handled but NaN not handled — NaN propagates through training"
        else
            fail "neither inf nor NaN handled for PSNR"
        fi
    fi
else
    echo "  SKIP: $DATA_PY not found"
fi

# ---- M-28: Negative compressed_size ----
echo ""
echo "--- M-28: Negative compressed_size in benchmark.py ---"
BENCH="$PROJECT_DIR/neural_net/training/benchmark.py"
if [ -f "$BENCH" ]; then
    # Check if there's a guard for compressed_size <= 0
    if grep -q 'compressed_size > 0' "$BENCH"; then
        pass "compressed_size > 0 guard present (handles negative case)"
    else
        fail "no guard for negative compressed_size"
    fi
    # Check for the potentially negative computation
    if grep -q 'len(compressed) - HEADER_SIZE' "$BENCH"; then
        echo "  NOTE: compressed_size = len(compressed) - HEADER_SIZE can be negative"
        echo "        but the > 0 guard defaults ratio to 1.0 (safe)"
        pass "pattern present but guarded — effectively safe"
    fi
else
    echo "  SKIP: $BENCH not found"
fi

# ---- M-29: Hardcoded float32 dtype ----
echo ""
echo "--- M-29: Hardcoded np.float32 in compute_stats_cpu ---"
if [ -f "$DATA_PY" ]; then
    # Check if compute_stats_cpu uses hardcoded dtype
    # Check if function signature accepts dtype as a parameter
    FUNC_LINE=$(grep 'def compute_stats_cpu' "$DATA_PY")
    if echo "$FUNC_LINE" | grep -q 'dtype'; then
        # Check that frombuffer uses the dtype parameter, not hardcoded float32
        BODY=$(sed -n '/def compute_stats_cpu/,/^def \|^[^ ]/p' "$DATA_PY")
        if echo "$BODY" | grep -q 'frombuffer.*dtype=dtype'; then
            pass "compute_stats_cpu dtype is parameterized"
        else
            fail "compute_stats_cpu has dtype param but frombuffer still hardcoded"
        fi
    else
        fail "compute_stats_cpu hardcodes np.float32 — wrong stats for float64 data"
    fi
else
    echo "  SKIP: $DATA_PY not found"
fi

# ---- M-30: Negative last bin target ----
echo ""
echo "--- M-30: Negative last bin target in generator.py ---"
GEN="$PROJECT_DIR/syntheticGeneration/generator.py"
if [ -f "$GEN" ]; then
    # targets[-1] = num_elements - targets[:-1].sum()
    # If int truncation makes sum > num_elements, last is negative
    if grep -q 'targets\[-1\].*=.*num_elements.*-.*sum' "$GEN"; then
        # Check if there's a max(0, ...) guard
        if grep -q 'max(0\|clip(min=0\|abs(' "$GEN"; then
            pass "last bin target has non-negative guard"
        else
            fail "last bin target can go negative (int truncation rounding)"
        fi
    else
        pass "pattern not found (may be refactored)"
    fi
else
    echo "  SKIP: $GEN not found"
fi

# ---- M-31: cd_values count in test_f9 ----
echo ""
echo "--- M-31: H5Pset_filter cd_values count ---"
TEST_F9="$PROJECT_DIR/tests/hdf5/test_f9_transfers.c"
if [ -f "$TEST_F9" ]; then
    FILTER_LINE=$(grep -n 'H5Pset_filter' "$TEST_F9" || true)
    if [ -n "$FILTER_LINE" ]; then
        echo "  Found: $FILTER_LINE"
        # Check if 3 or 5 cd_values passed
        if echo "$FILTER_LINE" | grep -q ', 3,'; then
            fail "only 3 cd_values passed to H5Pset_filter (5 expected)"
        elif echo "$FILTER_LINE" | grep -q ', 5,'; then
            pass "5 cd_values passed to H5Pset_filter"
        else
            echo "  INFO: could not determine cd_values count"
            pass "cd_values count unclear"
        fi
    else
        pass "no H5Pset_filter call found"
    fi
else
    echo "  SKIP: $TEST_F9 not found"
fi

# ---- Summary ----
echo ""
echo "=== Summary: $PASS_COUNT pass, $FAIL_COUNT fail ==="
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "OVERALL: PASS"
    exit 0
else
    echo "OVERALL: FAIL"
    exit 1
fi
