#!/bin/bash
# test_m19_m25_config_issues.sh
#
# Static analysis tests for build/config/script issues:
#   M-19: hid_t forward-declared as int (ABI mismatch with HDF5 1.14+)
#   M-20: -use_fast_math in release undermines quantization bounds
#   M-21: Missing -Wall -Wextra
#   M-22: CLI tools unconditionally link cufile
#   M-23: run_tests.sh references non-existent benchmark targets
#   M-24: run_eval_pipeline.sh passes --experience flag not accepted
#   M-25: run_eval_pipeline.sh reads non-existent CSV column
#
# Usage: ./tests/regression/test_m19_m25_config_issues.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "=== M-19 through M-25: Build/Config/Script issues ==="
echo ""

# ---- M-19: hid_t typedef ----
echo "--- M-19: hid_t forward declaration ---"
HDF5_H="$PROJECT_DIR/include/gpucompress_hdf5.h"
if [ -f "$HDF5_H" ]; then
    if grep -q 'typedef int hid_t' "$HDF5_H"; then
        fail "hid_t forward-declared as 'int' (ABI mismatch with HDF5 1.14+ int64_t)"
    else
        pass "hid_t not typedef'd as int"
    fi
else
    echo "  SKIP: $HDF5_H not found"
fi

# ---- M-20: -use_fast_math ----
echo ""
echo "--- M-20: -use_fast_math in release builds ---"
CMAKE_FILE="$PROJECT_DIR/CMakeLists.txt"
if [ -f "$CMAKE_FILE" ]; then
    if grep -q 'use_fast_math' "$CMAKE_FILE"; then
        fail "-use_fast_math found in CMakeLists.txt (undermines quantization precision)"
    else
        pass "no -use_fast_math in CMakeLists.txt"
    fi
else
    echo "  SKIP: CMakeLists.txt not found"
fi

# ---- M-21: Missing -Wall -Wextra ----
echo ""
echo "--- M-21: Compiler warning flags ---"
if [ -f "$CMAKE_FILE" ]; then
    if grep -q -- '-Wall' "$CMAKE_FILE" && grep -q -- '-Wextra' "$CMAKE_FILE"; then
        pass "-Wall and -Wextra present"
    else
        fail "Missing -Wall and/or -Wextra compiler warning flags"
    fi
fi

# ---- M-22: CLI tools unconditionally link cufile ----
echo ""
echo "--- M-22: cufile link dependency ---"
CORE_CMAKE="$PROJECT_DIR/cmake/CoreLibrary.cmake"
if [ -f "$CORE_CMAKE" ]; then
    CUFILE_REFS=$(grep -c 'cufile' "$CORE_CMAKE" 2>/dev/null | tail -1 || echo 0)
    CONDITIONAL=$(grep -c 'if.*cufile\|find.*cufile\|CUFILE_FOUND' "$CORE_CMAKE" 2>/dev/null | tail -1 || echo 0)

    if [ "$CUFILE_REFS" -gt 0 ] && [ "$CONDITIONAL" -eq 0 ]; then
        fail "cufile linked unconditionally ($CUFILE_REFS refs, no conditionals)"
    elif [ "$CUFILE_REFS" -gt 0 ]; then
        pass "cufile link is conditional"
    else
        pass "no cufile references"
    fi
else
    echo "  SKIP: CoreLibrary.cmake not found"
fi

# ---- M-23: Non-existent benchmark targets in run_tests.sh ----
echo ""
echo "--- M-23: Phantom benchmark targets ---"
RUN_TESTS="$PROJECT_DIR/scripts/run_tests.sh"
if [ -f "$RUN_TESTS" ]; then
    PHANTOM_TARGETS=$(grep -oP 'benchmark_\w+' "$RUN_TESTS" 2>/dev/null | sort -u)
    if [ -n "$PHANTOM_TARGETS" ]; then
        # Check if any of these targets exist in CMake (add_executable, add_vol_demo, etc.)
        FOUND=0
        for target in $PHANTOM_TARGETS; do
            if grep -rqE "add_executable\(${target}|add_vol_demo\(${target}" "$PROJECT_DIR/cmake/" "$PROJECT_DIR/CMakeLists.txt" 2>/dev/null; then
                FOUND=$((FOUND + 1))
            fi
        done
        TOTAL=$(echo "$PHANTOM_TARGETS" | wc -l)
        MISSING=$((TOTAL - FOUND))
        if [ "$MISSING" -gt 0 ]; then
            echo "  Phantom targets: $PHANTOM_TARGETS"
            fail "$MISSING benchmark targets referenced but not defined in CMake"
        else
            pass "all benchmark targets exist"
        fi
    else
        pass "no benchmark targets referenced"
    fi
else
    echo "  SKIP: run_tests.sh not found"
fi

# ---- M-24: --experience flag ----
echo ""
echo "--- M-24: --experience flag in eval pipeline ---"
EVAL_PIPELINE="$PROJECT_DIR/eval/run_eval_pipeline.sh"
if [ -f "$EVAL_PIPELINE" ]; then
    if grep -q '\-\-experience' "$EVAL_PIPELINE"; then
        # Check if eval_simulation accepts it
        EVAL_SIM="$PROJECT_DIR/eval/eval_simulation.cpp"
        if [ -f "$EVAL_SIM" ]; then
            if grep -q '"experience"\|--experience\|experience.*optarg\|experience.*getopt' "$EVAL_SIM"; then
                pass "--experience flag accepted by eval_simulation"
            else
                fail "--experience flag in pipeline but NOT accepted by eval_simulation"
            fi
        else
            echo "  SKIP: eval_simulation.cpp not found"
        fi
    else
        pass "no --experience flag in pipeline"
    fi
else
    echo "  SKIP: eval pipeline not found"
fi

# ---- M-25: experience_delta CSV column ----
echo ""
echo "--- M-25: experience_delta CSV column ---"
EVAL_SIM="$PROJECT_DIR/eval/eval_simulation.cpp"
if [ -f "$EVAL_PIPELINE" ] && [ -f "$EVAL_SIM" ]; then
    PIPELINE_READS=$(grep -c 'experience_delta' "$EVAL_PIPELINE" 2>/dev/null || true)
    CSV_OUTPUTS=$(grep -c 'experience_delta' "$EVAL_SIM" 2>/dev/null || true)
    if [ "${PIPELINE_READS:-0}" -gt 0 ] && [ "${CSV_OUTPUTS:-0}" -gt 0 ]; then
        pass "experience_delta in both CSV output and pipeline reader"
    elif [ "${PIPELINE_READS:-0}" -gt 0 ] && [ "${CSV_OUTPUTS:-0}" -eq 0 ]; then
        fail "pipeline reads 'experience_delta' but eval_simulation doesn't output it"
    else
        pass "no experience_delta mismatch"
    fi
else
    echo "  SKIP: eval pipeline or eval_simulation not found"
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
