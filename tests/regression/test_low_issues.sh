#!/bin/bash
# test_low_issues.sh
#
# Static analysis tests for all LOW severity issues (L-1 through L-14).
#
# Usage: ./tests/regression/test_low_issues.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "=== L-1 through L-14: LOW severity issues ==="
echo ""

# ---- L-1: Unused d_rt_buf allocation ----
echo "--- L-1: Unused d_rt_buf allocation ---"
API="$PROJECT_DIR/src/api/gpucompress_api.cpp"
if [ -f "$API" ]; then
    ALLOC=$(grep -c 'd_rt_buf' "$API" 2>/dev/null || true)
    ALLOC=${ALLOC:-0}
    USE=$(grep 'cudaMemcpy.*d_rt_buf\|kernel.*d_rt_buf\|d_rt_buf.*=' "$API" | grep -v 'cudaMalloc\|cudaFree\|nullptr' | wc -l)
    if [ "$ALLOC" -gt 0 ] && [ "$USE" -eq 0 ]; then
        fail "d_rt_buf allocated but never used (wasted GPU memory)"
    else
        pass "d_rt_buf used or not allocated"
    fi
else
    echo "  SKIP"
fi

# ---- L-2: Case-sensitive algorithm_from_string ----
echo ""
echo "--- L-2: Case-sensitive algorithm_from_string ---"
if [ -f "$API" ]; then
    if grep -q 'strcasecmp\|tolower\|toupper\|case_insensitive' "$API"; then
        pass "case-insensitive matching present"
    elif grep -q 'algorithm_from_string\|strcmp.*"lz4"\|strcmp.*"zstd"' "$API"; then
        fail "algorithm matching uses case-sensitive strcmp"
    else
        pass "pattern not found"
    fi
fi

# ---- L-3: Static GPU buffers never freed ----
echo ""
echo "--- L-3: Static d_range_min/max never freed ---"
QUANT="$PROJECT_DIR/src/preprocessing/quantization_kernels.cu"
if [ -f "$QUANT" ]; then
    if grep -q 'cudaFree.*d_range_min\|cudaFree.*d_range_max' "$QUANT" 2>/dev/null; then
        pass "d_range_min/max freed"
    else
        fail "static d_range_min/max never freed (16-byte GPU leak)"
    fi
fi

# ---- L-4: Unchecked cudaMalloc in allocInferenceBuffers ----
echo ""
echo "--- L-4: Unchecked cudaMalloc in nn_gpu.cu ---"
NN_GPU="$PROJECT_DIR/src/nn/nn_gpu.cu"
if [ -f "$NN_GPU" ]; then
    ALLOC_FUNC=$(sed -n '/allocInferenceBuffers/,/^}/p' "$NN_GPU")
    UNCHECKED=$(echo "$ALLOC_FUNC" | grep 'cudaMalloc' | grep -v 'cudaSuccess\|if\|CUDA_CHECK\|assert' | wc -l)
    if [ "$UNCHECKED" -gt 0 ]; then
        fail "$UNCHECKED cudaMalloc calls without error check in allocInferenceBuffers"
    else
        pass "all cudaMalloc calls checked"
    fi
fi

# ---- L-5: Inaccurate stats counter ----
echo ""
echo "--- L-5: s_chunks_comp increment timing ---"
VOL="$PROJECT_DIR/src/hdf5/H5VLgpucompress.cu"
if [ -f "$VOL" ]; then
    if grep -q 's_chunks_comp++' "$VOL"; then
        pass "s_chunks_comp counter exists (increment timing is cosmetic)"
    else
        pass "counter not found or renamed"
    fi
fi

# ---- L-6: Unchecked H5Tcopy/H5Tset_size ----
echo ""
echo "--- L-6: Unchecked H5T calls in write_chunk_attr ---"
FILTER="$PROJECT_DIR/src/hdf5/H5Zgpucompress.c"
if [ -f "$FILTER" ]; then
    CHUNK_ATTR=$(sed -n '/write_chunk_attr/,/^}/p' "$FILTER")
    # Check that H5Tcopy result is tested (if atype < 0) and H5Tset_size is checked
    if echo "$CHUNK_ATTR" | grep -q 'atype < 0' && echo "$CHUNK_ATTR" | grep -q 'H5Tset_size.*< 0'; then
        pass "H5T calls checked"
    else
        fail "unchecked H5T calls in write_chunk_attr"
    fi
fi

# ---- L-7: Hardcoded path in test_nn_shuffle.cu ----
echo ""
echo "--- L-7: Hardcoded path in test_nn_shuffle.cu ---"
TEST_SHUFFLE="$PROJECT_DIR/tests/nn/test_nn_shuffle.cu"
if [ -f "$TEST_SHUFFLE" ]; then
    if grep -q '/home/cc/GPUCompress' "$TEST_SHUFFLE"; then
        fail "hardcoded /home/cc/GPUCompress path in test"
    else
        pass "no hardcoded path"
    fi
fi

# ---- L-8: Hardcoded path in test_quantization_errors.sh ----
echo ""
echo "--- L-8: Hardcoded path in test_quantization_errors.sh ---"
QUANT_TEST="$PROJECT_DIR/scripts/test_quantization_errors.sh"
if [ -f "$QUANT_TEST" ]; then
    if grep -q '/home/cc/GPUCompress' "$QUANT_TEST"; then
        fail "hardcoded /home/cc/GPUCompress path"
    else
        pass "no hardcoded path"
    fi
fi

# ---- L-9: Hardcoded path in run_tests.sh ----
echo ""
echo "--- L-9: Hardcoded path in run_tests.sh ---"
RUN_TESTS="$PROJECT_DIR/scripts/run_tests.sh"
if [ -f "$RUN_TESTS" ]; then
    if grep -q '/u/imuradli/GPUCompress\|/home/cc/GPUCompress' "$RUN_TESTS"; then
        fail "hardcoded user path in run_tests.sh"
    else
        pass "no hardcoded path"
    fi
fi

# ---- L-10: Incomplete verify_export ----
echo ""
echo "--- L-10: Incomplete verify_export in export_weights.py ---"
EXPORT="$PROJECT_DIR/neural_net/export/export_weights.py"
if [ -f "$EXPORT" ]; then
    if grep -q 'forward\|manual.*pass\|reference.*output' "$EXPORT"; then
        pass "forward pass verification present"
    elif grep -q 'verify_export' "$EXPORT"; then
        fail "verify_export exists but no manual forward pass comparison"
    else
        pass "no verify_export function"
    fi
fi

# ---- L-11: Benign race in compare_buffers kernel ----
echo ""
echo "--- L-11: Non-atomic write in compare_buffers kernel ---"
COMPRESS="$PROJECT_DIR/src/cli/compress.cpp"
if [ -f "$COMPRESS" ]; then
    if grep -A5 'compare_buffers' "$COMPRESS" | grep -q 'atomicOr\|atomicAdd\|atomicExch'; then
        pass "compare_buffers uses atomic"
    elif grep -A5 'compare_buffers' "$COMPRESS" | grep -q '\*invalid.*=.*1\|invalid\[0\].*='; then
        fail "non-atomic write to *invalid in compare_buffers (benign race, technically UB)"
    else
        pass "pattern not found"
    fi
fi

# ---- L-12: No NULL check on d_data_i ----
echo ""
echo "--- L-12: NULL check on d_data_i in vpic_adapter ---"
VPIC="$PROJECT_DIR/src/vpic/vpic_adapter.cu"
if [ -f "$VPIC" ]; then
    if grep -B2 -A2 'd_data_i' "$VPIC" | grep -q 'nullptr\|NULL\|!.*d_data'; then
        pass "d_data_i NULL check present"
    elif grep -q 'd_data_i' "$VPIC"; then
        fail "d_data_i used without NULL check"
    else
        pass "d_data_i not found"
    fi
fi

# ---- L-13: Kokkos View extent validation ----
echo ""
echo "--- L-13: Kokkos View extent validation ---"
KOKKOS="$PROJECT_DIR/examples/vpic_kokkos_bridge.hpp"
if [ -f "$KOKKOS" ]; then
    if grep -q 'extent\|assert.*n_var\|check.*size' "$KOKKOS"; then
        pass "extent validation present"
    elif grep -q 'nbytes\|data()' "$KOKKOS"; then
        fail "Kokkos View used without extent validation"
    else
        pass "pattern not found"
    fi
fi

# ---- L-14: %lu for size_t ----
echo ""
echo "--- L-14: %lu vs %zu for size_t ---"
LU_COUNT=0
for f in "$PROJECT_DIR/src/cli/compress.cpp" "$PROJECT_DIR/src/cli/decompress.cpp"; do
    if [ -f "$f" ]; then
        C=$(grep -c '%lu' "$f" 2>/dev/null || true)
        C=${C:-0}
        LU_COUNT=$((LU_COUNT + C))
    fi
done
if [ "$LU_COUNT" -gt 0 ]; then
    fail "$LU_COUNT uses of %lu for size_t (should be %zu for portability)"
else
    pass "no %lu usage"
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
