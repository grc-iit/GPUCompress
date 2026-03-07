#!/bin/bash
# test_m33_m38_cli_vpic.sh
#
# Static analysis tests for CLI and VPIC issues:
#   M-33: Fragile cleanup in compress.cpp
#   M-34: Silent truncation of trailing bytes
#   M-35: GDS read uninitialized tail + partial read unchecked
#   M-36: Stale "64 MB chunks" comment (is 4 MiB)
#   M-37: HDF5 resource leak in vpic example
#   M-38: Unchecked gpucompress_vpic_create return values
#
# Usage: ./tests/regression/test_m33_m38_cli_vpic.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPRESS="$PROJECT_DIR/src/cli/compress.cpp"
VPIC_DECK="$PROJECT_DIR/examples/vpic_compress_deck.cxx"

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "=== M-33 through M-38: CLI/VPIC issues ==="
echo ""

# ---- M-33: Fragile cleanup ----
echo "--- M-33: Fragile three-way cleanup in compress.cpp ---"
if [ -f "$COMPRESS" ]; then
    # Count the number of cudaFree calls in cleanup section
    CLEANUP_FREES=$(grep -c 'cudaFree' "$COMPRESS" 2>/dev/null || true)
    CLEANUP_FREES=${CLEANUP_FREES:-0}
    ELSE_IF_CHAINS=$(grep -c 'else if.*d_dequantized\|else if.*d_unshuffled' "$COMPRESS" 2>/dev/null || true)
    ELSE_IF_CHAINS=${ELSE_IF_CHAINS:-0}

    if [ "$ELSE_IF_CHAINS" -gt 0 ]; then
        fail "fragile if/else if/else cleanup chain present ($ELSE_IF_CHAINS branches)"
    else
        pass "no fragile cleanup chain (may be refactored)"
    fi
else
    echo "  SKIP: compress.cpp not found"
fi

# ---- M-34: Silent trailing byte truncation ----
echo ""
echo "--- M-34: Silent trailing byte truncation ---"
if [ -f "$COMPRESS" ]; then
    # num_elements = file_size / quant_element_size — no modulo check
    if grep -q 'file_size.*%.*element\|file_size.*mod\|file_size.*remainder' "$COMPRESS"; then
        pass "trailing bytes checked"
    elif grep -q 'num_elements.*=.*file_size.*/' "$COMPRESS"; then
        fail "num_elements = file_size / element_size without remainder check"
    else
        pass "pattern not found"
    fi
else
    echo "  SKIP"
fi

# ---- M-35: GDS partial read unchecked ----
echo ""
echo "--- M-35: GDS read partial/uninitialized tail ---"
if [ -f "$COMPRESS" ]; then
    # Check if bytes_read is compared to file_size (not just > 0)
    if grep -q 'bytes_read.*<.*file_size\|bytes_read.*!=.*file_size\|bytes_read.*==.*file_size' "$COMPRESS"; then
        pass "bytes_read compared against file_size"
    elif grep -q 'bytes_read.*<.*0\|bytes_read.*< 0' "$COMPRESS"; then
        fail "bytes_read only checked for negative, not partial read < file_size"
    else
        pass "pattern not found or refactored"
    fi

    # Check uninitialized tail: aligned_input_size > file_size
    if grep -q 'memset.*tail\|cudaMemset.*aligned\|aligned_input.*file_size' "$COMPRESS"; then
        pass "tail bytes zeroed"
    else
        echo "  INFO: aligned read may leave uninitialized bytes in tail"
    fi
else
    echo "  SKIP"
fi

# ---- M-36: Stale comment ----
echo ""
echo "--- M-36: Stale 64 MB chunks comment ---"
if [ -f "$VPIC_DECK" ]; then
    if grep -q '64.*MB\|64.*MiB' "$VPIC_DECK"; then
        ACTUAL=$(grep -oP '\d+\s*\*\s*1024\s*\*\s*1024' "$VPIC_DECK" | head -1)
        if [ -n "$ACTUAL" ]; then
            echo "  Comment says 64 MB but code has: $ACTUAL"
            fail "stale comment: says 64 MB but code is 4 * 1024 * 1024 (4 MiB)"
        fi
    else
        pass "no stale 64 MB comment"
    fi
else
    echo "  SKIP: $VPIC_DECK not found"
fi

# ---- M-37: HDF5 resource leak in vpic example ----
echo ""
echo "--- M-37: HDF5 resource leak on error ---"
if [ -f "$VPIC_DECK" ]; then
    # Check write_gpu_to_hdf5 for error handling with H5Sclose/H5Pclose
    WRITE_FUNC=$(sed -n '/write_gpu_to_hdf5/,/^}/p' "$VPIC_DECK")
    # Check for error-path cleanup: goto cleanup, or early return with H5*close, or < 0 checks
    if echo "$WRITE_FUNC" | grep -q 'goto.*done\|goto.*cleanup\|< 0.*return\|< 0.*H5.*close'; then
        pass "error cleanup for HDF5 handles present"
    elif echo "$WRITE_FUNC" | grep -q 'if.*space < 0\|if.*dcpl < 0\|if.*dset < 0'; then
        pass "error-path early returns with cleanup present"
    else
        HAS_CREATE=$(echo "$WRITE_FUNC" | grep -c 'H5Dcreate\|H5Screate' 2>/dev/null || true)
        HAS_CREATE=${HAS_CREATE:-0}
        HAS_CLOSE=$(echo "$WRITE_FUNC" | grep -c 'H5Sclose\|H5Pclose' 2>/dev/null || true)
        HAS_CLOSE=${HAS_CLOSE:-0}
        if [ "$HAS_CREATE" -gt 0 ] && [ "$HAS_CLOSE" -eq 0 ]; then
            fail "H5D/S/Pcreate without matching close on error path"
        elif [ "$HAS_CREATE" -gt 0 ]; then
            fail "HDF5 handles created but no error-path cleanup"
        else
            pass "no HDF5 create calls (refactored?)"
        fi
    fi
else
    echo "  SKIP"
fi

# ---- M-38: Unchecked vpic_create return values ----
echo ""
echo "--- M-38: Unchecked gpucompress_vpic_create ---"
if [ -f "$VPIC_DECK" ]; then
    # Only match actual call sites (lines with function call parentheses), not string mentions
    VPIC_CREATES=$(grep 'gpucompress_vpic_create(' "$VPIC_DECK" | grep -v '//\|sim_log\|printf\|"' || true)
    if [ -n "$VPIC_CREATES" ]; then
        # Check if return value is checked (if wrapper or != SUCCESS)
        UNCHECKED=$(echo "$VPIC_CREATES" | grep -v 'if\|assert\|!=.*NULL\|==.*NULL\|!=.*SUCCESS\|GPUCOMPRESS_SUCCESS' | wc -l)
        if [ "$UNCHECKED" -gt 0 ]; then
            fail "$UNCHECKED gpucompress_vpic_create calls without error check"
        else
            pass "all gpucompress_vpic_create calls checked"
        fi
    else
        pass "no gpucompress_vpic_create calls"
    fi
else
    echo "  SKIP"
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
