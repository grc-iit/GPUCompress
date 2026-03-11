#!/bin/bash
# test_issue6_oracle_uses_vol.sh
#
# Issue #6: VPIC benchmark oracle phase uses raw POSIX I/O instead of HDF5 VOL.
# This gives the oracle an unfair throughput advantage over NN phases.
#
# This test checks:
#   1. Oracle temp file is .h5 (HDF5), not .bin (raw binary)
#   2. Oracle Stage 2 uses H5Fcreate/H5Dwrite (HDF5 VOL), not open()/write()
#   3. Oracle Stage 3 uses H5Fopen/H5Dread, not open()/read()
#   4. Oracle ratio uses file size, not raw compressed sum
#   5. make_dcpl_fixed() helper exists for the fixed-algo DCPL

PASS=0
FAIL=0
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VPIC_DECK="$SCRIPT_DIR/../../benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx"

check() {
    local desc="$1"
    local result="$2"
    if [ "$result" = "0" ]; then
        echo "  PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $desc"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Issue #6: VPIC Oracle Must Use HDF5 VOL ==="
echo ""

if [ ! -f "$VPIC_DECK" ]; then
    echo "  FAIL: VPIC benchmark deck not found at $VPIC_DECK"
    echo ""
    echo "$PASS pass, 1 fail"
    exit 1
fi

# 1. TMP_ORACLE should be .h5, not .bin
if grep -q 'TMP_ORACLE.*\.bin' "$VPIC_DECK"; then
    check "TMP_ORACLE uses .h5 extension (not .bin)" 1
else
    if grep -q 'TMP_ORACLE.*\.h5' "$VPIC_DECK"; then
        check "TMP_ORACLE uses .h5 extension (not .bin)" 0
    else
        check "TMP_ORACLE uses .h5 extension (not .bin)" 1
    fi
fi

# 2. Oracle Stage 2 should NOT use POSIX open() for writing
#    Look for open(TMP_ORACLE...) pattern inside run_oracle_pass
oracle_section=$(sed -n '/run_oracle_pass/,/^}/p' "$VPIC_DECK")
if echo "$oracle_section" | grep -q 'open(TMP_ORACLE.*O_WRONLY'; then
    check "Oracle Stage 2 does not use POSIX open() for writing" 1
else
    check "Oracle Stage 2 does not use POSIX open() for writing" 0
fi

# 3. Oracle Stage 2 should use H5Fcreate for writing
if echo "$oracle_section" | grep -q 'H5Fcreate'; then
    check "Oracle Stage 2 uses H5Fcreate (HDF5 VOL)" 0
else
    check "Oracle Stage 2 uses H5Fcreate (HDF5 VOL)" 1
fi

# 4. Oracle Stage 2 should use H5Dwrite
if echo "$oracle_section" | grep -q 'H5Dwrite'; then
    check "Oracle Stage 2 uses H5Dwrite" 0
else
    check "Oracle Stage 2 uses H5Dwrite" 1
fi

# 5. Oracle Stage 3 should NOT use POSIX open() for reading
if echo "$oracle_section" | grep -q 'open(TMP_ORACLE.*O_RDONLY'; then
    check "Oracle Stage 3 does not use POSIX open() for reading" 1
else
    check "Oracle Stage 3 does not use POSIX open() for reading" 0
fi

# 6. Oracle Stage 3 should use H5Fopen + H5Dread
if echo "$oracle_section" | grep -q 'H5Fopen'; then
    check "Oracle Stage 3 uses H5Fopen (HDF5 VOL)" 0
else
    check "Oracle Stage 3 uses H5Fopen (HDF5 VOL)" 1
fi

# 7. Oracle ratio should use file size, not raw compressed sum
#    Check for get_file_size or file_size pattern in ratio calculation
if echo "$oracle_section" | grep -q 'get_file_size\|file_size'; then
    check "Oracle ratio uses file size (includes HDF5 overhead)" 0
else
    check "Oracle ratio uses file size (includes HDF5 overhead)" 1
fi

# 8. make_dcpl_fixed() helper should exist
if grep -q 'make_dcpl_fixed' "$VPIC_DECK"; then
    check "make_dcpl_fixed() helper exists" 0
else
    check "make_dcpl_fixed() helper exists" 1
fi

echo ""
echo "$PASS pass, $FAIL fail"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
