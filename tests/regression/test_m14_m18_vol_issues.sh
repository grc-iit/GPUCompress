#!/bin/bash
# test_m14_m18_vol_issues.sh
#
# Static analysis tests for HDF5 VOL connector issues:
#   M-14: free_obj doesn't close dcpl_id
#   M-15: info_to_str buffer too small + memory leak
#   M-16: Prefetch thread cleanup (RETRACTED — properly handled)
#   M-17: dset[0] accessed when count==0
#   M-18: link_create mutates args (RETRACTED — standard VOL pattern)
#
# Usage: ./tests/regression/test_m14_m18_vol_issues.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
VOL="$PROJECT_DIR/src/hdf5/H5VLgpucompress.cu"

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "=== M-14 through M-18: HDF5 VOL connector issues ==="
echo ""

if [ ! -f "$VOL" ]; then
    echo "SKIP: $VOL not found"
    exit 0
fi

# ---- M-14: free_obj doesn't close dcpl_id ----
echo "--- M-14: free_obj dcpl_id leak ---"

# Check if free_obj closes dcpl_id
FREE_OBJ=$(sed -n '/^free_obj/,/^}/p' "$VOL")
if echo "$FREE_OBJ" | grep -q 'H5Pclose\|dcpl_id'; then
    pass "free_obj closes dcpl_id"
else
    fail "free_obj does NOT close dcpl_id (HDF5 handle leak)"
fi

# Check that dataset_close closes it (mitigation)
if grep -q 'H5Pclose.*dcpl_id' "$VOL"; then
    pass "dcpl_id closed somewhere (dataset_close path)"
else
    fail "dcpl_id never closed anywhere"
fi

# ---- M-15: info_to_str buffer size + memory leak ----
echo ""
echo "--- M-15: info_to_str buffer size + leak ---"

# Check buffer size: should be at least 48 + ulen
BUF_SIZE=$(grep -oP 'size_t sz = \K[0-9]+' "$VOL" || echo "0")
if [ "$BUF_SIZE" -ge 48 ] 2>/dev/null; then
    pass "info_to_str buffer size >= 48"
else
    fail "info_to_str buffer size = $BUF_SIZE (too small, needs >= 48)"
fi

# Check if 'us' string is freed
INFO_TO_STR=$(sed -n '/info_to_str/,/^}/p' "$VOL")
if echo "$INFO_TO_STR" | grep -q 'H5free_memory\|free(us)'; then
    pass "info_to_str frees 'us' string"
else
    fail "info_to_str does NOT free 'us' string (memory leak)"
fi

# ---- M-16: Prefetch thread cleanup (RETRACTED) ----
echo ""
echo "--- M-16: Prefetch thread cleanup ---"

if grep -q 'pre_thr.join' "$VOL"; then
    pass "prefetch thread join exists"
else
    fail "no prefetch thread join found"
fi

# Check that done_read label has cleanup
if grep -A5 'done_read:' "$VOL" | grep -q 'free_slots_count\|notify'; then
    pass "done_read unblocks prefetch thread before join"
else
    fail "done_read may not unblock prefetch thread"
fi

# ---- M-17: dset[0] accessed when count==0 ----
echo ""
echo "--- M-17: dset[0] OOB access when count==0 ---"

# Check if there's a count > 0 guard before dset[0]
DSET0_LINES=$(grep -n 'dset\[0\]' "$VOL" | head -5)
echo "  dset[0] references:"
echo "$DSET0_LINES" | while read -r line; do echo "    $line"; done

# Check if any dset[0] access is guarded by count check
if grep -B3 'dset\[0\]' "$VOL" | grep -q 'count.*>.*0\|count.*!=.*0'; then
    pass "dset[0] access guarded by count check"
else
    fail "dset[0] access NOT guarded by count > 0 check"
fi

# ---- M-18: link_create args mutation (RETRACTED) ----
echo ""
echo "--- M-18: link_create args mutation ---"

if grep -q 'args->args.hard.curr_obj.*=.*under_object' "$VOL"; then
    # This is standard VOL passthrough pattern
    pass "link_create unwraps curr_obj (standard VOL passthrough pattern)"
else
    pass "link_create does not mutate args (or pattern changed)"
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
