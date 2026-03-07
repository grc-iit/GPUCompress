#!/bin/bash
# test_h13_tmp_lib_paths.sh
#
# H-13: CMakeLists.txt hardcodes /tmp/include and /tmp/lib as include/link
#       directories. On shared systems, any user can place malicious libraries
#       there, which would be linked into the build.
#
# Test strategy:
#   1. Static analysis: check CMakeLists.txt and cmake/ for /tmp paths in
#      include_directories() and link_directories().
#   2. Verify /tmp is world-writable (confirms the risk).
#   3. Check if paths are overridable via CMake variables.
#   4. Count total /tmp references across build files.
#
# Usage: ./tests/regression/test_h13_tmp_lib_paths.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "=== H-13: World-writable /tmp paths in build configuration ==="
echo ""

# ---- Test 1: /tmp in include_directories ----
echo "--- Test 1: /tmp in include_directories() ---"

INCLUDE_TMP=$(grep -rn 'include_directories' "$PROJECT_DIR/CMakeLists.txt" \
              "$PROJECT_DIR/cmake/"*.cmake 2>/dev/null | grep '/tmp' || true)

if [ -n "$INCLUDE_TMP" ]; then
    echo "$INCLUDE_TMP" | while read -r line; do
        echo "  $line"
    done
    fail "/tmp found in include_directories"
else
    pass "no /tmp in include_directories"
fi

# ---- Test 2: /tmp in link_directories ----
echo ""
echo "--- Test 2: /tmp in link_directories() ---"

LINK_TMP=$(grep -rn 'link_directories' "$PROJECT_DIR/CMakeLists.txt" \
           "$PROJECT_DIR/cmake/"*.cmake 2>/dev/null | grep '/tmp' || true)

if [ -n "$LINK_TMP" ]; then
    echo "$LINK_TMP" | while read -r line; do
        echo "  $line"
    done
    fail "/tmp found in link_directories"
else
    pass "no /tmp in link_directories"
fi

# ---- Test 3: /tmp in hardcoded library paths ----
echo ""
echo "--- Test 3: /tmp in hardcoded set() paths ---"

SET_TMP=$(grep -rn 'set(.*_LIB\|set(.*_INCLUDE\|set(.*_PATH' \
          "$PROJECT_DIR/CMakeLists.txt" "$PROJECT_DIR/cmake/"*.cmake 2>/dev/null \
          | grep '/tmp' || true)

if [ -n "$SET_TMP" ]; then
    echo "$SET_TMP" | while read -r line; do
        echo "  $line"
    done
    fail "/tmp in hardcoded CMake path variables"
else
    pass "no /tmp in CMake path variables"
fi

# ---- Test 4: /tmp is world-writable (risk confirmation) ----
echo ""
echo "--- Test 4: /tmp is world-writable ---"

TMP_PERMS=$(stat -c '%a' /tmp 2>/dev/null || stat -f '%Lp' /tmp 2>/dev/null || echo "unknown")

if [ "$TMP_PERMS" = "1777" ] || [ "$TMP_PERMS" = "777" ]; then
    echo "  /tmp permissions: $TMP_PERMS (world-writable)"
    fail "/tmp is world-writable — library injection possible"
else
    echo "  /tmp permissions: $TMP_PERMS"
    pass "/tmp is not world-writable on this system"
fi

# ---- Test 5: Total /tmp references in build files ----
echo ""
echo "--- Test 5: Total /tmp references in build configuration ---"

TOTAL=$(grep -rn '/tmp' "$PROJECT_DIR/CMakeLists.txt" \
        "$PROJECT_DIR/cmake/"*.cmake 2>/dev/null \
        | grep -v ':#\|message\|STATUS\|CACHE' | wc -l || true)
TOTAL=${TOTAL:-0}

echo "  Non-comment, non-CACHE /tmp references: $TOTAL"

if [ "$TOTAL" -gt 0 ]; then
    fail "$TOTAL hardcoded /tmp references in build files"
else
    pass "no hardcoded /tmp references (all /tmp paths are CACHE-overridable)"
fi

# ---- Test 6: Check if overridable via CMake variables ----
echo ""
echo "--- Test 6: Paths overridable via CMake cache variables ---"

# A good pattern: if(NOT DEFINED NVCOMP_ROOT) set(NVCOMP_ROOT /tmp) endif()
OVERRIDABLE=$(grep -c 'if.*NOT.*DEFINED\|option\|CACHE' \
              "$PROJECT_DIR/CMakeLists.txt" 2>/dev/null || true)
OVERRIDABLE=${OVERRIDABLE:-0}

# Check specifically for nvcomp/hdf5 path overrides
NVCOMP_VAR=$(grep -rl 'NVCOMP_ROOT\|NVCOMP_DIR\|NVCOMP_PATH\|NVCOMP_PREFIX' \
             "$PROJECT_DIR/CMakeLists.txt" "$PROJECT_DIR/cmake/"*.cmake 2>/dev/null | wc -l || true)
NVCOMP_VAR=${NVCOMP_VAR:-0}

if [ "$NVCOMP_VAR" -gt 0 ]; then
    pass "nvCOMP path configurable via CMake variable"
else
    fail "nvCOMP path not configurable — hardcoded to /tmp"
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
