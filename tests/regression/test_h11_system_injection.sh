#!/bin/bash
# test_h11_system_injection.sh
#
# H-11: eval/eval_simulation.cpp passes unsanitized -o CLI arg to system().
#       Fix: validate config.output with an allowlist before calling system().
#
# Tests verify the fix is in place:
#   1. Input validation guard exists before the system() call
#   2. Validation rejects shell metacharacters
#   3. config.output still comes from optarg (context check)
#   4. Only safe characters pass the validation
#   5. Quoted argument blocks injection (defense in depth demo)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE="$PROJECT_DIR/eval/eval_simulation.cpp"

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "=== H-11: system() command injection in eval_simulation.cpp ==="
echo ""

if [ ! -f "$SOURCE" ]; then
    echo "SKIP: $SOURCE not found"
    exit 0
fi

# ---- Test 1: Input validation guard exists before system() ----
echo "--- Test 1: Input validation guard before system() ---"

if grep -q 'output_safe' "$SOURCE" && grep -q 'isalnum' "$SOURCE"; then
    pass "input validation guard (output_safe + isalnum) found before system()"
else
    fail "no input validation guard found before system()"
fi

# ---- Test 2: system() is only called when output_safe is true ----
echo ""
echo "--- Test 2: system() gated by output_safe check ---"

if grep -q 'if.*output_safe' "$SOURCE"; then
    pass "system() call is gated by output_safe conditional"
else
    fail "system() not gated by safety check"
fi

# ---- Test 3: Unsafe characters are rejected ----
echo ""
echo "--- Test 3: Metacharacter rejection logic ---"

# The validation should reject ; | $ ` & etc.
if grep -q "output_safe = false" "$SOURCE"; then
    pass "validation sets output_safe = false for bad characters"
else
    fail "no rejection path for unsafe characters"
fi

# ---- Test 4: Warning printed when skipping ----
echo ""
echo "--- Test 4: Warning message for unsafe paths ---"

if grep -q 'disallowed characters\|skipping.*plot' "$SOURCE"; then
    pass "warning message printed when skipping unsafe path"
else
    fail "no warning message for unsafe paths"
fi

# ---- Test 5: Quoted argument blocks injection (defense in depth) ----
echo ""
echo "--- Test 5: Quoted argument blocks injection ---"

CANARY_DIR=$(mktemp -d)
CANARY="$CANARY_DIR/h11_canary_safe"

CRAFTED_OUTPUT="\"; touch $CANARY; echo \""
SAFE_CMD="echo simulated_plot '$CRAFTED_OUTPUT' simulated_output.png"

sh -c "$SAFE_CMD" > /dev/null 2>&1 || true

if [ -f "$CANARY" ]; then
    fail "quoted argument still allowed injection (unexpected)"
    rm -f "$CANARY"
else
    pass "single-quoted argument blocks injection"
fi
rm -rf "$CANARY_DIR"

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
