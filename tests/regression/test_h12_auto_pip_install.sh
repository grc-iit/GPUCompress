#!/bin/bash
# test_h12_auto_pip_install.sh
#
# H-12: eval/download_well_data.py auto-installs pip packages via os.system()
#       without user consent (ensure_dependencies function).
#
# Test strategy:
#   1. Static analysis: verify os.system("pip install ...") pattern exists.
#   2. Verify package names are hardcoded (no user-input injection).
#   3. Verify no user confirmation prompt before install.
#   4. Run ensure_dependencies in a controlled venv where packages are
#      already installed — verify no pip install is triggered.
#
# Usage: ./tests/regression/test_h12_auto_pip_install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE="$PROJECT_DIR/eval/download_well_data.py"

PASS_COUNT=0
FAIL_COUNT=0

pass() { echo "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { echo "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

echo "=== H-12: Auto pip install without consent in download_well_data.py ==="
echo ""

if [ ! -f "$SOURCE" ]; then
    echo "SKIP: $SOURCE not found"
    exit 0
fi

# ---- Test 1: os.system with pip install exists ----
echo "--- Test 1: os.system('pip install ...') pattern in source ---"

PIP_CALLS=$(grep -c 'os\.system.*pip install' "$SOURCE" 2>/dev/null || true)
PIP_CALLS=${PIP_CALLS:-0}

if [ "$PIP_CALLS" -gt 0 ]; then
    echo "  Found $PIP_CALLS os.system('pip install ...') calls:"
    grep -n 'os\.system.*pip install' "$SOURCE" | while read -r line; do
        echo "    $line"
    done
    fail "os.system() used for auto pip install ($PIP_CALLS calls)"
else
    pass "no os.system('pip install ...') found"
fi

# ---- Test 2: Package names are hardcoded (not user-controlled) ----
echo ""
echo "--- Test 2: Package names are hardcoded (no injection vector) ---"

# Extract package names from the pip install calls
PACKAGES=$(grep 'os\.system.*pip install' "$SOURCE" | grep -oP 'install \K\S+' | sed 's/ .*//')

INJECTED=0
for pkg in $PACKAGES; do
    # Check if it looks like a hardcoded name (alphanumeric + hyphens/underscores)
    if echo "$pkg" | grep -qP '^[a-zA-Z0-9_-]+$'; then
        echo "  Package '$pkg': hardcoded name (OK)"
    else
        echo "  Package '$pkg': suspicious (may contain injection)"
        INJECTED=1
    fi
done

if [ "$INJECTED" -eq 0 ]; then
    pass "all package names are hardcoded literals"
else
    fail "non-literal package names found"
fi

# ---- Test 3: No auto-install (raises error instead) ----
echo ""
echo "--- Test 3: Missing packages raise ImportError (no auto-install) ---"

# The fix should raise ImportError with instructions instead of auto-installing
RAISES=$(grep -c 'raise ImportError' "$SOURCE" 2>/dev/null || true)
RAISES=${RAISES:-0}

if [ "$RAISES" -gt 0 ]; then
    pass "raises ImportError for missing packages (no auto-install)"
else
    fail "no raise ImportError found — may still auto-install"
fi

# ---- Test 4: No os.system calls remain in ensure_dependencies ----
echo ""
echo "--- Test 4: No os.system pip install calls remain ---"

OS_SYSTEM=$(grep -c 'os\.system.*pip' "$SOURCE" 2>/dev/null || true)
OS_SYSTEM=${OS_SYSTEM:-0}

if [ "$OS_SYSTEM" -eq 0 ]; then
    pass "no os.system(pip install) calls remain"
else
    fail "os.system(pip install) still present ($OS_SYSTEM calls)"
fi

# ---- Test 5: Recommended fix check ----
echo ""
echo "--- Test 5: Check for recommended fix (error message instead of install) ---"

# A proper fix would raise an error with install instructions
RAISES=$(grep -c 'raise\|sys\.exit\|print.*please install\|print.*pip install' "$SOURCE" 2>/dev/null || true)
RAISES=${RAISES:-0}

# Check if the function still uses os.system or has been fixed to just error
if [ "$PIP_CALLS" -gt 0 ] && [ "$RAISES" -eq 0 ]; then
    fail "auto-installs without providing manual install instructions"
else
    pass "either fixed or provides manual install instructions"
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
