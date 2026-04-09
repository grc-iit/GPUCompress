#!/bin/bash
# ============================================================
# Run the 3 VOL bypass / program_wall test binaries on a Delta A100.
#
# Submit from the repo root:
#   sbatch tests/hdf5/run_bypass_tests.sbatch
#
# Each test prints a PASS/FAIL summary; the sbatch exit code is the
# number of failing tests.
# ============================================================
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=gpucomp_bypass_tests
#SBATCH --output=deleteAfterwards/slurm-%x-%j.out

set -uo pipefail

module load cuda gcc-native/13.2 cray-mpich/8.1.32

GPUCOMPRESS_DIR=/u/$USER/GPUCompress
HDF5_DIR=$GPUCOMPRESS_DIR/.deps/hdf5
NVCOMP_DIR=$GPUCOMPRESS_DIR/.deps
BUILD_DIR=$GPUCOMPRESS_DIR/build

cd "$GPUCOMPRESS_DIR"

export LD_LIBRARY_PATH=$NVCOMP_DIR/lib:$HDF5_DIR/lib:$BUILD_DIR:$GPUCOMPRESS_DIR/examples:${LD_LIBRARY_PATH:-}

nvidia-smi | head -20 || true
echo ""
echo "============================================================"
echo "VOL bypass + program_wall test suite"
echo "  build dir: $BUILD_DIR"
echo "============================================================"

TESTS=(
    test_vol_bypass_roundtrip
    test_vol_bypass_timing
    test_vol_program_wall
)

declare -A RESULTS
fail=0

for t in "${TESTS[@]}"; do
    bin="$BUILD_DIR/$t"
    if [ ! -x "$bin" ]; then
        echo ""
        echo "!!! MISSING: $bin — did you rebuild? (cmake --build $BUILD_DIR -j 8)"
        RESULTS[$t]="MISSING"
        fail=$((fail + 1))
        continue
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo ">>> $t"
    echo "────────────────────────────────────────────────────────────"
    if "$bin"; then
        RESULTS[$t]="PASS"
    else
        rc=$?
        RESULTS[$t]="FAIL(rc=$rc)"
        fail=$((fail + 1))
    fi
done

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
for t in "${TESTS[@]}"; do
    printf "  %-30s %s\n" "$t" "${RESULTS[$t]}"
done
echo ""
echo "$((${#TESTS[@]} - fail))/${#TESTS[@]} PASS"

exit $fail
