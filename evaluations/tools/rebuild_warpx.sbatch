#!/bin/bash
# ============================================================
# Rebuild WarpX with the latest GPUCompress patches.
#
# The existing binary at /u/imuradli/src/warpx/build-cuda/bin/warpx.3d
# (linked Apr 7 15:28) is stale: it predates the Apr 7 17:01 patch
# commit bb6500c "Read back enabled for simulations", which adds:
#   - predicted_decomp_ms / actual_decomp_ms_raw / mape_decomp cols
#   - predicted_psnr_db / actual_psnr_db / mape_psnr cols
#   - in-situ H5Dread after every H5Dwrite (unconditional readback)
# Without this rebuild, sweep heatmaps and the sc26 per-metric
# breakdown will show empty decomp/psnr bars for WarpX.
#
# Steps:
#   1. Copy fresh benchmarks/warpx/patches/ files into the WarpX tree
#      (same canonical-overlay step deploy/run.sh setup_warpx does).
#   2. cmake --build the existing build-cuda/ directory (incremental,
#      only a few .o files will recompile).
#   3. Print binary age + chunk-CSV header probe so you can verify
#      the new decomp/psnr columns are in place.
#
# No GPU required for the build. CPU partition only.
#
# Submit:   sbatch deleteAfterwards/delta_rebuild_warpx.sbatch
# ============================================================
#SBATCH --account=bekn-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=warpx_rebuild
#SBATCH --output=deleteAfterwards/slurm-%x-%j.out

set -uo pipefail

module load cuda gcc-native/13.2 cray-mpich/8.1.32

GPUC_DIR=/u/$USER/GPUCompress
WARPX_SRC=/u/$USER/src/warpx
BUILD_DIR=$WARPX_SRC/build-cuda

cd "$GPUC_DIR"

echo "============================================================"
echo "WarpX rebuild — applying latest GPUCompress patches"
echo "  Source:  $WARPX_SRC"
echo "  Build:   $BUILD_DIR"
echo "  Patches: $GPUC_DIR/benchmarks/warpx/patches/"
echo "============================================================"
echo ""

# ── Step 1: overlay canonical patch files ──
echo ">>> Step 1: copying patch files into WarpX tree"
cp -v "$GPUC_DIR/benchmarks/warpx/patches/FlushFormatGPUCompress.H" \
      "$WARPX_SRC/Source/Diagnostics/FlushFormats/FlushFormatGPUCompress.H"
cp -v "$GPUC_DIR/benchmarks/warpx/patches/FlushFormatGPUCompress.cpp" \
      "$WARPX_SRC/Source/Diagnostics/FlushFormats/FlushFormatGPUCompress.cpp"
cp -v "$GPUC_DIR/benchmarks/warpx/patches/Diagnostics.cpp" \
      "$WARPX_SRC/Source/Diagnostics/Diagnostics.cpp"
cp -v "$GPUC_DIR/benchmarks/warpx/patches/FullDiagnostics.cpp" \
      "$WARPX_SRC/Source/Diagnostics/FullDiagnostics.cpp"
cp -v "$GPUC_DIR/benchmarks/warpx/patches/GPUCompress.cmake" \
      "$WARPX_SRC/cmake/dependencies/GPUCompress.cmake"
echo ""

# ── Step 2: incremental rebuild ──
echo ">>> Step 2: cmake --build $BUILD_DIR -j$(nproc)"
cmake --build "$BUILD_DIR" -j"$(nproc)"
rc=$?
echo ""

if [ "$rc" -ne 0 ]; then
    echo "!!! BUILD FAILED (rc=$rc) — check log above"
    exit $rc
fi

# ── Step 3: verify ──
NEW_BIN="$BUILD_DIR/bin/warpx.3d"
echo "============================================================"
echo ">>> Step 3: verification"
echo ""
if [ -L "$NEW_BIN" ]; then
    echo "Binary (symlink):"
    ls -la "$NEW_BIN"
    ls -la "$(readlink -f "$NEW_BIN")"
else
    echo "Binary:"
    ls -la "$NEW_BIN"
fi
echo ""
echo "Source patch file:"
stat -c '  %y %n' "$WARPX_SRC/Source/Diagnostics/FlushFormats/FlushFormatGPUCompress.cpp"
echo ""
echo "Quick header check — chunk CSV should have decomp + psnr columns:"
grep -E 'mape_decomp|actual_decomp_ms_raw|mape_psnr|actual_psnr_db' \
    "$WARPX_SRC/Source/Diagnostics/FlushFormats/FlushFormatGPUCompress.cpp" \
    | head -5
echo ""
echo "============================================================"
echo "WarpX rebuild complete. New binary ready at:"
echo "  $NEW_BIN"
echo ""
echo "Next steps:"
echo "  1. Verify WARPX_BIN in sweep sbatch points at this path:"
echo "     grep WARPX_BIN benchmarks/Paper_Evaluations/4/threshold_sweep/delta_threshold_sweep_allworkloads.sbatch"
echo "  2. Resubmit 7x7 threshold sweep:"
echo "     sbatch benchmarks/Paper_Evaluations/4/threshold_sweep/delta_threshold_sweep_allworkloads.sbatch"
echo "============================================================"
