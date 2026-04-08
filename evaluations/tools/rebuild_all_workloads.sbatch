#!/bin/bash
# ============================================================
# Rebuild ALL workload binaries with the latest GPUCompress patches:
#   1. VPIC        (benchmark deck at benchmarks/vpic-kokkos/)
#   2. LAMMPS      (build tree at ~/lammps/build/)
#   3. Nyx         (build tree at ~/Nyx/build/)
#   4. WarpX       (build tree at ~/src/warpx/build-cuda/)
#
# Each workload goes through two steps:
#   (a) Overlay the canonical patch files from benchmarks/<sim>/patches/
#       onto the source tree (same mechanism as deploy/run.sh setup_<sim>).
#   (b) Incremental rebuild — reuse the existing build dir so only changed
#       .cpp/.cu/.h files recompile.
#
# Why this is needed:
#   - VPIC   deck source at benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx
#            is canonical; deck binary lives in the GPUCompress tree.
#   - LAMMPS fix_gpucompress_kokkos emits per-chunk decomp/psnr diagnostics
#            only if the fix was built from the latest patch sources.
#   - Nyx    nyx_amrex_bridge.hpp readback path must match the current patch.
#   - WarpX  binary linked Apr 7 15:28 predates commit bb6500c (Apr 7 17:01)
#            "Read back enabled for simulations" which added decomp + psnr
#            columns to benchmark_warpx_timestep_chunks.csv and made
#            in-situ readback unconditional.
#
# After this completes, every workload's chunks CSV will carry:
#   - predicted/actual/MAPE for compression time
#   - predicted/actual/MAPE for decompression time   ← needs this rebuild
#   - predicted/actual/MAPE for PSNR (lossy only)    ← needs this rebuild
#   - predicted/actual/MAPE for compression ratio
# which is what figure8_cross_workload_regret's Figure 3 bar chart
# (per-metric MAPE breakdown) consumes.
#
# No GPU is required for the builds — CPU partition is sufficient.
#
# Submit:   sbatch evaluations/tools/rebuild_all_workloads.sbatch
# ============================================================
#SBATCH --account=bekn-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=rebuild_all
#SBATCH --output=deleteAfterwards/slurm-%x-%j.out

set -uo pipefail

module load cuda gcc-native/13.2 cray-mpich/8.1.32

GPUC_DIR=/u/$USER/GPUCompress
HDF5_DIR=$GPUC_DIR/.deps/hdf5
NVCOMP_DIR=$GPUC_DIR/.deps

VPIC_SRC=/u/$USER/vpic-kokkos
LAMMPS_SRC=/u/$USER/lammps
NYX_SRC=/u/$USER/Nyx
WARPX_SRC=/u/$USER/src/warpx

LAMMPS_BUILD=$LAMMPS_SRC/build
NYX_BUILD=$NYX_SRC/build
WARPX_BUILD=$WARPX_SRC/build-cuda

cd "$GPUC_DIR"

# Build tree must find libnvcomp/hdf5 headers at compile time.
export LD_LIBRARY_PATH=$NVCOMP_DIR/lib:$HDF5_DIR/lib:$GPUC_DIR/build:${LD_LIBRARY_PATH:-}

STATUS_FILE=$(mktemp)
trap 'rm -f "$STATUS_FILE"' EXIT

run_step() {
    local name="$1"; shift
    echo ""
    echo "============================================================"
    echo ">>> $name"
    echo "============================================================"
    local t0=$SECONDS
    if "$@"; then
        local el=$((SECONDS - t0))
        echo "  OK  $name  (${el}s)"
        echo "$name ok ${el}s" >> "$STATUS_FILE"
    else
        local rc=$?
        local el=$((SECONDS - t0))
        echo "  FAIL  $name  (rc=$rc, ${el}s)"
        echo "$name FAIL(rc=$rc) ${el}s" >> "$STATUS_FILE"
        return $rc
    fi
}

# ════════════════════════════════════════════════════════════════
# 1. VPIC benchmark deck
# ════════════════════════════════════════════════════════════════
rebuild_vpic() {
    [ -d "$VPIC_SRC" ] || { echo "ERROR: VPIC source not at $VPIC_SRC"; return 1; }
    [ -f "$VPIC_SRC/build-compress/libvpic.so" ] || \
    [ -f "$VPIC_SRC/build-compress/libvpic.a" ]  || {
        echo "ERROR: VPIC core not built at $VPIC_SRC/build-compress"; return 1;
    }
    export VPIC_DIR="$VPIC_SRC"
    export GPU_DIR="$GPUC_DIR"
    export VPIC_BUILD="$VPIC_SRC/build-compress"
    export HDF5_PREFIX="$HDF5_DIR"
    export NVCOMP_LIB="$NVCOMP_DIR/lib"
    export CUDA_ARCH="${CUDA_ARCH:-80}"
    bash "$GPUC_DIR/benchmarks/vpic-kokkos/build_vpic_pm.sh"
    ls -la "$GPUC_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux"
}

# ════════════════════════════════════════════════════════════════
# 2. LAMMPS with KOKKOS GPUCompress fix
# ════════════════════════════════════════════════════════════════
rebuild_lammps() {
    [ -d "$LAMMPS_BUILD" ] || { echo "ERROR: LAMMPS build dir not at $LAMMPS_BUILD"; return 1; }
    echo ">>> Overlaying LAMMPS patch sources"
    cp -v "$GPUC_DIR/benchmarks/lammps/patches/fix_gpucompress_kokkos.h"   "$LAMMPS_SRC/src/KOKKOS/"
    cp -v "$GPUC_DIR/benchmarks/lammps/patches/fix_gpucompress_kokkos.cpp" "$LAMMPS_SRC/src/KOKKOS/"
    cp -v "$GPUC_DIR/benchmarks/lammps/patches/lammps_ranking_profiler.cu" "$LAMMPS_SRC/src/KOKKOS/"
    cp -v "$GPUC_DIR/benchmarks/lammps/patches/KOKKOS.cmake"               "$LAMMPS_SRC/cmake/Modules/Packages/KOKKOS.cmake"

    echo ">>> Incremental build: cmake --build $LAMMPS_BUILD -j$(nproc)"
    cmake --build "$LAMMPS_BUILD" -j"$(nproc)"
    ls -la "$LAMMPS_BUILD/lmp"
}

# ════════════════════════════════════════════════════════════════
# 3. Nyx (AMReX) with GPUCompress bridge
# ════════════════════════════════════════════════════════════════
rebuild_nyx() {
    [ -d "$NYX_BUILD" ] || { echo "ERROR: Nyx build dir not at $NYX_BUILD"; return 1; }
    echo ">>> Overlaying Nyx patch sources"
    cp -v "$GPUC_DIR/benchmarks/nyx/patches/Nyx.H"          "$NYX_SRC/Source/Driver/Nyx.H"
    cp -v "$GPUC_DIR/benchmarks/nyx/patches/Nyx.cpp"        "$NYX_SRC/Source/Driver/Nyx.cpp"
    cp -v "$GPUC_DIR/benchmarks/nyx/patches/Nyx_output.cpp" "$NYX_SRC/Source/IO/Nyx_output.cpp"

    echo ">>> Incremental build: cmake --build $NYX_BUILD --target nyx_HydroTests -j$(nproc)"
    cmake --build "$NYX_BUILD" --target nyx_HydroTests -j"$(nproc)"
    ls -la "$NYX_BUILD/Exec/HydroTests/nyx_HydroTests"
}

# ════════════════════════════════════════════════════════════════
# 4. WarpX with GPUCompress FlushFormat
# ════════════════════════════════════════════════════════════════
rebuild_warpx() {
    [ -d "$WARPX_BUILD" ] || { echo "ERROR: WarpX build dir not at $WARPX_BUILD"; return 1; }
    echo ">>> Overlaying WarpX patch sources"
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

    echo ">>> Incremental build: cmake --build $WARPX_BUILD -j$(nproc)"
    cmake --build "$WARPX_BUILD" -j"$(nproc)"
    ls -la "$WARPX_BUILD/bin/warpx.3d"
}

# ════════════════════════════════════════════════════════════════
# Run them sequentially; continue on error so a single failure
# doesn't block the other rebuilds.
# ════════════════════════════════════════════════════════════════
OVERALL_START=$SECONDS

echo "============================================================"
echo "Rebuild all workload binaries"
echo "  GPUC_DIR:   $GPUC_DIR"
echo "  VPIC_SRC:   $VPIC_SRC"
echo "  LAMMPS_SRC: $LAMMPS_SRC (build: $LAMMPS_BUILD)"
echo "  NYX_SRC:    $NYX_SRC (build: $NYX_BUILD)"
echo "  WARPX_SRC:  $WARPX_SRC (build: $WARPX_BUILD)"
echo "============================================================"

run_step "VPIC"   rebuild_vpic   || true
run_step "LAMMPS" rebuild_lammps || true
run_step "NYX"    rebuild_nyx    || true
run_step "WarpX"  rebuild_warpx  || true

total=$((SECONDS - OVERALL_START))
echo ""
echo "============================================================"
echo "Rebuild all complete in ${total}s"
echo "  workload  status"
echo "  --------  ------"
while read -r line; do
    printf "  %s\n" "$line"
done < "$STATUS_FILE"
echo ""
echo "Binaries:"
for f in \
    "$GPUC_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux" \
    "$LAMMPS_BUILD/lmp" \
    "$NYX_BUILD/Exec/HydroTests/nyx_HydroTests" \
    "$WARPX_BUILD/bin/warpx.3d"; do
    if [ -e "$f" ]; then
        stat -c '  %y  %n' "$f"
    else
        echo "  MISSING  $f"
    fi
done
echo ""
echo "Next step — submit the figure8 cross-workload regret sweep:"
echo "  sbatch evaluations/figure8_cross_workload_regret/submit.sbatch"
echo "============================================================"
