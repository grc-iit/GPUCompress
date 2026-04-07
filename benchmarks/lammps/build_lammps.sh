#!/bin/bash
# ============================================================
# Build LAMMPS + KOKKOS (CUDA) + GPUCompress fix
#
# Clones LAMMPS, copies the fix sources from
# benchmarks/lammps/patches/, applies the KOKKOS.cmake patch,
# configures CMake with the GPUCompress include / link flags,
# and builds lmp.
#
# Usage:
#   bash benchmarks/lammps/build_lammps.sh
#
# Environment overrides:
#   LAMMPS_SRC      $HOME/sims/lammps      Where to clone LAMMPS
#   LAMMPS_TAG      develop                git tag/branch to check out
#   KOKKOS_ARCH     AMPERE80               Kokkos GPU arch macro
#   KOKKOS_PREC     SINGLE                 SINGLE | DOUBLE | MIXED
#   GPUCOMPRESS_DIR /home/cc/GPUCompress
#   HDF5_DIR        /tmp/hdf5-install
#   CUDA_DIR        /usr/local/cuda
#   JOBS            $(nproc)
# ============================================================
set -eo pipefail

GPUCOMPRESS_DIR="${GPUCOMPRESS_DIR:-/home/cc/GPUCompress}"
HDF5_DIR="${HDF5_DIR:-/tmp/hdf5-install}"
CUDA_DIR="${CUDA_DIR:-/usr/local/cuda}"
LAMMPS_SRC="${LAMMPS_SRC:-$HOME/sims/lammps}"
LAMMPS_TAG="${LAMMPS_TAG:-develop}"
KOKKOS_ARCH="${KOKKOS_ARCH:-AMPERE80}"
KOKKOS_PREC="${KOKKOS_PREC:-SINGLE}"
JOBS="${JOBS:-$(nproc)}"

PATCH_DIR="$GPUCOMPRESS_DIR/benchmarks/lammps/patches"

echo "============================================================"
echo "LAMMPS+KOKKOS+GPUCompress build"
echo "  GPUCompress:  $GPUCOMPRESS_DIR"
echo "  HDF5:         $HDF5_DIR"
echo "  CUDA:         $CUDA_DIR"
echo "  LAMMPS src:   $LAMMPS_SRC ($LAMMPS_TAG)"
echo "  Kokkos arch:  $KOKKOS_ARCH"
echo "  Kokkos prec:  $KOKKOS_PREC"
echo "  Jobs:         $JOBS"
echo "============================================================"

# ── Sanity checks ──
[ -f "$GPUCOMPRESS_DIR/build/libgpucompress.so" ] || {
    echo "ERROR: libgpucompress.so missing — build GPUCompress first"; exit 1; }
[ -f "$GPUCOMPRESS_DIR/examples/liblammps_gpucompress_udf.so" ] || {
    echo "ERROR: liblammps_gpucompress_udf.so missing — build the bridge first"; exit 1; }
[ -f "$HDF5_DIR/lib/libhdf5.so" ] || {
    echo "ERROR: HDF5 not found at $HDF5_DIR"; exit 1; }
[ -f "$PATCH_DIR/fix_gpucompress_kokkos.cpp" ] || {
    echo "ERROR: patch sources missing at $PATCH_DIR"; exit 1; }

mkdir -p "$(dirname "$LAMMPS_SRC")"

# ── Clone (or update) ──
if [ ! -d "$LAMMPS_SRC/.git" ]; then
    echo ""
    echo ">>> Cloning LAMMPS ($LAMMPS_TAG) into $LAMMPS_SRC ..."
    git clone --depth 1 --branch "$LAMMPS_TAG" \
        https://github.com/lammps/lammps.git "$LAMMPS_SRC"
else
    echo ""
    echo ">>> LAMMPS source already present at $LAMMPS_SRC — reusing."
fi

cd "$LAMMPS_SRC"

# ── Drop in fix sources ──
echo ""
echo ">>> Copying fix_gpucompress_kokkos.{h,cpp} into src/KOKKOS/ ..."
cp "$PATCH_DIR/fix_gpucompress_kokkos.h"   src/KOKKOS/
cp "$PATCH_DIR/fix_gpucompress_kokkos.cpp" src/KOKKOS/

# ── Build the Kendall-tau profiler wrapper as a shared library ──
# fix_gpucompress_kokkos.cpp references 4 C-linkage symbols
# (lammps_run_ranking_profiler, lammps_is_ranking_milestone,
#  lammps_write_ranking_csv_header, lammps_write_ranking_costs_csv_header)
# defined in benchmarks/lammps/patches/lammps_ranking_profiler.cu, which
# wraps benchmarks/kendall_tau_profiler.cuh. Compile it once into
# liblammps_ranking_profiler.so so the LAMMPS link line can pull it in.
PROFILER_SO="$PATCH_DIR/liblammps_ranking_profiler.so"
echo ""
echo ">>> Building $(basename "$PROFILER_SO") ..."
nvcc -O3 -std=c++17 \
    -gencode arch=compute_80,code=sm_80 \
    -Xcompiler -fPIC -shared \
    -I"$GPUCOMPRESS_DIR/include" \
    -I"$GPUCOMPRESS_DIR/benchmarks" \
    -I"$CUDA_DIR/include" \
    -L"$GPUCOMPRESS_DIR/build" -lgpucompress \
    -Xlinker -rpath -Xlinker "$GPUCOMPRESS_DIR/build" \
    -o "$PROFILER_SO" \
    "$PATCH_DIR/lammps_ranking_profiler.cu"
[ -f "$PROFILER_SO" ] || { echo "ERROR: profiler shared lib build failed"; exit 1; }

# ── Register fix in KOKKOS.cmake (idempotent) ──
KCMAKE="cmake/Modules/Packages/KOKKOS.cmake"
if grep -q "fix_gpucompress_kokkos.cpp" "$KCMAKE"; then
    echo ">>> KOKKOS.cmake already patched — skipping patch."
else
    echo ">>> Patching KOKKOS.cmake ..."
    if ! git apply --check "$PATCH_DIR/lammps-gpucompress.patch" 2>/dev/null; then
        echo "    git apply --check failed; falling back to in-place edit."
        # Insert "list(APPEND ...)" right before the line that sets the global
        # KOKKOS_PKG_SOURCES property. Insert RegisterFixStyle right after the
        # "register kokkos-only styles" comment.
        python3 - "$KCMAKE" <<'PYEOF'
import sys, re, pathlib
p = pathlib.Path(sys.argv[1])
src = p.read_text()
if "fix_gpucompress_kokkos.cpp" in src:
    sys.exit(0)
src = src.replace(
    'set_property(GLOBAL PROPERTY "KOKKOS_PKG_SOURCES" "${KOKKOS_PKG_SOURCES}")',
    '# GPUCompress fix (KOKKOS-only, no base style)\n'
    'list(APPEND KOKKOS_PKG_SOURCES ${KOKKOS_PKG_SOURCES_DIR}/fix_gpucompress_kokkos.cpp)\n\n'
    'set_property(GLOBAL PROPERTY "KOKKOS_PKG_SOURCES" "${KOKKOS_PKG_SOURCES}")',
    1)
src = src.replace(
    '# register kokkos-only styles\n',
    '# register kokkos-only styles\n'
    'RegisterFixStyle(${KOKKOS_PKG_SOURCES_DIR}/fix_gpucompress_kokkos.h)\n',
    1)
p.write_text(src)
PYEOF
    else
        git apply "$PATCH_DIR/lammps-gpucompress.patch"
    fi
fi

# ── Configure ──
BUILD_DIR="$LAMMPS_SRC/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

GPUC_INC="-I${GPUCOMPRESS_DIR}/include -I${GPUCOMPRESS_DIR}/examples -I${HDF5_DIR}/include -I${CUDA_DIR}/include"
GPUC_LINK="-L${GPUCOMPRESS_DIR}/examples -llammps_gpucompress_udf -L${PATCH_DIR} -llammps_ranking_profiler -L${GPUCOMPRESS_DIR}/build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress -L${HDF5_DIR}/lib -lhdf5 -L${CUDA_DIR}/lib64 -lcudart -Wl,-rpath,${GPUCOMPRESS_DIR}/build -Wl,-rpath,${GPUCOMPRESS_DIR}/examples -Wl,-rpath,${PATCH_DIR} -Wl,-rpath,${HDF5_DIR}/lib"

echo ""
echo ">>> Running cmake ..."
cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPKG_KOKKOS=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ARCH_${KOKKOS_ARCH}=ON \
    -DKOKKOS_PREC=${KOKKOS_PREC} \
    -DBUILD_MPI=ON \
    -DCMAKE_CXX_FLAGS="${GPUC_INC}" \
    -DCMAKE_EXE_LINKER_FLAGS="${GPUC_LINK}"

# ── Build ──
echo ""
echo ">>> Building (jobs=$JOBS) ..."
cmake --build . -j "$JOBS"

# ── Verify ──
LMP_BIN="$BUILD_DIR/lmp"
if [ -x "$LMP_BIN" ]; then
    echo ""
    echo "============================================================"
    echo "Build OK: $LMP_BIN"
    "$LMP_BIN" -h 2>&1 | head -3 || true
    echo ""
    echo "Verify gpucompress fix is registered:"
    if "$LMP_BIN" -h 2>&1 | grep -i gpucompress; then
        echo "  -> fix gpucompress present"
    else
        echo "  WARNING: 'gpucompress' not found in lmp -h output"
    fi
    echo "============================================================"
else
    echo "ERROR: lmp binary not produced at $LMP_BIN"
    exit 1
fi
