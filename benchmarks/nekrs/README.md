# nekRS Spectral-Element CFD Integration

GPU-accelerated lossless compression for [nekRS](https://github.com/Nek5000/nekRS)
GPU-capable CFD solver. nekRS uses OCCA for GPU abstraction; with the CUDA backend,
`occa::memory::ptr<void>()` returns raw CUDA device pointers that are passed directly
through the GPUCompress HDF5 VOL connector without CPU round-trips.

**Zero nekRS source code changes** — the integration is entirely at the case/UDF level,
using nekRS's standard extension mechanism (`UDF_ExecuteStep` + `udf.cmake`).

## Architecture

```
nekRS GPU Simulation (OCCA + CUDA)
    |
    v  (on checkpoint steps)
UDF_ExecuteStep()
    |
    |--- nrs->fluid->o_U.ptr<void>()  (velocity, CUDA device pointer)
    |--- nrs->fluid->o_P.ptr<void>()  (pressure, CUDA device pointer)
    |--- o_qcriterion.ptr()            (Q-criterion, CUDA device pointer)
    |
    v
gpucompress_nekrs_write_field()  (C bridge library)
    |
    v  (GPU device pointer passed to H5Dwrite)
HDF5 VOL Connector (libH5VLgpucompress.so)
    |--- detects CUDA device pointer via cudaPointerGetAttributes()
    |--- splits into 4 MiB chunks (H5Pset_chunk)
    |--- per chunk: stats -> NN inference -> algorithm selection -> nvCOMP compress
    v
H5Dwrite_chunk() --- pre-compressed bytes written to HDF5 file
```

## Files

### In GPUCompress (adapter + bridge, no nekRS dependency at build time)

| File | Description |
|------|-------------|
| `include/gpucompress_nekrs.h` | nekRS adapter C API (opaque handle, borrows GPU pointers) |
| `src/nekrs/nekrs_adapter.cu` | Adapter implementation (built into libgpucompress.so) |
| `examples/nekrs_gpucompress_udf.h` | C bridge API for nekRS UDFs |
| `examples/nekrs_gpucompress_udf.cpp` | Bridge implementation (compiled as libnekrs_gpucompress_udf.so) |
| `examples/nekrs_gpucompress_bridge.hpp` | Header-only bridge with HDF5 VOL + MPI support |
| `benchmarks/nekrs/README.md` | This file |
| `benchmarks/nekrs/patches/` | nekRS case files (UDF, par, udf.cmake) |

### In nekRS (zero source changes)

No nekRS source files are modified. The integration lives entirely in the case directory:

| File | Description |
|------|-------------|
| `tgv.udf` | Modified UDF with GPUCompress hooks in `UDF_Setup()` and `UDF_ExecuteStep()` |
| `tgv.par` | Parameter file for benchmark configuration |
| `udf.cmake` | Links GPUCompress libraries into the UDF shared library |

## Prerequisites

- CUDA 12.x with an NVIDIA GPU (tested on A100-40GB)
- MPI (OpenMPI or MPICH)
- CMake 3.21+
- Fortran compiler (gfortran)
- HDF5 1.14+ with VOL support
- nvCOMP 5.x
- GPUCompress (libgpucompress.so, libH5VLgpucompress.so, libH5Zgpucompress.so)

## Step 1: Build GPUCompress

```bash
cd /path/to/GPUCompress
mkdir -p build && cd build
cmake -S .. -B . \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DHDF5_ROOT=/path/to/hdf5-install
cmake --build . -j$(nproc)
```

## Step 2: Build the GPUCompress bridge library

```bash
cd /path/to/GPUCompress/examples
g++ -shared -fPIC -o libnekrs_gpucompress_udf.so nekrs_gpucompress_udf.cpp \
    -I../include -I/path/to/hdf5-install/include -I/usr/local/cuda/include \
    -L../build -lgpucompress -lH5VLgpucompress -lH5Zgpucompress \
    -L/path/to/hdf5-install/lib -lhdf5 \
    -L/usr/local/cuda/lib64 -lcudart \
    -Wl,-rpath,/path/to/GPUCompress/build \
    -Wl,-rpath,/path/to/hdf5-install/lib
```

## Step 3: Build nekRS

```bash
git clone https://github.com/Nek5000/nekRS.git
cd nekRS

CC=mpicc CXX=mpic++ FC=mpif77 cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=$HOME/.local/nekrs \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo

cmake --build ./build --target install -j$(nproc)

export NEKRS_HOME=$HOME/.local/nekrs
export PATH=$NEKRS_HOME/bin:$PATH
```

nekRS builds both fp64 (`nekrs`) and fp32 (`nekrs-fp32`) executables.
Use `nekrs-fp32` for compression benchmarks.

**Note:** The first run of any case will JIT-compile OCCA kernels (~5-10 min).
Subsequent runs reuse cached kernels and start in seconds.

## Step 4: Create a GPUCompress-enabled case

Copy the TGV (Taylor-Green Vortex) example from the nekRS installation:

```bash
cp -a $NEKRS_HOME/examples/tgv /path/to/mycase
cd /path/to/mycase
```

### Patch files and where to place them

The `benchmarks/nekrs/patches/` directory contains all files needed. **No nekRS source
files are modified** — everything goes into the case directory:

| Patch file | Place in case directory at | Action |
|-----------|--------------------------|--------|
| `tgv.udf` | `<case>/tgv.udf` | **Replace** — original UDF with GPUCompress hooks added |
| `tgv.par` | `<case>/tgv.par` | **Replace** — configured for benchmark (20 steps, checkpoint every 10) |
| `udf.cmake` | `<case>/udf.cmake` | **New file** — links GPUCompress libraries into UDF |

The original `tgv.re2` (mesh file) and `tgv.usr` (Nek5000 routines) are kept unchanged
from the nekRS installation.

```bash
# Copy all 3 case files
cp /path/to/GPUCompress/benchmarks/nekrs/patches/tgv.udf .
cp /path/to/GPUCompress/benchmarks/nekrs/patches/tgv.par .
cp /path/to/GPUCompress/benchmarks/nekrs/patches/udf.cmake .
```

### Configure paths

Edit `udf.cmake` or set environment variables to point to your installations:

```bash
export GPUCOMPRESS_DIR=/path/to/GPUCompress
export HDF5_DIR=/path/to/hdf5-install
```

Or edit the paths directly in `udf.cmake`:

```cmake
set(GPUCOMPRESS_DIR "/path/to/GPUCompress")
set(GPUCOMPRESS_BUILD "${GPUCOMPRESS_DIR}/build")
set(HDF5_DIR "/path/to/hdf5-install")
```

## Step 5: Run with GPUCompress

Set the library path:

```bash
export LD_LIBRARY_PATH=/path/to/GPUCompress/build:/path/to/GPUCompress/examples:/path/to/hdf5-install/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### TGV fp32 benchmark (recommended)

```bash
export GPUCOMPRESS_ALGO=auto
export GPUCOMPRESS_POLICY=ratio
export GPUCOMPRESS_VERIFY=1
export GPUCOMPRESS_WEIGHTS=/path/to/GPUCompress/neural_net/weights/model.nnwt

mpirun -np 2 nekrs-fp32 --setup tgv.par --backend CUDA --device-id 0
```

### TGV fp64 benchmark

```bash
mpirun -np 2 nekrs --setup tgv.par --backend CUDA --device-id 0
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPUCOMPRESS_ALGO` | `auto` | Algorithm: `auto`, `lz4`, `snappy`, `deflate`, `gdeflate`, `zstd`, `ans`, `cascaded`, `bitcomp` |
| `GPUCOMPRESS_POLICY` | `ratio` | NN ranking policy: `speed`, `balanced`, `ratio` |
| `GPUCOMPRESS_VERIFY` | `0` | Set to `1` for lossless round-trip verification |
| `GPUCOMPRESS_WEIGHTS` | (hardcoded path) | Path to NN model weights file (.nnwt) |

### Compare all algorithms

```bash
for ALGO in lz4 snappy deflate gdeflate zstd ans cascaded bitcomp auto; do
    rm -rf gpuc_step_*
    GPUCOMPRESS_ALGO=$ALGO mpirun -np 2 nekrs-fp32 --setup tgv.par --backend CUDA --device-id 0 \
        2>&1 | grep "GPUCompress"
    echo "=== $ALGO ==="
    du -sh gpuc_step_000010/
done
```

### Par file configuration

Key settings in `tgv.par`:

```ini
[GENERAL]
stopAt = numSteps
numSteps = 20              # Total simulation steps
checkpointControl = steps
checkpointInterval = 10    # GPUCompress writes at each checkpoint
```

Increase `numSteps` and `checkpointInterval` for longer simulations.

## Benchmark Results

Taylor-Green Vortex, polynomial order 7, ~12M mesh points/rank, fp32, A100-40GB:

### Fixed algorithms (lossless, verified)

| Algorithm | Compressed (MB) | Ratio | Status |
|-----------|----------------|-------|--------|
| Zstd | 295.3 | 1.54x | pass |
| Deflate | 326.9 | 1.39x | pass |
| GDeflate | 328.3 | 1.39x | pass |
| Snappy | 341.6 | 1.33x | pass |
| LZ4 | 344.8 | 1.32x | pass |
| Bitcomp | 348.3 | 1.31x | pass |
| Cascaded | 375.5 | 1.21x | pass |
| ANS | — | — | OOM (nekRS uses ~16GB, insufficient for ANS overhead) |

Original data: 455.6 MB per checkpoint (5 fields × 91.1 MB × 2 ranks).

### NN policy comparison (auto algorithm, lossless, verified)

| Policy | Compressed (MB) | Ratio |
|--------|----------------|-------|
| ratio | 295.9 | 1.54x |
| balanced | 383.8 | 1.19x |
| speed | 385.9 | 1.18x |

### Data characteristics

nekRS spectral-element data:
- **High-order fields** — polynomial order 7 means (7+1)^3 = 512 points per element
- **5 fields** — velocity (Ux, Uy, Uz), pressure (P), Q-criterion
- **11.9M points per rank** — ~45.6 MB per field per rank (fp32)
- **Smooth flow** — early TGV timesteps have regular velocity fields
- **Compute-heavy** — GPU solves Navier-Stokes; compression at I/O boundaries

## Output Format

Compressed checkpoints are written to `gpuc_step_<NNNNNN>/`:

```
gpuc_step_000010/
    Ux_rank0000.h5    # Velocity x-component (compressed HDF5)
    Ux_rank0001.h5
    Uy_rank0000.h5
    Uy_rank0001.h5
    Uz_rank0000.h5
    Uz_rank0001.h5
    P_rank0000.h5     # Pressure
    P_rank0001.h5
    Q_rank0000.h5     # Q-criterion
    Q_rank0001.h5
```

## How it works (zero nekRS source changes)

1. `tgv.udf` includes `nekrs_gpucompress_udf.h` and calls `gpucompress_nekrs_init()` in `UDF_Setup()`
2. In `UDF_ExecuteStep()`, when `nrs->checkpointStep` is true:
   - Casts `nrs->fluid->o_U` (OCCA memory) to `occa::memory`, calls `.ptr<void>()` to get CUDA pointer
   - Uses `nrs->fieldOffset` to compute pointer offsets for Ux, Uy, Uz components
   - Calls `gpucompress_nekrs_write_field()` for each field
3. `udf.cmake` links GPUCompress libraries into the JIT-compiled UDF shared library
4. nekRS's native checkpoint writer still runs after GPUCompress (for reference output)

The key OCCA → CUDA pointer extraction:
```cpp
occa::memory o_U_occa = nrs->fluid->o_U;
const void* d_Ux = o_U_occa.ptr<void>();
const void* d_Uy = (const char*)d_Ux + nrs->fieldOffset * sizeof(dfloat);
```

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `libnekrs_gpucompress_udf.so: cannot open` | Bridge library not found | Build bridge and set LD_LIBRARY_PATH |
| `ptr<void>()` template error | UDF compiled as C++ with OCCA headers | Use `occa::memory` intermediate: `occa::memory o = nrs->fluid->o_U; o.ptr<void>();` |
| Long first run (~10 min) | OCCA JIT kernel compilation | Normal; cached in `.cache/occa/` for subsequent runs |
| Low GPU utilization during setup | HYPRE preconditioner setup + JIT | Normal; GPU utilization increases during timestep loop |
| `MPI_Win_create` error with -np 1 | Single-rank MPI issue with HYPRE | Use `-np 2` or `--oversubscribe -np 2` |
