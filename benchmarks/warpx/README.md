# WarpX + GPUCompress Integration Benchmark

## Overview

WarpX is a Particle-In-Cell (PIC) plasma simulation built on AMReX.
This integration compresses WarpX's GPU-resident field data (E, B, J, rho)
and particle arrays in-situ using GPUCompress, avoiding device-to-host
round-trips.

## Prerequisites

- CUDA toolkit (11.0+)
- CMake 3.24+
- C++17 compiler
- MPI (optional, for multi-GPU runs)
- HDF5 with parallel support (optional, for VOL connector path)
- nvcomp (already part of GPUCompress build)

## Building WarpX from source

```bash
git clone https://github.com/BLAST-WarpX/warpx.git $HOME/src/warpx
cd $HOME/src/warpx

# CPU-only smoke test (no GPU/MPI complexity)
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DWarpX_COMPUTE=OMP \
  -DWarpX_MPI=OFF \
  -DWarpX_DIMS=3
cmake --build build-cpu -j $(nproc)

# CUDA build (production)
export CC=$(which gcc)
export CXX=$(which g++)
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=$(which g++)

cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DWarpX_COMPUTE=CUDA \
  -DWarpX_MPI=ON \
  -DWarpX_DIMS=3
cmake --build build-cuda -j $(nproc)
```

AMReX and PICSAR are fetched automatically by WarpX's CMake superbuild.

## Building GPUCompress with WarpX adapter

```bash
cd /path/to/GPUCompress
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j $(nproc)
```

The WarpX adapter (`warpx_adapter.cu`) is compiled into `libgpucompress.so`
alongside all other adapters.

## Integration approaches

### Approach A: Direct adapter API

Use `gpucompress_warpx.h` to borrow device pointers and compress directly:

```c
#include "gpucompress_warpx.h"

WarpxSettings settings = warpx_default_settings();
settings.data_type    = WARPX_DATA_EFIELD;
settings.n_components = 1;       /* one component per staggered MultiFab */
settings.element_size = 8;       /* AMReX Real = double */

gpucompress_warpx_t handle;
gpucompress_warpx_create(&handle, &settings);

/* Borrow device pointer from AMReX FArrayBox */
gpucompress_warpx_attach(handle, fab.dataPtr(), ncells);

/* Compress */
gpucompress_config_t config = gpucompress_default_config();
config.algorithm = GPUCOMPRESS_ALGO_AUTO;
gpucompress_stats_t stats;
gpucompress_compress_warpx(fab.dataPtr(), nbytes, d_output, &out_size, &config, &stats);

gpucompress_warpx_destroy(handle);
```

### Approach B: AMReX bridge with HDF5 VOL connector

Use `warpx_amrex_bridge.hpp` to write compressed HDF5 from GPU MultiFabs:

```cpp
#include "warpx_amrex_bridge.hpp"

hid_t fapl = gpucompress_warpx_bridge::init("weights.nnwt");

/* Write each field MultiFab */
gpucompress_warpx_bridge::write_field_compressed(
    "output/step_00100", "Ex", *Ex_mf, fapl,
    4*1024*1024,                        /* 4 MiB chunks */
    GPUCOMPRESS_ALGO_AUTO, 0.0, true);  /* lossless + verify */
```

## Validation

Run a stock WarpX 3D example to verify the build, then attach GPUCompress:

```bash
cd $HOME/src/warpx
# Pick a 3D example input
mpirun -np 4 build-cuda/bin/warpx Examples/Physics_applications/laser_acceleration/inputs_3d
```

Check for output in the configured diagnostic format (plotfiles by default).

## WarpX field data layout

WarpX uses a staggered Yee grid with separate MultiFabs per component:

| Field      | Components | Staggering      | Typical size per box |
|------------|-----------|-----------------|---------------------|
| Ex, Ey, Ez | 1 each    | Edge-centered   | ncells * sizeof(Real) |
| Bx, By, Bz | 1 each    | Face-centered   | ncells * sizeof(Real) |
| jx, jy, jz | 1 each    | Edge-centered   | ncells * sizeof(Real) |
| rho        | 1         | Node-centered   | ncells * sizeof(Real) |

Particle data is stored in AMReX Structure-of-Arrays (SoA) format:
x, y, z, ux, uy, uz, w (7 real components per particle).
