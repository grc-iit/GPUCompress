# GPUCompress — Quickstart Guide

Install dependencies, build, and run tests on an NCSA Delta GPU node.

---

## 1. Allocate a GPU Node

Start a tmux session first so you can reconnect if disconnected:

```bash
tmux new -s gpu
```

Request an interactive GPU node:

```bash
srun --account=bekn-delta-gpu --partition=gpuA100x4-interactive \
     --nodes=1 --gpus=1 --tasks=1 --cpus-per-task=16 \
     --mem=64g --time=00:30:00 --pty bash
```

> If disconnected, reconnect with: `tmux attach -t gpu`

---

## 2. Install Dependencies

Dependencies install to node-local `/tmp` and must be reinstalled on each new node.

```bash
cd /u/imuradli/GPUCompress
bash scripts/install_dependencies.sh
```

This installs:
- **nvcomp 5.1.0** → `/tmp/include`, `/tmp/lib`
- **HDF5 2.0.0** → `/tmp/hdf5-install`
- Builds the entire project → `build/`

Override CUDA architecture if needed (default is sm_80 for A100):

```bash
CUDA_ARCH=90 bash scripts/install_dependencies.sh   # H100
```

---

## 3. Build

### Standard build

```bash
cd /u/imuradli/GPUCompress
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j$(nproc)
```

### Build with custom dependency paths

nvCOMP and HDF5 paths are configurable via CMake cache variables:

```bash
cmake -B build \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DNVCOMP_PREFIX=/path/to/nvcomp \
  -DHDF5_VOL_PREFIX=/path/to/hdf5
cmake --build build -j$(nproc)
```

### Clean rebuild

```bash
rm -rf build
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j$(nproc)
```

> **Important:** Always pass `-DCMAKE_CUDA_ARCHITECTURES=80` (or higher) explicitly.
> A stale cache with `sm_52` will cause `atomicAdd(double*, double)` compile errors.

---

## 4. Set Up Environment

```bash
source scripts/setup_env.sh
```

Or manually:

```bash
export LD_LIBRARY_PATH=/tmp/lib:/tmp/hdf5-install/lib:$PWD/build:$LD_LIBRARY_PATH
```

---

## 5. Verify the Build

```bash
ls build/libgpucompress.so build/libH5Zgpucompress.so build/libH5VLgpucompress.so
```

CLI tools (`gpu_compress`, `gpu_decompress`) are only built if cuFile/GDS is available:

```bash
ls build/gpu_compress build/gpu_decompress 2>/dev/null || echo "CLI tools not built (cuFile not found — this is OK)"
```

---

## 6. Run Tests

### Shell-based tests (no GPU required)

Static analysis regression tests that verify code fixes:

```bash
bash tests/run_all_tests.sh
```

### Full test suite (GPU required)

Includes CUDA runtime tests, unit tests, HDF5 tests, and VOL tests:

```bash
bash tests/run_all_tests.sh --gpu
```

### Quick smoke test

```bash
./build/test_quantization_roundtrip
./build/test_nn_pipeline
```

### Individual test categories

**Unit / Regression:**

```bash
./build/test_quantization_roundtrip
./build/test_nn
./build/test_nn_pipeline
./build/test_nn_reinforce
./build/test_sgd_weight_update
./build/test_bug3_sgd_gradients
./build/test_bug4_format_string
./build/test_bug5_truncated_nnwt
./build/test_bug7_concurrent_quantize
./build/test_bug8_sgd_concurrent
./build/test_perf2_sort_speedup
./build/test_perf4_batched_dh
./build/test_perf14_atomic_double
```

**HDF5 filter:**

```bash
./build/test_hdf5_configs
./build/test_h5z_8mb
./build/test_f9_transfers
./build/test_design6_chunk_tracker
```

**VOL connector:**

```bash
./build/test_vol_gpu_write
./build/test_vol2_gpu_fallback
./build/test_vol_8mb
./build/test_vol_comprehensive
./build/test_correctness_vol
```

---

## 7. Run Gray-Scott Benchmark

Runs a Gray-Scott reaction-diffusion simulation on the GPU, then benchmarks
writing the 3D field to HDF5 via the GPUCompress VOL connector under five
compression phases (no-comp, static, nn, nn-rl, nn-rl+exploration).

```bash
./build/benchmark_grayscott_vol neural_net/weights/model.nnwt
```

**Options:**

```bash
# 64 MB dataset, 4 MB chunks (default)
./build/benchmark_grayscott_vol neural_net/weights/model.nnwt --L 256 --chunk-mb 4

# 4 GB dataset, 64 MB chunks
./build/benchmark_grayscott_vol neural_net/weights/model.nnwt --L 1000 --chunk-mb 64

# Custom simulation parameters
./build/benchmark_grayscott_vol neural_net/weights/model.nnwt --L 512 --steps 1000 --F 0.04 --k 0.06075
```

**Dataset size reference:**

| `--L` | Dataset Size |
|-------|-------------|
| 128   | 8 MB        |
| 256   | 64 MB       |
| 512   | 512 MB      |
| 640   | 1 GB        |
| 1000  | 4 GB        |
| 1260  | 8 GB        |

---

## 8. VPIC Benchmark (Real Harris Sheet Simulation)

Runs a real VPIC-Kokkos Harris sheet reconnection simulation, then benchmarks
GPU-resident field compression through the GPUCompress VOL connector.

### Prerequisites

- VPIC-Kokkos built with CUDA support at `/u/imuradli/vpic-kokkos`
- GPUCompress already built

### Build the deck

```bash
cd /u/imuradli/vpic-kokkos
bash /u/imuradli/GPUCompress/scripts/build_vpic_benchmark.sh
```

### Run the deck

```bash
export LD_LIBRARY_PATH=/u/imuradli/GPUCompress/build:/tmp/lib:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
export GPUCOMPRESS_WEIGHTS=/u/imuradli/GPUCompress/neural_net/weights/model.nnwt
./vpic_benchmark_deck.Linux
```

> **Note:** `mpirun` is not available on Cray/MPICH nodes. Run the binary directly
> for single-rank, or use `srun -n 1 ./vpic_benchmark_deck.Linux` under SLURM.

### Output

- Console: per-phase ratio, write/read MB/s, verification, SGD/exploration stats
- CSV: `benchmark_vpic_deck_results/benchmark_vpic_deck.csv`
- Temp files: `/tmp/bm_vpic_*.h5` (removed after each phase)

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `libnvcomp.so.5: cannot open shared object file` | `export LD_LIBRARY_PATH=/tmp/lib:$LD_LIBRARY_PATH` |
| `libhdf5.so.320: cannot open shared object file` | `export LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH` |
| `atomicAdd(double*, double)` compile error | `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=80` |
| `cuFile not found — skipping CLI tools` | Normal on systems without GDS. Library and tests still work. |
| Dependencies missing on new node | Re-run `bash scripts/install_dependencies.sh` |
| tmux session lost | `tmux attach -t gpu` |
| h5diff can't read compressed file | `export HDF5_PLUGIN_PATH=$PWD/build` |
