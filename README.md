# GPUCompress

A GPU-accelerated compression library with neural network–driven algorithm selection and online reinforcement learning, designed for HPC in-situ I/O.

## Key Innovation

GPUCompress replaces static compression choices with a lightweight neural network that evaluates **all 32 compression configurations** (8 algorithms × 2 preprocessing options) in a single GPU kernel (~0.22 ms), selecting the best one per data chunk based on learned data characteristics.  An online SGD loop adapts the model in real time — demonstrated MAPE drops from ~700% to ~10% within 20 timesteps on unseen VPIC plasma data.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Application / HDF5                    │
│         (Gray-Scott, VPIC-Kokkos, SDRBench, ...)        │
├─────────────────────────────────────────────────────────┤
│              HDF5 VOL Connector (2,744 LOC)             │
│   Intercepts H5Dwrite/H5Dread, detects GPU pointers,   │
│   routes to GPU-native compress/decompress pipeline     │
├─────────────────────────────────────────────────────────┤
│                   GPUCompress C API                      │
│   gpucompress_compress_gpu() / gpucompress_decompress() │
├────────┬────────┬──────────┬────────────┬───────────────┤
│ Stats  │   NN   │ Cost     │ Compress/  │   Online      │
│Kernels │Infer-  │ Model    │ Decompress │   Learning    │
│entropy,│ence    │log-space │ via nvCOMP │   SGD +       │
│MAD,    │15→128→ │policy-   │ 8 algos    │   Exploration │
│2nd-deriv│128→4  │controlled│            │               │
├────────┴────────┴──────────┴────────────┴───────────────┤
│              NVIDIA nvCOMP 5.1.0 + CUDA 12.8+          │
└─────────────────────────────────────────────────────────┘
```

## Compression Algorithms

All 8 algorithms are GPU-accelerated via nvCOMP:

| Algorithm  | Speed     | Ratio | Notes                              |
|------------|-----------|-------|------------------------------------|
| LZ4        | Fastest   | Low   | General-purpose, always competitive|
| Snappy     | Very Fast | Low   | Byte-oriented, fast decode         |
| Deflate    | Medium    | Med   | CPU-style, slower on GPU           |
| Gdeflate   | Medium    | Med   | GPU-optimized deflate variant      |
| Zstd       | Slow      | High  | Best ratio for low-entropy data    |
| ANS        | Slow      | High  | Entropy coding, structured data    |
| Cascaded   | Slow      | High  | Floating-point specific            |
| Bitcomp    | Slow      | High  | Bit-level compression              |

**Action encoding:** `action = algo_idx + quant*8 + shuffle*16` → 32 total configurations per chunk.

## Neural Network Model

### Architecture

```
Input [15 features] → ReLU [128] → ReLU [128] → Output [4 predictions]
```

**15 input features:**
- One-hot algorithm encoding (8 values)
- Quantization flag, shuffle flag
- `log10(error_bound)`, `log2(data_size)`
- Shannon entropy, normalized MAD, normalized 2nd derivative

**4 output predictions:**
- `log1p(compression_time_ms)`
- `log1p(decompression_time_ms)`
- `log1p(compression_ratio)` — primary ranking target
- PSNR (clamped to 120 dB)

**Parameters:** ~19,076 floats (~76 KB), evaluated for all 32 configs in parallel via 32-thread GPU kernel.

### Cost Model (Scale-Invariant Ranking)

```
cost = α · log(ct + γ · dt) + β · log(data_size / (ratio · bw_eff)) − δ · log(ratio)
```

Policy presets:
- **Speed** (α=1, β=0, δ=0): minimize compute time
- **Balanced** (α=1, β=1, δ=0.5): balance compute + I/O + ratio
- **Ratio-First** (α=0.3, β=1, δ=1): maximize compression

## Online Learning System

Three-level adaptation that runs during compression:

1. **Experience Logging** — every chunk records (action, features, actual ratio/time)
2. **Exploration** — when |predicted − actual| / actual > threshold, tries K alternative configurations and picks the best
3. **SGD Reinforcement** — updates output-layer weights using measured ground truth; gradient-clipped, heads-only for stability

Convergence on VPIC data: MAPE ~700% (cold start) → ~10% after 20 timesteps.

## HDF5 Integration

### VOL Connector (`src/hdf5/H5VLgpucompress.cu`)

Transparently intercepts `H5Dwrite()`/`H5Dread()`:
- Detects GPU device pointers via `cudaPointerGetAttributes()`
- GPU data → compressed on GPU, written via `H5VL_NATIVE_DATASET_CHUNK_WRITE`
- Host data → forwarded to underlying VOL unchanged
- Pool of 8 reusable compression contexts (thread-safe)
- Per-chunk diagnostic history (timing, predictions, exploration results)

### HDF5 Filter (`src/hdf5/H5Zgpucompress.c`)

Standard filter plugin (ID 305) for chunk-level compression in existing HDF5 workflows.

## Project Structure

```
GPUCompress/
├── include/                    # Public C API headers
│   ├── gpucompress.h           #   Main API (687 lines)
│   ├── gpucompress_hdf5.h      #   HDF5 filter
│   ├── gpucompress_hdf5_vol.h  #   HDF5 VOL connector
│   ├── gpucompress_vpic.h      #   VPIC integration
│   └── gpucompress_grayscott.h #   Gray-Scott simulation
│
├── src/
│   ├── api/                    # Core implementation (~7K lines)
│   │   ├── gpucompress_api.cpp       # Init, compress, decompress
│   │   ├── gpucompress_compress.cpp  # GPU compression pipeline + NN
│   │   ├── gpucompress_learning.cpp  # Online learning, SGD, exploration
│   │   ├── gpucompress_pool.cpp      # Compression context pool
│   │   └── gpucompress_diagnostics.cpp # Per-chunk diagnostic history
│   ├── nn/nn_gpu.cu            # GPU NN inference + SGD kernels
│   ├── stats/                  # Feature extraction (entropy, MAD, 2nd deriv)
│   ├── compression/            # nvCOMP wrapper factory (8 algorithms)
│   ├── preprocessing/          # Byte shuffle + quantization kernels
│   ├── selection/heuristic.cu  # Entropy-threshold baseline selector
│   ├── hdf5/                   # VOL connector + filter plugin
│   ├── gray-scott/             # Reaction-diffusion simulation
│   ├── vpic/                   # VPIC plasma physics adapter
│   └── cli/                    # Command-line compress/decompress tools
│
├── neural_net/
│   ├── core/                   # PyTorch model, data loading, configs
│   ├── training/               # Train, cross-validate, retrain
│   ├── inference/              # CPU-side prediction + evaluation
│   ├── export/                 # PyTorch → binary .nnwt export
│   ├── xgboost/                # Alternative model exploration
│   ├── weights/model.nnwt      # Pre-trained weights (shipped)
│   └── docs/                   # Architecture, tutorial, execution flow
│
├── benchmarks/
│   ├── grayscott/              # Gray-Scott benchmark driver
│   ├── vpic-kokkos/            # VPIC-Kokkos benchmark driver
│   ├── sdrbench/               # SDRBench (Hurricane, Nyx, CESM) driver
│   └── visualize.py            # Publication-quality figure generation
│
├── tests/
│   ├── regression/             # 50+ regression tests
│   └── run_all_tests.sh        # Test runner
│
├── cmake/                      # Modular CMake build system
├── scripts/                    # Setup, dependency install, smoke tests
├── CostModel.md                # Cost model design rationale
└── roadMapToSC.md              # SC submission roadmap
```

## Benchmark Pipeline

An 8-phase evaluation pipeline compares all system modes:

| Phase | Description |
|-------|-------------|
| `no-comp` | Uncompressed baseline |
| `cpu-zstd` | CPU zstd reference |
| `fixed-lz4` | Always LZ4 (speed extreme) |
| `fixed-gdeflate` | Always Gdeflate (GPU-optimized middle) |
| `fixed-zstd` | Always Zstd (ratio extreme) |
| `entropy-heuristic` | Rule-based selector (baseline) |
| `best` | Exhaustive search K=31 (oracle upper bound) |
| `nn` | NN inference (static, no learning) |
| `nn-rl` | NN + SGD (online learning) |
| `nn-rl+exp50` | NN + SGD + exploration (full system) |

Three benchmark drivers:
- **Gray-Scott** — reaction-diffusion simulation (controllable data complexity)
- **VPIC-Kokkos** — plasma physics particle-in-cell (real HPC workload)
- **SDRBench** — Hurricane Isabel, Nyx cosmology, CESM-ATM climate datasets

Each produces per-phase, per-timestep, and per-chunk CSV files. Multi-run support (`--runs N`) provides error bars.

## Building

```bash
# Prerequisites: CUDA 12.8+, nvCOMP 5.1.0
# Optional: HDF5 (for VOL/filter), PyTorch (for NN training)

cmake -B build \
  -DNVCOMP_PREFIX=/path/to/nvcomp \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j

# Run smoke test
bash scripts/smoke_test.sh
```

### Build Targets

| Target | Output | Description |
|--------|--------|-------------|
| `gpucompress` | `libgpucompress.so` | Core compression library |
| `gpu_compress` | CLI tool | Command-line compressor |
| `gpu_decompress` | CLI tool | Command-line decompressor |
| `H5Zgpucompress` | `libH5Zgpucompress.so` | HDF5 filter plugin |
| `H5VLgpucompress` | `libH5VLgpucompress.so` | HDF5 VOL connector |

## Running Benchmarks

```bash
# Gray-Scott
L=256 TIMESTEPS=50 RUNS=5 bash benchmarks/grayscott/run_gs_eval.sh

# VPIC
NX=64 TIMESTEPS=50 RUNS=3 bash benchmarks/vpic-kokkos/run_vpic_eval.sh

# SDRBench
bash benchmarks/sdrbench/run_sdrbench_eval.sh

# Generate figures
python3 benchmarks/visualize.py --view summary --view timesteps
```

## Quick API Example

```c
#include "gpucompress.h"

// Initialize with pre-trained NN weights
gpucompress_init("neural_net/weights/model.nnwt");

// Enable online learning
gpucompress_enable_online_learning();
gpucompress_set_exploration(1);

// Configure cost model (balanced policy)
gpucompress_set_ranking_weights(1.0, 1.0, 0.5);

// Compress GPU data — NN selects best algorithm automatically
gpucompress_config_t cfg = {.error_bound = 0.0, .chunk_size = 16*1024*1024};
gpucompress_stats_t stats;
gpucompress_compress_gpu(d_input, n_bytes, d_output, &out_size, &cfg, &stats, stream);

// Decompress — algorithm auto-detected from header
gpucompress_decompress_gpu(d_compressed, comp_size, d_output, &out_size, stream);
```

## Tests

```bash
# Run all tests
bash tests/run_all_tests.sh

# Run specific regression test
./build/tests/test_nn_ratio_prediction
```

50+ regression tests covering concurrency, memory safety, NN accuracy, SGD convergence, integer overflow, and RAII cleanup.

## Documentation

- [`CostModel.md`](CostModel.md) — Cost model design and policy rationale
- [`roadMapToSC.md`](roadMapToSC.md) — SC submission roadmap and status
- [`SC_BENCHMARK_GAPS.md`](SC_BENCHMARK_GAPS.md) — Benchmark gap analysis
- [`neural_net/docs/ARCHITECTURE.md`](neural_net/docs/ARCHITECTURE.md) — NN architecture details
- [`neural_net/docs/NN_EXECUTION_FLOW.md`](neural_net/docs/NN_EXECUTION_FLOW.md) — 5-timestep execution walkthrough
- [`neural_net/docs/TUTORIAL.md`](neural_net/docs/TUTORIAL.md) — Training tutorial
