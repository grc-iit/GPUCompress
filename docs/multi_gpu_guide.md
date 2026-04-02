# Multi-GPU / Multi-Node Benchmark Guide

Covers running VPIC benchmarks across 1-4 nodes with 1-4 GPUs each.

Tested on:
- **NCSA Delta**: A100-SXM4-40GB, Cray MPICH 8.1.32, CUDA 12.8, x86_64

## Quick Start (Tested & Working)

### One-Time Setup (Login Node)

Build the project and install dependencies to the shared filesystem:

```bash
cd /u/$USER/GPUCompress

# 1. Build the project (requires a compute node with GPU)
srun --account=bekn-delta-gpu --partition=gpuA100x4-interactive \
  --nodes=1 --gpus=1 --ntasks=1 --cpus-per-task=16 --mem=64G --time=00:30:00 \
  bash scripts/install_dependencies.sh

# 2. Install nvcomp + HDF5 to shared filesystem (.deps/)
#    This avoids slow/unreliable downloads on compute nodes.
NVCOMP_INSTALL_DIR=/u/$USER/GPUCompress/.deps \
HDF5_INSTALL_DIR=/u/$USER/GPUCompress/.deps/hdf5 \
  bash scripts/install_dependencies.sh --node-local-only

# 3. Build the VPIC binary (one-time, on login node)
module load gcc-native/13.2 cray-mpich/8.1.32
cd benchmarks/vpic-kokkos && bash build_vpic_pm.sh && cd ../..
```

### Run VPIC Multi-GPU (Interactive)

```bash
# Allocate 1 node with 2 GPUs
salloc --account=bekn-delta-gpu --partition=gpuA100x4 \
  -N1 --gpus-per-node=2 --ntasks-per-node=2 --cpus-per-task=16 \
  --mem=0 --time=00:30:00

# On the compute node:
cd /u/$USER/GPUCompress
export LD_LIBRARY_PATH=/u/$USER/GPUCompress/.deps/lib:/u/$USER/GPUCompress/.deps/hdf5/lib:$LD_LIBRARY_PATH
export SLURM_JOB_ID=${SLURM_JOB_ID:-1}
MPI_NP=2 GPUS_PER_NODE=2 BENCHMARKS=vpic VPIC_NX=160 CHUNK_MB=16 \
  TIMESTEPS=5 VERIFY=1 POLICIES=balanced \
  bash benchmarks/benchmark.sh

# When done:
exit
```

### Generate Plots (Login Node)

Compute nodes may not have matplotlib. Generate plots from the login node:

```bash
cd /u/$USER/GPUCompress
VPIC_DIR=benchmarks/vpic-kokkos/results/eval_NX160_chunk16mb_ts5/balanced_w1-1-1 \
  python3 benchmarks/plots/generate_dataset_figures.py \
  --dataset vpic --policy balanced_w1-1-1
```

Plots go to `benchmarks/results/per_dataset/vpic/`.

## How It Works

### Dependencies on Shared Filesystem

Compute nodes on Delta have unreliable external internet. Instead of downloading deps on each compute node, we:

1. Build nvcomp + HDF5 once on the login node into `.deps/` (shared filesystem)
2. Set `LD_LIBRARY_PATH` to point to `.deps/` on compute nodes
3. No `/tmp` copies, no per-node installs, no downloads on compute nodes

```bash
# This is all you need on the compute node:
export LD_LIBRARY_PATH=/u/$USER/GPUCompress/.deps/lib:/u/$USER/GPUCompress/.deps/hdf5/lib:$LD_LIBRARY_PATH
```

The VPIC binary has `-Wl,-rpath,/tmp/lib` baked in, but `LD_LIBRARY_PATH` takes precedence over rpath. Libs not found in `/tmp` fall through to `.deps/`.

### MPI Execution Flow

```
benchmark.sh (on compute node)
  |
  MPI_NP=2, LAUNCHER=auto
  |
  mpi_launch() detects SLURM_JOB_ID → uses srun
  |
  srun --ntasks=2 --gpus-per-task=1 vpic_binary
  |
  +-- rank 0: GPU 0, Y-domain [0, ny/2)
  +-- rank 1: GPU 1, Y-domain [ny/2, ny)
  |
  Both run in parallel, each writes rank-suffixed CSVs
  |
  Post-processing (rank 0 only): merge CSVs, split per-policy, generate plots
```

### VPIC Domain Decomposition

VPIC splits the grid along the Y-axis: `define_periodic_grid(..., 1, nproc(), 1)`.

**VPIC_NX must be divisible by the number of MPI ranks.**

`DATA_MB` auto-computes `VPIC_NX`, which may not be divisible. Always set `VPIC_NX` explicitly for multi-GPU:

| Ranks | Good VPIC_NX values |
|-------|-------------------|
| 1 | Any (e.g., 159, 200) |
| 2 | Even: 100, 160, 200, 256 |
| 4 | Div by 4: 100, 160, 200, 256 |
| 8 | Div by 8: 160, 200, 256 |
| 16 | Div by 16: 160, 256 |

Data size per rank: `(VPIC_NX+2)^3 * 64 / nproc` bytes. For VPIC_NX=160, 2 ranks: ~134 MB/rank.

## Scale Examples

### 1 Node, 2 GPUs

```bash
salloc --account=bekn-delta-gpu --partition=gpuA100x4 \
  -N1 --gpus-per-node=2 --ntasks-per-node=2 --cpus-per-task=16 \
  --mem=0 --time=00:30:00

cd /u/$USER/GPUCompress
export LD_LIBRARY_PATH=/u/$USER/GPUCompress/.deps/lib:/u/$USER/GPUCompress/.deps/hdf5/lib:$LD_LIBRARY_PATH
export SLURM_JOB_ID=${SLURM_JOB_ID:-1}
MPI_NP=2 GPUS_PER_NODE=2 BENCHMARKS=vpic VPIC_NX=160 CHUNK_MB=16 \
  TIMESTEPS=5 VERIFY=1 POLICIES=balanced \
  bash benchmarks/benchmark.sh
```

### 1 Node, 4 GPUs

```bash
salloc --account=bekn-delta-gpu --partition=gpuA100x4 \
  -N1 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=16 \
  --mem=0 --time=00:30:00

cd /u/$USER/GPUCompress
export LD_LIBRARY_PATH=/u/$USER/GPUCompress/.deps/lib:/u/$USER/GPUCompress/.deps/hdf5/lib:$LD_LIBRARY_PATH
export SLURM_JOB_ID=${SLURM_JOB_ID:-1}
MPI_NP=4 GPUS_PER_NODE=4 BENCHMARKS=vpic VPIC_NX=200 CHUNK_MB=16 \
  TIMESTEPS=5 VERIFY=1 POLICIES=balanced \
  bash benchmarks/benchmark.sh
```

### 2 Nodes, 2 GPUs Each (4 ranks)

```bash
salloc --account=bekn-delta-gpu --partition=gpuA100x4 \
  -N2 --gpus-per-node=2 --ntasks-per-node=2 --cpus-per-task=16 \
  --mem=0 --time=00:30:00

# From login node (NOT ssh into compute node):
cd /u/$USER/GPUCompress
srun --ntasks=4 --gpus-per-task=1 bash -c "\
  export LD_LIBRARY_PATH=/u/\$USER/GPUCompress/.deps/lib:/u/\$USER/GPUCompress/.deps/hdf5/lib:\$LD_LIBRARY_PATH; \
  cd /u/\$USER/GPUCompress; \
  BENCHMARKS=vpic VPIC_NX=200 CHUNK_MB=16 TIMESTEPS=5 \
  VERIFY=1 POLICIES=balanced \
  bash benchmarks/benchmark.sh"
```

For multi-node, use `srun` from the login node (not SSH). `benchmark.sh` auto-detects `SLURM_STEP_ID` and sets `LAUNCHER=none` to avoid nesting srun.

### 2 Nodes, 4 GPUs Each (8 ranks)

```bash
salloc --account=bekn-delta-gpu --partition=gpuA100x4 \
  -N2 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=16 \
  --mem=0 --time=00:30:00

cd /u/$USER/GPUCompress
srun --ntasks=8 --gpus-per-task=1 bash -c "\
  export LD_LIBRARY_PATH=/u/\$USER/GPUCompress/.deps/lib:/u/\$USER/GPUCompress/.deps/hdf5/lib:\$LD_LIBRARY_PATH; \
  cd /u/\$USER/GPUCompress; \
  BENCHMARKS=vpic VPIC_NX=256 CHUNK_MB=16 TIMESTEPS=5 \
  VERIFY=1 POLICIES=balanced \
  bash benchmarks/benchmark.sh"
```

## Output Structure

```
benchmarks/vpic-kokkos/results/eval_NX160_chunk16mb_ts5/
  benchmark_vpic_deck_rank0.csv                # rank 0 aggregate
  benchmark_vpic_deck_rank1.csv                # rank 1 aggregate
  benchmark_vpic_deck_timesteps_rank0.csv      # rank 0 per-timestep
  benchmark_vpic_deck_timesteps_rank1.csv      # rank 1 per-timestep
  benchmark_vpic_deck_timesteps.csv            # merged (all ranks)
  benchmark_vpic_deck_aggregate_multi_rank.csv # aggregate throughput across ranks
  vpic_benchmark.log                           # combined stdout/stderr
  balanced_w1-1-1/                             # per-policy split
    benchmark_vpic_deck_timesteps.csv          # merged, policy-filtered
    benchmark_vpic_deck.csv                    # per-phase averages

benchmarks/results/per_dataset/vpic/           # plots (generated on login node)
  eval_NX160_chunk16mb_ts5/balanced_w1-1-1/
    1_summary.png
    3_algorithm_evolution.png
    ...
```

### Aggregate Multi-Rank CSV

`benchmark_vpic_deck_aggregate_multi_rank.csv` computes distributed throughput:

- `avg_agg_write_mibps`: total data across all ranks / max write time (true system throughput)
- `avg_per_rank_write_mibps`: average single-rank throughput
- Near-linear scaling means `agg ≈ n_ranks × per_rank`

## Verifying Results

```bash
# Check all ranks produced output
ls benchmarks/vpic-kokkos/results/eval_NX160_chunk16mb_ts5/benchmark_vpic_deck_rank*.csv

# Check for errors (mismatches column should be 0)
grep "mismatches" benchmarks/vpic-kokkos/results/eval_NX160_chunk16mb_ts5/benchmark_vpic_deck_rank0.csv

# Compare rank throughputs (should be similar)
head -3 benchmarks/vpic-kokkos/results/eval_NX160_chunk16mb_ts5/benchmark_vpic_deck_rank0.csv
head -3 benchmarks/vpic-kokkos/results/eval_NX160_chunk16mb_ts5/benchmark_vpic_deck_rank1.csv
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARKS` | `grayscott,vpic,sdrbench` | Benchmarks to run |
| `VPIC_NX` | auto from DATA_MB | Grid cells per dimension (**must be divisible by MPI_NP**) |
| `DATA_MB` | `512` | Target data size (used if VPIC_NX not set) |
| `CHUNK_MB` | `16` | Compression chunk size |
| `TIMESTEPS` | `50` | Number of simulation timesteps |
| `POLICIES` | `balanced,ratio,speed` | NN compression policies |
| `VERIFY` | `1` | Bitwise decompression verification |
| `MPI_NP` | `1` or auto | Total MPI ranks |
| `GPUS_PER_NODE` | `1` or auto | GPUs per node |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Bad resolution (NxNxN) for domain decomposition` | VPIC_NX not divisible by MPI_NP. Set VPIC_NX explicitly (e.g., 160, 200, 256) |
| `libnvcomp.so.5: cannot open shared object` | `export LD_LIBRARY_PATH=/u/$USER/GPUCompress/.deps/lib:$LD_LIBRARY_PATH` |
| `libhdf5.so.320: cannot open shared object` | `export LD_LIBRARY_PATH=/u/$USER/GPUCompress/.deps/hdf5/lib:$LD_LIBRARY_PATH` |
| VPIC hangs (no output) | Use `salloc` + `srun`, not `srun --pty bash`. Set `SLURM_JOB_ID` if SSH'd directly. |
| `Out Of Memory` | Use `--mem=0` in salloc. Reduce VPIC_NX. |
| Nested srun error | Already inside srun. `benchmark.sh` auto-detects and sets `LAUNCHER=none`. |
| No plots generated | Run plot generation from login node where matplotlib is available |
| `n_ranks=1` in aggregate CSV | Old single-GPU results in the directory. Delete results dir and rerun. |
| SSL/download errors on compute node | Don't install deps on compute nodes. Use `.deps/` on shared filesystem. |

## Delta vs DeltaAI

| | Delta | DeltaAI |
|---|-------|---------|
| GPU | A100-SXM4-40GB | GH200 (120GB HBM3) |
| Arch | x86_64 (sm_80) | aarch64 (sm_90) |
| Account | `bekn-delta-gpu` | `bekn-dtai-gh` |
| Partition | `gpuA100x4` | `ghx4` |
| Repo path | `/u/$USER/GPUCompress` | `/work/hdd/bekn/$USER/GPUCompress` |
| GPUs/node | 4x A100 | 4x GH200 |
| CUDA_ARCH | 80 | 90 |
