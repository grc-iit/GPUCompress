#!/bin/bash
# ============================================================
# GPUCompress Multi-GPU Benchmark — DeltaAI (GH200)
#
# Submit: sbatch benchmarks/slurm/deltaai_benchmark.sbatch
#
# Customize via environment variables before sbatch:
#   NODES=2 GPUS=4 BENCHMARKS=vpic sbatch benchmarks/slurm/deltaai_benchmark.sbatch
#
# DeltaAI specifics:
#   - ARM aarch64, GH200 (sm_90), CUDA 12.8
#   - 4x GH200 per node, 120GB HBM3 each
#   - Shared filesystem: /work/hdd/bekn/YOUR_USERNAME/
#   - Node-local /tmp: 3.9TB (deleted after job)
#   - No mpirun — use srun for everything
# ============================================================

#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=gpucompress-bench
#SBATCH --output=gpucompress-bench-%j.out
#SBATCH --error=gpucompress-bench-%j.err
#SBATCH --time=02:00:00

# ── Node/GPU configuration ──
# Defaults below are for 1 node × 4 GPUs. Override via sbatch CLI flags:
#   sbatch -N2 --gpus-per-node=2 --ntasks-per-node=2 benchmarks/slurm/deltaai_benchmark.sbatch
# Or use wrapper scripts: bash benchmarks/slurm/deltaai_2n2g.sh
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0

# ── Progress logging helper ──
_step=0
log_step() {
    _step=$((_step + 1))
    echo ""
    echo "[STEP $_step] $(date '+%H:%M:%S') $1"
    echo "────────────────────────────────────────"
}

# ── Resolve configuration ──
NODES=${NODES:-${SLURM_JOB_NUM_NODES:-1}}
GPUS_PER_NODE=${GPUS:-${SLURM_GPUS_ON_NODE:-4}}
TOTAL_RANKS=$((NODES * GPUS_PER_NODE))

# ── Project paths (shared filesystem) ──
# Adjust GPU_DIR to your clone location on /work/hdd/bekn/
GPU_DIR="${GPU_DIR:-/work/hdd/bekn/${USER}/GPUCompress}"
cd "$GPU_DIR" || { echo "ERROR: GPU_DIR=$GPU_DIR not found"; exit 1; }

# ── Environment setup ──
source scripts/setup_env.sh
export LD_LIBRARY_PATH="$GPU_DIR/build:$LD_LIBRARY_PATH"

echo "============================================================"
echo "  GPUCompress Multi-GPU Benchmark"
echo "============================================================"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Started     : $(date)"
echo "  Nodes       : $NODES ($SLURM_JOB_NODELIST)"
echo "  GPUs/node   : $GPUS_PER_NODE"
echo "  Total ranks : $TOTAL_RANKS"
echo "  Partition   : $SLURM_JOB_PARTITION"
echo "  GPU_DIR     : $GPU_DIR"
echo "  Architecture: $(uname -m)"
echo "  Benchmarks  : ${BENCHMARKS:-vpic}"
echo "  DATA_MB     : ${DATA_MB:-256}"
echo "  TIMESTEPS   : ${TIMESTEPS:-25}"
echo "  VERIFY      : ${VERIFY:-0}"
echo "============================================================"

# ── Step 1: Verify build ──
log_step "Verifying build on shared filesystem ..."
if [ ! -f "$GPU_DIR/build/libgpucompress.so" ]; then
    echo "  ERROR: Build not found at $GPU_DIR/build/libgpucompress.so"
    echo "  Run: CUDA_ARCH=90 bash scripts/install_dependencies.sh"
    exit 1
fi
echo "  OK: libgpucompress.so found"
ls -lh "$GPU_DIR/build/libgpucompress.so"

# ── Step 2: Install node-local deps ──
log_step "Installing node-local deps (nvcomp + HDF5) on $NODES node(s) ..."
echo "  Running: srun --ntasks=$NODES --ntasks-per-node=1 bash scripts/install_dependencies.sh --node-local-only"
_t0=$(date +%s)
srun --ntasks=$NODES --ntasks-per-node=1 \
     bash scripts/install_dependencies.sh --node-local-only
_rc=$?
_t1=$(date +%s)
echo "  Completed in $((_t1 - _t0))s (exit code: $_rc)"
if [ "$_rc" -ne 0 ]; then
    echo "  ERROR: Node-local dep install failed on one or more nodes"
    exit 1
fi

# ── Step 3: Verify VPIC binary ──
log_step "Checking VPIC binary ..."
VPIC_BIN="$GPU_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux"
if [ -f "$VPIC_BIN" ]; then
    echo "  OK: $(ls -lh "$VPIC_BIN")"
else
    echo "  WARNING: VPIC binary not found at $VPIC_BIN"
    echo "  VPIC benchmark will be skipped"
fi

# ── Step 4: Run benchmark ──
log_step "Running benchmark: ${BENCHMARKS:-vpic} (MPI_NP=$TOTAL_RANKS) ..."
echo "  DATA_MB=${DATA_MB:-256} CHUNK_MB=${CHUNK_MB:-4} TIMESTEPS=${TIMESTEPS:-25}"
echo "  POLICIES=${POLICIES:-balanced} VERIFY=${VERIFY:-0}"
_t0=$(date +%s)

MPI_NP=$TOTAL_RANKS \
GPUS_PER_NODE=$GPUS_PER_NODE \
BENCHMARKS=${BENCHMARKS:-"vpic"} \
DATA_MB=${DATA_MB:-256} \
CHUNK_MB=${CHUNK_MB:-4} \
TIMESTEPS=${TIMESTEPS:-25} \
POLICIES=${POLICIES:-"balanced"} \
VERIFY=${VERIFY:-0} \
bash benchmarks/benchmark.sh
_rc=$?

_t1=$(date +%s)

# ── Step 5: Summary ──
log_step "Done"
echo "  Benchmark exit code: $_rc"
echo "  Benchmark duration : $((_t1 - _t0))s"
echo "  Total job duration : ${SECONDS}s"
echo ""
echo "  Results:"
ls -lh benchmarks/vpic-kokkos/results/eval_*/benchmark_vpic_deck*.csv 2>/dev/null || echo "    (no VPIC results found)"
ls -lh benchmarks/vpic-kokkos/results/eval_*/benchmark_vpic_deck_aggregate_multi_rank.csv 2>/dev/null || true
echo ""
echo "============================================================"
echo "  Job $SLURM_JOB_ID complete at $(date)"
echo "============================================================"
