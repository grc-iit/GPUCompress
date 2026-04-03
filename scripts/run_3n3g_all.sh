#!/bin/bash
# ============================================================
# 3n3g Benchmark: Lossless + Lossy
# ~128 MB/rank, 100 timesteps, 9 ranks (3 nodes x 3 GPUs), 4 MB chunks
# ============================================================
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=vpic-3n3g-all
#SBATCH --output=vpic-3n3g-all-%j.out
#SBATCH --error=vpic-3n3g-all-%j.err
#SBATCH --nodes=3
#SBATCH --gpus-per-node=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=04:00:00

_total_start=$(date +%s)

# ── Build hostfile for MPI distribution across all nodes ──
HOSTFILE="$HOME/.slurm_hostfile_${SLURM_JOB_ID}"
GPUS_PER_NODE=3
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read -r node; do
    for _g in $(seq 1 $GPUS_PER_NODE); do
        echo "$node"
    done
done > "$HOSTFILE"
export SLURM_HOSTFILE="$HOSTFILE"
echo "Hostfile: $HOSTFILE"
cat "$HOSTFILE"
echo ""

echo "============================================================"
echo "  3n3g Full Benchmark"
echo "  Run 1: Lossless (balanced + ratio)"
echo "  Run 2: Lossy    (balanced + ratio, error_bound=0.01)"
echo "  NX=198 (~120 MB/rank, ~1 GB total), 100 timesteps, 4 MB chunks"
echo "  Ranks: 9 (3 nodes x 3 GPUs)"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ── Run 1: Lossless ──
echo ">>> [1/2] Lossless benchmark starting..."
_t0=$(date +%s)
MPI_NP=9 GPUS_PER_NODE=3 VPIC_NX=198 CHUNK_MB=4 TIMESTEPS=100 \
VERIFY=1 POLICIES=balanced,ratio LOSSY=0 \
bash scripts/run_vpic_scaling.sh
_t1=$(date +%s)
echo ">>> [1/2] Lossless done in $((_t1 - _t0))s"
echo ""

# ── Run 2: Lossy ──
echo ">>> [2/2] Lossy benchmark starting..."
_t0=$(date +%s)
MPI_NP=9 GPUS_PER_NODE=3 VPIC_NX=198 CHUNK_MB=4 TIMESTEPS=100 \
VERIFY=1 POLICIES=balanced,ratio LOSSY=0.01 \
bash scripts/run_vpic_scaling.sh
_t1=$(date +%s)
echo ">>> [2/2] Lossy done in $((_t1 - _t0))s"

_total_end=$(date +%s)
_total_elapsed=$((_total_end - _total_start))
echo ""
echo "============================================================"
echo "  3n3g Full Benchmark Complete"
echo "  Total wall time: ${_total_elapsed}s ($((_total_elapsed/60))m $((_total_elapsed%60))s)"
echo "  Finished: $(date)"
echo ""
echo "  Lossless results:"
ls -d benchmarks/vpic-kokkos/results/eval_NX198_chunk4mb_ts100/ 2>/dev/null
echo "  Lossy results:"
ls -d benchmarks/vpic-kokkos/results/eval_NX198_chunk4mb_ts100_lossy0.01/ 2>/dev/null
echo "============================================================"

# Cleanup hostfile
rm -f "$HOSTFILE"
