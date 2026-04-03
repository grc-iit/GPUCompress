#!/bin/bash
# ============================================================
# 2n2g Smoke Test: Lossless only
# ~128 MB/rank, 10 timesteps, 4 ranks, 16 MB chunks
# ============================================================
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --job-name=vpic-2n2g-smoke
#SBATCH --output=vpic-2n2g-smoke-%j.out
#SBATCH --error=vpic-2n2g-smoke-%j.err
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:30:00

_total_start=$(date +%s)

# ── Build hostfile for MPI distribution across all nodes ──
HOSTFILE="$HOME/.slurm_hostfile_${SLURM_JOB_ID}"
GPUS_PER_NODE=2
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
echo "  2n2g Smoke Test (Lossless)"
echo "  NX=200 (~128 MB/rank), 10 timesteps, 16 MB chunks"
echo "  Policies: balanced,ratio"
echo "  Started: $(date)"
echo "============================================================"
echo ""

MPI_NP=4 GPUS_PER_NODE=2 VPIC_NX=200 CHUNK_MB=16 TIMESTEPS=10 \
VERIFY=1 POLICIES=balanced,ratio LOSSY=0 \
bash scripts/run_vpic_scaling.sh

_total_end=$(date +%s)
_total_elapsed=$((_total_end - _total_start))
echo ""
echo "============================================================"
echo "  2n2g Smoke Test Complete"
echo "  Wall time: ${_total_elapsed}s ($((_total_elapsed/60))m $((_total_elapsed%60))s)"
echo "  Finished: $(date)"
echo "  Results:"
ls -d benchmarks/vpic-kokkos/results/eval_NX200_chunk16mb_ts10/ 2>/dev/null
echo "============================================================"

# Cleanup hostfile
rm -f "$HOSTFILE"
