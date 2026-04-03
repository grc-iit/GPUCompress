#!/bin/bash
# 2n2g Lossy 15 timesteps — run from interactive allocation
cd /u/$USER/GPUCompress

HOSTFILE="$HOME/.slurm_hostfile_$$"
GPUS_PER_NODE=2
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read -r node; do
    for _g in $(seq 1 $GPUS_PER_NODE); do echo "$node"; done
done > "$HOSTFILE"
export SLURM_HOSTFILE="$HOSTFILE"

MPI_NP=4 GPUS_PER_NODE=2 VPIC_NX=200 CHUNK_MB=16 TIMESTEPS=15 \
VERIFY=1 POLICIES=balanced,ratio LOSSY=0.01 \
bash scripts/run_vpic_scaling.sh

rm -f "$HOSTFILE"
