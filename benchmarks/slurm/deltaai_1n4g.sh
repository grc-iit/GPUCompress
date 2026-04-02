#!/bin/bash
# 1 node × 4 GPUs (4 ranks) — DeltaAI GH200
# Usage: bash benchmarks/slurm/deltaai_1n4g.sh
NODES=1 GPUS=4 sbatch -N1 --gpus-per-node=4 --ntasks-per-node=4 benchmarks/slurm/deltaai_benchmark.sbatch "$@"
