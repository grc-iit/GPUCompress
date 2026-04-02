#!/bin/bash
# 1 node × 2 GPUs (2 ranks) — DeltaAI GH200
# Usage: bash benchmarks/slurm/deltaai_1n2g.sh
NODES=1 GPUS=2 sbatch -N1 --gpus-per-node=2 --ntasks-per-node=2 benchmarks/slurm/deltaai_benchmark.sbatch "$@"
