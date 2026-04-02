#!/bin/bash
# 2 nodes × 4 GPUs each (8 ranks) — DeltaAI GH200
# Usage: bash benchmarks/slurm/deltaai_2n4g.sh
NODES=2 GPUS=4 sbatch -N2 --gpus-per-node=4 --ntasks-per-node=4 benchmarks/slurm/deltaai_benchmark.sbatch "$@"
