#!/bin/bash
# 2 nodes × 2 GPUs each (4 ranks) — DeltaAI GH200
# Usage: bash benchmarks/slurm/deltaai_2n2g.sh
NODES=2 GPUS=2 sbatch -N2 --gpus-per-node=2 --ntasks-per-node=2 benchmarks/slurm/deltaai_benchmark.sbatch "$@"
