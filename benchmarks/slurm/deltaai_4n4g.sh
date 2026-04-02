#!/bin/bash
# 4 nodes × 4 GPUs each (16 ranks) — DeltaAI GH200
# Usage: bash benchmarks/slurm/deltaai_4n4g.sh
NODES=4 GPUS=4 sbatch -N4 --gpus-per-node=4 --ntasks-per-node=4 benchmarks/slurm/deltaai_benchmark.sbatch "$@"
