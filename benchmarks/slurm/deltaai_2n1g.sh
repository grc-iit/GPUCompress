#!/bin/bash
# 2 nodes × 1 GPU each (2 ranks) — DeltaAI GH200
# Usage: bash benchmarks/slurm/deltaai_2n1g.sh
NODES=2 GPUS=1 sbatch -N2 --gpus-per-node=1 --ntasks-per-node=1 benchmarks/slurm/deltaai_benchmark.sbatch "$@"
