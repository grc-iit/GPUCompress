#!/bin/bash
# ============================================================
# Full threshold sweep — all 4 workloads × 49 (X1, delta) cells (7x7 grid).
#
# Runs evaluations/figure6_threshold_sweep/run_all.sh
# which invokes the vpic / warpx / lammps / nyx sweep scripts back
# to back. Each child script actually executes its simulator (one
# live run per cell, 49 cells per workload), runs GPUCompress over
# the dumps, aggregates per-cell CSVs, and at the end calls
# plot.py to render the 9 heatmaps into the
# per-workload results dir.
#
# Total runs: 4 × 49 = 196 simulations (+ 1 one-shot Nyx dump).
#
# Rough per-workload wall-clock estimates (one cell):
#   VPIC   ~2-3 min    (NX=100, 50 timesteps)
#   WarpX  ~2-4 min    (32x32x256, 200 steps, diag every 10)
#   LAMMPS ~1-2 min    (120^3 FCC, 500 warmup + 50×190 measured)
#   NYX    ~20-30 min for Phase 1 dump, then ~30-90s per cell replay
# Total: ~2-3 h with head-room. If it blows the wall, rerun with
# a subset via WORKLOADS env (see below) — each workload skips
# already-completed cells (benchmark_*.csv sentinel).
#
# Submit:   sbatch evaluations/figure6_threshold_sweep/submit.sbatch
#
# Knobs (all env-forwarded to child scripts):
#   WORKLOADS  "vpic warpx lammps nyx"  subset selection
#   POLICY     balanced                 balanced|ratio|speed
#   SGD_LR     0.2
#   EXPLORE_K  4
#   DRY_RUN    0                        set 1 for a no-sim smoke test
# Per-workload sizing overrides (VPIC_NX, WARPX_MAX_STEP, LMP_ATOMS,
# NYX_NCELL, NYX_MAX_STEP, CHUNK_MB, ERROR_BOUND, ...) are honored.
# ============================================================
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=gpucomp_sweep_all
#SBATCH --output=deleteAfterwards/slurm-%x-%j.out

set -uo pipefail

module load cuda gcc-native/13.2 cray-mpich/8.1.32

GPUCOMPRESS_DIR=/u/$USER/GPUCompress
HDF5_DIR=$GPUCOMPRESS_DIR/.deps/hdf5
NVCOMP_DIR=$GPUCOMPRESS_DIR/.deps

cd "$GPUCOMPRESS_DIR"

export MPICH_DIR=/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2
export LD_LIBRARY_PATH=$NVCOMP_DIR/lib:$HDF5_DIR/lib:$GPUCOMPRESS_DIR/build:$GPUCOMPRESS_DIR/examples:${LD_LIBRARY_PATH:-}
export GPUCOMPRESS_WEIGHTS=$GPUCOMPRESS_DIR/neural_net/weights/model.nnwt

# Real run — no dry-run
export DRY_RUN=${DRY_RUN:-0}

# Shared threshold-sweep knobs
export POLICY=${POLICY:-balanced}
export SGD_LR=${SGD_LR:-0.2}
export EXPLORE_K=${EXPLORE_K:-4}

# Workload binaries (Delta paths override sweep-script defaults).
# VPIC_BIN MUST point at the freshly-built deck under benchmarks/vpic-kokkos/
# (the legacy top-level vpic_benchmark_deck.Linux is a Mar 22 binary that
# predates VPIC_PHASE= single-phase support and will run all 8 fixed phases
# + the exhaustive `best` profiler, crashing in its H5Dread.)
export VPIC_BIN=$GPUCOMPRESS_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux
export NYX_BIN=/u/$USER/Nyx/build/Exec/HydroTests/nyx_HydroTests
export WARPX_BIN=/u/$USER/src/warpx/build-cuda/bin/warpx.3d
export WARPX_INPUTS=/u/$USER/src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d
export LMP_BIN=/u/$USER/lammps/build/lmp

# Reuse the shared dump across NYX cells across reruns (saves the
# 20–30 min Phase 1 Nyx simulation if a prior job already dumped)
export REUSE_DUMPS=${REUSE_DUMPS:-1}

nvidia-smi | head -20 || true
echo ""
echo "============================================================"
echo "Threshold sweep: WORKLOADS=${WORKLOADS:-vpic warpx lammps nyx}"
echo "  POLICY=$POLICY  SGD_LR=$SGD_LR  EXPLORE_K=$EXPLORE_K  DRY_RUN=$DRY_RUN"
echo "============================================================"

bash evaluations/figure6_threshold_sweep/run_all.sh
rc=$?

echo ""
echo "============================================================"
echo "Wrapper exited with rc=$rc"
echo "Result dirs + heatmaps:"
for d in benchmarks/Paper_Evaluations/4/results/*threshold_sweep_*/; do
    [ -d "$d" ] || continue
    n_png=$(ls "$d"threshold_sweep_*.png 2>/dev/null | wc -l)
    n_cells=$(ls -d "$d"x1_*_delta_* 2>/dev/null | wc -l)
    echo "  $d  cells=$n_cells  plots=$n_png"
done
echo "============================================================"

exit $rc
