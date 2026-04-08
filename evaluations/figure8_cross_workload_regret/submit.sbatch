#!/bin/bash
# ============================================================
# SC26 all 4 simulation workloads (VPIC, NYX, WarpX, LAMMPS) via
# evaluations/figure8_cross_workload_regret/run_all.sh, with VPIC configured for a 512 MiB /
# 50-timestep / 8 MiB-chunk lossy sweep. AI workload skipped.
#
# Binaries are pointed at their Delta build locations (overriding
# sc26's hardcoded Chameleon Cloud paths that would otherwise fail).
#
# VPIC overrides (only VPIC has direct "dataset size" knobs):
#   VPIC_NX=201           → (201+2)^3 * 64 B ≈ 510.6 MiB per dump
#   VPIC_TIMESTEPS=50
#   CHUNK_MB=8            → 64 chunks per dump for VPIC
#                           (CHUNK_MB is global; other workloads
#                           inherit it too)
#
# Inherited (no override):
#   VPIC: NPPC=2 mi_me=25 wpe_wce=1 Ti_Te=5 warmup=500 sim_int=190
#         (new deck defaults after the 20:06 rebuild)
#   NYX:   NCELL=88 max_step=500 plot_int=10   (~33 MiB/dump)
#   WarpX: ncell=128 128 384 max_step=200 diag_int=25 (~80–120 MiB)
#   LAMMPS: atoms=80³ (2.05M) ts=12 sim_int=400 (~70 MiB/dump)
#
# sc26 contract:
#   PHASE=nn-rl+exp50     ERROR_BOUND=0.01 (lossy 1%)
#   SGD_LR=0.2  SGD_MAPE=0.10
#   EXPLORE_K=4  EXPLORE_THRESH=0.25
#
# Wall: 3 hours.
#   Per-policy estimate:
#     VPIC   ~12-14 min  (NX=201, 50 ts)
#     NYX    ~20-40 min  (~50 dumps)
#     WarpX  ~8-12 min   (8 dumps)
#     LAMMPS ~3-5 min    (12 dumps)
#     total  ~45-70 min per policy
#   Both policies + combined plotter ~95-145 min
#   3 h wall provides ~1.5-2× headroom.
# ============================================================
#SBATCH --account=bekn-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=gpucomp_sc26_all
#SBATCH --output=deleteAfterwards/slurm-%x-%j.out

set -uo pipefail

module load cuda gcc-native/13.2 cray-mpich/8.1.32

GPUCOMPRESS_DIR=/u/$USER/GPUCompress
HDF5_DIR=$GPUCOMPRESS_DIR/.deps/hdf5
NVCOMP_DIR=$GPUCOMPRESS_DIR/.deps

cd "$GPUCOMPRESS_DIR"

export MPICH_DIR=/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2
export LD_LIBRARY_PATH=$NVCOMP_DIR/lib:$HDF5_DIR/lib:$GPUCOMPRESS_DIR/build:$LD_LIBRARY_PATH
export GPUCOMPRESS_WEIGHTS=$GPUCOMPRESS_DIR/neural_net/weights/model.nnwt

# ── Workload binaries (Delta paths override sc26's Chameleon defaults) ──
export VPIC_BIN=$GPUCOMPRESS_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux
export NYX_BIN=/u/$USER/Nyx/build/Exec/HydroTests/nyx_HydroTests
export WARPX_BIN=/u/$USER/src/warpx/build-cuda/bin/warpx.3d
export WARPX_INPUTS=/u/$USER/src/warpx/Examples/Physics_applications/laser_acceleration/inputs_base_3d
export LMP_BIN=/u/$USER/lammps/build/lmp

# ── Workload skips (only AI is skipped) ──
export SKIP_VPIC=0
export SKIP_NYX=0
export SKIP_WARPX=0
export SKIP_LAMMPS=0
export SKIP_AI=1

# ── Per-workload sizing (50 dumps × ~16 MiB/dump, equalized across all 4) ──
export CHUNK_MB=4              # 4 chunks per 16 MiB dump

export VPIC_NX=62              # (62+2)^3 * 64 B = exactly 16 MiB per dump
export VPIC_TIMESTEPS=50
export VPIC_WARMUP_STEPS=500
export VPIC_SIM_INTERVAL=190

export NYX_NCELL=88            # 88^3 * 24 B ≈ 15.6 MiB per dump
export NYX_MAX_STEP=500        # 50 dumps at plot_int=10
export NYX_PLOT_INT=10

export WARPX_NCELL="64 64 128"  # 524288 cells * 40 B/cell ≈ 20 MiB/FAB so the
                                # Kendall-tau ranking profiler iterates ≥1 chunk
                                # (smaller grids fall below CHUNK_MB and the
                                # profiler early-returns -1 with empty CSV).
export WARPX_MAX_STEP=500      # 50 dumps at diag_interval=10
export WARPX_DIAG_INTERVAL=10

export LMP_ATOMS=49            # 4*49^3 atoms * 36 B ≈ 16 MiB per dump
export LMP_TIMESTEPS=50
export LMP_SIM_INTERVAL=400

# ── Output scoping ──
export SC26_DIR=$GPUCOMPRESS_DIR/deleteAfterwards/sc26_allworkloads
rm -rf "$SC26_DIR"
mkdir -p "$SC26_DIR"

echo "=== host: $(hostname) ==="
nvidia-smi -L
echo

echo "=== binaries ==="
for b in "$VPIC_BIN" "$NYX_BIN" "$WARPX_BIN" "$LMP_BIN"; do
    if [ -x "$b" ]; then
        printf "  OK  %s  %s\n" "$(stat -c '%y' "$b" | cut -d. -f1)" "$b"
    else
        printf "  ??  MISSING: %s\n" "$b"
    fi
done
[ -f "$WARPX_INPUTS" ] && echo "  OK  $WARPX_INPUTS" || echo "  ??  MISSING WARPX_INPUTS"
echo

echo "=== effective run parameters (50 dumps × ~16 MiB/dump, equalized) ==="
echo "  VPIC  : NX=$VPIC_NX ts=$VPIC_TIMESTEPS warmup=$VPIC_WARMUP_STEPS sim_int=$VPIC_SIM_INTERVAL chunk=${CHUNK_MB}MB"
echo "          -> ~16 MiB per dump (64^3 * 64 B), ~800 MiB per cell"
echo "  NYX   : NC=$NYX_NCELL max_step=$NYX_MAX_STEP plot_int=$NYX_PLOT_INT"
echo "          -> ~15.6 MiB per dump (88^3 * 24 B), ~780 MiB per cell"
echo "  WarpX : ncell=\"$WARPX_NCELL\" max_step=$WARPX_MAX_STEP diag=$WARPX_DIAG_INTERVAL"
echo "          -> 32^3 grid; 50 dumps; size has particle overhead"
echo "  LAMMPS: atoms=$LMP_ATOMS ts=$LMP_TIMESTEPS sim_int=$LMP_SIM_INTERVAL"
echo "          -> ~16 MiB per dump (4*49^3 atoms * 36 B), ~800 MiB per cell"
echo "  sc26  : phase=nn-rl+exp50 eb=0.01 LR=0.2 explore_thresh=0.25"
echo "  pols  : balanced, ratio (sc26 sweeps both)"
echo

echo "=== launching evaluations/figure8_cross_workload_regret/run_all.sh (4 workloads × 2 policies = 8 runs) ==="
bash evaluations/figure8_cross_workload_regret/run_all.sh

echo
echo "=== SC26 output layout ==="
find "$SC26_DIR" -maxdepth 2 -type d 2>&1 | head -40
echo
echo "=== timing.csv ==="
[ -f "$SC26_DIR/timing.csv" ] && column -t -s, "$SC26_DIR/timing.csv"
echo
echo "=== aggregated figures ==="
ls -la "$SC26_DIR/figures/" 2>&1 | head -30 || true
echo
echo "=== aggregated CSVs ==="
ls -la "$SC26_DIR/csv/" 2>&1 | head -30 || true
echo "=== DONE ==="
