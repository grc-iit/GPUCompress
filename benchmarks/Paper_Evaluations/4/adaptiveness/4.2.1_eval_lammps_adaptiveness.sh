#!/bin/bash
# ============================================================
# LAMMPS RL Adaptiveness Evaluation
#
# Single-phase nn-rl+exp50 run on a live LAMMPS Kokkos LJ
# molecular-dynamics simulation. The simulator continuously
# integrates Newton's equations under NVE; the GPUCompress fix
# (fix_gpucompress_kokkos) borrows the live device-resident
# atom positions/velocities/forces every SIM_INTERVAL physics
# steps and feeds them straight into the GPUCompress HDF5 VOL
# (examples/lammps_gpucompress_udf.cpp), which chunks the field
# and runs the NN-guided algorithm selector + Kendall-tau
# prediction profiler on every chunk.
#
# Why "adaptiveness": the initial condition is a hot expanding
# sphere embedded in a cold FCC lattice. As the simulation
# evolves, the velocity/force distributions diversify radically
# from step to step (shock front, melt zone, cold periphery),
# so each incoming chunk has a different optimal algorithm and
# the online learner must adapt continuously.
#
# Field size is sized so each dump produces dozens of chunks (fp32 build):
#   LMP_ATOMS=120  -> 4 * 120^3 = 6,912,000 atoms
#   3 fields (pos/vel/force) * 3 components * 4 bytes (float)
#   ~= 237 MiB per dump  ~=  60 chunks @ 4 MiB
#
# REQUIREMENT: LAMMPS must be built with KOKKOS_PREC=SINGLE. The project-wide
# rule is "GPUCompress benchmarks always use float32"; the build_lammps.sh
# script defaults to SINGLE for this reason.
#
# Usage:
#   bash benchmarks/Paper_Evaluations/4/adaptiveness/4.2.1_eval_lammps_adaptiveness.sh
#
# Environment overrides:
#   LMP_BIN           $HOME/lammps/build/lmp
#   LMP_ATOMS         120       FCC lattice side (~6.9M atoms)
#   CHUNK_MB          4         HDF5 chunk size in MiB
#   TIMESTEPS         50        Number of measured dumps
#   WARMUP_STEPS      500       Physics steps before first measured dump
#   SIM_INTERVAL      190       Physics steps between measured dumps
#   ERROR_BOUND       0.01      (informational; LAMMPS fix is lossless-pass)
#   SGD_LR            0.2       Online SGD learning rate
#   SGD_MAPE          0.30      X1: SGD MAPE threshold
#   EXPLORE_THRESH    0.50      X2: exploration threshold
#   EXPLORE_K         4         Exploration alternatives
#   POLICY            balanced  Cost model policy (balanced|ratio|speed)
#   DRY_RUN           0         Print commands without running
# ============================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Parameters ──
LMP_BIN="${LMP_BIN:-$HOME/lammps/build/lmp}"
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPU_DIR/neural_net/weights/model.nnwt}"
LMP_ATOMS=${LMP_ATOMS:-120}
CHUNK_MB=${CHUNK_MB:-4}
TIMESTEPS=${TIMESTEPS:-50}
WARMUP_STEPS=${WARMUP_STEPS:-500}
SIM_INTERVAL=${SIM_INTERVAL:-190}
ERROR_BOUND=${ERROR_BOUND:-0.01}
SGD_LR=${SGD_LR:-0.2}
SGD_MAPE=${SGD_MAPE:-0.10}
EXPLORE_THRESH=${EXPLORE_THRESH:-0.20}
EXPLORE_K=${EXPLORE_K:-4}
POLICY=${POLICY:-balanced}
DRY_RUN=${DRY_RUN:-0}

# Policy weights (informational — fix derives them from POLICY internally)
case "$POLICY" in
    balanced) W0=1.0; W1=1.0; W2=1.0 ;;
    ratio)    W0=0.0; W1=0.0; W2=1.0 ;;
    speed)    W0=1.0; W1=1.0; W2=0.0 ;;
    *) echo "ERROR: unknown policy '$POLICY' (use balanced, ratio, speed)"; exit 1 ;;
esac

# Error bound tag
if [ "$ERROR_BOUND" = "0" ] || [ "$ERROR_BOUND" = "0.0" ]; then
    EB_TAG="lossless"
else
    EB_TAG="eb${ERROR_BOUND}"
fi
RESULTS_DIR="$PARENT_DIR/results/lammps_adaptiveness_${POLICY}_${EB_TAG}_lr${SGD_LR}"

# Derived sizing — fp32 (LAMMPS must be built with KOKKOS_PREC=SINGLE,
# project-wide rule: GPUCompress benchmarks always use float32).
NATOMS=$(python3 -c "print(4 * $LMP_ATOMS**3)" 2>/dev/null || echo "?")
FIELD_MIB=$(python3 -c "print(f'{$NATOMS * 3 * 4 / 1048576:.1f}')" 2>/dev/null || echo "?")
DUMP_MIB=$(python3 -c "print(f'{$NATOMS * 3 * 4 * 3 / 1048576:.1f}')" 2>/dev/null || echo "?")
CHUNKS_PER_DUMP=$(python3 -c "import math; print(3 * math.ceil($NATOMS*3*4 / ($CHUNK_MB*1048576)))" 2>/dev/null || echo "?")

# ── Validate ──
if [ ! -x "$LMP_BIN" ]; then
    echo "ERROR: lmp binary not found or not executable at $LMP_BIN"
    echo "  Build with: bash benchmarks/lammps/build_lammps.sh  (or set LMP_BIN env)"
    exit 1
fi
if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: NN weights not found at $WEIGHTS"
    exit 1
fi

echo "============================================================"
echo "LAMMPS RL Adaptiveness Evaluation"
echo "  Box:           ${LMP_ATOMS}^3 FCC (~${NATOMS} atoms)"
echo "  Field/dump:    ~${FIELD_MIB} MiB per field, ~${DUMP_MIB} MiB total"
echo "  Chunk size:    ${CHUNK_MB} MiB  (~${CHUNKS_PER_DUMP} chunks per dump)"
echo "  Warmup:        ${WARMUP_STEPS} physics steps (no measurement)"
echo "  Dumps:         ${TIMESTEPS} measured, every ${SIM_INTERVAL} sim steps"
echo "  Total steps:   $((WARMUP_STEPS + TIMESTEPS * SIM_INTERVAL))"
echo "  Policy:        $POLICY (w0=$W0 w1=$W1 w2=$W2)"
echo "  SGD LR:        $SGD_LR    MAPE: $SGD_MAPE"
echo "  Explore K:     $EXPLORE_K  thresh: $EXPLORE_THRESH"
echo "  Output:        $RESULTS_DIR"
echo "============================================================"

# Skip if results already exist
if [ -f "$RESULTS_DIR/benchmark_lammps_ranking.csv" ]; then
    echo ""
    echo "Already done, skipping. (rm -rf $RESULTS_DIR to re-run)"
    exit 0
fi

mkdir -p "$RESULTS_DIR"

# ── LAMMPS input deck ──
# Two-phase run:
#   1. Equilibration: WARMUP_STEPS of NVE LJ MD with no GPUCompress fix.
#      Lets the hot sphere start expanding into the cold lattice so the
#      first measured dump already shows non-trivial heterogeneity.
#   2. reset_timestep 0, then add fix gpucompress with nevery=SIM_INTERVAL
#      and run TIMESTEPS*SIM_INTERVAL steps. The fix's end_of_step fires
#      at multiples of SIM_INTERVAL, producing exactly TIMESTEPS dumps.
INPUT="$RESULTS_DIR/in.lammps_adaptive"
cat > "$INPUT" <<EOF
# LAMMPS LJ adaptiveness deck for GPUCompress paper eval (auto-generated)
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 ${LMP_ATOMS} 0 ${LMP_ATOMS} 0 ${LMP_ATOMS}
create_box      1 box
create_atoms    1 box
mass            1 1.0

# Hot sphere expanding into cold lattice => continuously evolving,
# diversified velocity/force distributions step after step.
region          hot sphere $((LMP_ATOMS/2)) $((LMP_ATOMS/2)) $((LMP_ATOMS/2)) $((LMP_ATOMS/4))
group           hot region hot
group           cold subtract all hot
velocity        cold create 0.01 87287 loop geom
velocity        hot create 10.0 12345 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    every 10 delay 0 check no

fix             1 all nve
timestep        0.003

# ── Phase 1: warmup (no GPUCompress, no measurement) ──
thermo          ${WARMUP_STEPS}
run             ${WARMUP_STEPS}

# ── Phase 2: live measurement ──
# reset so dumps land at clean multiples of SIM_INTERVAL.
reset_timestep  0
fix             gpuc all gpucompress ${SIM_INTERVAL} positions velocities forces
thermo          ${SIM_INTERVAL}
run             $((TIMESTEPS * SIM_INTERVAL))
EOF

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY_RUN: would run:"
    echo "  $LMP_BIN -k on g 1 -sf kk -in $INPUT"
    echo "with env:"
    echo "  GPUCOMPRESS_WEIGHTS=$WEIGHTS"
    echo "  GPUCOMPRESS_POLICY=$POLICY  GPUCOMPRESS_ALGO=auto"
    echo "  GPUCOMPRESS_SGD=1 GPUCOMPRESS_EXPLORE=1"
    echo "  GPUCOMPRESS_LR=$SGD_LR GPUCOMPRESS_MAPE=$SGD_MAPE"
    echo "  GPUCOMPRESS_EXPLORE_K=$EXPLORE_K GPUCOMPRESS_EXPLORE_THRESH=$EXPLORE_THRESH"
    echo "  GPUCOMPRESS_CHUNK_MB=$CHUNK_MB GPUCOMPRESS_TOTAL_WRITES=$TIMESTEPS"
    echo "  LAMMPS_LOG_CHUNKS=1 LAMMPS_LOG_DIR=$RESULTS_DIR"
    exit 0
fi

# Make sure GPUCompress shared libs are findable by lmp at runtime.
export LD_LIBRARY_PATH="$GPU_DIR/build:$GPU_DIR/examples:/tmp/hdf5-install/lib:/tmp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ── Run LAMMPS once. The fix opens benchmark_lammps_ranking.csv and
#    writes a row at every Kendall-tau milestone (T=0, 5, 10, then
#    every 10% of TIMESTEPS), so this single run produces the full
#    regret-vs-time convergence trajectory. ──
(
    cd "$RESULTS_DIR"
    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    GPUCOMPRESS_POLICY="$POLICY" \
    GPUCOMPRESS_ALGO="auto" \
    GPUCOMPRESS_SGD=1 \
    GPUCOMPRESS_EXPLORE=1 \
    GPUCOMPRESS_LR="$SGD_LR" \
    GPUCOMPRESS_MAPE="$SGD_MAPE" \
    GPUCOMPRESS_EXPLORE_K="$EXPLORE_K" \
    GPUCOMPRESS_EXPLORE_THRESH="$EXPLORE_THRESH" \
    GPUCOMPRESS_CHUNK_MB="$CHUNK_MB" \
    GPUCOMPRESS_TOTAL_WRITES="$TIMESTEPS" \
    GPUCOMPRESS_VERIFY=0 \
    LAMMPS_LOG_CHUNKS=1 \
    LAMMPS_LOG_DIR="$RESULTS_DIR" \
    "$LMP_BIN" -k on g 1 -sf kk -in "$INPUT" \
        > "$RESULTS_DIR/lammps.log" 2>&1
) || echo "WARNING: lmp exited non-zero (teardown SIGABRT from Kokkos is harmless if ranking CSV is present)"

if [ ! -s "$RESULTS_DIR/benchmark_lammps_ranking.csv" ]; then
    echo "ERROR: ranking CSV missing or empty — see $RESULTS_DIR/lammps.log"
    exit 1
fi

N_RANK=$(($(wc -l < "$RESULTS_DIR/benchmark_lammps_ranking.csv") - 1))
N_CHUNKS=$(($(wc -l < "$RESULTS_DIR/benchmark_lammps_timestep_chunks.csv" 2>/dev/null || echo 1) - 1))
echo ""
echo "Ranking milestones recorded: $N_RANK"
echo "Per-chunk diagnostics rows:   $N_CHUNKS"

# ── Optional: render regret-vs-timestep plot using the shared
#    adaptiveness plotter, which understands benchmark_<ds>_ranking.csv. ──
PLOT_SCRIPT="$SCRIPT_DIR/4.2.1_plot_rl_adaptiveness.py"
if [ -f "$PLOT_SCRIPT" ] && python3 -c "import matplotlib" 2>/dev/null; then
    echo ""
    echo ">>> Generating regret-vs-timestep plot..."
    python3 "$PLOT_SCRIPT" "$RESULTS_DIR" 2>&1 | grep -E "Saved:|Done|ERROR" || true
fi

echo ""
echo "============================================================"
echo "LAMMPS adaptiveness evaluation complete."
echo "Results: $RESULTS_DIR"
echo "  benchmark_lammps_ranking.csv         (top1_regret per milestone)"
echo "  benchmark_lammps_ranking_costs.csv   (predicted vs actual costs)"
echo "  benchmark_lammps_timestep_chunks.csv (per-chunk MAPE / sgd / explore)"
echo "  lammps.log                           (full LAMMPS stdout)"
echo "============================================================"
