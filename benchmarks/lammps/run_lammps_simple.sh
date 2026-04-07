#!/bin/bash
# ============================================================
# LAMMPS Simple Run — long evolution, diverse data
#
# Mirrors the VPIC benchmark structure but stripped down:
#   Phase A: WARMUP — let physics evolve before any dumps
#            (hot sphere expands, temperature ramps up,
#             stripes start mixing) so the first benchmark
#             dump already has non-trivial features.
#   Phase B: BENCHMARK — dump positions/velocities/forces
#            every SIM_INTERVAL steps for TIMESTEPS dumps
#            using GPUCompress (single algo, single policy).
#
# This is a SINGLE run, no multi-phase sweep. Use it to:
#   1. Sanity-check that the diverse deck actually produces
#      varying entropy/MAD/deriv across dumps and chunks.
#   2. Generate raw .f32 fields for later offline benchmarking
#      (set LAMMPS_DUMP_FIELDS=1).
#
# Diversity sources baked into the input:
#   - 2 atom types (mass 1.0 vs 2.5) in stripes along x
#   - Hot sphere (T=8) expanding into cold bulk (T=0.05)
#   - NVT temperature ramp 0.5 -> 4.0 over the whole run
#
# Usage:
#   bash benchmarks/lammps/run_lammps_simple.sh
#
# Environment variables:
#   LMP_BIN          Path to lmp binary [$HOME/lammps/build/lmp]
#   LMP_ATOMS        Box size per dimension [80 -> ~2M atoms]
#   CHUNK_MB         HDF5 chunk size [4]
#   TIMESTEPS        Number of benchmark dumps [50]
#   SIM_INTERVAL     Physics steps between dumps [30]
#   WARMUP_STEPS     Physics steps BEFORE first dump [300]
#   ALGO             GPUCompress algo [auto]
#   POLICY           NN policy [ratio]
#   VERIFY           Lossless verification [0]
#   DUMP_FIELDS      Also write raw .f32 fields [0]
#   LOG_CHUNKS       Write per-chunk stats CSV [1]
#   RESULTS_DIR      Output directory [auto]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# -- Defaults --
LMP_BIN="${LMP_BIN:-$HOME/lammps/build/lmp}"
LMP_ATOMS=${LMP_ATOMS:-80}
CHUNK_MB=${CHUNK_MB:-4}
TIMESTEPS=${TIMESTEPS:-50}
SIM_INTERVAL=${SIM_INTERVAL:-30}
WARMUP_STEPS=${WARMUP_STEPS:-300}
ALGO=${ALGO:-auto}
POLICY=${POLICY:-ratio}
VERIFY=${VERIFY:-0}
DUMP_FIELDS=${DUMP_FIELDS:-0}
LOG_CHUNKS=${LOG_CHUNKS:-1}
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"

# Derived
BENCH_STEPS=$((TIMESTEPS * SIM_INTERVAL))
TOTAL_STEPS=$((WARMUP_STEPS + BENCH_STEPS))
HALF=$((LMP_ATOMS / 2))
QUARTER=$((LMP_ATOMS / 4))
STRIPE1=$((LMP_ATOMS / 4))
STRIPE2=$((LMP_ATOMS / 2))
STRIPE3=$((3 * LMP_ATOMS / 4))
NATOMS_APPROX=$(python3 -c "print(4 * $LMP_ATOMS**3)" 2>/dev/null || echo "?")
DATA_MB=$(python3 -c "print(f'{4 * $LMP_ATOMS**3 * 12 / 1048576:.1f}')" 2>/dev/null || echo "?")
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/results/lammps_simple_box${LMP_ATOMS}_warm${WARMUP_STEPS}_ts${TIMESTEPS}x${SIM_INTERVAL}}"

export LD_LIBRARY_PATH="$GPUC_DIR/build:$GPUC_DIR/examples:/tmp/hdf5-install/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "============================================================"
echo "LAMMPS Simple Run (long evolution, diverse data)"
echo "============================================================"
echo "  Binary:        $LMP_BIN"
echo "  Box:           ${LMP_ATOMS}^3 (~$NATOMS_APPROX atoms)"
echo "  Data/field:    ~$DATA_MB MB per field (fp32)"
echo "  Warmup:        $WARMUP_STEPS steps  (NO dumps)"
echo "  Benchmark:     $TIMESTEPS dumps every $SIM_INTERVAL steps  ($BENCH_STEPS steps)"
echo "  Total steps:   $TOTAL_STEPS"
echo "  Chunk:         ${CHUNK_MB} MB"
echo "  Algo / policy: $ALGO / $POLICY"
echo "  Verify:        $VERIFY"
echo "  Dump fields:   $DUMP_FIELDS"
echo "  Log chunks:    $LOG_CHUNKS"
echo "  Results:       $RESULTS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"
rm -rf gpuc_step_* raw_fields chunks.csv lammps.log

# -- Generate input deck --
# Two-phase deck: warmup run (no compression fix) then benchmark run.
# This guarantees the FIRST benchmark dump already sees evolved data.
INPUT_FILE="$RESULTS_DIR/input.lmp"
cat > "$INPUT_FILE" << EOF
# Diverse LAMMPS deck for GPUCompress NN training
# - 2 atom types in 4 stripes along x (composition gradient)
# - hot central sphere vs cold bulk (propagating front)
# - NVT temperature ramp 0.5 -> 4.0 over the full run
# - WARMUP first ($WARMUP_STEPS steps, no dumps), THEN benchmark dumps
# All quantities float32 (KOKKOS_PREC=SINGLE).

units           lj
atom_style      atomic
lattice         fcc 0.8442

region          box block 0 $LMP_ATOMS 0 $LMP_ATOMS 0 $LMP_ATOMS
create_box      2 box
create_atoms    1 box

# ---- composition stripes (alternating type along x) ----
region          s2 block $STRIPE1 $STRIPE2 0 $LMP_ATOMS 0 $LMP_ATOMS
region          s4 block $STRIPE3 $LMP_ATOMS 0 $LMP_ATOMS 0 $LMP_ATOMS
group           gs2 region s2
group           gs4 region s4
set             group gs2 type 2
set             group gs4 type 2

mass            1 1.0
mass            2 2.5

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
pair_coeff      2 2 1.4 1.1 2.5
pair_coeff      1 2 1.2 1.05 2.5

neighbor        0.3 bin
neigh_modify    every 10 delay 0 check yes

# ---- hot sphere initial condition ----
region          hot sphere $HALF $HALF $HALF $QUARTER units lattice
group           hot region hot
group           cold subtract all hot
velocity        cold create 0.05 87287 loop geom
velocity        hot  create 8.00 12345 loop geom

timestep        0.004

# ============================================================
# PHASE A: WARMUP — heat from 0.5 -> 4.0 (liquid), NO dumps.
# Hot sphere expands, stripes mix, velocity distribution
# broadens. By the end, the first benchmark dump already sees
# fully evolved, diverse data.
# ============================================================
fix             1 all nvt temp 0.5 4.0 0.1
thermo          $((WARMUP_STEPS / 5))
thermo_style    custom step temp press pe ke etotal
run             $WARMUP_STEPS

# ============================================================
# PHASE B: BENCHMARK — slow quench from 4.0 -> 0.3 across the
# whole benchmark phase. This crosses the LJ liquid/solid
# boundary (~T=0.7), so data distribution KEEPS evolving:
#   * early dumps   (T~4 ): liquid, random positions, large forces
#   * middle dumps  (T~1 ): cooling liquid, structuring begins
#   * late dumps    (T~0.3): partially crystallized, ordered
# Guarantees the NN sees a sweep of feature regimes, not just
# an equilibrated steady state.
# ============================================================
unfix           1
fix             1 all nvt temp 4.0 0.3 0.1
fix             gpuc all gpucompress $SIM_INTERVAL positions velocities forces
thermo          $SIM_INTERVAL
run             $BENCH_STEPS
EOF

# -- GPUCompress environment --
export GPUCOMPRESS_ALGO="$ALGO"
export GPUCOMPRESS_POLICY="$POLICY"
export GPUCOMPRESS_VERIFY="$VERIFY"
export GPUCOMPRESS_WEIGHTS="$WEIGHTS"
export GPUCOMPRESS_CHUNK_MB="$CHUNK_MB"
export GPUCOMPRESS_SGD=1
export GPUCOMPRESS_EXPLORE=1

if [ "$DUMP_FIELDS" = "1" ]; then
    mkdir -p raw_fields
    export LAMMPS_DUMP_FIELDS=1
    export LAMMPS_DUMP_DIR="$RESULTS_DIR/raw_fields"
fi

if [ "$LOG_CHUNKS" = "1" ]; then
    export LAMMPS_LOG_CHUNKS=1
fi

# -- Sanity checks --
if [ ! -x "$LMP_BIN" ]; then
    echo "ERROR: LAMMPS binary not found: $LMP_BIN"
    exit 1
fi
if [ ! -f "$WEIGHTS" ] && [ "$ALGO" = "auto" ]; then
    echo "ERROR: NN weights not found: $WEIGHTS"
    echo "       (needed because ALGO=auto). Set ALGO=lz4 to bypass."
    exit 1
fi

# -- Run --
echo ">>> Running LAMMPS ($TOTAL_STEPS steps total)..."
START_NS=$(date +%s%N)
"$LMP_BIN" -k on g 1 -sf kk -in "$INPUT_FILE" > lammps.log 2>&1 || {
    echo "WARNING: LAMMPS exited with error — see lammps.log"
}
END_NS=$(date +%s%N)
WALL_S=$(python3 -c "print(f'{($END_NS - $START_NS)/1e9:.1f}')")

echo "  done in ${WALL_S}s"
echo ""

# -- Report --
N_DUMPS=$(ls -d gpuc_step_* 2>/dev/null | wc -l)
echo ">>> Results"
echo "  Dumps written: $N_DUMPS  (expected $TIMESTEPS)"

if [ "$N_DUMPS" -gt 0 ]; then
    NATOMS=$(grep "Created.*atoms$" lammps.log | head -1 | awk '{print $2}')
    [ -z "$NATOMS" ] && NATOMS=0
    ORIG_PER_FIELD=$((NATOMS * 3 * 4))
    ORIG_PER_DUMP=$((ORIG_PER_FIELD * 3))
    ORIG_MB=$(python3 -c "print(f'{$ORIG_PER_DUMP/1048576:.1f}')")

    echo "  Atoms:         $NATOMS"
    echo "  Orig/dump:     ${ORIG_MB} MB (3 fields x 3 components x fp32)"
    echo ""
    printf "  %-22s %-12s %-10s\n" "step_dir" "comp_bytes" "ratio"
    for d in $(ls -d gpuc_step_* 2>/dev/null | sort); do
        CB=$(du -sb "$d" | awk '{print $1}')
        R=$(python3 -c "print(f'{$ORIG_PER_DUMP/$CB:.2f}x')" 2>/dev/null)
        printf "  %-22s %-12s %-10s\n" "$d" "$CB" "$R"
    done | head -20
    if [ "$N_DUMPS" -gt 20 ]; then
        echo "  ... ($((N_DUMPS - 20)) more)"
    fi
fi

if [ "$LOG_CHUNKS" = "1" ] && [ -f benchmark_lammps_timestep_chunks.csv ]; then
    N_CHUNK_ROWS=$(($(wc -l < benchmark_lammps_timestep_chunks.csv) - 1))
    echo ""
    echo "  Per-chunk CSV: benchmark_lammps_timestep_chunks.csv ($N_CHUNK_ROWS rows)"
    echo "  -> inspect entropy/mad/deriv columns to verify feature diversity"
fi

echo ""
echo "============================================================"
echo "Done. Results in: $RESULTS_DIR"
echo "============================================================"
