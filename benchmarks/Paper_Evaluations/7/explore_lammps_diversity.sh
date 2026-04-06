#!/bin/bash
# ============================================================
# LAMMPS Parameter Diversity Explorer
#
# Systematically explores LAMMPS simulation configurations to
# find setups where MD data changes compression characteristics
# DRAMATICALLY between consecutive dumps — high ratio variance
# and MAPE variation across timesteps.
#
# The current hot-sphere default (T_hot=10, T_cold=0.01,
# sim_interval=50) produces stationary distributions (~1.09x
# across all dumps). This script tests configurations that push
# the simulation through phase transitions, shock fronts, or
# rapid melting to create genuine data diversity.
#
# Strategy:
#   A) Extreme temperature ratio  — violent expansion (T=100 hot sphere)
#   B) Slow equilibration capture — many short intervals (sim_interval=10)
#   C) Global melt through Tm     — uniform high T, crystallization during run
#   D) Shock wave                 — velocity pulse into cold lattice
#
# Usage:
#   bash benchmarks/Paper_Evaluations/7/explore_lammps_diversity.sh
#
# Environment variables:
#   LMP_BIN      Path to lmp binary [$HOME/lammps/build/lmp]
#   OUT_DIR      Output directory [auto under this script's dir]
#   N_DUMPS      Dump count per config [12]
#   ATOMS        Box size per dim [40]  (256K atoms, ~8.8 MB/dump)
# ============================================================
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPUC_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LMP_BIN="${LMP_BIN:-$HOME/lammps/build/lmp}"
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPUC_DIR/neural_net/weights/model.nnwt}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/lammps_diversity_$(date +%Y%m%d_%H%M%S)}"

# ── Per-config tuning ──────────────────────────────────────────────────────
# Small box for speed: 40^3 = 256,000 atoms, ~8.8 MB per dump (3 fields fp32)
ATOMS="${ATOMS:-40}"
N_DUMPS="${N_DUMPS:-12}"   # dump count per config (excluding warmup)

export LD_LIBRARY_PATH="$GPUC_DIR/build:$GPUC_DIR/examples:/tmp/hdf5-install/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$OUT_DIR"

# ── Utilities ──────────────────────────────────────────────────────────────
note() { echo "[explore] $*"; }

hr() { printf '%s\n' "────────────────────────────────────────────────────────────"; }

# Compute per-timestep average actual_ratio from the chunk CSV.
# The CSV timestep column is the write_count index (0-based).
# Output: one "T=N ratio=R.RR" line per write, then summary stats.
summarize_chunks() {
    local csv="$1"
    local label="$2"

    if [ ! -f "$csv" ]; then
        echo "    (no chunk CSV found at $csv)"
        return
    fi

    python3 - "$csv" "$label" <<'PY'
import sys, csv, statistics

path, label = sys.argv[1], sys.argv[2]

# Accumulate per-timestep ratios across all chunks
ts_ratios = {}   # timestep -> [actual_ratio, ...]

with open(path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            t = int(row["timestep"])
            r = float(row["actual_ratio"])
            if r > 0:
                ts_ratios.setdefault(t, []).append(r)
        except (ValueError, KeyError):
            continue

if not ts_ratios:
    print("    (no valid rows)")
    sys.exit(0)

sorted_ts = sorted(ts_ratios.keys())
per_ts_avg = [statistics.mean(ts_ratios[t]) for t in sorted_ts]

print(f"    Per-timestep mean ratio ({label}):")
for t, r in zip(sorted_ts, per_ts_avg):
    bar_len = max(0, int((r - 1.0) * 40))
    bar = "#" * bar_len
    print(f"      T={t:02d}  ratio={r:.4f}  {bar}")

if len(per_ts_avg) >= 2:
    spread = max(per_ts_avg) - min(per_ts_avg)
    stdev  = statistics.stdev(per_ts_avg) if len(per_ts_avg) > 1 else 0.0
    print(f"    Summary: min={min(per_ts_avg):.4f}  max={max(per_ts_avg):.4f}"
          f"  spread={spread:.4f}  stdev={stdev:.4f}")
else:
    print(f"    Summary: only {len(per_ts_avg)} timestep(s) recorded")
PY
}

# Run one LAMMPS configuration.
# Sets the global LAST_TC_CSV variable to the chunk CSV path so callers
# can read it without capturing stdout (which would swallow all note/summary
# output). All diagnostic output goes directly to the terminal.
#
# Arguments:
#   $1  config label (no spaces)
#   $2  T_hot       (hot sphere initial velocity temperature)
#   $3  T_cold      (cold background temperature)
#   $4  sim_interval (physics steps between dumps)
#   $5  warmup_steps
#   $6  lattice_density  (e.g. 0.8442 = solid, 1.2 = compressed, 0.6 = near-melting)
#   $7  extra_description (printed only)
LAST_TC_CSV=""

run_config() {
    local label="$1"
    local T_hot="$2"
    local T_cold="$3"
    local sim_interval="$4"
    local warmup="$5"
    local density="$6"
    local desc="$7"

    local total_steps=$(( warmup + N_DUMPS * sim_interval ))
    local half=$(( ATOMS / 2 ))
    local radius=$(( ATOMS / 4 ))

    local work="$OUT_DIR/$label/work"
    local out="$OUT_DIR/$label"
    mkdir -p "$work" "$out"

    note "Config: $label"
    note "  $desc"
    note "  T_hot=$T_hot  T_cold=$T_cold  interval=$sim_interval  warmup=$warmup  density=$density"
    note "  atoms=${ATOMS}^3  dumps=$N_DUMPS  total_steps=$total_steps"

    # Generate LAMMPS input
    cat > "$work/input.lmp" <<LMPEOF
units           lj
atom_style      atomic
lattice         fcc ${density}
region          box block 0 ${ATOMS} 0 ${ATOMS} 0 ${ATOMS}
create_box      1 box
create_atoms    1 box
mass            1 1.0

region          hot sphere ${half} ${half} ${half} ${radius}
group           hot region hot
group           cold subtract all hot
velocity        cold create ${T_cold} 87287 loop geom
velocity        hot  create ${T_hot}  12345 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    every 10 delay 0 check no

fix             1 all nve
fix             gpuc all gpucompress ${sim_interval} positions velocities forces
thermo          ${sim_interval}
timestep        0.003
run             ${total_steps}
LMPEOF

    local t_start
    t_start=$(date +%s)

    (
        cd "$work"
        GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
        GPUCOMPRESS_ALGO="auto" \
        GPUCOMPRESS_POLICY="ratio" \
        GPUCOMPRESS_VERIFY="0" \
        GPUCOMPRESS_SGD="1" \
        GPUCOMPRESS_EXPLORE="1" \
        GPUCOMPRESS_LR="0.2" \
        GPUCOMPRESS_MAPE="0.10" \
        GPUCOMPRESS_EXPLORE_K="4" \
        GPUCOMPRESS_EXPLORE_THRESH="0.20" \
        GPUCOMPRESS_CHUNK_MB="4" \
        GPUCOMPRESS_TOTAL_WRITES="$N_DUMPS" \
        LAMMPS_LOG_CHUNKS="1" \
        LAMMPS_LOG_DIR="$out" \
        "$LMP_BIN" -k on g 1 -sf kk -in input.lmp \
            > "$out/lammps.log" 2>&1 || true
    )

    local t_end
    t_end=$(date +%s)
    local elapsed=$(( t_end - t_start ))

    local tc_csv="$out/benchmark_lammps_timestep_chunks.csv"
    local n_rows=0
    [ -f "$tc_csv" ] && n_rows=$(tail -n +2 "$tc_csv" | wc -l)

    note "  Elapsed: ${elapsed}s   Chunk rows: $n_rows"
    summarize_chunks "$tc_csv" "$label"

    # Communicate the path back via a global — avoids swallowing stdout
    # from note() and summarize_chunks() in a $(...) subshell.
    LAST_TC_CSV="$tc_csv"
}

# ── Config A: Extreme temperature ratio (violent hot-sphere expansion) ─────
# T_hot=100 vs T_cold=0.01 (10000x ratio) at density 0.8442 (solid FCC)
# The hot sphere atoms have massive KE, blow through the lattice causing
# a rapidly expanding shock front. First dumps are ordered lattice,
# middle dumps are shock front, final dumps are partial melt.
# Prediction: ratio drops sharply from ~1.15 (ordered cold region)
#             to ~1.05-1.07 (turbulent shock zone) then stabilizes.
hr
note "CONFIG A: Extreme T_hot=100 — violent shock expansion"
hr
run_config \
    "A_extreme_Thot100" \
    "100.0" "0.01" \
    "10" "20" \
    "0.8442" \
    "Violent hot-sphere shock (T_hot=100), short interval=10 to capture transient"
CSV_A="$LAST_TC_CSV"

echo ""

# ── Config B: Ultra-slow dumps, moderate heat — transient lattice disorder ─
# T_hot=50, but very long sim_interval=200 so each dump captures a
# different equilibration stage. The simulation runs long enough that:
#   early dumps: lattice still cold/ordered in bulk → high ratio
#   mid dumps:   heat diffuses radially, mixed entropy → medium ratio
#   late dumps:  near-equilibrated warm solid → uniform low ratio
# This maximizes ratio *trend* (monotonic decrease).
hr
note "CONFIG B: Slow equilibration capture (T_hot=50, interval=200)"
hr
run_config \
    "B_slow_equil_Thot50" \
    "50.0" "0.01" \
    "200" "100" \
    "0.8442" \
    "Heat diffusion transient: large interval=200 to capture each equilibration stage"
CSV_B="$LAST_TC_CSV"

echo ""

# ── Config C: Global melt — all atoms above Tm, crystallization during run ─
# LJ Tm ≈ 1.15 at density 0.8442. Set all atoms to T=3.0 (well above Tm).
# Then reduce density to 0.6 (expanded gas phase) to force re-condensation.
# Wait: with NVE we cannot change temperature mid-run, but we CAN set
# initial T=3.0 (liquid) with density 0.8442 (solid). The mismatch causes
# rapid pressure buildup and phase transition through the run.
# Hot sphere radius=0 workaround: set T_hot=T_cold=3.0 with density=1.2
# (over-compressed solid at liquid T — explosive decompression).
# We achieve uniform initial conditions by setting hot sphere to ATOMS/2
# with T_hot=T_cold=3.0 (both groups same T), density=1.2.
hr
note "CONFIG C: Global melt / explosive decompression (T=3.0, density=1.2)"
hr
# Note: both hot and cold get T=3.0 (Tm for LJ ≈ 1.15 at ρ=0.8442;
# ρ=1.2 is over-compressed → immediate phase explosion).
# sim_interval=20 to catch the explosive early dynamics.
run_config \
    "C_global_melt_rho1p2" \
    "3.0" "3.0" \
    "20" "0" \
    "1.2" \
    "Over-compressed liquid (density=1.2, T=3.0): explosive decompression from t=0"
CSV_C="$LAST_TC_CSV"

echo ""

# ── Config D: Shock pulse — cold solid, tiny hot needle at center ──────────
# Small hot sphere (radius = ATOMS/8 instead of ATOMS/4) with very high T
# concentrates energy into a needle-like shock source.  The shock wave
# expands spherically; each dump captures a different wavefront position.
# Use sim_interval=15 to sample frequently through the wavefront passage.
# Half-sphere radius is ATOMS/8 — we achieve this by overriding directly
# in the input (not parameterized above, so inline here).
hr
note "CONFIG D: Narrow hot needle (radius=ATOMS/8, T_hot=50, interval=15)"
hr

LABEL_D="D_needle_shock"
T_HOT_D="50.0"
T_COLD_D="0.01"
SIM_INT_D="15"
WARMUP_D="0"
DENSITY_D="0.8442"
half_D=$(( ATOMS / 2 ))
radius_D=$(( ATOMS / 8 ))       # half the default radius — narrow needle
total_D=$(( WARMUP_D + N_DUMPS * SIM_INT_D ))

work_D="$OUT_DIR/$LABEL_D/work"
out_D="$OUT_DIR/$LABEL_D"
mkdir -p "$work_D" "$out_D"

note "Config: $LABEL_D"
note "  Narrow hot needle: radius=${radius_D} (ATOMS/8), T_hot=$T_HOT_D, interval=$SIM_INT_D"
note "  atoms=${ATOMS}^3  dumps=$N_DUMPS  total_steps=$total_D"

cat > "$work_D/input.lmp" <<LMPEOF2
units           lj
atom_style      atomic
lattice         fcc ${DENSITY_D}
region          box block 0 ${ATOMS} 0 ${ATOMS} 0 ${ATOMS}
create_box      1 box
create_atoms    1 box
mass            1 1.0

region          hot sphere ${half_D} ${half_D} ${half_D} ${radius_D}
group           hot region hot
group           cold subtract all hot
velocity        cold create ${T_COLD_D} 87287 loop geom
velocity        hot  create ${T_HOT_D}  12345 loop geom

pair_style      lj/cut 2.5
pair_coeff      1 1 1.0 1.0 2.5
neighbor        0.3 bin
neigh_modify    every 10 delay 0 check no

fix             1 all nve
fix             gpuc all gpucompress ${SIM_INT_D} positions velocities forces
thermo          ${SIM_INT_D}
timestep        0.003
run             ${total_D}
LMPEOF2

t_start_D=$(date +%s)
(
    cd "$work_D"
    GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
    GPUCOMPRESS_ALGO="auto" \
    GPUCOMPRESS_POLICY="ratio" \
    GPUCOMPRESS_VERIFY="0" \
    GPUCOMPRESS_SGD="1" \
    GPUCOMPRESS_EXPLORE="1" \
    GPUCOMPRESS_LR="0.2" \
    GPUCOMPRESS_MAPE="0.10" \
    GPUCOMPRESS_EXPLORE_K="4" \
    GPUCOMPRESS_EXPLORE_THRESH="0.20" \
    GPUCOMPRESS_CHUNK_MB="4" \
    GPUCOMPRESS_TOTAL_WRITES="$N_DUMPS" \
    LAMMPS_LOG_CHUNKS="1" \
    LAMMPS_LOG_DIR="$out_D" \
    "$LMP_BIN" -k on g 1 -sf kk -in input.lmp \
        > "$out_D/lammps.log" 2>&1 || true
)
t_end_D=$(date +%s)
elapsed_D=$(( t_end_D - t_start_D ))

CSV_D="$out_D/benchmark_lammps_timestep_chunks.csv"
n_rows_D=0
[ -f "$CSV_D" ] && n_rows_D=$(tail -n +2 "$CSV_D" | wc -l)
note "  Elapsed: ${elapsed_D}s   Chunk rows: $n_rows_D"
summarize_chunks "$CSV_D" "$LABEL_D"

echo ""

# ── Final cross-config comparison ─────────────────────────────────────────
hr
note "FINAL COMPARISON — ratio spread across timesteps (higher spread = more diverse data)"
hr

python3 - "$CSV_A" "$CSV_B" "$CSV_C" "$CSV_D" <<'PY'
import sys, csv, statistics, os

labels_and_paths = [
    ("A: T_hot=100 shock    (interval=10 )", sys.argv[1]),
    ("B: T_hot=50 equil     (interval=200)", sys.argv[2]),
    ("C: global melt rho1.2 (interval=20 )", sys.argv[3]),
    ("D: needle shock       (interval=15 )", sys.argv[4]),
]

results = []   # (label, spread, stdev, n_ts, per_ts)

for label, path in labels_and_paths:
    ts_ratios = {}
    if os.path.exists(path):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    t = int(row["timestep"])
                    r = float(row["actual_ratio"])
                    if r > 0:
                        ts_ratios.setdefault(t, []).append(r)
                except (ValueError, KeyError):
                    continue

    sorted_ts = sorted(ts_ratios.keys())
    per_ts = [statistics.mean(ts_ratios[t]) for t in sorted_ts]
    spread = (max(per_ts) - min(per_ts)) if len(per_ts) >= 2 else 0.0
    stdev  = statistics.stdev(per_ts) if len(per_ts) > 1 else 0.0
    results.append((label, spread, stdev, len(per_ts), per_ts))

print()
print(f"  {'Config':<40}  {'n_ts':>4}  {'spread':>7}  {'stdev':>7}  {'min':>6}  {'max':>6}  {'verdict'}")
print(f"  {'-'*40}  {'----':>4}  {'-------':>7}  {'-------':>7}  {'------':>6}  {'------':>6}")
best_spread = max(r[1] for r in results) if results else 0.0
for label, spread, stdev, n_ts, per_ts in results:
    lo = min(per_ts) if per_ts else 0.0
    hi = max(per_ts) if per_ts else 0.0
    verdict = "<<< BEST" if (n_ts >= 2 and spread == best_spread and spread > 0) else ""
    print(f"  {label:<40}  {n_ts:>4}  {spread:>7.4f}  {stdev:>7.4f}  {lo:>6.4f}  {hi:>6.4f}  {verdict}")

print()
# Winner recommendation
best = max(results, key=lambda x: (x[1], x[2]))  # primary: spread, secondary: stdev
print(f"  Recommendation: {best[0].strip()}")
print(f"    spread={best[1]:.4f}  stdev={best[2]:.4f}  over {best[3]} timestep(s)")
print()
print("  Interpretation guide:")
print("    spread > 0.05  — meaningful ratio variation, NN has something to learn")
print("    spread > 0.10  — strong variation, ideal for algorithm-selection diversity")
print("    spread < 0.02  — near-stationary (current default behavior)")
print()
PY

hr
note "Results written to: $OUT_DIR"
note "To inspect raw chunk CSVs:"
note "  ls $OUT_DIR/*/benchmark_lammps_timestep_chunks.csv"
