#!/bin/bash
# ============================================================
# LAMMPS Exploration Threshold Sweep
#
# Mirrors eval_{vpic,warpx}.sh: re-runs the
# live Kokkos LJ MD simulation once per (X1, X2-X1) cell so each
# cell trains the online learner on fresh, evolving
# position/velocity/force fields from a hot-sphere-in-cold-lattice
# initial condition (the same deck as the LAMMPS adaptiveness
# eval). The fix_gpucompress_kokkos plugin runs the NN-guided
# algorithm selector + Kendall-tau profiler on every chunk and
# honors the per-cell GPUCOMPRESS_MAPE / GPUCOMPRESS_EXPLORE_THRESH
# env vars.
#
# X1 (SGD MAPE):    {0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 10.00}
# delta (X2-X1):    {0.05, 0.10, 0.20, 0.30, 0.50, 1.00, 10.00}
# Total: 7 x 7 = 49 runs
#
# Per-cell output (under
#   PARENT_DIR/results/lammps_threshold_sweep_<...>/x1_<x1>_delta_<delta>/):
#   benchmark_lammps_timestep_chunks.csv  (per-chunk MAPE + sgd + explore)
#   benchmark_lammps_ranking.csv          (per-milestone top1_regret)
#   benchmark_lammps_ranking_costs.csv
#   benchmark_lammps.csv                  (synthesized aggregate)
#   lammps.log                            (full lmp stdout)
#
# Pair with plot.py to render the heatmaps for
# the paper's "Effect of online learning thresholds" figure.
#
# NOTE: LAMMPS must be built with KOKKOS_PREC=SINGLE (project-wide
# rule: all GPUCompress benchmarks use float32). See
# benchmarks/lammps/build_lammps.sh.
#
# Environment overrides (data target: 25 dumps × ~128 MiB/dump):
#   LMP_BIN           $HOME/lammps/build/lmp
#   LMP_ATOMS         98        FCC lattice side (4*98^3 = 3.77M atoms,
#                               3 fields * 3 comp * 4 B = 36 B/atom → ~129 MiB/dump)
#   CHUNK_MB          8         HDF5 chunk size in MiB
#   TIMESTEPS         25        Number of measured dumps
#   WARMUP_STEPS      500       Physics steps before first measured dump
#   SIM_INTERVAL      190       Physics steps between measured dumps
#   ERROR_BOUND       0.01      (informational; fix is lossless-pass)
#   EXPLORE_K         4         Exploration alternatives K
#   SGD_LR            0.2       SGD learning rate
#   POLICY            balanced  Cost policy (balanced|ratio|speed)
#   DRY_RUN           0         Print commands without running
# ============================================================
set -o pipefail
# NOTE: do NOT use `set -e` — quota/IO failures on a single cell would
# abort the entire 49-cell sweep. The for-loop already validates each cell
# via the benchmark_lammps.csv sentinel and skips broken cells.

command -v bc >/dev/null 2>&1 || { echo "ERROR: bc not found"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARENT_DIR="$GPU_DIR/benchmarks/Paper_Evaluations/4"

# ── Fixed parameters ──
LMP_BIN="${LMP_BIN:-$HOME/lammps/build/lmp}"
WEIGHTS="${GPUCOMPRESS_WEIGHTS:-$GPU_DIR/neural_net/weights/model.nnwt}"

LMP_ATOMS=${LMP_ATOMS:-98}      # 4*98^3 = 3.77M atoms → ~129 MiB/dump
CHUNK_MB=${CHUNK_MB:-8}
TIMESTEPS=${TIMESTEPS:-15}
WARMUP_STEPS=${WARMUP_STEPS:-500}
SIM_INTERVAL=${SIM_INTERVAL:-190}
ERROR_BOUND=${ERROR_BOUND:-0.01}
EXPLORE_K=${EXPLORE_K:-4}
SGD_LR=${SGD_LR:-0.2}
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

SWEEP_DIR="$PARENT_DIR/results/lammps_threshold_sweep_${POLICY}_${EB_TAG}_lr${SGD_LR}"

# ── Validate ──
[ -x "$LMP_BIN" ]  || { echo "ERROR: lmp binary not found or not executable at $LMP_BIN"; exit 1; }
[ -f "$WEIGHTS" ] || { echo "ERROR: NN weights not found at $WEIGHTS"; exit 1; }

mkdir -p "$SWEEP_DIR"

export LD_LIBRARY_PATH="$GPU_DIR/build:$GPU_DIR/examples:/tmp/hdf5-install/lib:/tmp/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Derived sizing
NATOMS=$(python3 -c "print(4 * $LMP_ATOMS**3)" 2>/dev/null || echo "?")
DUMP_MIB=$(python3 -c "print(f'{$NATOMS * 3 * 4 * 3 / 1048576:.1f}')" 2>/dev/null || echo "?")

echo "============================================================"
echo "LAMMPS Threshold Sweep"
echo "  Box:         ${LMP_ATOMS}^3 FCC (~${NATOMS} atoms, ~${DUMP_MIB} MiB/dump)"
echo "  Policy:      $POLICY (w0=$W0 w1=$W1 w2=$W2)"
echo "  Dumps:       $TIMESTEPS measured, every $SIM_INTERVAL sim steps (warmup=$WARMUP_STEPS)"
echo "  Chunk:       ${CHUNK_MB} MB"
echo "  Error bound: $ERROR_BOUND ($EB_TAG)"
echo "  SGD LR:      $SGD_LR    Explore K: $EXPLORE_K"
echo "  Output:      $SWEEP_DIR"
echo "============================================================"
echo ""

# ── Threshold values — 7x7 = 49 cells ──
X1_VALUES=(0.05 0.10 0.20 0.30 0.50 1.00 10.00)
DELTA_VALUES=(0.05 0.10 0.20 0.30 0.50 1.00 10.00)

# ── Build randomized parameter pairs (Fisher-Yates) ──
PAIRS=()
for x1 in "${X1_VALUES[@]}"; do
    for delta in "${DELTA_VALUES[@]}"; do
        PAIRS+=("${x1},${delta}")
    done
done
TOTAL=${#PAIRS[@]}
for (( i=TOTAL-1; i>0; i-- )); do
    j=$(( RANDOM % (i+1) ))
    tmp="${PAIRS[$i]}"; PAIRS[$i]="${PAIRS[$j]}"; PAIRS[$j]="$tmp"
done

RUN=0
for pair in "${PAIRS[@]}"; do
    x1="${pair%%,*}"
    delta="${pair##*,}"
    RUN=$((RUN + 1))
    x2=$(echo "$x1 + $delta" | bc -l)

    OUT_DIR="$SWEEP_DIR/x1_${x1}_delta_${delta}"

    if [ -f "$OUT_DIR/benchmark_lammps.csv" ]; then
        echo "[$RUN/$TOTAL] X1=$x1 delta=$delta — already done, skipping."
        continue
    fi

    echo ""
    echo "[$RUN/$TOTAL] X1=$x1 delta=$delta (X2=$x2)"
    mkdir -p "$OUT_DIR"

    # ── Emit a fresh LAMMPS deck per cell (same as adaptiveness deck). ──
    INPUT="$OUT_DIR/in.lammps_sweep"
    cat > "$INPUT" <<EOF
# LAMMPS LJ threshold-sweep deck (auto-generated)
units           lj
atom_style      atomic
lattice         fcc 0.8442
region          box block 0 ${LMP_ATOMS} 0 ${LMP_ATOMS} 0 ${LMP_ATOMS}
create_box      1 box
create_atoms    1 box
mass            1 1.0

# Hot sphere expanding into cold lattice => continuously evolving
# velocity/force distributions, so each incoming chunk has a
# different optimal algorithm and the online learner must adapt.
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
reset_timestep  0
fix             gpuc all gpucompress ${SIM_INTERVAL} positions velocities forces
thermo          ${SIM_INTERVAL}
run             $((TIMESTEPS * SIM_INTERVAL))
EOF

    if [ "$DRY_RUN" = "1" ]; then
        echo "  DRY_RUN: lmp -in $INPUT  MAPE=$x1  EXPLORE_THRESH=$x2"
        continue
    fi

    # ── Run LAMMPS for this (X1, X2). Chunk + ranking CSVs are written
    #    by fix_gpucompress_kokkos into LAMMPS_LOG_DIR. ──
    (
        cd "$OUT_DIR"
        GPUCOMPRESS_WEIGHTS="$WEIGHTS" \
        GPUCOMPRESS_POLICY="$POLICY" \
        GPUCOMPRESS_ALGO="auto" \
        GPUCOMPRESS_SGD=1 \
        GPUCOMPRESS_EXPLORE=1 \
        GPUCOMPRESS_LR="$SGD_LR" \
        GPUCOMPRESS_MAPE="$x1" \
        GPUCOMPRESS_EXPLORE_K="$EXPLORE_K" \
        GPUCOMPRESS_EXPLORE_THRESH="$x2" \
        GPUCOMPRESS_CHUNK_MB="$CHUNK_MB" \
        GPUCOMPRESS_TOTAL_WRITES="$TIMESTEPS" \
        GPUCOMPRESS_VERIFY=0 \
        LAMMPS_LOG_CHUNKS=1 \
        LAMMPS_LOG_DIR="$OUT_DIR" \
        "$LMP_BIN" -k on g 1 -sf kk -in "$INPUT" \
            > lammps.log 2>&1
    ) || echo "  WARNING: lmp exited non-zero (teardown SIGABRT from Kokkos is harmless if ranking CSV is present)"

    if [ ! -s "$OUT_DIR/benchmark_lammps_timestep_chunks.csv" ]; then
        echo "  ERROR: chunks CSV missing or empty — see $OUT_DIR/lammps.log"
        # Even on failure, drop the gpuc_step_*/ dirs so a crashed cell
        # doesn't leak ~7 GiB to /u quota until the whole sweep ends.
        find "$OUT_DIR" -maxdepth 1 -type d -name 'gpuc_step_*' -exec rm -rf {} + 2>/dev/null
        continue
    fi

    # ── Synthesize benchmark_lammps.csv aggregate from the per-chunk CSV.
    #
    # LAMMPS fix does not emit a gpucompress_vol_summary.txt, so this
    # aggregator derives:
    #   - avg ratio                from nonzero actual_ratio rows
    #   - mape_{ratio,comp,decomp} from nonzero mape_* rows
    #   - mape_psnr_pct            from nonzero mape_psnr rows (lossless → 0)
    #   - write_mibps (kernel)     = total_bytes / sum(actual_comp_ms_raw)
    #   - read_mibps  (kernel)     = total_bytes / sum(actual_decomp_ms_raw)
    #   - sgd_fires, explorations  from column sums
    #
    # total_bytes is estimated as n_chunks * CHUNK_MB * 1 MiB (chunks are
    # full-size for this deck given CHUNK_MB=4 and ~237 MiB dumps). This
    # gives a kernel-only throughput proxy, the same semantics as
    # warpx_threshold_sweep's kernel read_mibps. NOT directly comparable
    # to end-to-end H5Dwrite bandwidth, but the units (MiB/s) and the
    # numerator (bytes compressed by the kernel) match across the three
    # workloads so the heatmaps are internally consistent.
    python3 - "$OUT_DIR" "$x1" "$x2" "$CHUNK_MB" <<'PYEOF'
import os, sys, csv, math

run_dir, x1, x2, chunk_mb = sys.argv[1:5]
chunk_mb = float(chunk_mb)

chunk_csv = os.path.join(run_dir, "benchmark_lammps_timestep_chunks.csv")
n_chunks = 0
sum_ratio, n_ratio = 0.0, 0
sum_mr, n_mr = 0.0, 0
sum_mc, n_mc = 0.0, 0
sum_md, n_md = 0.0, 0
sum_mp, n_mp = 0.0, 0
sum_comp_ms_raw = 0.0
sum_decomp_ms_raw = 0.0
sgd_fires = 0
explorations = 0

with open(chunk_csv) as f:
    for r in csv.DictReader(f):
        try:
            ar  = float(r.get("actual_ratio", 0))
            mr  = float(r.get("mape_ratio",  0))
            mc  = float(r.get("mape_comp",   0))
            md  = float(r.get("mape_decomp", 0))
            ac  = float(r.get("actual_comp_ms_raw", 0))
            ad  = float(r.get("actual_decomp_ms_raw", 0))
            mp_raw = r.get("mape_psnr", "nan")
            try:
                mp = float(mp_raw)
            except ValueError:
                mp = float("nan")

            if ar > 0:
                sum_ratio += ar; n_ratio += 1
            if mr > 0:
                sum_mr += mr; n_mr += 1
            if mc > 0:
                sum_mc += mc; n_mc += 1
            if md > 0:
                sum_md += md; n_md += 1
            if math.isfinite(mp) and mp > 0:
                sum_mp += mp; n_mp += 1
            if ac > 0:
                sum_comp_ms_raw += ac
            if ad > 0:
                sum_decomp_ms_raw += ad
            sgd_fires    += int(float(r.get("sgd_fired", 0)))
            explorations += int(float(r.get("exploration_triggered", 0)))
            n_chunks += 1
        except (ValueError, TypeError):
            pass

avg_ratio  = (sum_ratio / n_ratio) if n_ratio else 0.0
avg_mr     = (sum_mr    / n_mr)    if n_mr    else 0.0
avg_mc     = (sum_mc    / n_mc)    if n_mc    else 0.0
avg_md     = (sum_md    / n_md)    if n_md    else 0.0
avg_mp     = (sum_mp    / n_mp)    if n_mp    else 0.0

total_mib = n_chunks * chunk_mb
write_mibps = (total_mib / (sum_comp_ms_raw / 1000.0)) if sum_comp_ms_raw > 0 else 0.0
read_mibps  = (total_mib / (sum_decomp_ms_raw / 1000.0)) if sum_decomp_ms_raw > 0 else 0.0

out_csv = os.path.join(run_dir, "benchmark_lammps.csv")
fields = [
    "rank", "source", "phase", "n_runs",
    "write_ms", "write_ms_std", "read_ms", "read_ms_std",
    "file_mib", "orig_mib", "ratio", "write_mibps", "read_mibps",
    "mismatches", "sgd_fires", "explorations", "n_chunks",
    "mape_ratio_pct", "mape_comp_pct", "mape_decomp_pct", "mape_psnr_pct",
    "x1_sgd_mape", "x2_explore_thresh",
]
row = {f: 0 for f in fields}
row["rank"] = 0
row["source"] = "lammps"
row["phase"] = "nn-rl+exp50"
row["n_runs"] = 1
row["file_mib"] = total_mib / max(avg_ratio, 1.0) if avg_ratio > 0 else 0.0
row["orig_mib"] = total_mib
row["ratio"] = avg_ratio
row["write_mibps"] = write_mibps
row["read_mibps"]  = read_mibps
row["sgd_fires"] = sgd_fires
row["explorations"] = explorations
row["n_chunks"] = n_chunks
row["mape_ratio_pct"]  = avg_mr
row["mape_comp_pct"]   = avg_mc
row["mape_decomp_pct"] = avg_md
row["mape_psnr_pct"]   = avg_mp
row["x1_sgd_mape"] = float(x1)
row["x2_explore_thresh"] = float(x2)

with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerow(row)

print(f"  ratio={avg_ratio:.2f}x  kcomp={write_mibps:.0f}MiB/s  kdecomp={read_mibps:.0f}MiB/s  "
      f"mape_ratio={avg_mr:.1f}%  sgd={sgd_fires}  expl={explorations}  chunks={n_chunks}")
PYEOF

    # ── Per-cell cleanup: delete the bulky HDF5 dump dirs now that all
    #    metric CSVs have been extracted. At LMP_ATOMS=98 each cell writes
    #    ~7 GiB of plotfiles (25 timesteps × 3 fields), which would blow
    #    the user /u quota after only a few cells. CSVs (benchmark_lammps*,
    #    ranking*, timestep_chunks, lammps.log) are preserved.
    find "$OUT_DIR" -maxdepth 1 -type d -name 'gpuc_step_*' -exec rm -rf {} + 2>/dev/null
done

echo ""
echo "============================================================"
echo "LAMMPS threshold sweep complete: $TOTAL configurations"
echo "Results: $SWEEP_DIR/"
echo ""
echo "Generate heatmaps:"
echo "  python3 $SCRIPT_DIR/plot.py $SWEEP_DIR"
echo "============================================================"

# ── Generate plots if matplotlib available ──
PLOT_SCRIPT="$SCRIPT_DIR/plot.py"
if [ -f "$PLOT_SCRIPT" ] && python3 -c "import matplotlib" 2>/dev/null; then
    echo ""
    echo ">>> Generating threshold sweep plots..."
    python3 "$PLOT_SCRIPT" "$SWEEP_DIR" || echo "WARNING: plotter failed"
fi
