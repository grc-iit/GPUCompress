#!/bin/bash
# ============================================================
# explore_vpic_diversity.sh
#
# Systematically explores VPIC physics and timing parameters to
# find configurations where compression ratio changes significantly
# across consecutive timesteps (avoids the flat ~1.6x plateau).
#
# Strategy:
#   The flat plateau happens because the Harris sheet has fully
#   reconnected and fields are homogeneous.  Diversity requires
#   catching the system mid-reconnection: tearing mode onset,
#   plasmoid formation, and post-reconnection turbulence each
#   have different field textures (and thus different compressibility).
#
#   Key levers:
#     VPIC_WARMUP_STEPS  — how evolved the sheet is before writes start.
#                          Too few: flat equilibrium; too many: flat post-recon.
#                          Sweet spot: arrival at the nonlinear tearing phase.
#     VPIC_SIM_INTERVAL  — physics steps between consecutive writes.
#                          More steps = data moves further between samples.
#     VPIC_MI_ME=5       — light ions → faster reconnection timescale.
#     VPIC_WPE_WCE=1     — stronger magnetization → sharper current sheets.
#     VPIC_TI_TE=5       — hot ions → more free energy → stronger instability.
#     VPIC_PERTURBATION  — seed amplitude for tearing mode.
#                          0.3+ skips the slow linear phase entirely.
#     VPIC_GUIDE_FIELD   — out-of-plane B; 0.3 creates 3D island structure.
#
#   Four configs are tested at NX=64 (fast), 4 timesteps each:
#
#   Config 1 — "baseline_slow":
#     Defaults (mi_me=25, wpe_wce=3, Ti_Te=1, pert=0.1).
#     Expected: very flat ratios (slow reconnection, barely evolved).
#
#   Config 2 — "fast_recon_early":
#     Fast physics (mi_me=5, wpe_wce=1, Ti_Te=5), large perturbation (0.3).
#     Warmup=200, short interval=50.  Catches the system just after tearing onset.
#     Expected: ratio rises from ~1.5 to ~3+ as reconnection develops.
#
#   Config 3 — "fast_recon_sweep":
#     Same fast physics, but warmup=350 + long interval=120.
#     Writes span from peak reconnection into turbulent exhaust phase.
#     Expected: ratio FALLS as turbulence re-randomises fields (compressible→noisy).
#
#   Config 4 — "guide_field_3d":
#     Fast physics + guide field 0.3 → creates flux ropes (3D structure).
#     Warmup=250, interval=80.
#     Expected: intermediate rising trend; 3D structure increases compressibility.
#
# Usage:
#   bash benchmarks/Paper_Evaluations/7/explore_vpic_diversity.sh
#
# Overrides:
#   NX=64                 TIMESTEPS=4      CHUNK_MB=4
#   RESULTS_BASE=...      SKIP_EXISTING=1
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

VPIC_BIN="$GPU_DIR/benchmarks/vpic-kokkos/vpic_benchmark_deck.Linux"
WEIGHTS="$GPU_DIR/neural_net/weights/model.nnwt"

NX="${NX:-64}"
TIMESTEPS="${TIMESTEPS:-4}"
CHUNK_MB="${CHUNK_MB:-4}"
RESULTS_BASE="${RESULTS_BASE:-$SCRIPT_DIR/results/diversity_$(date +%Y%m%d_%H%M%S)}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [ ! -f "$VPIC_BIN" ]; then
    echo "ERROR: VPIC binary not found: $VPIC_BIN"
    exit 1
fi

echo "============================================================"
echo "VPIC Parameter Diversity Exploration"
echo "  Goal: find configs with MAX ratio variation across timesteps"
echo "  NX=$NX  Timesteps=$TIMESTEPS  Chunk=${CHUNK_MB}MB"
echo "  Output: $RESULTS_BASE"
echo "============================================================"

# ── run_config name warmup sim_interval mi_me wpe_wce ti_te pert guide ──
#
# Arguments:
#   $1  config label (used for directory naming and summary table)
#   $2  VPIC_WARMUP_STEPS
#   $3  VPIC_SIM_INTERVAL
#   $4  VPIC_MI_ME
#   $5  VPIC_WPE_WCE
#   $6  VPIC_TI_TE
#   $7  VPIC_PERTURBATION
#   $8  VPIC_GUIDE_FIELD
run_config() {
    local LABEL="$1"
    local WARMUP="$2"
    local SIM_INT="$3"
    local MI_ME="$4"
    local WPE_WCE="$5"
    local TI_TE="$6"
    local PERT="$7"
    local GUIDE="$8"

    local OUT_DIR="$RESULTS_BASE/$LABEL"
    local TS_CSV="$OUT_DIR/benchmark_vpic_deck_timesteps.csv"

    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Config: $LABEL"
    echo "    Physics : mi_me=$MI_ME  wpe_wce=$WPE_WCE  Ti_Te=$TI_TE"
    echo "    Seed    : pert=$PERT  guide=$GUIDE"
    echo "    Timing  : warmup=$WARMUP  sim_interval=$SIM_INT  timesteps=$TIMESTEPS"
    echo "──────────────────────────────────────────────────────────"

    if [ "$SKIP_EXISTING" = "1" ] && [ -f "$TS_CSV" ]; then
        local n_rows
        # Count data rows (skip header); default to 0 if file unreadable
        n_rows=$(tail -n +2 "$TS_CSV" 2>/dev/null | wc -l) || n_rows=0
        n_rows=$(( n_rows + 0 ))  # coerce to integer; guards whitespace from wc
        if [ "$n_rows" -ge "$TIMESTEPS" ]; then
            echo "  Already done ($n_rows rows), skipping."
            return
        fi
        echo "  Incomplete results ($n_rows rows < $TIMESTEPS), re-running."
        rm -rf "$OUT_DIR"
    fi

    mkdir -p "$OUT_DIR"

    local START_S
    START_S=$(date +%s)

    GPUCOMPRESS_WEIGHTS="$WEIGHTS"         \
    VPIC_NX="$NX"                          \
    VPIC_TIMESTEPS="$TIMESTEPS"            \
    VPIC_WARMUP_STEPS="$WARMUP"            \
    VPIC_SIM_INTERVAL="$SIM_INT"           \
    VPIC_MI_ME="$MI_ME"                    \
    VPIC_WPE_WCE="$WPE_WCE"               \
    VPIC_TI_TE="$TI_TE"                   \
    VPIC_PERTURBATION="$PERT"             \
    VPIC_GUIDE_FIELD="$GUIDE"             \
    VPIC_CHUNK_MB="$CHUNK_MB"             \
    VPIC_PHASE="nn-rl+exp50"              \
    VPIC_VERIFY=0                          \
    VPIC_RESULTS_DIR="$OUT_DIR"            \
    "$VPIC_BIN" \
        >"$OUT_DIR/vpic.log" 2>&1 \
        || { echo "  WARNING: VPIC exited non-zero; partial results may still exist."; }

    local END_S ELAPSED
    END_S=$(date +%s)
    ELAPSED=$(( END_S - START_S ))
    echo "  Finished in ${ELAPSED}s."

    if [ -f "$TS_CSV" ]; then
        local n_rows
        n_rows=$(tail -n +2 "$TS_CSV" | wc -l)
        echo "  Timestep rows written: $n_rows"
    else
        echo "  WARNING: no timestep CSV produced.  Check $OUT_DIR/vpic.log"
    fi
}

# ── Config 1: Baseline slow reconnection ──────────────────────
# Default physics.  Reconnection timescale >>  our 4-timestep window.
# Expect flat ratios — this is the "bad" baseline we're trying to escape.
run_config "c1_baseline_slow"  \
    500 190                    \
    25 3 1                     \
    0.1 0.0

# ── Config 2: Fast reconnection, early phase ──────────────────
# Light ions (mi_me=5) with strong magnetization (wpe_wce=1) and
# hot ions (Ti_Te=5) make the tearing mode grow in ~200 steps.
# Large perturbation (0.3) seeds the mode immediately, bypassing
# the slow linear phase.  Short warmup=200 catches the fields just
# as the X-point forms.  Interval=50 means each write is 50 steps
# further into the reconnection jet development.
run_config "c2_fast_early"     \
    200 50                     \
    5 1 5                      \
    0.30 0.0

# ── Config 3: Fast reconnection, spanning peak → exhaust ──────
# Same fast physics, but warmup=350 places the start firmly inside
# the nonlinear reconnection phase (plasmoid formation).  Interval=120
# gives each snapshot time to see the system transition through:
#   t0: active reconnection jets (structured → high ratio)
#   t1: exhaust broadening (intermediate)
#   t2: turbulence onset (random-looking → lower ratio)
#   t3: post-reconnection field randomization (noisy → lowest ratio)
# This should produce the MOST diversity (falling trend).
run_config "c3_fast_sweep"     \
    350 120                    \
    5 1 5                      \
    0.30 0.0

# ── Config 4: Guide field — 3D flux rope structure ────────────
# Guide field (By=0.3*b0) breaks the 2D symmetry of the Harris sheet.
# Flux ropes form oblique to the reconnection plane, creating coherent
# 3D structure that is MORE compressible than turbulence but LESS than
# the equilibrium.  Warmup=250 hits the early flux-rope formation window.
# Interval=80 lets each write capture the rope coalescence process.
# Expected: gentle rising trend as ropes merge into larger structures.
run_config "c4_guide_3d"       \
    250 80                     \
    5 1 5                      \
    0.25 0.30

# ============================================================
# Summary analysis
# ============================================================
echo ""
echo "============================================================"
echo "SUMMARY: Per-timestep ratio variation by config"
echo "============================================================"

python3 - "$RESULTS_BASE" "$TIMESTEPS" <<'EOF'
import sys
import os
import csv

results_base = sys.argv[1]
n_ts = int(sys.argv[2])

# Configs in display order
configs = [
    ("c1_baseline_slow",  "Baseline slow (mi_me=25, pert=0.1, warmup=500, int=190)"),
    ("c2_fast_early",     "Fast early    (mi_me=5,  pert=0.30, warmup=200, int=50) "),
    ("c3_fast_sweep",     "Fast sweep    (mi_me=5,  pert=0.30, warmup=350, int=120)"),
    ("c4_guide_3d",       "Guide-field   (mi_me=5,  pert=0.25, guide=0.3, warmup=250, int=80)"),
]

col_w = max(len(d) for _, d in configs) + 2
ts_w = 9  # width per timestep column

sep = "-" * (col_w + ts_w * n_ts + 30)

# Header row
header = f"{'Config':{col_w}}"
for t in range(n_ts):
    header += f"{'t='+str(t):>{ts_w}}"
header += f"{'StdDev':>9}  {'MaxDelta':>9}  Winner?"
print(sep)
print(header)
print(sep)

best_label = None
best_std = -1.0
results = []

for cfg_key, cfg_label in configs:
    ts_csv = os.path.join(results_base, cfg_key, "benchmark_vpic_deck_timesteps.csv")
    if not os.path.isfile(ts_csv):
        print(f"{cfg_label:{col_w}}  (no CSV found — run may have failed)")
        continue

    ratios = []
    with open(ts_csv) as f:
        for row in csv.DictReader(f):
            try:
                ratios.append(float(row["ratio"]))
            except (KeyError, ValueError):
                pass

    if not ratios:
        print(f"{cfg_label:{col_w}}  (CSV empty or missing ratio column)")
        continue

    # Pad to n_ts if fewer rows (partial run)
    while len(ratios) < n_ts:
        ratios.append(float("nan"))

    ratios = ratios[:n_ts]

    valid = [r for r in ratios if r == r]  # filter NaN
    if len(valid) < 2:
        std_val = 0.0
        max_delta = 0.0
    else:
        mean_val = sum(valid) / len(valid)
        variance = sum((r - mean_val) ** 2 for r in valid) / len(valid)
        std_val = variance ** 0.5
        max_delta = max(valid) - min(valid)

    if std_val > best_std:
        best_std = std_val
        best_label = cfg_label

    results.append((cfg_key, cfg_label, ratios, std_val, max_delta))

    row_str = f"{cfg_label:{col_w}}"
    for r in ratios:
        if r != r:
            row_str += f"{'---':>{ts_w}}"
        else:
            row_str += f"{r:>{ts_w}.3f}"
    row_str += f"{std_val:>9.4f}  {max_delta:>9.4f}"
    print(row_str)

print(sep)
print()

if best_label:
    print(f"  MOST DIVERSE: {best_label}")
    print()

# Trend analysis
print("  Trend direction per config:")
for cfg_key, cfg_label, ratios, std_val, max_delta in results:
    valid = [(i, r) for i, r in enumerate(ratios) if r == r]
    if len(valid) < 2:
        trend = "insufficient data"
    else:
        # Simple linear regression slope
        n = len(valid)
        xs = [v[0] for v in valid]
        ys = [v[1] for v in valid]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        den = sum((xs[i] - x_mean) ** 2 for i in range(n))
        slope = num / den if den > 0 else 0.0
        if abs(slope) < 0.005:
            trend = "flat"
        elif slope > 0:
            trend = f"rising  (+{slope:.4f}/timestep)"
        else:
            trend = f"falling ({slope:.4f}/timestep)"
    print(f"    {cfg_key}: {trend}  (std={std_val:.4f}, delta={max_delta:.4f})")

print()
print(f"  Results written to: {results_base}")
print()

# Recommendation
print("  RECOMMENDATION:")
print("  Use the config with the highest StdDev for NN training diversity.")
print("  A falling trend (post-recon turbulence) creates the most pressure")
print("  on the NN to adapt: the optimal algorithm shifts as fields decorrelate.")
print("  A rising trend (early reconnection) is good for demonstrating NN")
print("  adaptation from an uninformative prior to a structured field regime.")
EOF

echo "============================================================"
echo "Full logs: $RESULTS_BASE/<config>/vpic.log"
echo "============================================================"
