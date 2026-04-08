#!/bin/bash
# ============================================================
# Run all threshold sweeps (VPIC, WarpX, LAMMPS, NYX) back-to-back.
#
# Each per-workload script does its own 7x7 (X1, delta) sweep (49 cells),
# writes results under benchmarks/Paper_Evaluations/4/results/
# <workload>_threshold_sweep_<policy>_<eb>_lr<lr>/, and invokes
# plot.py to render the heatmaps.
#
# Usage:
#   bash evaluations/figure6_threshold_sweep/run_all.sh
#
# Environment overrides:
#   WORKLOADS    "vpic warpx lammps nyx"   Space-separated subset
#   DRY_RUN      0                          Pass DRY_RUN=1 to skip sim execution
#   POLICY       balanced                   Cost policy (shared across workloads)
#   SGD_LR       0.2                        SGD learning rate
#   EXPLORE_K    4                          Exploration alternatives K
#   CONTINUE_ON_ERROR  1                    If 0, stop on first failing workload
#
# All other per-workload env vars (LMP_ATOMS, NYX_NCELL, WARPX_MAX_STEP,
# VPIC_NX, CHUNK_MB, ERROR_BOUND, ...) are forwarded to the child scripts
# through the environment — set them before invoking this wrapper.
# ============================================================
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WORKLOADS="${WORKLOADS:-vpic warpx lammps nyx}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

export POLICY="${POLICY:-balanced}"
export SGD_LR="${SGD_LR:-0.2}"
export EXPLORE_K="${EXPLORE_K:-4}"
export DRY_RUN="${DRY_RUN:-0}"

declare -A STATUS
declare -A ELAPSED
OVERALL_START=$SECONDS

echo "============================================================"
echo "Running threshold sweeps for: $WORKLOADS"
echo "  POLICY=$POLICY  SGD_LR=$SGD_LR  EXPLORE_K=$EXPLORE_K  DRY_RUN=$DRY_RUN"
echo "============================================================"

for w in $WORKLOADS; do
    child="$SCRIPT_DIR/eval_${w}.sh"
    if [ ! -f "$child" ]; then
        echo ""
        echo "!!! SKIP $w — no script at $child"
        STATUS[$w]="missing"
        ELAPSED[$w]="0s"
        continue
    fi

    echo ""
    echo "============================================================"
    echo ">>> [$w] $(basename "$child")"
    echo "============================================================"

    t0=$SECONDS
    if bash "$child"; then
        STATUS[$w]="ok"
    else
        rc=$?
        STATUS[$w]="FAIL(rc=$rc)"
        if [ "$CONTINUE_ON_ERROR" != "1" ]; then
            echo ""
            echo "!!! [$w] failed with rc=$rc — aborting (CONTINUE_ON_ERROR=0)"
            ELAPSED[$w]="$((SECONDS - t0))s"
            break
        fi
    fi
    ELAPSED[$w]="$((SECONDS - t0))s"
done

total=$((SECONDS - OVERALL_START))
echo ""
echo "============================================================"
echo "All threshold sweeps complete in ${total}s"
printf "  %-8s  %-12s  %s\n" "workload" "elapsed" "status"
printf "  %-8s  %-12s  %s\n" "--------" "-------" "------"
any_fail=0
for w in $WORKLOADS; do
    st="${STATUS[$w]:-not-run}"
    el="${ELAPSED[$w]:-0s}"
    printf "  %-8s  %-12s  %s\n" "$w" "$el" "$st"
    [[ "$st" == FAIL* ]] && any_fail=1
done
echo "============================================================"

exit $any_fail
