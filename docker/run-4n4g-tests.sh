#!/bin/bash
###############################################################################
# run-4n4g-tests.sh — Run all 4n4g benchmark scripts in the Docker cluster
#
# This script:
#   1. Builds and starts the 4-node Docker cluster (if not running)
#   2. Runs each 4n4g configuration inside the head container
#   3. Reports pass/fail for each
#
# Usage:
#   bash docker/run-4n4g-tests.sh [--quick]
#
# Options:
#   --quick    Use smaller grid (NX=64) and fewer timesteps (3) for fast test
#   --down     Tear down the cluster
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"

QUICK=false
TEARDOWN=false
NX=128
TS=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)    QUICK=true; NX=64; TS=3; shift ;;
        --down)     TEARDOWN=true; shift ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $TEARDOWN; then
    echo ">> Tearing down GPUCompress cluster..."
    docker compose -f "$COMPOSE_FILE" down -v
    echo "Done."
    exit 0
fi

# ---- Build & start cluster ----
echo "===== Building GPUCompress Docker images ====="
docker compose -f "$COMPOSE_FILE" build

echo ""
echo "===== Starting GPUCompress cluster (1 head + 3 workers) ====="
docker compose -f "$COMPOSE_FILE" up -d

echo ""
echo ">> Waiting for cluster to initialize..."
for i in $(seq 1 90); do
    if docker exec gpucompress-head test -f /workspace/hostfile 2>/dev/null; then
        LINES=$(docker exec gpucompress-head wc -l /workspace/hostfile | awk '{print $1}')
        if [ "$LINES" -ge 4 ]; then
            echo ">> Cluster ready! ($LINES nodes in hostfile)"
            break
        fi
    fi
    sleep 2
done

echo ""
docker exec gpucompress-head cat /workspace/hostfile
echo ""

# Verify GPU access inside container
echo ">> Verifying GPU access inside containers..."
docker exec gpucompress-head nvidia-smi --query-gpu=name --format=csv,noheader
echo ""

# ---- Test configurations ----
# Each test: name, POLICIES, LOSSY
declare -a TESTS=(
    "lossless_balanced|balanced|0"
    "lossless_ratio|ratio|0"
    "lossy_balanced|balanced|0.01"
    "lossy_ratio|ratio|0.01"
)

PASS=0
FAIL=0
RESULTS=()

for test_spec in "${TESTS[@]}"; do
    IFS='|' read -r NAME POLICY LOSSY <<< "$test_spec"

    echo ""
    echo "============================================================"
    echo "  TEST: 4n4g_${NAME}"
    echo "  Policy=$POLICY, Lossy=$LOSSY, NX=$NX, Timesteps=$TS"
    echo "============================================================"
    echo ""

    # Run inside the head container
    if docker exec gpucompress-head bash -c "
        cd /opt/GPUCompress

        export LD_LIBRARY_PATH=/tmp/hdf5-install/lib:/opt/GPUCompress/build:/tmp/lib:\${LD_LIBRARY_PATH}
        export OMPI_ALLOW_RUN_AS_ROOT=1
        export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        export OMP_NUM_THREADS=1

        # Tell OpenMPI to use the Docker cluster hostfile and allow SSH across containers
        export OMPI_MCA_orte_default_hostfile=/workspace/hostfile
        export OMPI_MCA_plm_rsh_no_tree_spawn=1
        export OMPI_MCA_btl_tcp_if_include=eth0
        export OMPI_MCA_plm=rsh

        MPI_NP=16 \
        GPUS_PER_NODE=1 \
        LAUNCHER=mpirun \
        VPIC_NX=$NX \
        CHUNK_MB=32 \
        TIMESTEPS=$TS \
        VERIFY=1 \
        POLICIES=$POLICY \
        LOSSY=$LOSSY \
        BENCHMARKS=vpic \
        PHASES=no-comp,lz4,zstd,nn,nn-rl \
        bash scripts/run_vpic_scaling.sh
    " 2>&1 | tee "/tmp/4n4g_${NAME}.log" | tail -30; then
        echo ""
        echo "  >> PASS: 4n4g_${NAME}"
        PASS=$((PASS + 1))
        RESULTS+=("PASS  4n4g_${NAME}")
    else
        echo ""
        echo "  >> FAIL: 4n4g_${NAME}"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  4n4g_${NAME}")
    fi
done

echo ""
echo "============================================================"
echo "  4n4g Test Summary"
echo "============================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "  Passed: $PASS / $((PASS + FAIL))"
echo "  Failed: $FAIL / $((PASS + FAIL))"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Logs saved to /tmp/4n4g_*.log"
    exit 1
fi
