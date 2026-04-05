#!/bin/bash
# Wrapper around mpirun that adds Docker cluster flags:
#   --hostfile, -x for env vars, --mca btl_tcp_if_include eth0
# This allows unmodified benchmark.sh to run across Docker containers.

EXTRA_ARGS=()

# Add hostfile if available and not already specified
if [ -f /workspace/hostfile ] && ! echo "$@" | grep -q -- "--hostfile"; then
    EXTRA_ARGS+=(--hostfile /workspace/hostfile)
fi

# Add network interface restriction for TCP BTL
EXTRA_ARGS+=(--mca btl_tcp_if_include eth0)
EXTRA_ARGS+=(--mca plm_rsh_no_tree_spawn 1)

# Map by slot (spread across nodes), don't bind (shared GPU)
EXTRA_ARGS+=(--map-by slot)
EXTRA_ARGS+=(--bind-to none)

# Forward all relevant environment variables to remote ranks
# Core MPI/runtime vars
for var in LD_LIBRARY_PATH OMPI_ALLOW_RUN_AS_ROOT OMPI_ALLOW_RUN_AS_ROOT_CONFIRM OMP_NUM_THREADS PATH; do
    [ -n "${!var}" ] && EXTRA_ARGS+=(-x "$var")
done

# Forward all VPIC_* GPUCOMPRESS_* CUDA_* SGD_* env vars (benchmark config)
while IFS='=' read -r key _; do
    case "$key" in
        VPIC_*|GPUCOMPRESS_*|CUDA_VISIBLE_DEVICES|SGD_*|EXPLORE_*)
            EXTRA_ARGS+=(-x "$key")
            ;;
    esac
done < <(env)

exec /usr/bin/mpirun "${EXTRA_ARGS[@]}" "$@"
