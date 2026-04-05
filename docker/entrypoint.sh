#!/bin/bash
set -e

ROLE="${1:-worker}"
NUM_WORKERS="${NUM_WORKERS:-3}"

# ---- Shared SSH keys across the cluster ----
SSH_SHARED="/workspace/.ssh"
mkdir -p "$SSH_SHARED"

if [ "$ROLE" = "head" ]; then
    if [ ! -f "$SSH_SHARED/id_rsa" ]; then
        ssh-keygen -t rsa -b 2048 -f "$SSH_SHARED/id_rsa" -N ""
    fi
fi

# Wait for shared keys to appear (workers may start before head)
for i in $(seq 1 30); do
    [ -f "$SSH_SHARED/id_rsa" ] && break
    sleep 1
done

# Install shared keys
cp "$SSH_SHARED/id_rsa"     /root/.ssh/id_rsa
cp "$SSH_SHARED/id_rsa.pub" /root/.ssh/id_rsa.pub
cat "$SSH_SHARED/id_rsa.pub" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/id_rsa /root/.ssh/authorized_keys

# ---- Environment for MPI-launched processes ----
# Enable SSH user environment so mpirun-spawned processes inherit library paths
sed -i 's/#PermitUserEnvironment.*/PermitUserEnvironment yes/' /etc/ssh/sshd_config
grep -q "PermitUserEnvironment" /etc/ssh/sshd_config || echo "PermitUserEnvironment yes" >> /etc/ssh/sshd_config

cat > /root/.ssh/environment <<'ENVEOF'
LD_LIBRARY_PATH=/tmp/hdf5-install/lib:/opt/GPUCompress/build:/tmp/lib:/usr/local/cuda/lib64
OMPI_ALLOW_RUN_AS_ROOT=1
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
OMP_NUM_THREADS=1
PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENVEOF

# ---- Start SSH daemon ----
/usr/sbin/sshd

echo "[$HOSTNAME] SSH daemon started (role=$ROLE)"

if [ "$ROLE" = "head" ]; then
    # ---- Build the hostfile ----
    echo "[$HOSTNAME] Waiting for worker nodes..."

    HOSTFILE="/workspace/hostfile"
    echo "$HOSTNAME slots=${SLOTS_PER_NODE:-4}" > "$HOSTFILE"

    for i in $(seq 1 "$NUM_WORKERS"); do
        WORKER="gpucompress-worker-${i}"
        echo "  Waiting for $WORKER ..."
        for attempt in $(seq 1 60); do
            if ssh -o ConnectTimeout=2 "$WORKER" hostname &>/dev/null; then
                echo "$WORKER slots=${SLOTS_PER_NODE:-4}" >> "$HOSTFILE"
                echo "  $WORKER is ready."
                break
            fi
            sleep 2
        done
    done

    echo ""
    echo "===== GPUCompress MPI Cluster Ready ====="
    echo "Hostfile:"
    cat "$HOSTFILE"
    echo ""
    echo "GPUCompress:  /opt/GPUCompress"
    echo "VPIC:         /opt/vpic-kokkos"
    echo "Workspace:    /workspace"
    echo "=========================================="
    echo ""

    # Keep the head node alive
    exec tail -f /dev/null
else
    # Worker: just keep SSH running
    exec tail -f /dev/null
fi
