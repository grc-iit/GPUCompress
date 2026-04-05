###############################################################################
# GPUCompress 4n4g Simulation Docker Image
#
# CUDA-enabled runtime image with MPI + SSH for multi-node simulation.
# Bind-mounts the host GPUCompress and vpic-kokkos directories at runtime.
###############################################################################

FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        gfortran \
        openmpi-bin \
        libopenmpi-dev \
        openssh-server \
        openssh-client \
        python3 \
        python3-pip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- SSH setup for MPI communication between containers ----
RUN mkdir -p /var/run/sshd \
    && sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#StrictModes.*/StrictModes no/' /etc/ssh/sshd_config \
    && echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config \
    && echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && echo "LogLevel ERROR" >> /etc/ssh/ssh_config

# Generate host keys and a shared SSH key pair
RUN ssh-keygen -A \
    && mkdir -p /root/.ssh \
    && ssh-keygen -t rsa -b 2048 -f /root/.ssh/id_rsa -N "" \
    && cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys \
    && chmod 600 /root/.ssh/authorized_keys

# ---- Environment ----
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMP_NUM_THREADS=1

# Shared workspace for simulation data
RUN mkdir -p /workspace
WORKDIR /workspace

COPY entrypoint.sh /entrypoint.sh
COPY mpirun-wrapper.sh /usr/local/bin/mpirun
RUN chmod +x /entrypoint.sh /usr/local/bin/mpirun

EXPOSE 22

ENTRYPOINT ["/entrypoint.sh"]
CMD ["head"]
