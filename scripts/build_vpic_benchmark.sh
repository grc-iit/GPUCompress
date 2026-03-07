#!/bin/bash
set -e

VPIC_DIR="/u/imuradli/vpic-kokkos"
GPU_DIR="/u/imuradli/GPUCompress"

export NVCC_WRAPPER_DEFAULT_COMPILER=/opt/cray/pe/craype/2.7.34/bin/CC

"${VPIC_DIR}/kokkos/bin/nvcc_wrapper" \
  -I"${GPU_DIR}/include" \
  -I"${GPU_DIR}/examples" \
  -I/tmp/hdf5-install/include \
  -I/opt/cray/pe/dsmml/0.3.1/dsmml/include \
  -I/opt/cray/pe/libsci/25.03.0/GNU/12.2/x86_64/include \
  -I/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/include \
  -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8/include \
  -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8/nvvm/include \
  -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8/extras/Debugger/include \
  -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8/extras/CUPTI/include \
  -I/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/math_libs/12.8/include \
  -I/opt/rh/gcc-toolset-13/root/usr/include/c++/13 \
  -I/opt/rh/gcc-toolset-13/root/usr/include/c++/13/x86_64-redhat-linux \
  -I/opt/rh/gcc-toolset-13/root/usr/include/c++/13/backward \
  -I/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13/include \
  -I/usr/local/include \
  -I/opt/rh/gcc-toolset-13/root/usr/include \
  -I/usr/include \
  -fopenmp \
  -I. \
  -I"${VPIC_DIR}/src" \
  -I"${VPIC_DIR}/build-benchmark/kokkos" \
  -I"${VPIC_DIR}/build-benchmark/kokkos/core/src" \
  -I"${VPIC_DIR}/kokkos/core/src" \
  -I"${VPIC_DIR}/kokkos/tpls/desul/include" \
  -I"${VPIC_DIR}/build-benchmark/kokkos/containers/src" \
  -I"${VPIC_DIR}/kokkos/containers/src" \
  -fopenmp -std=c++17 \
  -DINPUT_DECK='"'"${GPU_DIR}/tests/benchmarks/vpic_benchmark_deck.cxx"'"' \
  -DUSE_KOKKOS -DENABLE_KOKKOS_CUDA \
  -DUSE_LEGACY_PARTICLE_ARRAY -DVPIC_ENABLE_AUTO_TUNING \
  "${VPIC_DIR}/deck/main.cc" \
  "${VPIC_DIR}/deck/wrapper.cc" \
  -o vpic_benchmark_deck.Linux \
  -Wl,-rpath,"${VPIC_DIR}/build-benchmark" \
  -L"${VPIC_DIR}/build-benchmark" -lvpic \
  -lpthread -ldl \
  -L"${VPIC_DIR}/build-benchmark/kokkos" \
  -L"${VPIC_DIR}/build-benchmark/kokkos/core/src" \
  -lkokkoscore -lkokkoscontainers \
  -L"${GPU_DIR}/build" -lgpucompress -lH5VLgpucompress -lH5Zgpucompress \
  -L/tmp/hdf5-install/lib -lhdf5 \
  -L/tmp/lib -lnvcomp \
  -x cu -expt-extended-lambda -Wext-lambda-captures-this -arch=sm_80

echo "Built: vpic_benchmark_deck.Linux"
