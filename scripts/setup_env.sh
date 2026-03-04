#!/bin/bash
# Source this file to set up the GPUCompress environment
# Usage: source scripts/setup_env.sh

export LD_LIBRARY_PATH=/tmp/lib:/tmp/hdf5-install/lib:${LD_LIBRARY_PATH}
export NVCOMP_INCLUDE_DIR=/tmp/include
export NVCOMP_LIB_DIR=/tmp/lib
export HDF5_VOL_DIR=/tmp/hdf5-install

echo "GPUCompress environment configured:"
echo "  LD_LIBRARY_PATH includes: /tmp/lib, /tmp/hdf5-install/lib"
echo "  NVCOMP_INCLUDE_DIR: ${NVCOMP_INCLUDE_DIR}"
echo "  NVCOMP_LIB_DIR: ${NVCOMP_LIB_DIR}"
echo "  HDF5_VOL_DIR: ${HDF5_VOL_DIR}"
