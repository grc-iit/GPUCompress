# udf.cmake — link GPUCompress libraries into the nekRS UDF
#
# This file is automatically included by nekRS's UDF build system
# when present in the case directory.
#
# Edit the paths below to match your installation:

set(GPUCOMPRESS_DIR "$ENV{GPUCOMPRESS_DIR}" CACHE PATH "GPUCompress source directory")
set(GPUCOMPRESS_BUILD "${GPUCOMPRESS_DIR}/build" CACHE PATH "GPUCompress build directory")
set(HDF5_DIR "$ENV{HDF5_DIR}" CACHE PATH "HDF5 install directory")

if(NOT GPUCOMPRESS_DIR OR GPUCOMPRESS_DIR STREQUAL "")
    message(FATAL_ERROR "Set GPUCOMPRESS_DIR environment variable or edit udf.cmake")
endif()

# Include paths for GPUCompress headers and the UDF bridge header
target_include_directories(udf PRIVATE
    ${GPUCOMPRESS_DIR}/include
    ${GPUCOMPRESS_DIR}/examples
    ${HDF5_DIR}/include
)

# Link the precompiled bridge library and dependencies
target_link_libraries(udf PRIVATE
    ${GPUCOMPRESS_DIR}/examples/libnekrs_gpucompress_udf.so
    ${GPUCOMPRESS_BUILD}/libgpucompress.so
    ${GPUCOMPRESS_BUILD}/libH5VLgpucompress.so
    ${GPUCOMPRESS_BUILD}/libH5Zgpucompress.so
    ${HDF5_DIR}/lib/libhdf5.so
)

# CUDA runtime (needed for cudaDeviceSynchronize, cudaMalloc, etc.)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(udf PRIVATE CUDA::cudart)
else()
    target_link_libraries(udf PRIVATE -L/usr/local/cuda/lib64 -lcudart)
endif()
