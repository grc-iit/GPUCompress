/**
 * @file lammps_gpucompress_udf.h
 * @brief Simple C interface for calling GPUCompress from a LAMMPS fix
 *
 * This header provides a minimal C API that a LAMMPS custom fix can call.
 * The implementation lives in lammps_gpucompress_udf.cpp, compiled
 * separately as a shared library and linked via LD_PRELOAD or cmake.
 *
 * The fix extracts raw CUDA device pointers from KOKKOS views
 * via Kokkos::View::data() and passes them through this API.
 * GPUCompress's HDF5 VOL connector handles compression transparently.
 */

#ifndef LAMMPS_GPUCOMPRESS_UDF_H
#define LAMMPS_GPUCOMPRESS_UDF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize GPUCompress + HDF5 VOL connector.
 * Call once at simulation start.
 *
 * @param weights_path  Path to NN weights file
 * @param policy        Ranking policy: "speed", "balanced", or "ratio"
 * @return 0 on success, -1 on failure
 */
int gpucompress_lammps_init(const char* weights_path, const char* policy);

/**
 * Write a single GPU-resident field to compressed HDF5.
 *
 * @param filename      Output HDF5 file path
 * @param dset_name     Dataset name within the file
 * @param d_ptr         CUDA device pointer (from Kokkos::View::data())
 * @param n_elements    Number of elements (n_atoms * 3 for positions)
 * @param elem_bytes    Bytes per element (4 for float, 8 for double)
 * @param algo_name     Algorithm name string
 * @param error_bound   0.0 for lossless
 * @param verify        If nonzero, read back and verify bitwise
 * @return 0 on success, -1 on failure
 */
int gpucompress_lammps_write_field(const char* filename,
                                    const char* dset_name,
                                    const void* d_ptr,
                                    size_t n_elements,
                                    int elem_bytes,
                                    const char* algo_name,
                                    double error_bound,
                                    int verify);

/**
 * Finalize GPUCompress.
 */
void gpucompress_lammps_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* LAMMPS_GPUCOMPRESS_UDF_H */
