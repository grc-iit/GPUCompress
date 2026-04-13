/**
 * @file lammps_adapter.cu
 * @brief Host driver and C API for LAMMPS/KOKKOS data adapter
 *
 * Borrows device pointers from LAMMPS KOKKOS views and passes
 * them to gpucompress_compress_gpu(). No simulation logic lives here.
 *
 * Follows the same pattern as vpic_adapter.cu and nyx_adapter.cu.
 */

#include <cstdio>
#include <cstdlib>
#include <new>

#include <cuda_runtime.h>

#include "gpucompress_lammps.h"

/* ============================================================
 * Internal struct
 * ============================================================ */

struct gpucompress_lammps {
    LammpsSettings settings;
    const void*    d_data;     /* borrowed device pointer (from Kokkos::View::data()) */
    size_t         n_atoms;    /* atoms in attached field */
    size_t         nbytes;     /* n_atoms * n_components * element_size */
};

/* ============================================================
 * C API implementation
 * ============================================================ */

extern "C" {

gpucompress_lammps_t gpucompress_lammps_create(const LammpsSettings* settings)
{
    if (!settings) return NULL;
    if (settings->n_components <= 0) return NULL;
    if (settings->element_size != 4 && settings->element_size != 8)
        return NULL;

    gpucompress_lammps* adapter = new (std::nothrow) gpucompress_lammps;
    if (!adapter) return NULL;

    adapter->settings = *settings;
    adapter->d_data   = NULL;
    adapter->n_atoms  = 0;
    adapter->nbytes   = 0;

    return adapter;
}

void gpucompress_lammps_destroy(gpucompress_lammps_t handle)
{
    delete handle;
}

void gpucompress_lammps_attach(gpucompress_lammps_t handle,
                               const void* d_ptr,
                               size_t n_atoms)
{
    if (!handle) return;

    handle->d_data  = d_ptr;
    handle->n_atoms = n_atoms;
    handle->nbytes  = n_atoms
                    * (size_t)handle->settings.n_components
                    * (size_t)handle->settings.element_size;
}

const void* gpucompress_lammps_get_device_ptr(gpucompress_lammps_t handle)
{
    return handle ? handle->d_data : NULL;
}

size_t gpucompress_lammps_get_nbytes(gpucompress_lammps_t handle)
{
    return handle ? handle->nbytes : 0;
}

gpucompress_error_t gpucompress_compress_lammps(
    gpucompress_lammps_t handle,
    void** d_compressed,
    size_t* comp_bytes,
    gpucompress_algorithm_t algo,
    double error_bound)
{
    if (!handle || !d_compressed || !comp_bytes)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (!handle->d_data || handle->nbytes == 0)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;

    gpucompress_config_t config = gpucompress_default_config();
    config.algorithm     = algo;
    config.error_bound   = error_bound;
    config.preprocessing = GPUCOMPRESS_PREPROC_NONE;
    if (handle->settings.element_size == 4)
        config.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;

    gpucompress_stats_t stats;

    return gpucompress_compress_gpu(
        handle->d_data, handle->nbytes,
        *d_compressed, comp_bytes,
        &config, &stats, NULL);
}

} /* extern "C" */
