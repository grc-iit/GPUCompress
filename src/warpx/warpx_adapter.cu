/**
 * @file warpx_adapter.cu
 * @brief Host driver and C API for WarpX/AMReX PIC data adapter
 *
 * Borrows device pointers from AMReX MultiFab FArrayBoxes (EM fields,
 * current density, charge density) and ParticleContainers, then passes
 * them to gpucompress_compress_gpu(). No simulation logic lives here.
 *
 * Follows the same pattern as nyx_adapter.cu.
 */

#include <cstdio>
#include <cstdlib>
#include <new>

#include <cuda_runtime.h>

#include "gpucompress_warpx.h"

/* ============================================================
 * Internal struct
 * ============================================================ */

struct gpucompress_warpx {
    WarpxSettings settings;
    void*  d_data;       /* borrowed device pointer */
    size_t n_cells;      /* cells or particles in attached block */
    size_t nbytes;       /* n_cells * n_components * element_size */
};

/* ============================================================
 * C API implementation
 * ============================================================ */

extern "C" {

gpucompress_error_t gpucompress_warpx_create(
    gpucompress_warpx_t* handle,
    const WarpxSettings* settings)
{
    if (!handle || !settings) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (settings->n_components <= 0) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (settings->element_size != 4 && settings->element_size != 8)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;

    gpucompress_warpx* adapter = new (std::nothrow) gpucompress_warpx;
    if (!adapter) return GPUCOMPRESS_ERROR_OUT_OF_MEMORY;

    adapter->settings = *settings;
    adapter->d_data   = NULL;
    adapter->n_cells  = 0;
    adapter->nbytes   = 0;

    *handle = adapter;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_warpx_destroy(gpucompress_warpx_t handle)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    /* We don't own the device memory -- just delete the handle */
    delete handle;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_warpx_attach(
    gpucompress_warpx_t handle,
    void* d_data,
    size_t n_cells)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    if (!d_data && n_cells > 0) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    handle->d_data  = d_data;
    handle->n_cells = n_cells;
    handle->nbytes  = n_cells
                    * (size_t)handle->settings.n_components
                    * (size_t)handle->settings.element_size;

    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_warpx_get_device_ptr(
    gpucompress_warpx_t handle,
    void** d_data,
    size_t* nbytes)
{
    if (!handle) return GPUCOMPRESS_ERROR_INVALID_INPUT;

    if (d_data) *d_data = handle->d_data;
    if (nbytes) *nbytes = handle->nbytes;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_warpx_get_nbytes(
    gpucompress_warpx_t handle,
    size_t* nbytes)
{
    if (!handle || !nbytes) return GPUCOMPRESS_ERROR_INVALID_INPUT;
    *nbytes = handle->nbytes;
    return GPUCOMPRESS_SUCCESS;
}

gpucompress_error_t gpucompress_compress_warpx(
    const void* d_data,
    size_t nbytes,
    void* d_output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats)
{
    if (!d_data || !d_output || !output_size)
        return GPUCOMPRESS_ERROR_INVALID_INPUT;

    return gpucompress_compress_gpu(
        d_data, nbytes,
        d_output, output_size,
        config, stats, NULL);
}

} /* extern "C" */
