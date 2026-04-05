/**
 * @file gpucompress_warpx.h
 * @brief WarpX/AMReX Particle-In-Cell data adapter C API
 *
 * Thin data-access layer that borrows GPU device pointers from AMReX
 * MultiFab FArrayBoxes (fields) and ParticleContainers (particles),
 * then passes them to gpucompress_compress_gpu().
 * The adapter does NOT own GPU memory -- it borrows pointers via attach().
 *
 * Follows the same pattern as gpucompress_nyx.h.
 */

#ifndef GPUCOMPRESS_WARPX_H
#define GPUCOMPRESS_WARPX_H

#include "gpucompress.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Data types
 * ============================================================ */

/** WarpX data categories. */
typedef enum {
    WARPX_DATA_EFIELD    = 0,  /**< Electric field (Ex, Ey, Ez) */
    WARPX_DATA_BFIELD    = 1,  /**< Magnetic field (Bx, By, Bz) */
    WARPX_DATA_JFIELD    = 2,  /**< Current density (jx, jy, jz) */
    WARPX_DATA_RHO       = 3,  /**< Charge density */
    WARPX_DATA_PARTICLES = 4,  /**< Particle arrays (positions, momenta, weights) */
    WARPX_DATA_CUSTOM    = 5   /**< User-defined diagnostic fields */
} warpx_data_type_t;

/* ============================================================
 * Settings
 * ============================================================ */

/** WarpX adapter parameters. */
typedef struct {
    warpx_data_type_t data_type;    /**< Which WarpX data category */
    int               n_components; /**< Number of variables (e.g. 3 for E-field) */
    int               element_size; /**< Bytes per element: 8 for double, 4 for float */
} WarpxSettings;

/** Return default settings (E-field, 3 components, double precision). */
static inline WarpxSettings warpx_default_settings(void)
{
    WarpxSettings s;
    s.data_type    = WARPX_DATA_EFIELD;
    s.n_components = 3;
    s.element_size = 8;  /* AMReX Real = double by default */
    return s;
}

/* ============================================================
 * Opaque handle
 * ============================================================ */

/** Opaque handle to a WarpX adapter instance. */
typedef struct gpucompress_warpx* gpucompress_warpx_t;

/* ============================================================
 * Lifecycle
 * ============================================================ */

/**
 * Create a WarpX adapter instance.
 *
 * @param handle   Output: opaque handle
 * @param settings Adapter parameters
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_warpx_create(
    gpucompress_warpx_t* handle,
    const WarpxSettings* settings);

/**
 * Destroy a WarpX adapter instance.
 */
gpucompress_error_t gpucompress_warpx_destroy(
    gpucompress_warpx_t handle);

/**
 * Attach a borrowed device pointer from an AMReX FArrayBox or particle array.
 *
 * @param handle     Adapter handle
 * @param d_data     Device pointer to data (fab.dataPtr() or particle SoA)
 * @param n_cells    Number of cells/particles in this data block
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_warpx_attach(
    gpucompress_warpx_t handle,
    void* d_data,
    size_t n_cells);

/* ============================================================
 * Data access
 * ============================================================ */

/**
 * Get the currently attached device pointer and byte size.
 *
 * @param handle  Adapter handle
 * @param d_data  Output: device pointer (can be NULL)
 * @param nbytes  Output: total data size in bytes (can be NULL)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_warpx_get_device_ptr(
    gpucompress_warpx_t handle,
    void** d_data,
    size_t* nbytes);

/**
 * Return total data size in bytes.
 */
gpucompress_error_t gpucompress_warpx_get_nbytes(
    gpucompress_warpx_t handle,
    size_t* nbytes);

/* ============================================================
 * Convenience: compress in one call
 * ============================================================ */

/**
 * Compress WarpX data directly from GPU device pointer.
 *
 * @param d_data      Device pointer to input data
 * @param nbytes      Size of input data in bytes
 * @param d_output    Pre-allocated GPU output buffer for compressed data
 * @param output_size [in] max size, [out] actual compressed size
 * @param config      Compression configuration
 * @param stats       Optional compression stats (can be NULL)
 * @return GPUCOMPRESS_SUCCESS or error code
 */
gpucompress_error_t gpucompress_compress_warpx(
    const void* d_data,
    size_t nbytes,
    void* d_output,
    size_t* output_size,
    const gpucompress_config_t* config,
    gpucompress_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_WARPX_H */
