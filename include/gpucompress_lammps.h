/**
 * @file gpucompress_lammps.h
 * @brief LAMMPS/KOKKOS data adapter C API
 *
 * Thin data-access layer that borrows GPU device pointers from LAMMPS
 * KOKKOS views and passes them to gpucompress_compress_gpu().
 * The adapter does NOT own GPU memory — it borrows pointers via attach().
 *
 * Follows the same pattern as gpucompress_vpic.h and gpucompress_nyx.h.
 */

#ifndef GPUCOMPRESS_LAMMPS_H
#define GPUCOMPRESS_LAMMPS_H

#include "gpucompress.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Data types
 * ============================================================ */

/** LAMMPS field categories. */
typedef enum {
    LAMMPS_DATA_POSITION = 0,  /**< Atom positions (x,y,z) */
    LAMMPS_DATA_VELOCITY = 1,  /**< Atom velocities (vx,vy,vz) */
    LAMMPS_DATA_FORCE    = 2,  /**< Atom forces (fx,fy,fz) */
    LAMMPS_DATA_CUSTOM   = 3   /**< User-defined per-atom data */
} lammps_data_type_t;

/* ============================================================
 * Settings
 * ============================================================ */

/** LAMMPS adapter parameters. */
typedef struct {
    lammps_data_type_t data_type;    /**< Which LAMMPS data category */
    int                n_components; /**< Components per atom (3 for x/v/f) */
    int                element_size; /**< Bytes per element: 4 for float, 8 for double */
} LammpsSettings;

/** Return default settings (positions, 3 components, double). */
static inline LammpsSettings lammps_default_settings(void)
{
    LammpsSettings s;
    s.data_type    = LAMMPS_DATA_POSITION;
    s.n_components = 3;
    s.element_size = 8;  /* KK_FLOAT = double by default */
    return s;
}

/* ============================================================
 * Opaque handle
 * ============================================================ */

/** Opaque handle returned by create(). */
typedef struct gpucompress_lammps* gpucompress_lammps_t;

/* ============================================================
 * Lifecycle
 * ============================================================ */

gpucompress_lammps_t gpucompress_lammps_create(const LammpsSettings* settings);
void gpucompress_lammps_destroy(gpucompress_lammps_t handle);

/* ============================================================
 * Data attachment (borrow, not own)
 * ============================================================ */

/**
 * Attach a KOKKOS device pointer (from Kokkos::View::data()).
 * The adapter borrows this pointer — caller is responsible for lifetime.
 */
void gpucompress_lammps_attach(gpucompress_lammps_t handle,
                               const void* d_ptr,
                               size_t n_atoms);

const void* gpucompress_lammps_get_device_ptr(gpucompress_lammps_t handle);
size_t gpucompress_lammps_get_nbytes(gpucompress_lammps_t handle);

/* ============================================================
 * Compression (forwards to gpucompress_compress_gpu)
 * ============================================================ */

gpucompress_error_t gpucompress_compress_lammps(
    gpucompress_lammps_t handle,
    void** d_compressed,
    size_t* comp_bytes,
    gpucompress_algorithm_t algo,
    double error_bound);

#ifdef __cplusplus
}
#endif

#endif /* GPUCOMPRESS_LAMMPS_H */
