/**
 * tests/hdf5/test_vol_program_wall.cu
 *
 * Unit tests for H5VL_gpucompress_get_program_wall() and the derived
 * "compute_ms" value (program wall − accumulated VOL wall clock).
 *
 * Checks, all within a single process:
 *   (1) Before the first H5VL_gpucompress_register() call, the getter
 *       returns program_ms == 0 (timer not yet started).
 *   (2) Immediately after register() returns, program_ms > 0 and
 *       compute_ms ≈ program_ms (no VOL writes yet).
 *   (3) After an H5Dwrite, program_ms must be ≥ vol total_ms.
 *   (4) program_ms is monotonically non-decreasing across a sleep.
 *   (5) Across two sequential writes, both Δprogram and Δcompute are
 *       non-negative, and Δcompute ≤ Δprogram.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static herr_t dcpl_set_lz4(hid_t dcpl)
{
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS] = {0};
    cd[0] = 1;  /* LZ4 — no NN weights required */
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
}

static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl      = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

__global__ static void fill_kernel(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s) buf[i] = (float)i * 1e-6f;
}

static int do_one_write(const char *fname, float *d_data,
                         size_t n_floats, size_t chunk_floats)
{
    remove(fname);
    hsize_t dims[1]  = { (hsize_t)n_floats };
    hsize_t cdims[1] = { (hsize_t)chunk_floats };
    hid_t fapl = make_vol_fapl();
    hid_t f = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (f < 0) return -1;
    hid_t fs   = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    dcpl_set_lz4(dcpl);
    hid_t ds = H5Dcreate2(f, "data", H5T_NATIVE_FLOAT, fs,
                          H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl); H5Sclose(fs);
    if (ds < 0) { H5Fclose(f); return -1; }
    herr_t w = H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, d_data);
    cudaDeviceSynchronize();
    H5Dclose(ds); H5Fclose(f); remove(fname);
    return w < 0 ? -1 : 0;
}

int main(void)
{
    const char *tmp = "/tmp/test_vol_program_wall.h5";
    const size_t n_floats     = 1 * 1024 * 1024;   /* 4 MiB */
    const size_t chunk_floats =     512 * 1024;    /* 2 MiB chunks */
    const size_t total_bytes  = n_floats * sizeof(float);
    int fail = 0;

    printf("=== H5VL_gpucompress_get_program_wall() semantics ===\n\n");

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* (1) Before register: getter should return 0 */
    double prog0 = -1, comp0 = -1;
    H5VL_gpucompress_get_program_wall(&prog0, &comp0);
    printf("  (1) Before register: program_ms=%.3f compute_ms=%.3f\n", prog0, comp0);
    if (prog0 != 0) {
        printf("      FAIL: expected program_ms == 0 before register (got %.3f)\n", prog0);
        fail++;
    }
    if (comp0 != 0) {
        printf("      FAIL: expected compute_ms == 0 before register (got %.3f)\n", comp0);
        fail++;
    }

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "FATAL: H5VL_gpucompress_register failed\n"); return 1;
    }

    /* (2) After register: prog > 0, compute ≈ prog */
    double prog1 = 0, comp1 = 0;
    H5VL_gpucompress_get_program_wall(&prog1, &comp1);
    printf("  (2) After register:  program_ms=%.3f compute_ms=%.3f\n", prog1, comp1);
    if (prog1 <= 0) {
        printf("      FAIL: expected program_ms > 0 after register (got %.3f)\n", prog1);
        fail++;
    }
    if (prog1 - comp1 > 1.0) {
        printf("      FAIL: compute_ms (%.3f) diverges from program_ms (%.3f) "
               "by >1ms without any VOL writes\n", comp1, prog1);
        fail++;
    }

    /* (4) Monotonic: sleep, re-read, prog must not decrease */
    struct timespec ts = {0, 5 * 1000 * 1000};  /* 5 ms */
    nanosleep(&ts, NULL);
    double prog2 = 0, comp2 = 0;
    H5VL_gpucompress_get_program_wall(&prog2, &comp2);
    printf("  (4) After 5ms sleep: program_ms=%.3f (delta=%.3f) compute_ms=%.3f\n",
           prog2, prog2 - prog1, comp2);
    if (prog2 < prog1) {
        printf("      FAIL: program_ms went backwards (%.3f → %.3f)\n", prog1, prog2);
        fail++;
    }

    float *d_data = NULL;
    if (cudaMalloc(&d_data, total_bytes) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc failed\n");
        H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }
    fill_kernel<<<512, 256>>>(d_data, n_floats);
    cudaDeviceSynchronize();

    if (do_one_write(tmp, d_data, n_floats, chunk_floats) < 0) {
        fprintf(stderr, "FATAL: H5Dwrite #1 failed\n");
        cudaFree(d_data); H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }

    /* (3) After first write: prog_ms ≥ vol total_ms */
    double prog3 = 0, comp3 = 0;
    double vs1 = 0, vdrain = 0, viodrain = 0, vtotal = 0;
    H5VL_gpucompress_get_program_wall(&prog3, &comp3);
    H5VL_gpucompress_get_stage_timing(&vs1, &vdrain, &viodrain, &vtotal);
    printf("  (3) After write #1:  program_ms=%.3f compute_ms=%.3f "
           "vol_total_ms=%.3f\n", prog3, comp3, vtotal);
    if (prog3 < vtotal) {
        printf("      FAIL: program_ms (%.3f) < vol total_ms (%.3f)\n",
               prog3, vtotal);
        fail++;
    }
    if (comp3 < 0) {
        printf("      FAIL: compute_ms went negative (%.3f)\n", comp3);
        fail++;
    }

    /* (5) Across second write: Δprog ≥ 0, Δcomp ≥ 0, Δcomp ≤ Δprog */
    double prog_before_w2 = 0, comp_before_w2 = 0;
    H5VL_gpucompress_get_program_wall(&prog_before_w2, &comp_before_w2);

    if (do_one_write(tmp, d_data, n_floats, chunk_floats) < 0) {
        fprintf(stderr, "FATAL: H5Dwrite #2 failed\n");
        cudaFree(d_data); H5VLclose(vol_id); gpucompress_cleanup(); return 1;
    }

    double prog_after_w2 = 0, comp_after_w2 = 0;
    H5VL_gpucompress_get_program_wall(&prog_after_w2, &comp_after_w2);
    double dprog = prog_after_w2 - prog_before_w2;
    double dcomp = comp_after_w2 - comp_before_w2;
    printf("  (5) Δ across write #2: Δprogram=%.3fms Δcompute=%.3fms\n",
           dprog, dcomp);
    if (dprog < 0) {
        printf("      FAIL: program_ms went backwards across write #2 (%.3f → %.3f)\n",
               prog_before_w2, prog_after_w2); fail++;
    }
    if (dcomp < 0) {
        printf("      FAIL: compute_ms went backwards across write #2 (%.3f → %.3f)\n",
               comp_before_w2, comp_after_w2); fail++;
    }
    if (dcomp > dprog + 0.5) {
        printf("      FAIL: Δcompute (%.3f) > Δprogram (%.3f) — compute_ms "
               "is outrunning program_ms\n", dcomp, dprog); fail++;
    }

    cudaFree(d_data);
    H5VLclose(vol_id);
    gpucompress_cleanup();

    printf("\n  %s (%d failure(s))\n",
           fail == 0 ? "ALL PROGRAM-WALL CHECKS PASSED" : "PROGRAM-WALL CHECKS FAILED",
           fail);
    return fail == 0 ? 0 : 1;
}
