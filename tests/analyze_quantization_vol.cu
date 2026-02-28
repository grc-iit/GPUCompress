/**
 * tests/analyze_quantization_vol.cu
 *
 * Analysis of LZ4 + quantization through the VOL connector.
 * Uses POSITIVE-ONLY data patterns so original and restored values are both
 * positive and easy to compare directly.
 *
 * Four patterns (all positive):
 *   Chunk 0: ramp [0, 1)         — small values, tight range
 *   Chunk 1: ramp [0, 10)        — medium values, medium range
 *   Chunk 2: ramp [1000, 1001)   — large offset, range=1
 *   Chunk 3: |sin(x)| * 10       — [0, 10], always non-negative
 *
 * Tests both eb=0.1 and eb=0.01.
 * Shows first 20 elements of chunk 0 per bound: original, restored, error.
 *
 * Usage:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *   ./build/analyze_quantization_vol
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

#define N_CHUNKS   4
#define CHUNK_MB   4
#define SHOW_ROWS  20

#define OUT_PATH "/tmp/analyze_quant.h5"

/* ============================================================
 * Filter wiring
 * ============================================================ */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static void pack_double_cd(double v, unsigned int *lo, unsigned int *hi)
{
    uint64_t bits; memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static herr_t dcpl_set_lz4_quant(hid_t dcpl, double eb)
{
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 1;    /* ALGO_LZ4         */
    cd[1] = 0x10; /* PREPROC_QUANTIZE */
    cd[2] = 0;
    pack_double_cd(eb, &cd[3], &cd[4]);
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
}

/* ============================================================
 * GPU fill kernels — random positive values via Wang hash
 * ============================================================ */

/* Wang hash: maps index → pseudo-random uint32 */
__device__ static unsigned int wang_hash(unsigned int seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed ^= seed >> 4u;
    seed *= 0x27d4eb2du;
    seed ^= seed >> 15u;
    return seed;
}

/* Random float in [lo, hi) using wang_hash(i ^ seed) */
__device__ static float rand_float(size_t i, unsigned int seed, float lo, float hi)
{
    unsigned int h = wang_hash((unsigned int)(i ^ (size_t)seed));
    /* use 24 bits for float mantissa precision */
    float t = (float)(h >> 8) / (float)(1u << 24);
    return lo + t * (hi - lo);
}

/* Chunk 0: random [0.1, 1.0] */
__global__ static void rand01_kernel(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s)
        buf[i] = rand_float(i, 0xAABBu, 0.1f, 1.0f);
}

/* Chunk 1: random [0.1, 10.0] */
__global__ static void rand010_kernel(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s)
        buf[i] = rand_float(i, 0xCCDDu, 0.1f, 10.0f);
}

/* Chunk 2: random [1000, 1001] — large positive offset */
__global__ static void rand1000_kernel(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s)
        buf[i] = rand_float(i, 0xEEFFu, 1000.0f, 1001.0f);
}

/* Chunk 3: random [0.1, 10.0] with different seed */
__global__ static void rand010b_kernel(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s)
        buf[i] = rand_float(i, 0x1234u, 0.1f, 10.0f);
}

/* ============================================================
 * Helpers
 * ============================================================ */
static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl      = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

static size_t file_size_bytes(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (size_t)st.st_size;
}

static void chunk_stats(const float *orig, const float *rest,
                        size_t n, double eb,
                        double *max_err, size_t *viol,
                        float *dmin, float *dmax,
                        int *any_negative_restored)
{
    *max_err = 0.0; *viol = 0;
    *dmin = orig[0]; *dmax = orig[0];
    *any_negative_restored = 0;
    for (size_t i = 0; i < n; i++) {
        double e = fabs((double)orig[i] - (double)rest[i]);
        if (e > *max_err) *max_err = e;
        if (e > eb) (*viol)++;
        if (orig[i] < *dmin) *dmin = orig[i];
        if (orig[i] > *dmax) *dmax = orig[i];
        if (rest[i] < 0.0f) *any_negative_restored = 1;
    }
}

/* ============================================================
 * run_one_eb
 * ============================================================ */
static int run_one_eb(double eb,
                      const float *h_orig,
                      float *d_full,
                      size_t n_floats,
                      size_t chunk_floats,
                      size_t total_bytes)
{
    const char *pat_names[N_CHUNKS] = {"rand [0.1,1]", "rand [0.1,10]", "rand [1000,1001]", "rand [0.1,10]"};
    hsize_t dims[1]  = { (hsize_t)n_floats };
    hsize_t cdims[1] = { (hsize_t)chunk_floats };
    size_t raw_bytes = total_bytes;

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  Error bound: %.2f   step ≈ 2×eb×0.95 = %.4f%*s║\n",
           eb, 2.0*eb*0.95, (int)(eb < 0.05 ? 14 : 13), " ");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* Write */
    remove(OUT_PATH);
    {
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fcreate(OUT_PATH, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        if (file < 0) { fprintf(stderr, "  H5Fcreate failed\n"); return 0; }
        hid_t fsp  = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, cdims);
        dcpl_set_lz4_quant(dcpl, eb);
        hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, fsp,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Pclose(dcpl); H5Sclose(fsp);
        if (dset < 0) { H5Fclose(file); fprintf(stderr, "  H5Dcreate2 failed\n"); return 0; }
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_full);
        H5Dclose(dset);
        H5Fclose(file);
    }

    size_t comp_bytes = file_size_bytes(OUT_PATH);
    printf("  On-disk: %zu bytes  |  ratio: %.2fx\n\n",
           comp_bytes, (double)raw_bytes / (double)comp_bytes);

    /* Read back */
    float *d_rb = NULL;
    if (cudaMalloc(&d_rb, total_bytes) != cudaSuccess) {
        fprintf(stderr, "  cudaMalloc readback failed\n"); return 0;
    }
    {
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fopen(OUT_PATH, H5F_ACC_RDONLY, fapl); H5Pclose(fapl);
        hid_t dset = H5Dopen2(file, "data", H5P_DEFAULT);
        H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_rb);
        H5Dclose(dset); H5Fclose(file);
    }
    cudaDeviceSynchronize();

    float *h_rest = (float*)malloc(total_bytes);
    if (!h_rest) { cudaFree(d_rb); return 0; }
    cudaMemcpy(h_rest, d_rb, total_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_rb);

    /* Per-chunk stats */
    printf("  ┌──────────────────────┬──────────┬──────────┬─────────────┬──────────┬──────────┬────────┐\n");
    printf("  │ Pattern              │ Data min │ Data max │   Max error │  Violat. │ Neg.rest │ Result │\n");
    printf("  ├──────────────────────┼──────────┼──────────┼─────────────┼──────────┼──────────┼────────┤\n");

    int all_pass = 1;
    size_t total_viol = 0;
    for (int c = 0; c < N_CHUNKS; c++) {
        const float *oc = h_orig  + c * chunk_floats;
        const float *rc = h_rest  + c * chunk_floats;
        double max_err; size_t viol; float dmin, dmax; int neg;
        chunk_stats(oc, rc, chunk_floats, eb, &max_err, &viol, &dmin, &dmax, &neg);
        total_viol += viol;
        if (viol > 0) all_pass = 0;
        printf("  │ %-20s │ %8.4f │ %8.4f │ %11.6f │ %8zu │ %8s │ %6s │\n",
               pat_names[c], (double)dmin, (double)dmax,
               max_err, viol,
               neg ? "YES ✗" : "no  ✓",
               viol == 0 ? "PASS" : "FAIL");
    }
    printf("  └──────────────────────┴──────────┴──────────┴─────────────┴──────────┴──────────┴────────┘\n");
    printf("\n  Total violations: %zu / %zu   →  %s\n",
           total_viol, n_floats,
           all_pass ? "ALL WITHIN BOUND" : "BOUND VIOLATED");

    /* Element table for chunk 0 only — clearest to see */
    printf("\n  Chunk 0 (%s): first %d elements\n", pat_names[0], SHOW_ROWS);
    printf("  %8s  %12s  %12s  %12s  %5s\n",
           "index", "original", "restored", "error", "ok?");
    printf("  %8s  %12s  %12s  %12s  %5s\n",
           "--------", "------------", "------------", "------------", "-----");
    for (int i = 0; i < SHOW_ROWS; i++) {
        double err = fabs((double)h_orig[i] - (double)h_rest[i]);
        printf("  %8d  %12.6f  %12.6f  %12.6e  %s\n",
               i, (double)h_orig[i], (double)h_rest[i], err,
               err <= eb ? "PASS" : "FAIL");
    }


    free(h_rest);
    remove(OUT_PATH);
    return all_pass;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void)
{
    size_t chunk_floats = (size_t)CHUNK_MB * 1024 * 1024 / sizeof(float);
    size_t n_floats     = (size_t)N_CHUNKS * chunk_floats;
    size_t total_bytes  = n_floats * sizeof(float);

    printf("=== Quantization Analysis — Positive Data Only ===\n");
    printf("  %d chunks × %d MiB = %d MiB\n", N_CHUNKS, CHUNK_MB, N_CHUNKS*CHUNK_MB);
    printf("  Random positive floats via Wang hash — clearly distinct values per element\n\n");

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }

    float *d_full = NULL;
    if (cudaMalloc(&d_full, total_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n"); gpucompress_cleanup(); return 1;
    }

    /* Fill chunks with random positive patterns */
    float *p = d_full;
    rand01_kernel   <<<1024,256>>>(p, chunk_floats); p += chunk_floats;
    rand010_kernel  <<<1024,256>>>(p, chunk_floats); p += chunk_floats;
    rand1000_kernel <<<1024,256>>>(p, chunk_floats); p += chunk_floats;
    rand010b_kernel <<<1024,256>>>(p, chunk_floats);
    cudaDeviceSynchronize();

    float *h_orig = (float*)malloc(total_bytes);
    if (!h_orig) {
        fprintf(stderr, "malloc failed\n"); cudaFree(d_full); gpucompress_cleanup(); return 1;
    }
    cudaMemcpy(h_orig, d_full, total_bytes, cudaMemcpyDeviceToHost);

    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "H5VL_gpucompress_register failed\n");
        free(h_orig); cudaFree(d_full); gpucompress_cleanup(); return 1;
    }

    /* Warmup */
    printf("  Warmup...\n");
    {
        hsize_t dims[1] = {(hsize_t)n_floats}, cdims[1] = {(hsize_t)chunk_floats};
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fcreate("/tmp/aq_warmup.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        if (file >= 0) {
            hid_t fsp  = H5Screate_simple(1, dims, NULL);
            hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcpl, 1, cdims); dcpl_set_lz4_quant(dcpl, 0.1);
            hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, fsp,
                                     H5P_DEFAULT, dcpl, H5P_DEFAULT);
            H5Pclose(dcpl); H5Sclose(fsp);
            if (dset >= 0) {
                H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_full);
                H5Dclose(dset);
            }
            H5Fclose(file);
        }
        remove("/tmp/aq_warmup.h5");
    }

    double ebs[] = { 0.1, 0.01 };
    int overall = 1;
    for (int e = 0; e < 2; e++) {
        int pass = run_one_eb(ebs[e], h_orig, d_full,
                              n_floats, chunk_floats, total_bytes);
        if (!pass) overall = 0;
    }

    printf("\n══════════════════════════════════\n");
    printf("  Overall: %s\n", overall ? "ALL PASS" : "FAIL");
    printf("══════════════════════════════════\n\n");

    H5VLclose(vol_id);
    free(h_orig);
    cudaFree(d_full);
    gpucompress_cleanup();
    return overall ? 0 : 1;
}
