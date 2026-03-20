/**
 * @file test_nn_predict_vs_actual.cu
 * @brief Round-trip test: generate data on GPU, write via HDF5 VOL with
 *        ALGO_AUTO, read back, and compare NN predictions against actual
 *        metrics for all 4 outputs (ratio, comp_time, decomp_time, PSNR).
 *
 * Dataset: 2 MB (524288 floats), chunked at 1 MB (262144 floats) → 2 chunks.
 * PSNR is only shown when quantization is enabled (lossy mode).
 *
 * Usage:
 *   ./build/test_nn_predict_vs_actual model.nnwt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ── Constants ──────────────────────────────────────────────── */

#define DATASET_BYTES   (2 * 1024 * 1024)                  /* 2 MB */
#define CHUNK_BYTES     (1 * 1024 * 1024)                  /* 1 MB */
#define N_FLOATS        (DATASET_BYTES / sizeof(float))    /* 524288 */
#define CHUNK_FLOATS    (CHUNK_BYTES / sizeof(float))      /* 262144 */
#define N_CHUNKS        (N_FLOATS / CHUNK_FLOATS)          /* 2 */

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

#define TMP_FILE "/tmp/test_nn_predict.h5"

/* ── Action → string ────────────────────────────────────────── */

static const char *ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

static void action_str(int action, char *buf, size_t sz)
{
    if (action < 0) { snprintf(buf, sz, "none"); return; }
    int a = action % 8, q = (action/8)%2, s = (action/16)%2;
    snprintf(buf, sz, "%s%s%s", ALGO_NAMES[a], s?"+shuf":"", q?"+quant":"");
}

static int action_has_quant(int action)
{
    return (action >= 0) ? ((action / 8) % 2) : 0;
}

/* ── Timing ─────────────────────────────────────────────────── */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── GPU data generators ────────────────────────────────────── */

__global__ void gen_smooth_sine(float *out, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = sinf(6.2832f * 4.0f * (float)i / (float)n)
               * cosf(6.2832f * 7.0f * (float)i / (float)n);
}

__global__ void gen_random(float *out, size_t n, unsigned long long seed)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, 0, &st);
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = curand_normal(&st);
}

__global__ void gen_sparse(float *out, size_t n, unsigned long long seed)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, i, 0, &st);
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = (curand_uniform(&st) < 0.02f) ? curand_normal(&st) * 50.0f : 0.0f;
}

/* ── GPU bitwise compare ────────────────────────────────────── */

__global__ void count_mismatches(const float *a, const float *b,
                                  size_t n, unsigned long long *cnt)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long local = 0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (a[i] != b[i]) local++;
    atomicAdd(cnt, local);
}

/* ── GPU MSE kernel (for PSNR) ──────────────────────────────── */

__global__ void compute_mse_kernel(const float *orig, const float *recon,
                                    size_t n, double *sum_sq, double *sum_orig_sq)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    double local_se = 0.0, local_o2 = 0.0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        double diff = (double)orig[i] - (double)recon[i];
        local_se += diff * diff;
        local_o2 += (double)orig[i] * (double)orig[i];
    }
    atomicAdd(sum_sq, local_se);
    atomicAdd(sum_orig_sq, local_o2);
}

static double compute_psnr_gpu(const float *d_orig, const float *d_recon,
                                size_t n_floats)
{
    double *d_sum_sq, *d_sum_orig_sq;
    cudaMalloc(&d_sum_sq, sizeof(double));
    cudaMalloc(&d_sum_orig_sq, sizeof(double));
    cudaMemset(d_sum_sq, 0, sizeof(double));
    cudaMemset(d_sum_orig_sq, 0, sizeof(double));

    compute_mse_kernel<<<256,256>>>(d_orig, d_recon, n_floats,
                                     d_sum_sq, d_sum_orig_sq);
    cudaDeviceSynchronize();

    double h_se = 0.0, h_o2 = 0.0;
    cudaMemcpy(&h_se, d_sum_sq, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_o2, d_sum_orig_sq, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum_sq);
    cudaFree(d_sum_orig_sq);

    if (h_se == 0.0) return INFINITY;
    double mse = h_se / (double)n_floats;
    double max_sq = h_o2 / (double)n_floats;  /* signal power */
    return 10.0 * log10(max_sq / mse);
}

/* ── HDF5 helpers ───────────────────────────────────────────── */

static void pack_double_cd(double v, unsigned int *lo, unsigned int *hi)
{
    uint64_t bits; memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static hid_t make_dcpl_auto(void)
{
    hsize_t cdim = CHUNK_FLOATS;
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, &cdim);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0; cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

static hid_t make_vol_fapl(void)
{
    hid_t native = H5VLget_connector_id_by_name("native");
    hid_t fapl   = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native, NULL);
    H5VLclose(native);
    return fapl;
}

static size_t file_size_bytes(const char *path)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    off_t sz = lseek(fd, 0, SEEK_END);
    close(fd);
    return (sz < 0) ? 0 : (size_t)sz;
}

/* ── Run one pattern ────────────────────────────────────────── */

typedef struct {
    const char *name;
    void (*generate)(float*, size_t);
} Pattern;

static void launch_sine(float *d, size_t n)  { gen_smooth_sine<<<256,256>>>(d, n); }
static void launch_rand(float *d, size_t n)  { gen_random<<<256,256>>>(d, n, 42ULL); }
static void launch_sparse(float *d, size_t n){ gen_sparse<<<256,256>>>(d, n, 99ULL); }

static int run_pattern(const Pattern *pat, float *d_data, float *d_read,
                       unsigned long long *d_cnt)
{
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  Pattern: %s\n", pat->name);
    printf("════════════════════════════════════════════════════════════\n");

    /* Generate on GPU */
    pat->generate(d_data, N_FLOATS);
    cudaDeviceSynchronize();

    /* Reset diagnostics */
    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();

    /* ── Write via VOL ────────────────────────────────────── */
    remove(TMP_FILE);
    hid_t dcpl = make_dcpl_auto();
    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    hsize_t dims = N_FLOATS;
    hid_t fsp  = H5Screate_simple(1, &dims, NULL);
    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);

    double t0 = now_ms();
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    double write_ms = now_ms() - t0;
    H5Dclose(dset);
    H5Fclose(file);
    H5Pclose(dcpl);

    size_t fsize = file_size_bytes(TMP_FILE);
    double overall_ratio = (double)DATASET_BYTES / (double)(fsize > 0 ? fsize : 1);

    /* ── Read back via VOL (decompression) ────────────────── */
    fapl = make_vol_fapl();
    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);

    double t1 = now_ms();
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    double read_ms = now_ms() - t1;
    H5Dclose(dset);
    H5Fclose(file);

    /* ── Verify bitwise (lossless) or compute PSNR (lossy) ── */
    cudaMemset(d_cnt, 0, sizeof(unsigned long long));
    count_mismatches<<<256,256>>>(d_data, d_read, N_FLOATS, d_cnt);
    cudaDeviceSynchronize();
    unsigned long long mm = 0;
    cudaMemcpy(&mm, d_cnt, sizeof(mm), cudaMemcpyDeviceToHost);

    /* Compute per-chunk PSNR on GPU (only meaningful if lossy) */
    double chunk_psnr[N_CHUNKS];
    for (size_t c = 0; c < N_CHUNKS; c++) {
        chunk_psnr[c] = compute_psnr_gpu(
            d_data + c * CHUNK_FLOATS,
            d_read + c * CHUNK_FLOATS,
            CHUNK_FLOATS);
    }

    /* ── Print per-chunk NN predictions vs actual ─────────── */
    int n_hist = gpucompress_get_chunk_history_count();

    printf("\n  Per-chunk results (%d chunks, 1 MB each):\n", n_hist);
    printf("  ──────────────────────────────────────────────────────────────────────────────\n");

    for (int c = 0; c < n_hist; c++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(c, &d) != 0) continue;

        char algo[40];
        action_str(d.nn_action, algo, sizeof(algo));
        int lossy = action_has_quant(d.nn_action);

        printf("\n  Chunk %d/%d  [%s]%s\n", c+1, n_hist, algo,
               lossy ? "  (lossy)" : "  (lossless)");
        printf("  %-14s  %12s  %12s  %9s\n",
               "Metric", "Predicted", "Actual", "Error");
        printf("  ─────────────────────────────────────────────────\n");

        /* Ratio */
        double ratio_err = (d.actual_ratio > 0)
            ? (d.predicted_ratio - d.actual_ratio) / d.actual_ratio * 100.0 : 0.0;
        printf("  %-14s  %10.3fx  %10.3fx  %+7.1f%%\n",
               "ratio", (double)d.predicted_ratio, (double)d.actual_ratio, ratio_err);

        /* Compression time */
        double comp_err = (d.compression_ms > 0)
            ? (d.predicted_comp_time - d.compression_ms) / d.compression_ms * 100.0 : 0.0;
        printf("  %-14s  %9.3f ms  %9.3f ms  %+7.1f%%\n",
               "comp_time", (double)d.predicted_comp_time, (double)d.compression_ms, comp_err);

        /* Decompression time */
        double decomp_err = (d.decompression_ms > 0)
            ? (d.predicted_decomp_time - d.decompression_ms) / d.decompression_ms * 100.0 : 0.0;
        if (d.decompression_ms > 0) {
            printf("  %-14s  %9.3f ms  %9.3f ms  %+7.1f%%\n",
                   "decomp_time", (double)d.predicted_decomp_time,
                   (double)d.decompression_ms, decomp_err);
        } else {
            printf("  %-14s  %9.3f ms  %12s\n",
                   "decomp_time", (double)d.predicted_decomp_time, "n/a");
        }

        /* PSNR — only show if lossy (quantization enabled) */
        if (lossy && c < (int)N_CHUNKS) {
            double psnr_actual = chunk_psnr[c];
            double psnr_err = (psnr_actual > 0 && isfinite(psnr_actual))
                ? (d.predicted_psnr - psnr_actual) / psnr_actual * 100.0 : 0.0;
            if (isfinite(psnr_actual)) {
                printf("  %-14s  %8.1f dB  %8.1f dB  %+7.1f%%\n",
                       "psnr", (double)d.predicted_psnr, psnr_actual, psnr_err);
            } else {
                printf("  %-14s  %8.1f dB  %12s\n",
                       "psnr", (double)d.predicted_psnr, "inf (exact)");
            }
        }
    }

    /* ── Summary ──────────────────────────────────────────── */
    printf("\n  Overall:  write=%.1f ms (%.0f MiB/s)  read=%.1f ms (%.0f MiB/s)\n",
           write_ms, (double)DATASET_BYTES / (1<<20) / (write_ms/1000.0),
           read_ms,  (double)DATASET_BYTES / (1<<20) / (read_ms/1000.0));
    printf("            file=%.1f KB  ratio=%.2fx  mismatches=%llu  %s\n",
           (double)fsize / 1024.0, overall_ratio, mm,
           mm == 0 ? "PASS (lossless roundtrip)" : "LOSSY (mismatches expected if quant)");

    remove(TMP_FILE);
    return 0;  /* lossy mismatches are OK when quant is enabled */
}

/* ── Main ───────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    const char *weights = (argc > 1) ? argv[1] : getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) {
        fprintf(stderr, "Usage: %s <weights.nnwt>\n", argv[0]);
        return 1;
    }

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  NN Prediction vs Actual — 2 MB dataset, 1 MB chunks   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("  Dataset : %zu floats = %zu KB\n", (size_t)N_FLOATS, (size_t)DATASET_BYTES/1024);
    printf("  Chunks  : %zu x %zu KB\n", (size_t)N_CHUNKS, (size_t)CHUNK_BYTES/1024);
    printf("  Weights : %s\n", weights);

    if (gpucompress_init(weights) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    hid_t vol = H5VL_gpucompress_register();
    if (vol == H5I_INVALID_HID) {
        fprintf(stderr, "FATAL: VOL register failed\n");
        gpucompress_cleanup(); return 1;
    }
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* NN inference only — no SGD or exploration */
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);

    /* Allocate GPU buffers */
    float *d_data = NULL, *d_read = NULL;
    unsigned long long *d_cnt = NULL;
    cudaMalloc(&d_data, DATASET_BYTES);
    cudaMalloc(&d_read, DATASET_BYTES);
    cudaMalloc(&d_cnt,  sizeof(unsigned long long));

    Pattern patterns[] = {
        { "smooth_sine",     launch_sine   },
        { "gaussian_random", launch_rand   },
        { "sparse_2pct",     launch_sparse },
    };
    int n_patterns = sizeof(patterns) / sizeof(patterns[0]);

    for (int i = 0; i < n_patterns; i++)
        run_pattern(&patterns[i], d_data, d_read, d_cnt);

    /* Cleanup */
    cudaFree(d_data);
    cudaFree(d_read);
    cudaFree(d_cnt);
    H5VLclose(vol);
    gpucompress_cleanup();

    printf("\n=== DONE ===\n");
    return 0;
}
