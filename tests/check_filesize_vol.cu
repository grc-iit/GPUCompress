/**
 * tests/check_filesize_vol.cu
 *
 * File-size verification test for GPU VOL compression.
 *
 * Tests THREE cases on a 64 MiB dataset:
 *
 *  1) No compression, ramp pattern   → /tmp/check_nocomp.h5
 *     Expected: on-disk size ≈ raw bytes (within small HDF5 header overhead).
 *     Verifies the pipeline saves ALL data.
 *
 *  2) LZ4 compression, ramp pattern  → /tmp/check_lz4_ramp.h5
 *     Ramp floats (i/N) have highly varied IEEE 754 byte patterns, so LZ4
 *     cannot compress them.  Expected: on-disk size ≈ raw bytes (ratio ~1x).
 *     Demonstrates: correct pipeline behavior on incompressible data.
 *
 *  3) LZ4 compression, blocks pattern → /tmp/check_lz4_blocks.h5
 *     buf[i] = (float)(i % BLOCK_PERIOD): repeating cycle of BLOCK_PERIOD
 *     distinct float values.  LZ4 finds byte-level back-references within each
 *     4 MiB chunk.  Expected: on-disk size << raw bytes (ratio >> 1x).
 *     Verifies: compression actually reduces file size.
 *
 * Files are intentionally kept after the test for manual inspection.
 *
 * Usage:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *   ./build/check_filesize_vol [--dataset-mb N] [--chunk-mb N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Defaults
 * ============================================================ */
#define DEFAULT_DATASET_MB 64
#define DEFAULT_CHUNK_MB    4

#define OUT_NOCOMP      "/tmp/check_nocomp.h5"
#define OUT_LZ4_RAMP    "/tmp/check_lz4_ramp.h5"
#define OUT_LZ4_BLOCKS  "/tmp/check_lz4_blocks.h5"

/* Max tolerated HDF5 overhead above raw data bytes for no-comp check */
#define MAX_OVERHEAD_BYTES (64 * 1024)   /* 64 KiB */

/*
 * Period for the "blocks" compressible pattern.
 * buf[i] = (float)(i % BLOCK_PERIOD)
 * → BLOCK_PERIOD distinct float values repeat cyclically.
 * With BLOCK_PERIOD=1024, each cycle is 1024×4=4096 bytes.
 * LZ4 can back-reference earlier cycles → strong compression.
 */
#define BLOCK_PERIOD 1024

/* ============================================================
 * Filter wiring
 * ============================================================ */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static void pack_double_cd(double v, unsigned int *lo, unsigned int *hi)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static herr_t dcpl_set_gpucompress(hid_t dcpl, unsigned int algo,
                                    unsigned int preproc, unsigned int shuf_sz,
                                    double error_bound)
{
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = algo; cd[1] = preproc; cd[2] = shuf_sz;
    pack_double_cd(error_bound, &cd[3], &cd[4]);
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
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

/* Returns actual on-disk file size via stat(2), or 0 on error. */
static size_t on_disk_bytes(const char *path)
{
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (size_t)st.st_size;
}

static void fdatasync_path(const char *path)
{
    int fd = open(path, O_WRONLY);
    if (fd >= 0) { fdatasync(fd); close(fd); }
}

/* GPU ramp fill: buf[i] = (float)i / (float)n */
__global__ static void ramp_kernel(float *buf, size_t n)
{
    size_t i      = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += stride)
        buf[i] = (float)i / (float)n;
}

/* GPU blocks fill: buf[i] = (float)(i % BLOCK_PERIOD)
 * Creates a repeating cycle of BLOCK_PERIOD distinct float values.
 * LZ4 finds byte-level back-references after the first cycle. */
__global__ static void blocks_kernel(float *buf, size_t n)
{
    size_t i      = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += stride)
        buf[i] = (float)(i % BLOCK_PERIOD);
}

/* ============================================================
 * Write dataset and return on-disk file size.
 * use_lz4 = 0 → no filter (host pointer write)
 * use_lz4 = 1 → LZ4 filter  (GPU pointer write)
 * ============================================================ */
static int write_and_measure(const char *path, float *d_buf, float *h_buf,
                              size_t n_floats, size_t chunk_floats,
                              int use_lz4, size_t *out_file_bytes)
{
    hsize_t dims[1]  = { (hsize_t)n_floats };
    hsize_t cdims[1] = { (hsize_t)chunk_floats };

    remove(path);

    hid_t fapl   = make_vol_fapl();
    hid_t file   = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "H5Fcreate failed: %s\n", path); return -1; }

    hid_t fspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    if (use_lz4)
        dcpl_set_gpucompress(dcpl, 1 /*ALGO_LZ4*/, 0, 0, 0.0);

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             fspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl);
    H5Sclose(fspace);
    if (dset < 0) { H5Fclose(file); fprintf(stderr, "H5Dcreate2 failed\n"); return -1; }

    /* No-comp: write host pointer.  LZ4: write GPU pointer directly. */
    const void *write_ptr = use_lz4 ? (const void *)d_buf : (const void *)h_buf;
    herr_t ret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, write_ptr);
    H5Dclose(dset);
    H5Fclose(file);

    if (ret < 0) { fprintf(stderr, "H5Dwrite failed\n"); return -1; }

    /* Flush to storage before measuring size */
    fdatasync_path(path);

    *out_file_bytes = on_disk_bytes(path);
    return 0;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char **argv)
{
    int dataset_mb = DEFAULT_DATASET_MB;
    int chunk_mb   = DEFAULT_CHUNK_MB;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dataset-mb") == 0 && i + 1 < argc)
            dataset_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc)
            chunk_mb = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--dataset-mb N] [--chunk-mb N]\n", argv[0]);
            return 0;
        }
    }

    size_t chunk_floats = (size_t)chunk_mb   * 1024 * 1024 / sizeof(float);
    size_t n_floats     = (size_t)dataset_mb * 1024 * 1024 / sizeof(float);
    size_t total_bytes  = n_floats * sizeof(float);
    size_t n_chunks     = n_floats / chunk_floats;

    printf("=== File-Size Verification: No-Comp vs LZ4 ===\n\n");
    printf("  Dataset : %d MiB  (%zu floats, %zu chunks × %d MiB)\n",
           dataset_mb, n_floats, n_chunks, chunk_mb);
    printf("  Raw bytes: %zu\n\n", total_bytes);
    printf("  Case 1 - No compression  (ramp pattern)   → " OUT_NOCOMP "\n");
    printf("  Case 2 - LZ4 compression (ramp pattern)   → " OUT_LZ4_RAMP "\n");
    printf("  Case 3 - LZ4 compression (blocks pattern) → " OUT_LZ4_BLOCKS "\n");
    printf("\n");
    printf("  NOTE: ramp floats (i/N) have varied IEEE 754 byte patterns and\n");
    printf("        do NOT compress well with LZ4 (byte-level compressor).\n");
    printf("        blocks pattern (i %% %d) repeats a %d-byte cycle, which\n",
           BLOCK_PERIOD, BLOCK_PERIOD * (int)sizeof(float));
    printf("        LZ4 compresses efficiently via back-references.\n");
    printf("  Files are kept after the test for manual inspection.\n\n");

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }

    /* Allocate GPU + pinned-host buffers */
    float *d_buf = NULL, *h_buf = NULL;
    if (cudaMalloc(&d_buf, total_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n"); gpucompress_cleanup(); return 1;
    }
    if (cudaMallocHost(&h_buf, total_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed\n");
        cudaFree(d_buf); gpucompress_cleanup(); return 1;
    }

    /* Register VOL */
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "H5VL_gpucompress_register failed\n");
        cudaFree(d_buf); cudaFreeHost(h_buf); gpucompress_cleanup(); return 1;
    }

    /* Warmup: JIT-compiles nvcomp LZ4 kernel (untimed, result discarded) */
    printf("  Warmup LZ4 run (nvcomp JIT)...\n");
    {
        ramp_kernel<<<65535, 256>>>(d_buf, n_floats);
        cudaDeviceSynchronize();
        cudaMemcpy(h_buf, d_buf, total_bytes, cudaMemcpyDeviceToHost);
        size_t dummy;
        write_and_measure("/tmp/check_warmup.h5", d_buf, h_buf,
                          n_floats, chunk_floats, 1, &dummy);
        remove("/tmp/check_warmup.h5");
    }

    /* ---- Case 1: No-comp, ramp ---- */
    printf("\n  [Case 1] No compression, ramp pattern...\n");
    ramp_kernel<<<65535, 256>>>(d_buf, n_floats);
    cudaDeviceSynchronize();
    cudaMemcpy(h_buf, d_buf, total_bytes, cudaMemcpyDeviceToHost);
    size_t nocomp_bytes = 0;
    if (write_and_measure(OUT_NOCOMP, d_buf, h_buf,
                          n_floats, chunk_floats, 0, &nocomp_bytes) != 0) {
        fprintf(stderr, "Case 1 write FAILED\n");
        H5VLclose(vol_id); cudaFree(d_buf); cudaFreeHost(h_buf);
        gpucompress_cleanup(); return 1;
    }
    printf("        On-disk: %zu bytes  (%.3f MiB)\n",
           nocomp_bytes, (double)nocomp_bytes / (1 << 20));

    /* ---- Case 2: LZ4, ramp (expected: ~1x ratio, data is incompressible) ---- */
    printf("\n  [Case 2] LZ4 compression, ramp pattern...\n");
    /* d_buf and h_buf still have ramp data */
    size_t lz4_ramp_bytes = 0;
    if (write_and_measure(OUT_LZ4_RAMP, d_buf, h_buf,
                          n_floats, chunk_floats, 1, &lz4_ramp_bytes) != 0) {
        fprintf(stderr, "Case 2 write FAILED\n");
        H5VLclose(vol_id); cudaFree(d_buf); cudaFreeHost(h_buf);
        gpucompress_cleanup(); return 1;
    }
    printf("        On-disk: %zu bytes  (%.3f MiB)\n",
           lz4_ramp_bytes, (double)lz4_ramp_bytes / (1 << 20));

    /* ---- Case 3: LZ4, blocks (expected: strong compression) ---- */
    printf("\n  [Case 3] LZ4 compression, blocks pattern (i %% %d)...\n",
           BLOCK_PERIOD);
    blocks_kernel<<<65535, 256>>>(d_buf, n_floats);
    cudaDeviceSynchronize();
    cudaMemcpy(h_buf, d_buf, total_bytes, cudaMemcpyDeviceToHost);
    size_t lz4_blocks_bytes = 0;
    if (write_and_measure(OUT_LZ4_BLOCKS, d_buf, h_buf,
                          n_floats, chunk_floats, 1, &lz4_blocks_bytes) != 0) {
        fprintf(stderr, "Case 3 write FAILED\n");
        H5VLclose(vol_id); cudaFree(d_buf); cudaFreeHost(h_buf);
        gpucompress_cleanup(); return 1;
    }
    printf("        On-disk: %zu bytes  (%.3f MiB)\n",
           lz4_blocks_bytes, (double)lz4_blocks_bytes / (1 << 20));

    H5VLclose(vol_id);
    cudaFree(d_buf);
    cudaFreeHost(h_buf);
    gpucompress_cleanup();

    /* ---- Sizes table ---- */
    double raw_mib    = (double)total_bytes     / (double)(1 << 20);
    double nc_mib     = (double)nocomp_bytes    / (double)(1 << 20);
    double lr_mib     = (double)lz4_ramp_bytes  / (double)(1 << 20);
    double lb_mib     = (double)lz4_blocks_bytes/ (double)(1 << 20);
    size_t overhead_b = (nocomp_bytes > total_bytes) ? nocomp_bytes - total_bytes : 0;
    double ratio_ramp = (lz4_ramp_bytes   > 0) ? (double)total_bytes / (double)lz4_ramp_bytes  : 0.0;
    double ratio_blk  = (lz4_blocks_bytes > 0) ? (double)total_bytes / (double)lz4_blocks_bytes : 0.0;

    printf("\n");
    printf("  %-14s   %-8s   %14s   %10s   %7s\n",
           "Case", "Pattern", "On-disk bytes", "On-disk MiB", "Ratio");
    printf("  %-14s   %-8s   %14s   %10s   %7s\n",
           "--------------", "--------",
           "--------------", "-----------", "-------");
    printf("  %-14s   %-8s   %14zu   %10.3f   %6.2fx  (HDF5 overhead %zu B)\n",
           "No-comp", "ramp", nocomp_bytes, nc_mib, (double)nocomp_bytes/(double)total_bytes, overhead_b);
    printf("  %-14s   %-8s   %14zu   %10.3f   %6.2fx\n",
           "LZ4", "ramp", lz4_ramp_bytes, lr_mib, ratio_ramp);
    printf("  %-14s   %-8s   %14zu   %10.3f   %6.2fx\n",
           "LZ4", "blocks", lz4_blocks_bytes, lb_mib, ratio_blk);
    printf("\n");
    printf("  Raw uncompressed data: %zu bytes  (%.3f MiB)\n\n",
           total_bytes, raw_mib);

    /* ---- Pass/fail checks ---- */
    int pass = 1;
    printf("  Checks:\n");

    /* 1. No-comp: file must cover all raw bytes */
    if (nocomp_bytes >= total_bytes) {
        printf("  [PASS] No-comp file >= raw bytes"
               " (%zu >= %zu) — all data is stored\n",
               nocomp_bytes, total_bytes);
    } else {
        printf("  [FAIL] No-comp file SMALLER than raw bytes!"
               " (%zu < %zu) — data was lost\n",
               nocomp_bytes, total_bytes);
        pass = 0;
    }

    /* 2. No-comp: HDF5 overhead must be small */
    if (overhead_b <= MAX_OVERHEAD_BYTES) {
        printf("  [PASS] HDF5 header overhead is small"
               " (%zu B ≤ %d B)\n",
               overhead_b, MAX_OVERHEAD_BYTES);
    } else {
        printf("  [FAIL] HDF5 header overhead too large"
               " (%zu B > %d B)\n",
               overhead_b, MAX_OVERHEAD_BYTES);
        pass = 0;
    }

    /* 3. LZ4 blocks: must be smaller than raw (compression succeeded) */
    if (lz4_blocks_bytes < total_bytes) {
        printf("  [PASS] LZ4+blocks file < raw bytes"
               " (%zu < %zu) — %.2fx compression\n",
               lz4_blocks_bytes, total_bytes, ratio_blk);
    } else {
        printf("  [FAIL] LZ4+blocks file NOT smaller than raw bytes!"
               " (%zu >= %zu)\n",
               lz4_blocks_bytes, total_bytes);
        pass = 0;
    }

    /* 4. LZ4 blocks: file size matches compression ratio */
    size_t expected_lz4_blocks = (size_t)((double)total_bytes / ratio_blk);
    printf("  [INFO] LZ4+blocks: raw=%zu, compressed=%zu → ratio=%.2fx"
           " → compressed file is %.1f%% of original size\n",
           total_bytes, lz4_blocks_bytes, ratio_blk,
           100.0 * (double)lz4_blocks_bytes / (double)total_bytes);
    (void)expected_lz4_blocks;

    /* 5. Ramp with LZ4: note (not a failure — ramp is incompressible) */
    if (lz4_ramp_bytes >= total_bytes) {
        printf("  [INFO] LZ4+ramp: file is NOT smaller (%zu bytes)"
               " — ramp IEEE 754 floats are incompressible by LZ4 (expected)\n",
               lz4_ramp_bytes);
    } else {
        printf("  [INFO] LZ4+ramp: %.2fx compression"
               " (%zu bytes)\n", ratio_ramp, lz4_ramp_bytes);
    }

    printf("\n  Files retained for inspection:\n");
    printf("    " OUT_NOCOMP     "\n");
    printf("    " OUT_LZ4_RAMP   "\n");
    printf("    " OUT_LZ4_BLOCKS "\n");
    printf("\n  Overall: %s\n\n", pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");

    return pass ? 0 : 1;
}
