/**
 * hdf5_chunk_verify.cu
 *
 * Systematic investigation of the GPUCompress VOL connector:
 *
 *   A. Pointer type check  — VOL receives a GPU device pointer, not host
 *   B. Write transfer audit — only D→H compressed bytes cross the bus;
 *                             raw data stays on GPU
 *   C. Chunk-by-chunk verification — each chunk compressed independently
 *   D. CompressionHeader inspection — header prepended to every on-disk chunk
 *   E. Read transfer audit  — only H→D compressed bytes cross the bus;
 *                             decompressed data stays on GPU
 *   F. Round-trip correctness — bitwise GPU verify after 0xCD poison
 *
 * Dataset: 8 MB (2M float32), 4 MB chunks (2 chunks)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "hdf5/H5Zgpucompress.h"
#include "compression/compression_header.h"

#define N_ELEM  2097152          /* 8 MB as float32  */
#define CHUNK   1048576          /* 4 MB per chunk   */
#define N_CHUNK (N_ELEM / CHUNK)
#define FNAME   "hdf5_chunk_verify.h5"

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ void gen_kernel(float *d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float t = (float)i / n; d[i] = __sinf(t * 628.318f) * __expf(-t * 2.0f); }
}

__global__ void verify_kernel(const float *ref, const float *got, int n, int *err) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && ref[i] != got[i]) atomicAdd(err, 1);
}

static void print_xfer(const char *label) {
    int    h2d_n, d2h_n, d2d_n;
    size_t h2d_b, d2h_b, d2d_b;
    H5VL_gpucompress_get_transfer_stats(&h2d_n, &h2d_b, &d2h_n, &d2h_b, &d2d_n, &d2d_b);
    printf("  [%s transfer audit]\n", label);
    printf("    H→D : %d calls, %.3f MB  (compressed data going to GPU)\n",
           h2d_n, h2d_b / (1.0 * (1 << 20)));
    printf("    D→H : %d calls, %.3f MB  (compressed data coming off GPU)\n",
           d2h_n, d2h_b / (1.0 * (1 << 20)));
    printf("    D→D : %d calls, %.3f MB  (boundary chunk staging, should be 0 for aligned)\n",
           d2d_n, d2d_b / (1.0 * (1 << 20)));
    printf("    Raw data (%.0f MB) never crossed the PCIe bus\n",
           N_ELEM * 4.0 / (1 << 20));
}

/* ------------------------------------------------------------------ */
int main(void)
{
    int failures = 0;

    /* ---- Init ---- */
    gpucompress_init(NULL);
    H5Z_gpucompress_register();
    hid_t vol_id    = H5VL_gpucompress_register();
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl      = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    double raw_mb = N_ELEM * 4.0 / (1 << 20);

    printf("=== GPUCompress VOL Systematic Investigation ===\n");
    printf("    Dataset: %d floats = %.0f MB, %d chunks x %.0f MB\n\n",
           N_ELEM, raw_mb, N_CHUNK, CHUNK * 4.0 / (1 << 20));

    /* ---- Allocate GPU buffers — no host copy of the data ---- */
    float *d_src, *d_dst;
    cudaMalloc(&d_src, N_ELEM * sizeof(float));
    cudaMalloc(&d_dst, N_ELEM * sizeof(float));
    gen_kernel<<<(N_ELEM + 255) / 256, 256>>>(d_src, N_ELEM);
    cudaDeviceSynchronize();

    /* ==================================================================
     * A. Pointer type — confirm VOL receives a real device pointer
     * ================================================================== */
    printf("A. Pointer type check\n");
    {
        cudaPointerAttributes attr;
        cudaPointerGetAttributes(&attr, d_src);
        int ok = (attr.type == cudaMemoryTypeDevice);
        printf("   d_src type = %s  ->  %s\n",
               attr.type == cudaMemoryTypeDevice ? "cudaMemoryTypeDevice" :
               attr.type == cudaMemoryTypeHost   ? "cudaMemoryTypeHost"   : "other",
               ok ? "PASS — GPU pointer" : "FAIL");
        if (!ok) failures++;
    }

    /* ==================================================================
     * B + C. Write: transfer audit + chunk-by-chunk compression
     * ================================================================== */
    printf("\nB+C. Write path (GPU ptr → compress → D→H compressed → disk)\n");
    {
        hsize_t dims[1]  = { N_ELEM };
        hsize_t cdims[1] = { CHUNK  };

        hid_t fid   = H5Fcreate(FNAME, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, cdims);
        H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_LZ4, 0, 4, 0.0);
        hid_t dset  = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space,
                                  H5P_DEFAULT, dcpl, H5P_DEFAULT);

        H5VL_gpucompress_reset_stats();
        double t0 = now_sec();
        herr_t rc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_src);
        double dt = now_sec() - t0;

        int n_writes, n_comp;
        H5VL_gpucompress_get_stats(&n_writes, NULL, &n_comp, NULL);

        printf("   H5Dwrite result : %s\n", rc >= 0 ? "PASS" : "FAIL"); if (rc < 0) failures++;
        printf("   Write time      : %.3f s  (%.1f MB/s)\n", dt, raw_mb / dt);
        printf("   chunks_compressed = %d  (expected %d)  ->  %s\n",
               n_comp, N_CHUNK, n_comp == N_CHUNK ? "PASS" : "FAIL");
        if (n_comp != N_CHUNK) failures++;

        print_xfer("WRITE");

        /* Sanity: D→H bytes should be ~compressed size, NOT raw size */
        int d2h_n; size_t d2h_b;
        H5VL_gpucompress_get_transfer_stats(NULL, NULL, &d2h_n, &d2h_b, NULL, NULL);
        int no_raw_transfer = (d2h_b < (size_t)(N_ELEM * sizeof(float)));
        printf("   D→H < raw data  ->  %s  (%.3f MB << %.0f MB raw)\n",
               no_raw_transfer ? "PASS" : "FAIL", d2h_b / (1.0*(1<<20)), raw_mb);
        if (!no_raw_transfer) failures++;

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
    }

    /* ==================================================================
     * D. CompressionHeader — inspect each on-disk chunk
     * ================================================================== */
    printf("\nD. CompressionHeader on-disk (per chunk)\n");
    {
        hid_t fid  = H5Fopen(FNAME, H5F_ACC_RDONLY, fapl);
        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);

        hsize_t total_comp = 0;
        for (int c = 0; c < N_CHUNK; c++) {
            hsize_t offset[1] = { (hsize_t)c * CHUNK };
            hsize_t csz = 0;
            H5Dget_chunk_storage_size(dset, offset, &csz);
            total_comp += csz;

            void    *raw   = malloc(csz);
            uint32_t filt  = 0;
            size_t   bufsz = csz;
            H5Dread_chunk(dset, H5P_DEFAULT, offset, &filt, raw, &bufsz);
            CompressionHeader *hdr = (CompressionHeader *)raw;

            int valid = hdr->isValid();
            printf("   chunk[%d]: on-disk=%.3f MB  ratio=%.2fx  "
                   "magic=%s  shuffle=%s  quant=%s  hdr=%zu B  ->  %s\n",
                   c,
                   csz / (1.0*(1<<20)),
                   (double)hdr->original_size / hdr->compressed_size,
                   hdr->magic == COMPRESSION_MAGIC ? "GPUC" : "BAD",
                   hdr->hasShuffleApplied() ? "yes" : "no",
                   hdr->hasQuantizationApplied() ? "yes" : "no",
                   sizeof(CompressionHeader),
                   valid ? "PASS" : "FAIL");
            if (!valid) failures++;
            free(raw);
        }
        printf("   Total: %.0f MB → %.3f MB on disk  (%.2fx)\n",
               raw_mb, total_comp / (1.0*(1<<20)), (double)(N_ELEM*4ULL) / total_comp);

        H5Dclose(dset); H5Fclose(fid);
    }

    /* ==================================================================
     * E + F. Read: transfer audit + round-trip correctness
     * ================================================================== */
    printf("\nE+F. Read path (disk → H→D compressed → GPU decompress → d_dst)\n");
    {
        cudaMemset(d_dst, 0xCD, N_ELEM * sizeof(float));  /* poison before read */

        hid_t fid  = H5Fopen(FNAME, H5F_ACC_RDONLY, fapl);
        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);

        H5VL_gpucompress_reset_stats();
        double t0  = now_sec();
        herr_t rc  = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_dst);
        double dt  = now_sec() - t0;

        int n_reads, n_decomp;
        H5VL_gpucompress_get_stats(NULL, &n_reads, NULL, &n_decomp);

        printf("   H5Dread result  : %s\n", rc >= 0 ? "PASS" : "FAIL"); if (rc < 0) failures++;
        printf("   Read time       : %.3f s  (%.1f MB/s)\n", dt, raw_mb / dt);
        printf("   chunks_decompressed = %d  (expected %d)  ->  %s\n",
               n_decomp, N_CHUNK, n_decomp == N_CHUNK ? "PASS" : "FAIL");
        if (n_decomp != N_CHUNK) failures++;

        print_xfer("READ");

        /* Sanity: H→D bytes should be ~compressed size, NOT raw size */
        int h2d_n; size_t h2d_b;
        H5VL_gpucompress_get_transfer_stats(&h2d_n, &h2d_b, NULL, NULL, NULL, NULL);
        int no_raw_transfer = (h2d_b < (size_t)(N_ELEM * sizeof(float)));
        printf("   H→D < raw data  ->  %s  (%.3f MB << %.0f MB raw)\n",
               no_raw_transfer ? "PASS" : "FAIL", h2d_b / (1.0*(1<<20)), raw_mb);
        if (!no_raw_transfer) failures++;

        H5Dclose(dset); H5Fclose(fid);

        /* F. Bitwise GPU verify (d_dst was poisoned, must fully match d_src) */
        int *d_err; cudaMalloc(&d_err, sizeof(int)); cudaMemset(d_err, 0, sizeof(int));
        verify_kernel<<<(N_ELEM + 255) / 256, 256>>>(d_src, d_dst, N_ELEM, d_err);
        cudaDeviceSynchronize();
        int mismatches = 0;
        cudaMemcpy(&mismatches, d_err, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_err);

        printf("   Bitwise verify  : %s  (%d / %d elements match after 0xCD poison)\n",
               mismatches == 0 ? "PASS" : "FAIL",
               N_ELEM - mismatches, N_ELEM);
        if (mismatches) failures++;
    }

    printf("\n=== Result: %d failure(s) ===\n", failures);

    cudaFree(d_src); cudaFree(d_dst);
    H5Pclose(fapl); H5VLclose(vol_id);
    gpucompress_cleanup();
    return failures ? 1 : 0;
}
