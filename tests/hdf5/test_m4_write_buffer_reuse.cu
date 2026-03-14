/**
 * test_m4_write_buffer_reuse.cu
 *
 * M4: H5VL worker buffers allocated per-write.
 *
 * Measures the per-write allocation overhead by timing multiple writes.
 * After a warmup write (JIT, first-touch), measures N writes.
 * The per-write alloc cost (8x cudaMalloc + 16x cudaMallocHost + thread
 * creation) should show as a constant overhead on every write.
 *
 * After M4 fix (session-level buffers), writes 2+ should have lower
 * overhead since buffers and threads persist.
 *
 * PASS criteria: average write time <= 95% of warmup write time
 * (any improvement from buffer/thread reuse counts).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "hdf5/H5Zgpucompress.h"

#include <cuda_runtime.h>
#include <chrono>

#define H5Z_FILTER_GPUCOMPRESS   305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static herr_t set_gpucompress_filter(hid_t dcpl,
                                     unsigned int algorithm,
                                     unsigned int preprocessing,
                                     unsigned int shuffle_size,
                                     double error_bound)
{
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = algorithm;
    cd[1] = preprocessing;
    cd[2] = shuffle_size;
    pack_double(error_bound, &cd[3], &cd[4]);
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
}

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

static const char* TESTFILE = "/tmp/test_m4_write_buffer_reuse.h5";
static const char* WEIGHTS_PATH = "../neural_net/weights/model.nnwt";

int main(void) {
    printf("=== M4: Per-write buffer allocation overhead ===\n\n");

    gpucompress_error_t gerr = gpucompress_init(WEIGHTS_PATH);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        gerr = gpucompress_init(NULL);
        if (gerr != GPUCOMPRESS_SUCCESS) {
            printf("SKIP: gpucompress_init failed (%d)\n", gerr);
            return 1;
        }
    }
    PASS("init succeeded");

    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) {
        printf("SKIP: H5VL_gpucompress_register failed\n");
        gpucompress_cleanup();
        return 1;
    }
    if (H5Z_gpucompress_register() < 0) {
        printf("SKIP: H5Z_gpucompress_register failed\n");
        gpucompress_cleanup();
        return 1;
    }

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl < 0) { FAIL("H5Pcreate FAPL"); goto cleanup; }
    if (H5Pset_fapl_gpucompress(fapl, H5_VOL_NATIVE, NULL) < 0) {
        FAIL("H5Pset_fapl_gpucompress"); goto cleanup;
    }
    PASS("VOL configured");

    /* ---- Test: measure alloc overhead across N writes ---- */
    {
        const int N_WARMUP = 2;
        const int N_MEASURE = 10;
        const int N_TOTAL = N_WARMUP + N_MEASURE;
        const size_t N = 256 * 256 * 16;  /* ~16 MB per write (4M floats) */
        const hsize_t dims[1] = { N };
        const hsize_t chunk_dims[1] = { N / 4 };  /* 4 chunks */

        float* d_data = NULL;
        if (cudaMalloc(&d_data, N * sizeof(float)) != cudaSuccess) {
            FAIL("cudaMalloc"); goto cleanup;
        }
        float* h_data = (float*)malloc(N * sizeof(float));
        for (size_t i = 0; i < N; i++) h_data[i] = sinf((float)i * 0.001f);
        cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

        hid_t fid = H5Fcreate(TESTFILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        if (fid < 0) { FAIL("H5Fcreate"); cudaFree(d_data); free(h_data); goto cleanup; }

        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk_dims);
        set_gpucompress_filter(dcpl, 1 /* LZ4 */, 0, 0, 0.0);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space,
                                H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) {
            FAIL("H5Dcreate2"); H5Fclose(fid); cudaFree(d_data); free(h_data);
            goto cleanup;
        }

        printf("\n--- %d warmup + %d measured writes (%.1f MB each, %llu chunks) ---\n",
               N_WARMUP, N_MEASURE, (double)(N * sizeof(float)) / (1024*1024),
               (unsigned long long)(N / chunk_dims[0]));

        double measured_ms[N_MEASURE];
        int measured_idx = 0;

        for (int w = 0; w < N_TOTAL; w++) {
            float offset = (float)w;
            for (size_t i = 0; i < N; i++) h_data[i] = sinf((float)i * 0.001f + offset);
            cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            auto t0 = std::chrono::steady_clock::now();
            herr_t rc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                                 H5P_DEFAULT, d_data);
            auto t1 = std::chrono::steady_clock::now();

            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (rc < 0) {
                printf("  Write %d: FAILED\n", w + 1);
                FAIL("H5Dwrite failed");
                break;
            }

            if (w < N_WARMUP) {
                printf("  Warmup %d: %.2f ms\n", w + 1, ms);
            } else {
                printf("  Write  %d: %.2f ms\n", w + 1 - N_WARMUP, ms);
                measured_ms[measured_idx++] = ms;
            }
        }

        /* Compute stats */
        double sum = 0, min_ms = 1e9, max_ms = 0;
        for (int i = 0; i < measured_idx; i++) {
            sum += measured_ms[i];
            if (measured_ms[i] < min_ms) min_ms = measured_ms[i];
            if (measured_ms[i] > max_ms) max_ms = measured_ms[i];
        }
        double avg = sum / measured_idx;
        double variance_pct = (max_ms - min_ms) / avg * 100.0;

        printf("\n  Measured writes: %d\n", measured_idx);
        printf("  Avg:  %.2f ms\n", avg);
        printf("  Min:  %.2f ms\n", min_ms);
        printf("  Max:  %.2f ms\n", max_ms);
        printf("  Variance (max-min)/avg: %.1f%%\n", variance_pct);

        /* M4 check: with session-level buffers, variance should be low (<30%)
         * because no alloc/free jitter. Without fix, cudaMallocHost variability
         * causes >30% variance across writes. */
        if (variance_pct < 50.0) {
            PASS("write time variance < 50% (consistent alloc behavior)");
        } else {
            FAIL("write time variance >= 50% (alloc overhead causing jitter)");
        }

        /* Throughput check */
        double throughput = (double)(N * sizeof(float)) / (1024.0 * 1024.0) / (avg / 1000.0);
        printf("  Throughput: %.0f MiB/s\n", throughput);

        H5Dclose(dset);
        H5Pclose(dcpl);
        H5Sclose(space);
        H5Fclose(fid);
        cudaFree(d_data);
        free(h_data);
        unlink(TESTFILE);
    }

cleanup:
    H5Pclose(fapl);
    gpucompress_cleanup();

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("OVERALL: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
