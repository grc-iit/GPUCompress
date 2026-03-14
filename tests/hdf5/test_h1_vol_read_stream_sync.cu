/**
 * test_h1_vol_read_stream_sync.cu
 *
 * H1: cudaDeviceSynchronize() in VOL read loop.
 *
 * The VOL read path calls cudaDeviceSynchronize() twice per chunk,
 * which serializes ALL GPU streams. This should be replaced with
 * stream-scoped synchronization.
 *
 * Test: Write a multi-chunk dataset via the VOL, read it back with a
 * GPU pointer (exercises the decompress loop), and verify round-trip
 * correctness across many chunks.
 *
 * Run: ./test_h1_vol_read_stream_sync
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "hdf5/H5Zgpucompress.h"

#include <cuda_runtime.h>

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

static const char* TESTFILE = "/tmp/test_h1_vol_read_stream.h5";
static const char* WEIGHTS_PATH = "../neural_net/weights/model.nnwt";

int main(void) {
    printf("=== H1: VOL read stream sync (multi-chunk) ===\n\n");

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
        FAIL("H5Pset_fapl_gpucompress");
        goto cleanup;
    }
    PASS("VOL configured");

    /* ---- Test 1: Multi-chunk write + GPU read round-trip ---- */
    printf("\n--- Test 1: 16-chunk write + GPU read round-trip ---\n");
    {
        const int N = 16384;           /* 16384 floats = 64 KB */
        const int CHUNK = 1024;        /* 1024 floats/chunk = 16 chunks */

        hsize_t dims[1] = {(hsize_t)N};
        hsize_t chunk[1] = {(hsize_t)CHUNK};

        /* Generate test data */
        float* h_data = (float*)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++)
            h_data[i] = sinf((float)i * 0.01f) * 100.0f;

        float* d_data = NULL;
        cudaMalloc(&d_data, N * sizeof(float));
        cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

        /* Create file + chunked dataset */
        hid_t fid = H5Fcreate(TESTFILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        if (fid < 0) { FAIL("H5Fcreate"); goto cleanup; }

        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        set_gpucompress_filter(dcpl, 1 /* LZ4 */, 0, 0, 0.0);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space,
                                H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) {
            FAIL("H5Dcreate2");
            H5Pclose(dcpl); H5Sclose(space); H5Fclose(fid);
            goto cleanup;
        }

        /* Write from GPU */
        herr_t werr = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, d_data);
        if (werr < 0) {
            FAIL("H5Dwrite GPU");
        } else {
            PASS("16-chunk GPU write succeeded");
        }

        H5Dclose(dset);
        H5Pclose(dcpl);
        H5Sclose(space);
        H5Fclose(fid);
        cudaFree(d_data);

        /* Re-open and read back to GPU — this exercises the decompress loop */
        fid = H5Fopen(TESTFILE, H5F_ACC_RDONLY, fapl);
        if (fid < 0) { FAIL("H5Fopen"); free(h_data); goto cleanup; }

        dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        if (dset < 0) {
            FAIL("H5Dopen2");
            H5Fclose(fid);
            free(h_data);
            goto cleanup;
        }

        float* d_read = NULL;
        cudaMalloc(&d_read, N * sizeof(float));
        cudaMemset(d_read, 0, N * sizeof(float));

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        herr_t rerr = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, d_read);

        clock_gettime(CLOCK_MONOTONIC, &t1);
        float read_ms = (float)((t1.tv_sec - t0.tv_sec) * 1000.0
                       + (t1.tv_nsec - t0.tv_nsec) / 1e6);

        if (rerr < 0) {
            FAIL("H5Dread GPU");
        } else {
            printf("  16-chunk GPU read took %.2f ms\n", read_ms);

            float* h_read = (float*)malloc(N * sizeof(float));
            cudaMemcpy(h_read, d_read, N * sizeof(float), cudaMemcpyDeviceToHost);

            int match = 1;
            int first_mismatch = -1;
            for (int i = 0; i < N; i++) {
                if (fabsf(h_read[i] - h_data[i]) > 1e-5f) {
                    match = 0;
                    first_mismatch = i;
                    break;
                }
            }

            if (match) {
                PASS("16-chunk GPU read round-trip verified");
            } else {
                char msg[128];
                snprintf(msg, sizeof(msg),
                         "mismatch at [%d]: got %f expected %f",
                         first_mismatch,
                         h_read[first_mismatch], h_data[first_mismatch]);
                FAIL(msg);
            }
            free(h_read);
        }

        cudaFree(d_read);
        H5Dclose(dset);
        H5Fclose(fid);
        free(h_data);
    }

    /* ---- Test 2: GPU-pointer re-read (re-open file, verify round-trip) ---- */
    printf("\n--- Test 2: GPU-pointer multi-chunk re-read ---\n");
    {
        hid_t fid = H5Fopen(TESTFILE, H5F_ACC_RDONLY, fapl);
        if (fid < 0) {
            FAIL("H5Fopen for GPU re-read");
        } else {
            hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
            if (dset < 0) {
                FAIL("H5Dopen2 for GPU re-read");
            } else {
                const int N = 16384;
                float* d_rbuf = NULL;
                cudaMalloc(&d_rbuf, N * sizeof(float));
                cudaMemset(d_rbuf, 0, N * sizeof(float));
                herr_t rerr = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                                      H5P_DEFAULT, d_rbuf);
                if (rerr < 0) {
                    FAIL("H5Dread GPU pointer (re-open)");
                } else {
                    float* h_buf = (float*)malloc(N * sizeof(float));
                    cudaMemcpy(h_buf, d_rbuf, N * sizeof(float), cudaMemcpyDeviceToHost);
                    int ok = 1;
                    for (int i = 0; i < N; i++) {
                        float expected = sinf((float)i * 0.01f) * 100.0f;
                        if (fabsf(h_buf[i] - expected) > 1e-5f) {
                            ok = 0;
                            break;
                        }
                    }
                    if (ok) PASS("GPU re-read round-trip verified");
                    else FAIL("GPU re-read data mismatch");
                    free(h_buf);
                }
                cudaFree(d_rbuf);
                H5Dclose(dset);
            }
            H5Fclose(fid);
        }
    }

cleanup:
    H5Pclose(fapl);
    unlink(TESTFILE);
    gpucompress_cleanup();
    PASS("cleanup completed");

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
