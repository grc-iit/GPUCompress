/**
 * test_vol_c4c8h7_defensive.cu
 *
 * Functional test for VOL defensive fixes:
 *   C4: H5Pcopy() return validated in file_create/open/specific
 *   C8: new_obj() checks calloc() return for NULL
 *   H7: cudaStreamCreate checked in gather/scatter paths
 *
 * Tests the happy path: VOL file create, dataset write (GPU pointer),
 * dataset read (GPU pointer), file close — exercises all fixed code paths.
 *
 * Run: ./test_vol_c4c8h7_defensive
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

static const char* TESTFILE = "/tmp/test_vol_c4c8h7.h5";

int main(void) {
    printf("=== C4/C8/H7: VOL defensive checks (functional) ===\n\n");

    gpucompress_error_t gerr = gpucompress_init(NULL);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        printf("SKIP: gpucompress_init failed (%d)\n", gerr);
        return 1;
    }

    /* Register VOL */
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) {
        printf("SKIP: H5VL_gpucompress_register failed\n");
        gpucompress_cleanup();
        return 1;
    }

    /* Register H5Z filter (needed for host-pointer reads) */
    if (H5Z_gpucompress_register() < 0) {
        printf("SKIP: H5Z_gpucompress_register failed\n");
        gpucompress_cleanup();
        return 1;
    }

    /* Create FAPL with gpucompress VOL */
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl < 0) { FAIL("H5Pcreate FAPL"); goto cleanup; }

    /* This exercises H5Pcopy inside file_create (C4 fix) */
    if (H5Pset_fapl_gpucompress(fapl, H5_VOL_NATIVE, NULL) < 0) {
        FAIL("H5Pset_fapl_gpucompress");
        goto cleanup;
    }
    PASS("FAPL configured with gpucompress VOL");

    /* ---- Test 1: File create + close (exercises C4 H5Pcopy in file_create) ---- */
    printf("\n--- Test 1: File create/close ---\n");
    {
        hid_t fid = H5Fcreate(TESTFILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        if (fid < 0) {
            FAIL("H5Fcreate with VOL");
            goto cleanup;
        }
        PASS("H5Fcreate with gpucompress VOL succeeded");

        /* Create chunked dataset with gpucompress filter */
        hsize_t dims[1] = {4096};
        hsize_t chunk[1] = {1024};
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);
        set_gpucompress_filter(dcpl, 1 /* LZ4 */, 0, 0, 0.0);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space,
                                H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) {
            FAIL("H5Dcreate2");
        } else {
            PASS("dataset created with gpucompress filter");

            /* GPU write — exercises gather_stream (H7 fix) and new_obj (C8 fix) */
            float* d_data = NULL;
            cudaMalloc(&d_data, 4096 * sizeof(float));
            float* h_data = (float*)malloc(4096 * sizeof(float));
            for (int i = 0; i < 4096; i++) h_data[i] = (float)i * 0.1f;
            cudaMemcpy(d_data, h_data, 4096 * sizeof(float), cudaMemcpyHostToDevice);

            herr_t werr = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                                   H5P_DEFAULT, d_data);
            if (werr < 0) {
                FAIL("H5Dwrite with GPU pointer");
            } else {
                PASS("GPU write succeeded (gather_stream + new_obj exercised)");
            }

            /* GPU read — exercises scatter_stream */
            float* d_read = NULL;
            cudaMalloc(&d_read, 4096 * sizeof(float));
            cudaMemset(d_read, 0, 4096 * sizeof(float));

            herr_t rerr = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                                  H5P_DEFAULT, d_read);
            if (rerr < 0) {
                FAIL("H5Dread with GPU pointer");
            } else {
                /* Verify round-trip */
                float* h_read = (float*)malloc(4096 * sizeof(float));
                cudaMemcpy(h_read, d_read, 4096 * sizeof(float), cudaMemcpyDeviceToHost);
                int match = 1;
                for (int i = 0; i < 4096; i++) {
                    if (fabsf(h_read[i] - h_data[i]) > 1e-6f) {
                        match = 0;
                        break;
                    }
                }
                if (match) {
                    PASS("GPU read round-trip verified");
                } else {
                    FAIL("GPU read data mismatch");
                }
                free(h_read);
            }

            cudaFree(d_data);
            cudaFree(d_read);
            free(h_data);
            H5Dclose(dset);
        }

        H5Pclose(dcpl);
        H5Sclose(space);
        H5Fclose(fid);
        PASS("file closed cleanly");
    }

    /* ---- Test 2: File open (exercises C4 H5Pcopy in file_open) ---- */
    printf("\n--- Test 2: File re-open/read ---\n");
    {
        hid_t fid = H5Fopen(TESTFILE, H5F_ACC_RDONLY, fapl);
        if (fid < 0) {
            FAIL("H5Fopen with VOL");
        } else {
            PASS("H5Fopen with gpucompress VOL succeeded");

            hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
            if (dset < 0) {
                FAIL("H5Dopen2");
            } else {
                PASS("dataset opened");

                /* Read back and verify */
                float* h_buf = (float*)malloc(4096 * sizeof(float));
                herr_t rerr = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                                      H5P_DEFAULT, h_buf);
                if (rerr < 0) {
                    FAIL("H5Dread host pointer");
                } else {
                    int ok = 1;
                    for (int i = 0; i < 4096; i++) {
                        if (fabsf(h_buf[i] - (float)i * 0.1f) > 1e-6f) {
                            ok = 0;
                            break;
                        }
                    }
                    if (ok) PASS("host read round-trip verified");
                    else FAIL("host read data mismatch");
                }
                free(h_buf);
                H5Dclose(dset);
            }
            H5Fclose(fid);
        }
    }

cleanup:
    H5Pclose(fapl);
    unlink(TESTFILE);
    gpucompress_cleanup();

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
