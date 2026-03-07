/**
 * test_h7_null_calloc.cu
 *
 * H-7: new_obj() in H5VLgpucompress.cu dereferences calloc() result without
 *       NULL check. Under OOM, calloc returns NULL → SIGSEGV.
 *
 * Test strategy:
 *   Use LD_PRELOAD with a calloc interposer (calloc_fault.so) that forces
 *   calloc to return NULL after a configurable number of successful calls.
 *   The test performs an H5Fcreate + H5Dcreate2 through the VOL connector,
 *   which triggers new_obj() several times. When one of those calloc calls
 *   returns NULL, the code should handle it gracefully (return error) rather
 *   than crashing with SIGSEGV.
 *
 * Without the interposer (normal run), the test just verifies basic VOL
 * functionality as a sanity check.
 *
 * Usage:
 *   # Normal run (sanity):
 *   ./test_h7_null_calloc
 *
 *   # Fault injection (requires calloc_fault.so):
 *   CALLOC_FAIL_AFTER=50 LD_PRELOAD=./calloc_fault.so ./test_h7_null_calloc
 *
 *   If the code crashes with SIGSEGV under fault injection, H-7 is confirmed.
 *   If it returns an error code cleanly, the bug has been fixed.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <csignal>
#include <setjmp.h>
#include <stdint.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ------------------------------------------------------------------ */
/* Config                                                               */
/* ------------------------------------------------------------------ */
#define CHUNK_ELEMS   1024
#define TOTAL_ELEMS   4096
#define TMP_FILE      "/tmp/test_h7_null_calloc.h5"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits; memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/* ------------------------------------------------------------------ */
/* SIGSEGV handler for fault injection mode                             */
/* ------------------------------------------------------------------ */
static volatile sig_atomic_t g_caught_segv = 0;
static jmp_buf g_jmp;

static void segv_handler(int sig) {
    (void)sig;
    g_caught_segv = 1;
    longjmp(g_jmp, 1);
}

/* ------------------------------------------------------------------ */
/* Helpers                                                               */
/* ------------------------------------------------------------------ */
static hid_t open_vol_file(const char* path, unsigned flags) {
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) return H5I_INVALID_HID;
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
    hid_t fid = (flags & H5F_ACC_TRUNC)
        ? H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)
        : H5Fopen(path, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    H5VLclose(vol_id);
    return fid;
}

static int do_vol_operation(void) {
    hid_t fid = open_vol_file(TMP_FILE, H5F_ACC_TRUNC);
    if (fid < 0) return -1;

    hsize_t dims[1]  = { (hsize_t)TOTAL_ELEMS };
    hsize_t cdims[1] = { (hsize_t)CHUNK_ELEMS };
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS] = {0};
    cd[0] = 2; /* LZ4 */
    cd[1] = 0; /* no preprocessing */
    cd[2] = 0;
    pack_double(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL,
                  H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                             space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

    int ret = 0;
    if (dset >= 0) {
        /* Write some data */
        float* h_data = (float*)malloc(TOTAL_ELEMS * sizeof(float));
        if (h_data) {
            for (int i = 0; i < TOTAL_ELEMS; i++)
                h_data[i] = (float)i * 0.01f;
            herr_t we = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                                 H5P_DEFAULT, h_data);
            if (we < 0) ret = -2;
            free(h_data);
        }
        H5Dclose(dset);
    } else {
        ret = -3;
    }

    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(fid);
    return ret;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */
int main(void) {
    printf("=== H-7: NULL calloc check in new_obj() ===\n\n");

    int fault_injection = (getenv("CALLOC_FAIL_AFTER") != NULL);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init() failed\n");
        return 1;
    }

    if (!fault_injection) {
        /* Normal mode: just verify VOL works end-to-end */
        printf("Mode: normal (no fault injection)\n");
        printf("  To test H-7, run with:\n");
        printf("    CALLOC_FAIL_AFTER=50 LD_PRELOAD=./calloc_fault.so %s\n\n",
               "test_h7_null_calloc");

        int rc = do_vol_operation();
        if (rc == 0) PASS("VOL write completed normally");
        else { printf("  FAIL: VOL write returned %d\n", rc); g_fail++; }
    } else {
        /* Fault injection mode: expect either graceful error or SIGSEGV */
        int fail_after = atoi(getenv("CALLOC_FAIL_AFTER"));
        printf("Mode: fault injection (CALLOC_FAIL_AFTER=%d)\n", fail_after);
        printf("  If new_obj() lacks NULL check, this will SIGSEGV.\n");
        printf("  If fixed, it should return an error gracefully.\n\n");

        /* Install SIGSEGV handler */
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = segv_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGSEGV, &sa, NULL);

        if (setjmp(g_jmp) == 0) {
            int rc = do_vol_operation();
            if (rc == 0) {
                printf("  VOL operation succeeded (calloc didn't fail on new_obj path)\n");
                printf("  Try a lower CALLOC_FAIL_AFTER value.\n");
                PASS("no crash (calloc threshold too high or fixed)");
            } else {
                PASS("VOL returned error gracefully (new_obj NULL handled)");
            }
        } else {
            /* Landed here from SIGSEGV handler */
            printf("\n  *** CAUGHT SIGSEGV ***\n");
            printf("  new_obj() dereferenced NULL calloc result.\n");
            printf("  This confirms H-7: missing NULL check.\n");
            FAIL("SIGSEGV from NULL calloc in new_obj()");
        }
    }

    remove(TMP_FILE);

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    gpucompress_cleanup();
    return g_fail == 0 ? 0 : 1;
}
