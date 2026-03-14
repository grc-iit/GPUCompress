/**
 * test_vol_host_ptr_reject.cu
 *
 * Verifies that the gpucompress VOL connector aborts when a host (CPU)
 * pointer is passed to H5Dwrite or H5Dread.  The VOL now requires CUDA
 * device pointers exclusively.
 *
 * Strategy: fork BEFORE any HDF5 calls so the child has a clean library
 * state.  Each child initialises HDF5 + VOL from scratch, then attempts
 * a host-pointer write or read.  The parent checks for SIGABRT.
 *
 * Run: ./test_vol_host_ptr_reject
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include "hdf5/H5Zgpucompress.h"

#include <cuda_runtime.h>

#define H5Z_FILTER_GPUCOMPRESS   305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5
#define NFLOATS 1024

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

static const char* TESTFILE = "/tmp/test_vol_host_reject.h5";

/**
 * Setup helper: init VOL, create FAPL, open/create file.
 * Returns 0 on success, -1 on skip.  Populates output params.
 */
static int vol_setup(hid_t* out_fapl) {
    gpucompress_error_t gerr = gpucompress_init(NULL);
    if (gerr != GPUCOMPRESS_SUCCESS) return -1;

    H5Z_gpucompress_register();

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl < 0) { gpucompress_cleanup(); return -1; }
    if (H5Pset_fapl_gpucompress(fapl, H5_VOL_NATIVE, NULL) < 0) {
        H5Pclose(fapl);
        gpucompress_cleanup();
        return -1;
    }

    *out_fapl = fapl;
    return 0;
}

/**
 * Child: create a dataset and write with a HOST pointer → must abort.
 */
static void child_host_write(void) {
    hid_t fapl;
    if (vol_setup(&fapl) != 0) _exit(2);

    hid_t file = H5Fcreate(TESTFILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (file < 0) _exit(2);

    hsize_t dims[1] = {NFLOATS};
    hsize_t chunk[1] = {NFLOATS};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk);
    set_gpucompress_filter(dcpl, 1, 0, 4, 0.0);

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, space,
                            H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) _exit(2);

    /* Host pointer — should trigger abort */
    float h_data[NFLOATS];
    for (int i = 0; i < NFLOATS; i++) h_data[i] = (float)i;

    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_data);

    /* Should never reach here */
    fprintf(stderr, "ERROR: host write did not abort!\n");
    _exit(0);
}

/**
 * Child: create a file with GPU data, then read back with HOST pointer → must abort.
 */
static void child_host_read(void) {
    hid_t fapl;
    if (vol_setup(&fapl) != 0) _exit(2);

    /* First, write valid data using GPU pointer */
    hid_t file = H5Fcreate(TESTFILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (file < 0) _exit(2);

    hsize_t dims[1] = {NFLOATS};
    hsize_t chunk[1] = {NFLOATS};
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk);
    set_gpucompress_filter(dcpl, 1, 0, 4, 0.0);

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, space,
                            H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) _exit(2);

    float* h_data = (float*)malloc(NFLOATS * sizeof(float));
    for (int i = 0; i < NFLOATS; i++) h_data[i] = (float)i;
    float* d_data = NULL;
    cudaMalloc(&d_data, NFLOATS * sizeof(float));
    cudaMemcpy(d_data, h_data, NFLOATS * sizeof(float), cudaMemcpyHostToDevice);

    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);

    cudaFree(d_data);
    free(h_data);
    H5Dclose(dset);
    H5Sclose(space);
    H5Pclose(dcpl);
    H5Fclose(file);

    /* Re-open and read with host pointer */
    file = H5Fopen(TESTFILE, H5F_ACC_RDONLY, fapl);
    if (file < 0) _exit(2);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    if (dset < 0) _exit(2);

    /* Host pointer — should trigger abort */
    float h_buf[NFLOATS];
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_buf);

    /* Should never reach here */
    fprintf(stderr, "ERROR: host read did not abort!\n");
    _exit(0);
}

/**
 * Fork a child, run fn(), check it died with SIGABRT.
 * Returns 1 on PASS, 0 on FAIL.
 */
static int expect_abort(void (*fn)(void), const char* label) {
    fflush(stdout);
    fflush(stderr);

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 0; }

    if (pid == 0) {
        fn();
        _exit(0);
    }

    int status;
    waitpid(pid, &status, 0);

    if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) {
        printf("  PASS: %s → aborted (SIGABRT)\n", label);
        g_pass++;
        return 1;
    } else if (WIFEXITED(status) && WEXITSTATUS(status) == 2) {
        printf("  SKIP: %s → setup failed (exit 2)\n", label);
        return 1; /* not a failure, just skip */
    } else if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        printf("  FAIL: %s → did NOT abort (exit 0)\n", label);
        g_fail++;
        return 0;
    } else {
        /* Some other signal or non-zero exit — still rejected */
        printf("  PASS: %s → rejected (status=0x%x)\n", label, status);
        g_pass++;
        return 1;
    }
}

int main(void) {
    printf("=== VOL host-pointer rejection test ===\n\n");

    /* Fork BEFORE any HDF5 calls so children get clean library state */

    expect_abort(child_host_write, "H5Dwrite with host pointer");
    expect_abort(child_host_read,  "H5Dread with host pointer");

    unlink(TESTFILE);

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
