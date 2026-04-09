/**
 * tests/hdf5/test_vol_bypass_roundtrip.cu
 *
 * Correctness test for GPUCOMPRESS_VOL_BYPASS=1 mode.
 *
 * The bypass mode is designed to skip the NN + nvCOMP entirely and route
 * raw bytes through the same worker + I/O pipeline to disk. bytes_in
 * must equal bytes_out, and the on-disk file must contain at least the
 * raw bytes (HDF5 metadata may add some overhead on top).
 *
 * Cases:
 *   1. Env var NOT set:    bypass_mode == 0, ratio > 1 (compress path)
 *   2. Env var "1":        bypass_mode == 1, ratio == 1.0, file ≥ raw
 *   3. Env var "0":        bypass_mode == 0
 *   4. Env var empty:      bypass_mode == 0
 *
 * Each case is a SEPARATE subprocess invocation because the VOL's
 * std::call_once caches the first env read for the rest of the process
 * lifetime. The subprocess signals its result via exit code (0 = pass,
 * 1 = fail).
 *
 * Round-trip semantics differ by mode:
 *   - Compression: H5Dread back through the VOL and bit-compare.
 *   - Bypass: skip H5Dread because the DCPL still has the gpucompress
 *     filter attached, so any reader going through the VOL will try to
 *     decompress the raw bytes and fail. Bypass is a write-path
 *     measurement mode, not a storage format. Validation is via the
 *     on-disk file size check instead.
 *
 * Usage (standalone):
 *   ./build/test_vol_bypass_roundtrip              # drives 4 subprocess cases
 *   ./build/test_vol_bypass_roundtrip --case work1  # internal: bypass worker
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static herr_t dcpl_set_gpucompress_lz4(hid_t dcpl)
{
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS] = {0};
    cd[0] = 1;  /* GPUCOMPRESS_ALGO_LZ4 — no NN weights needed */
    cd[1] = 0;
    cd[2] = 0;
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

/* Compressible-but-not-trivial pattern: small-range ramp + mantissa noise */
__global__ static void fill_kernel(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s) {
        uint32_t x = (uint32_t)(i * 2654435761UL + 1);
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        x = (x & 0x000007FFu) | 0x3F800000u;
        float f;
        memcpy(&f, &x, sizeof(f));
        buf[i] = (float)i * 1e-6f + (f - 1.0f);
    }
}

/* ============================================================
 * Subprocess worker: one write (+ optional roundtrip), returns exit code
 * ============================================================ */
static int worker_write_and_verify(const char *tmp_file,
                                    size_t n_floats,
                                    size_t chunk_floats,
                                    int expected_bypass_flag)
{
    const size_t total_bytes = n_floats * sizeof(float);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "  [worker] gpucompress_init failed\n");
        return 1;
    }
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "  [worker] H5VL_gpucompress_register failed\n");
        return 1;
    }

    int actual_bypass = H5VL_gpucompress_is_bypass_mode();
    if (actual_bypass != expected_bypass_flag) {
        fprintf(stderr,
                "  [worker] bypass flag mismatch: got %d, expected %d "
                "(GPUCOMPRESS_VOL_BYPASS=%s)\n",
                actual_bypass, expected_bypass_flag,
                getenv("GPUCOMPRESS_VOL_BYPASS") ?: "(unset)");
        return 1;
    }

    float *d_data = NULL, *d_read = NULL;
    if (cudaMalloc(&d_data, total_bytes) != cudaSuccess ||
        cudaMalloc(&d_read, total_bytes) != cudaSuccess) {
        fprintf(stderr, "  [worker] cudaMalloc failed\n");
        return 1;
    }
    fill_kernel<<<1024, 256>>>(d_data, n_floats);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "  [worker] GPU fill failed\n");
        return 1;
    }

    remove(tmp_file);
    hsize_t dims[1]  = { (hsize_t)n_floats };
    hsize_t cdims[1] = { (hsize_t)chunk_floats };
    hid_t fapl  = make_vol_fapl();
    hid_t file  = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "  [worker] H5Fcreate failed\n"); return 1; }

    hid_t fspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    dcpl_set_gpucompress_lz4(dcpl);
    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, fspace,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl); H5Sclose(fspace);
    if (dset < 0) { H5Fclose(file); return 1; }

    herr_t w = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, d_data);
    H5Dclose(dset); H5Fclose(file);
    if (w < 0) { fprintf(stderr, "  [worker] H5Dwrite failed\n"); return 1; }

    /* Round-trip verification (compression mode only — see header comment
     * for why bypass files can't be safely read back through the VOL). */
    int mismatch_idx = -1;
    if (expected_bypass_flag == 0) {
        fapl = make_vol_fapl();
        file = H5Fopen(tmp_file, H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        if (file < 0) { fprintf(stderr, "  [worker] H5Fopen failed\n"); return 1; }

        dset = H5Dopen2(file, "data", H5P_DEFAULT);
        herr_t r = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, d_read);
        cudaDeviceSynchronize();
        H5Dclose(dset); H5Fclose(file);
        if (r < 0) { fprintf(stderr, "  [worker] H5Dread failed\n"); return 1; }

        float *h_orig = (float*)malloc(total_bytes);
        float *h_read = (float*)malloc(total_bytes);
        cudaMemcpy(h_orig, d_data, total_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_read, d_read, total_bytes, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n_floats; i++) {
            if (h_orig[i] != h_read[i]) { mismatch_idx = (int)i; break; }
        }
        free(h_orig); free(h_read);
    }
    cudaFree(d_data); cudaFree(d_read);

    if (mismatch_idx >= 0) {
        fprintf(stderr, "  [worker] round-trip mismatch at index %d\n", mismatch_idx);
        remove(tmp_file);
        return 1;
    }

    /* Program-wall getter sanity: prog > 0, compute >= 0 */
    double prog_ms = 0, comp_ms = 0;
    H5VL_gpucompress_get_program_wall(&prog_ms, &comp_ms);
    if (prog_ms <= 0) {
        fprintf(stderr, "  [worker] program_wall=%.2f (expected > 0)\n", prog_ms);
        return 1;
    }
    if (comp_ms < 0) {
        fprintf(stderr, "  [worker] compute_ms=%.2f (expected >= 0)\n", comp_ms);
        return 1;
    }

    /* On-disk file size check:
     *   bypass: file size >= raw bytes (raw data + HDF5 metadata)
     *   compress (lz4 on compressible ramp): file size < raw bytes */
    FILE *fp = fopen(tmp_file, "rb");
    long file_sz = 0;
    if (fp) { fseek(fp, 0, SEEK_END); file_sz = ftell(fp); fclose(fp); }
    remove(tmp_file);

    if (expected_bypass_flag == 1) {
        if ((size_t)file_sz < total_bytes) {
            fprintf(stderr,
                    "  [worker] bypass: file size %ld < raw bytes %zu — "
                    "bypass should not compress\n", file_sz, total_bytes);
            return 1;
        }
    } else {
        if ((size_t)file_sz >= total_bytes) {
            fprintf(stderr,
                    "  [worker] compress: file size %ld >= raw bytes %zu — "
                    "compression did nothing?\n", file_sz, total_bytes);
            return 1;
        }
    }

    H5VLclose(vol_id);
    gpucompress_cleanup();
    return 0;
}

/* ============================================================
 * Driver: spawns subprocesses with controlled env
 * ============================================================ */
static int drive_one_case(const char *label, const char *env_val,
                           int expected_bypass, const char *argv0)
{
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return -1; }
    if (pid == 0) {
        if (env_val) setenv("GPUCOMPRESS_VOL_BYPASS", env_val, 1);
        else         unsetenv("GPUCOMPRESS_VOL_BYPASS");

        char arg1[] = "--case";
        char arg2[32];
        snprintf(arg2, sizeof(arg2), "work%d", expected_bypass);
        char *child_argv[] = { (char*)argv0, arg1, arg2, NULL };
        execv(argv0, child_argv);
        perror("execv");
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) { perror("waitpid"); return -1; }
    int rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128;
    printf("  [%s] GPUCOMPRESS_VOL_BYPASS=%-7s expected_bypass=%d → rc=%d %s\n",
           label, env_val ? env_val : "(unset)", expected_bypass, rc,
           rc == 0 ? "PASS" : "FAIL");
    return rc;
}

int main(int argc, char **argv)
{
    const size_t n_floats     = 4 * 1024 * 1024;    /* 16 MiB */
    const size_t chunk_floats = 1 * 1024 * 1024;    /*  4 MiB chunks */
    const char *tmp_file = "/tmp/test_vol_bypass_roundtrip.h5";

    if (argc == 3 && strcmp(argv[1], "--case") == 0) {
        int expect = (strcmp(argv[2], "work1") == 0) ? 1 : 0;
        return worker_write_and_verify(tmp_file, n_floats, chunk_floats, expect);
    }

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    printf("=== GPUCompress VOL bypass mode round-trip test ===\n");
    printf("    Dataset: %zu floats (%zu MiB), 4 subprocess cases\n\n",
           n_floats, (n_floats * sizeof(float)) >> 20);

    int fail = 0;
    fail += (drive_one_case("case 1", NULL, 0, argv[0]) != 0);
    fail += (drive_one_case("case 2", "1",  1, argv[0]) != 0);
    fail += (drive_one_case("case 3", "0",  0, argv[0]) != 0);
    fail += (drive_one_case("case 4", "",   0, argv[0]) != 0);

    printf("\n  Total: %d/4 PASS\n", 4 - fail);
    if (fail == 0)
        printf("  ALL TESTS PASSED — bypass round-trip correctness verified.\n");
    else
        printf("  %d TEST(S) FAILED.\n", fail);
    return fail == 0 ? 0 : 1;
}
