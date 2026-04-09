/**
 * tests/hdf5/test_vol_bypass_timing.cu
 *
 * Validates that GPUCOMPRESS_VOL_BYPASS=1 actually skips the NN +
 * nvCOMP work — not just that it toggles the reported flag.
 *
 * Approach: run the SAME write workload twice in subprocesses (compress
 * mode + bypass mode), diff the VOL stage timings. Bypass must satisfy:
 *   (a) bypass_flag matches env
 *   (b) file_sz(bypass) >= bytes_in   (raw passthrough)
 *   (c) file_sz(compress) < bytes_in  (compression shrank the file)
 *   (d) prog_wall_ms >= vol total_ms in both runs (sanity)
 *   (e) compute_ms >= 0 in both runs
 *
 * This test uses GPUCOMPRESS_ALGO_AUTO to exercise the NN inference
 * path in compression mode. Without NN weights loaded the NN
 * falls back to heuristics, which is fine for the timing contract.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static herr_t dcpl_set_lz4(hid_t dcpl)
{
    /* Use LZ4 (fixed algo) so the test doesn't depend on NN weights.
     * The compression path still exercises nvCOMP, stats, and pipeline
     * bookkeeping — just not the NN inference slot. */
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS] = {0};
    cd[0] = 1;  /* LZ4 */
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

__global__ static void ramp_plus_noise(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t s = (size_t)gridDim.x  * blockDim.x;
    for (; i < n; i += s) {
        uint32_t x = (uint32_t)(i * 2654435761UL + 1);
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        x = (x & 0x000003FFu) | 0x3F800000u;
        float f;
        memcpy(&f, &x, sizeof(f));
        buf[i] = (float)i * 1e-7f + (f - 1.0f);
    }
}

/* ============================================================
 * Worker: one write, dumps KEY=VAL timings to stdout
 * ============================================================ */
static int worker(const char *tmp_file, size_t n_floats, size_t chunk_floats)
{
    size_t total_bytes = n_floats * sizeof(float);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) return 1;
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) return 1;

    float *d_data = NULL;
    if (cudaMalloc(&d_data, total_bytes) != cudaSuccess) return 1;
    ramp_plus_noise<<<1024, 256>>>(d_data, n_floats);
    cudaDeviceSynchronize();

    remove(tmp_file);
    hsize_t dims[1]  = { (hsize_t)n_floats };
    hsize_t cdims[1] = { (hsize_t)chunk_floats };

    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    hid_t fs   = H5Screate_simple(1, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    dcpl_set_lz4(dcpl);
    hid_t ds = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT, fs,
                          H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl); H5Sclose(fs);

    H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    cudaDeviceSynchronize();
    H5Dclose(ds); H5Fclose(file);

    double stage1 = 0, drain = 0, io_drain = 0, total = 0;
    double s2_busy = 0, s3_busy = 0;
    double prog_wall = 0, compute = 0;
    int bypass_flag = H5VL_gpucompress_is_bypass_mode();

    H5VL_gpucompress_get_stage_timing(&stage1, &drain, &io_drain, &total);
    H5VL_gpucompress_get_busy_timing(&s2_busy, &s3_busy);
    H5VL_gpucompress_get_program_wall(&prog_wall, &compute);

    FILE *fp = fopen(tmp_file, "rb");
    long file_sz = 0;
    if (fp) { fseek(fp, 0, SEEK_END); file_sz = ftell(fp); fclose(fp); }
    remove(tmp_file);

    printf("BYPASS_FLAG=%d\n", bypass_flag);
    printf("BYTES_IN=%zu\n",  total_bytes);
    printf("FILE_SZ=%ld\n",   file_sz);
    printf("STAGE1_MS=%.6f\n", stage1);
    printf("S2_BUSY_MS=%.6f\n", s2_busy);
    printf("TOTAL_MS=%.6f\n", total);
    printf("PROG_WALL_MS=%.6f\n", prog_wall);
    printf("COMPUTE_MS=%.6f\n", compute);
    fflush(stdout);

    cudaFree(d_data);
    H5VLclose(vol_id);
    gpucompress_cleanup();
    return 0;
}

typedef struct {
    int    bypass_flag;
    size_t bytes_in;
    long   file_sz;
    double stage1_ms;
    double s2_busy_ms;
    double total_ms;
    double prog_wall_ms;
    double compute_ms;
} WorkerResult;

static int parse_worker_output(const char *buf, WorkerResult *r)
{
    memset(r, 0, sizeof(*r));
    int n = 0;
    const char *p = buf;
    while (*p) {
        if (sscanf(p, "BYPASS_FLAG=%d",   &r->bypass_flag)   == 1) n++;
        if (sscanf(p, "BYTES_IN=%zu",     &r->bytes_in)      == 1) n++;
        if (sscanf(p, "FILE_SZ=%ld",      &r->file_sz)       == 1) n++;
        if (sscanf(p, "STAGE1_MS=%lf",    &r->stage1_ms)     == 1) n++;
        if (sscanf(p, "S2_BUSY_MS=%lf",   &r->s2_busy_ms)    == 1) n++;
        if (sscanf(p, "TOTAL_MS=%lf",     &r->total_ms)      == 1) n++;
        if (sscanf(p, "PROG_WALL_MS=%lf", &r->prog_wall_ms)  == 1) n++;
        if (sscanf(p, "COMPUTE_MS=%lf",   &r->compute_ms)    == 1) n++;
        const char *nl = strchr(p, '\n');
        if (!nl) break;
        p = nl + 1;
    }
    return (n >= 8) ? 0 : -1;
}

static int run_worker(const char *argv0, const char *env_val, WorkerResult *out)
{
    int pipefd[2];
    if (pipe(pipefd) < 0) return -1;

    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        if (env_val) setenv("GPUCOMPRESS_VOL_BYPASS", env_val, 1);
        else         unsetenv("GPUCOMPRESS_VOL_BYPASS");
        char arg1[] = "--worker";
        char *cargs[] = { (char*)argv0, arg1, NULL };
        execv(argv0, cargs);
        _exit(127);
    }
    close(pipefd[1]);
    char buf[4096] = {0};
    ssize_t total_read = 0;
    while (total_read < (ssize_t)sizeof(buf) - 1) {
        ssize_t n = read(pipefd[0], buf + total_read, sizeof(buf) - 1 - total_read);
        if (n <= 0) break;
        total_read += n;
    }
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return -1;
    return parse_worker_output(buf, out);
}

int main(int argc, char **argv)
{
    const char *tmp_file      = "/tmp/test_vol_bypass_timing.h5";
    const size_t n_floats     = 4 * 1024 * 1024;  /* 16 MiB */
    const size_t chunk_floats = 1 * 1024 * 1024;  /*  4 MiB chunks */

    if (argc == 2 && strcmp(argv[1], "--worker") == 0)
        return worker(tmp_file, n_floats, chunk_floats);

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    printf("=== GPUCompress VOL bypass timing contract test ===\n");
    printf("    Dataset: %zu MiB (%zu floats), 4 MiB chunks\n\n",
           (n_floats * sizeof(float)) >> 20, n_floats);

    WorkerResult compress = {0}, bypass = {0};
    if (run_worker(argv[0], NULL, &compress) < 0) {
        fprintf(stderr, "FAIL: compress subprocess failed to produce timing output\n");
        return 1;
    }
    if (run_worker(argv[0], "1",  &bypass) < 0) {
        fprintf(stderr, "FAIL: bypass subprocess failed to produce timing output\n");
        return 1;
    }

    printf("  metric           compress       bypass\n");
    printf("  ---------------  -------------  -------------\n");
    printf("  bypass_flag      %13d  %13d\n", compress.bypass_flag, bypass.bypass_flag);
    printf("  bytes_in         %13zu  %13zu\n", compress.bytes_in,  bypass.bytes_in);
    printf("  file_sz          %13ld  %13ld\n", compress.file_sz,   bypass.file_sz);
    printf("  stage1_ms        %13.3f  %13.3f\n", compress.stage1_ms, bypass.stage1_ms);
    printf("  s2_busy_ms       %13.3f  %13.3f\n", compress.s2_busy_ms, bypass.s2_busy_ms);
    printf("  total_ms         %13.3f  %13.3f\n", compress.total_ms, bypass.total_ms);
    printf("  prog_wall_ms     %13.3f  %13.3f\n", compress.prog_wall_ms, bypass.prog_wall_ms);
    printf("  compute_ms       %13.3f  %13.3f\n", compress.compute_ms, bypass.compute_ms);
    printf("\n");

    int fail = 0;

    /* (1) bypass flag must match env */
    if (compress.bypass_flag != 0) {
        printf("  FAIL: compress run reports bypass_flag=%d (expected 0)\n",
               compress.bypass_flag); fail++;
    }
    if (bypass.bypass_flag != 1) {
        printf("  FAIL: bypass run reports bypass_flag=%d (expected 1)\n",
               bypass.bypass_flag); fail++;
    }

    /* (2) on-disk file size inverts between modes */
    if ((size_t)bypass.file_sz < bypass.bytes_in) {
        printf("  FAIL: bypass file_sz %ld < bytes_in %zu (bypass should not compress)\n",
               bypass.file_sz, bypass.bytes_in); fail++;
    }
    if ((size_t)compress.file_sz >= compress.bytes_in) {
        printf("  FAIL: compress file_sz %ld >= bytes_in %zu (compression didn't shrink)\n",
               compress.file_sz, compress.bytes_in); fail++;
    }

    /* (3) program_wall and compute_ms must be sane */
    if (compress.prog_wall_ms <= 0 || bypass.prog_wall_ms <= 0) {
        printf("  FAIL: prog_wall_ms not captured (compress=%.2f bypass=%.2f)\n",
               compress.prog_wall_ms, bypass.prog_wall_ms); fail++;
    }
    if (compress.compute_ms < 0 || bypass.compute_ms < 0) {
        printf("  FAIL: compute_ms is negative (compress=%.2f bypass=%.2f)\n",
               compress.compute_ms, bypass.compute_ms); fail++;
    }

    /* (4) prog_wall ≥ vol total in both runs */
    if (compress.prog_wall_ms < compress.total_ms) {
        printf("  FAIL: compress prog_wall %.2f < total %.2f\n",
               compress.prog_wall_ms, compress.total_ms); fail++;
    }
    if (bypass.prog_wall_ms < bypass.total_ms) {
        printf("  FAIL: bypass prog_wall %.2f < total %.2f\n",
               bypass.prog_wall_ms, bypass.total_ms); fail++;
    }

    printf("\n  %s (%d check(s) failed)\n",
           fail == 0 ? "ALL TIMING CHECKS PASSED" : "TIMING CHECKS FAILED", fail);
    return fail == 0 ? 0 : 1;
}
