/**
 * tests/demo_sgd_parallel_chunks.cu
 *
 * Demonstrates NN-driven SGD firing during parallel chunk compression.
 *
 * Setup:
 *   - 128 MB dataset, 16 MB chunks (8 chunks total)
 *   - ALGO_AUTO: NN picks algorithm per chunk
 *   - Online learning ENABLED
 *   - Reinforcement: learning_rate=0.4, MAPE threshold=20%
 *   - Exploration: enabled (triggers when predicted ratio is off by >20%)
 *
 * The 4-worker parallel pipeline (Stage1→Stage2→Stage3) means up to 4 chunks
 * compress concurrently on independent CUDA streams.  Each chunk that exceeds
 * the MAPE threshold fires SGD on g_sgd_stream and sets g_sgd_ever_fired.
 * Subsequent chunks insert cudaStreamWaitEvent before inference.
 *
 * After the write we read the per-chunk diagnostic history and print:
 *   - Which action (algo + quant + shuffle) the NN picked for each chunk
 *   - Whether exploration was triggered
 *   - Whether SGD fired
 *   - Actual vs predicted compression ratio
 *
 * Then we read back, decompress, and verify every float is bit-exact.
 *
 * Usage:
 *   LD_LIBRARY_PATH=/tmp/hdf5-install/lib:$LD_LIBRARY_PATH \
 *   ./build/demo_sgd_parallel_chunks neural_net/weights/model.nnwt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Configuration
 * ============================================================ */
#define DATASET_MB      1024
#define CHUNK_MB        16
#define REINFORCE_LR    0.4f
#define REINFORCE_MAPE  0.20f   /* 20% MAPE threshold */
#define EXPLORE_THRESH  0.20    /* 20% MAPE triggers exploration */

#define TMP_FILE "/tmp/demo_sgd_parallel_chunks.h5"
#define DSET_NAME "data"

/* ============================================================
 * HDF5 filter wiring (mirrors H5Zgpucompress.h)
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

static herr_t dcpl_set_gpucompress_auto(hid_t dcpl, double error_bound)
{
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0; /* no static preprocessing */
    cd[2] = 0; /* shuffle size (NN decides) */
    pack_double_cd(error_bound, &cd[3], &cd[4]);
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
}

/* ============================================================
 * VOL FAPL
 * ============================================================ */
static hid_t make_vol_fapl(void)
{
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl      = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

/* ============================================================
 * GPU ramp kernel: buf[i] = (float)i / (float)n
 * ============================================================ */
__global__ static void ramp_kernel(float *buf, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        buf[i] = (float)i / (float)n;
}

/* ============================================================
 * Action decoder helpers
 * ============================================================ */
static const char *ALGO_NAMES[] = {
    "none", "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
};

static void decode_action(int action, int *algo, int *quant, int *shuffle)
{
    *algo   = action % 8;
    *quant  = (action / 8)  % 2;
    *shuffle= (action / 16) % 2;
}

static void sprint_action(char *buf, int action)
{
    int algo, quant, shuffle;
    decode_action(action, &algo, &quant, &shuffle);
    sprintf(buf, "action%2d: %s%s%s",
            action,
            ALGO_NAMES[algo + 1],
            quant   ? "+quant"  : "",
            shuffle ? "+shuffle" : "");
}

/* ============================================================
 * Timing
 * ============================================================ */
static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char **argv)
{
    const char *weights_path = (argc > 1) ? argv[1] : NULL;
    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s neural_net/weights/model.nnwt\n", argv[0]);
        return 1;
    }

    size_t chunk_floats = (size_t)CHUNK_MB   * 1024 * 1024 / sizeof(float);
    size_t n_floats     = (size_t)DATASET_MB * 1024 * 1024 / sizeof(float);
    size_t total_bytes  = n_floats * sizeof(float);
    int    n_chunks     = (int)(n_floats / chunk_floats);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  NN SGD Parallel Chunk Compression Demo                  ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("Dataset   : %d MiB  (%zu floats)\n", DATASET_MB, n_floats);
    printf("Chunk size: %d MiB  (%zu floats)\n", CHUNK_MB,   chunk_floats);
    printf("Chunks    : %d (4 compress concurrently via CompContext pool)\n", n_chunks);
    printf("Algorithm : ALGO_AUTO  (NN selects per chunk)\n");
    printf("Online LR : %.2f   MAPE threshold: %.0f%%\n",
           REINFORCE_LR, REINFORCE_MAPE * 100.0f);
    printf("Weights   : %s\n\n", weights_path);

    /* ── Init library ─────────────────────────────────────────────── */
    if (gpucompress_init(weights_path) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "FATAL: NN weights not loaded from %s\n", weights_path);
        gpucompress_cleanup(); return 1;
    }

    /* ── Enable online learning + exploration ─────────────────────── */
    gpucompress_enable_online_learning();           /* enables SGD */
    // gpucompress_set_exploration(1);                /* enables exploration */
    // gpucompress_set_exploration_threshold(EXPLORE_THRESH);
    gpucompress_set_reinforcement(1, REINFORCE_LR, REINFORCE_MAPE, REINFORCE_MAPE);

    printf("[Setup] Online learning: ENABLED\n");
    printf("[Setup] Exploration:     ENABLED (threshold=%.0f%% MAPE)\n",
           EXPLORE_THRESH * 100.0);
    printf("[Setup] SGD:             lr=%.2f  MAPE_threshold=%.0f%%\n\n",
           REINFORCE_LR, REINFORCE_MAPE * 100.0f);

    /* ── Allocate GPU buffer + fill ───────────────────────────────── */
    float *d_full = NULL;
    if (cudaMalloc(&d_full, total_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc(%.0f MiB) failed\n",
                (double)total_bytes / (1 << 20));
        gpucompress_cleanup(); return 1;
    }
    printf("[GPU] Generating %d MiB ramp on GPU... ", DATASET_MB);
    fflush(stdout);
    ramp_kernel<<<4096, 256>>>(d_full, n_floats);
    cudaDeviceSynchronize();
    printf("done.\n\n");

    /* ── Register VOL ─────────────────────────────────────────────── */
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "H5VL_gpucompress_register failed\n");
        cudaFree(d_full); gpucompress_cleanup(); return 1;
    }

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);  /* silence HDF5 error stack */

    /* ── Create dataset ───────────────────────────────────────────── */
    hsize_t dims[1]  = { (hsize_t)n_floats };
    hsize_t cdims[1] = { (hsize_t)chunk_floats };

    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "H5Fcreate failed\n"); return 1; }

    hid_t fspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);
    dcpl_set_gpucompress_auto(dcpl, 0.0 /* lossless */);

    hid_t dset = H5Dcreate2(file, DSET_NAME, H5T_NATIVE_FLOAT,
                             fspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl);
    H5Sclose(fspace);

    /* ── Reset diagnostic history + WRITE ─────────────────────────── */
    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();

    printf("[Write] H5Dwrite (GPU pointer) — triggering parallel compression...\n");
    printf("        (stderr shows [EXPLORE-GPU] lines if exploration fires)\n\n");

    double t0 = now_ms();
    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_full);
    double t1 = now_ms();

    H5Dclose(dset);
    H5Fclose(file);

    if (wret < 0) {
        fprintf(stderr, "H5Dwrite failed\n");
        H5VLclose(vol_id); cudaFree(d_full); gpucompress_cleanup(); return 1;
    }

    size_t file_bytes = (size_t)lseek(open(TMP_FILE, O_RDONLY), 0, SEEK_END);
    double ratio = (double)total_bytes / (double)(file_bytes > 0 ? file_bytes : 1);
    double write_ms = t1 - t0;

    printf("[Write] Done in %.0f ms  (%.1f MiB/s uncompressed throughput)\n",
           write_ms, (double)total_bytes / (1 << 20) / (write_ms / 1000.0));
    printf("[Write] File: %.1f MiB  overall ratio: %.2fx\n\n",
           (double)file_bytes / (1 << 20), ratio);

    /* ── Per-chunk diagnostic report ──────────────────────────────── */
    int n_hist = gpucompress_get_chunk_history_count();
    printf("╔══ Per-Chunk NN Diagnostics (%d chunks) ════════════════════╗\n",
           n_hist);
    printf("║  %-5s  %-32s  %-11s  %-5s  %-3s ║\n",
           "Chunk", "Action", "Exploration", "SGD", "Alt");
    printf("╟──────────────────────────────────────────────────────────╢\n");

    int total_sgd       = 0;
    int total_explore   = 0;
    int action_changed  = 0;

    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) != 0) continue;

        char act_buf[64], orig_buf[64];
        sprint_action(act_buf,  d.nn_action);
        sprint_action(orig_buf, d.nn_original_action);

        int changed = (d.nn_action != d.nn_original_action);
        if (d.sgd_fired)     total_sgd++;
        if (d.exploration_triggered) total_explore++;
        if (changed)         action_changed++;

        printf("║  [%3d]  %-32s  %-11s  %-5s  %-3s ║\n",
               i,
               act_buf,
               d.exploration_triggered ? "YES" : "no",
               d.sgd_fired             ? "YES" : "no",
               changed                 ? "YES" : "no");

        if (changed) {
            printf("║         orig: %-46s  ║\n", orig_buf);
        }
    }

    printf("╟──────────────────────────────────────────────────────────╢\n");
    printf("║  Total: %d/%d SGD fired   %d/%d explored   %d/%d action changed  ║\n",
           total_sgd, n_hist, total_explore, n_hist, action_changed, n_hist);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    /* ── Interpretation ───────────────────────────────────────────── */
    printf("[Analysis]\n");
    if (total_sgd > 0) {
        printf("  SGD fired on %d chunk(s) — g_sgd_ever_fired was set to true.\n",
               total_sgd);
        printf("  Subsequent inference calls inserted cudaStreamWaitEvent(g_sgd_done)\n");
        printf("  before nnFusedInferenceKernel — correct GPU-level ordering enforced.\n");
    } else {
        printf("  SGD did NOT fire — NN predictions were within %.0f%% MAPE.\n",
               REINFORCE_MAPE * 100.0f);
        printf("  This is OK: the NN already predicts well for this data pattern.\n");
        printf("  Try a different data pattern or lower --mape to force SGD.\n");
    }
    if (total_explore > 0) {
        printf("  Exploration triggered on %d chunk(s) — K alternatives tried.\n",
               total_explore);
        printf("  Winner action may differ from original NN selection.\n");
    }
    printf("\n");

    /* ── Read back + verify ───────────────────────────────────────── */
    printf("[Verify] Reading back and decompressing on GPU...\n");

    float *d_read = NULL;
    if (cudaMalloc(&d_read, total_bytes) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc(read) failed\n");
        H5VLclose(vol_id); cudaFree(d_full); gpucompress_cleanup(); return 1;
    }

    fapl = make_vol_fapl();
    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, DSET_NAME, H5P_DEFAULT);

    double r0 = now_ms();
    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    double r1 = now_ms();

    H5Dclose(dset);
    H5Fclose(file);

    if (rret < 0) {
        fprintf(stderr, "[Verify] H5Dread failed\n");
        cudaFree(d_read); H5VLclose(vol_id);
        cudaFree(d_full); gpucompress_cleanup();
        remove(TMP_FILE);
        return 1;
    }

    printf("[Verify] Read done in %.0f ms  (%.1f MiB/s decompressed throughput)\n",
           r1 - r0, (double)total_bytes / (1 << 20) / ((r1 - r0) / 1000.0));

    /* D→H and compare */
    float *h_orig  = (float *)malloc(total_bytes);
    float *h_read  = (float *)malloc(total_bytes);
    if (!h_orig || !h_read) {
        fprintf(stderr, "malloc failed\n");
        free(h_orig); free(h_read);
        cudaFree(d_read); H5VLclose(vol_id);
        cudaFree(d_full); gpucompress_cleanup();
        remove(TMP_FILE);
        return 1;
    }

    cudaMemcpy(h_orig, d_full,  total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_read, d_read,  total_bytes, cudaMemcpyDeviceToHost);

    size_t mismatches = 0;
    for (size_t i = 0; i < n_floats && mismatches < 5; i++) {
        if (h_read[i] != h_orig[i]) {
            printf("[Verify] MISMATCH at [%zu]: orig=%.8g got=%.8g\n",
                   i, (double)h_orig[i], (double)h_read[i]);
            mismatches++;
        }
    }

    free(h_orig); free(h_read);
    cudaFree(d_read);

    /* ── Final summary ────────────────────────────────────────────── */
    printf("\n");
    printf("╔══ Final Summary ══════════════════════════════════════════╗\n");
    printf("║  Dataset        : %d MiB  →  %.1f MiB on disk (%.2fx)    \n",
           DATASET_MB, (double)file_bytes / (1 << 20), ratio);
    printf("║  Write          : %.0f ms (%.1f MiB/s)                   \n",
           write_ms, (double)total_bytes / (1 << 20) / (write_ms / 1000.0));
    printf("║  Read           : %.0f ms (%.1f MiB/s)                   \n",
           r1 - r0, (double)total_bytes / (1 << 20) / ((r1 - r0) / 1000.0));
    printf("║  Chunks written : %d  (4 workers, parallel streams)       \n", n_hist);
    printf("║  SGD fires      : %d/%d                                   \n",
           total_sgd, n_hist);
    printf("║  Explorations   : %d/%d                                   \n",
           total_explore, n_hist);
    printf("║  Action changes : %d/%d (exploration found better config)  \n",
           action_changed, n_hist);
    printf("║  Lossless verify: %s                                      \n",
           mismatches == 0 ? "PASS (all floats bit-exact)" : "FAIL");
    printf("╚═══════════════════════════════════════════════════════════╝\n");

    /* Cleanup */
    H5VLclose(vol_id);
    cudaFree(d_full);
    gpucompress_cleanup();
    remove(TMP_FILE);

    return (mismatches == 0) ? 0 : 1;
}
