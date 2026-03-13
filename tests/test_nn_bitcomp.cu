/**
 * @file test_nn_bitcomp.cu
 * @brief Test: does the NN select bitcomp (non-zstd) on low-bit-width data?
 *
 * Layout (128 chunks, 1 MB each = 128 MB total):
 *   Chunks  0-63:  random uint32 range [0,1]   -> bitcomp wins (~28.9x vs zstd ~19.7x)
 *   Chunks 64-127: smooth ramp + noise floats  -> zstd+shuf wins (~8-16x)
 *
 * Phases:
 *   1. Exhaustive search -- confirms bitcomp wins first half, zstd second
 *   2. NN-RL (ALGO_AUTO + SGD) -- does the NN pick bitcomp then switch to zstd?
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include <hdf5.h>
#include <cuda_runtime.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

#define CHUNK_FLOATS  (256 * 1024)   /* 1 MB per chunk (262144 x 4 bytes) */
#define NUM_CHUNKS    32
#define HALF          (NUM_CHUNKS / 2)
#define TOTAL_FLOATS  ((size_t)CHUNK_FLOATS * NUM_CHUNKS)
#define DATA_BYTES    (TOTAL_FLOATS * sizeof(float))
#define HDF5_FILE     "/tmp/test_nn_bitcomp.h5"

static const char* ALGO_NAMES[] = {
    "auto", "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while (0)

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/* ── Xorshift PRNG (CPU side) ────────────────────────────────── */

static uint32_t xor_state = 12345;
static uint32_t xorshift(void) {
    uint32_t x = xor_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xor_state = x;
    return x;
}

/* ── GPU kernel (second half only) ───────────────────────────── */

/* Second half: smooth ramp + noise — zstd+shuf wins */
__global__ void fill_zstd_region(float* out, size_t offset, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    /* Use idx/n (not offset+idx) to avoid float precision loss at large offsets */
    float x = (float)idx / (float)n;
    out[offset + idx] = x * 100.0f + 0.5f * sinf(x * 200.0f) + 0.1f * cosf(x * 37.0f);
}

/* ── CPU fill for bitcomp region ─────────────────────────────── */
/* Random uint32 in [0,1] — bitcomp wins ~28.9x vs zstd ~19.7x */
static void fill_bitcomp_cpu(uint32_t* h_buf, size_t n_elems)
{
    xor_state = 42;
    for (size_t i = 0; i < n_elems; i++)
        h_buf[i] = xorshift() & 1u;  /* 1-bit values: 0 or 1 */
}

/* Bitwise compare */
__global__ void compare_kernel(const float* a, const float* b, size_t n,
                                unsigned long long* mismatches)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    unsigned int va, vb;
    memcpy(&va, &a[idx], sizeof(unsigned int));
    memcpy(&vb, &b[idx], sizeof(unsigned int));
    if (va != vb) atomicAdd(mismatches, 1ULL);
}

/* ── VOL helpers ─────────────────────────────────────────────── */

static hid_t make_vol_fapl(void) {
    hid_t vol_id = H5VL_gpucompress_register();
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
    H5VLclose(vol_id);
    return fapl;
}

/* ── Exhaustive search ───────────────────────────────────────── */

struct ExhResult {
    int best_algo;
    int best_shuf;
    double best_ratio;
    int second_algo;
    int second_shuf;
    double second_ratio;
};

static void run_exhaustive(const float* d_data, ExhResult* results)
{
    size_t chunk_bytes = CHUNK_FLOATS * sizeof(float);
    size_t out_buf_size = chunk_bytes + 65536;

    void* d_comp = NULL;
    CUDA_CHECK(cudaMalloc(&d_comp, out_buf_size));

    printf("  chunk | best algo            | ratio    | 2nd-best algo        | 2nd ratio | gap\n");
    printf("  ------+----------------------+----------+----------------------+-----------+--------\n");

    for (int c = 0; c < NUM_CHUNKS; c++) {
        const float* d_chunk = d_data + (size_t)c * CHUNK_FLOATS;

        double best_ratio = 0, second_ratio = 0;
        int best_algo = 1, second_algo = 1;
        int best_shuf = 0, second_shuf = 0;

        for (int algo = 1; algo <= 8; algo++) {
            for (int shuf = 0; shuf <= 1; shuf++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)algo;
                cfg.preprocessing = shuf ? GPUCOMPRESS_PREPROC_SHUFFLE_4 : 0;
                cfg.error_bound = 0.0;

                size_t out_size = out_buf_size;
                gpucompress_stats_t stats;
                gpucompress_error_t err = gpucompress_compress_gpu(
                    d_chunk, chunk_bytes, d_comp, &out_size, &cfg, &stats, NULL);

                double ratio = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0;
                if (ratio > best_ratio) {
                    second_ratio = best_ratio;
                    second_algo  = best_algo;
                    second_shuf  = best_shuf;
                    best_ratio   = ratio;
                    best_algo    = algo;
                    best_shuf    = shuf;
                } else if (ratio > second_ratio) {
                    second_ratio = ratio;
                    second_algo  = algo;
                    second_shuf  = shuf;
                }
            }
        }

        results[c].best_algo    = best_algo;
        results[c].best_shuf    = best_shuf;
        results[c].best_ratio   = best_ratio;
        results[c].second_algo  = second_algo;
        results[c].second_shuf  = second_shuf;
        results[c].second_ratio = second_ratio;

        char best_str[32], second_str[32];
        snprintf(best_str, sizeof(best_str), "%s%s",
                 ALGO_NAMES[best_algo], best_shuf ? "+shuf" : "");
        snprintf(second_str, sizeof(second_str), "%s%s",
                 ALGO_NAMES[second_algo], second_shuf ? "+shuf" : "");
        double gap = best_ratio - second_ratio;

        const char* region = (c < HALF) ? "bitcomp-region" : "zstd-region";

        printf("  %5d | %-20s | %7.2fx | %-20s | %8.2fx | %+.2fx  (%s)\n",
               c, best_str, best_ratio, second_str, second_ratio, gap, region);
    }

    /* Summary: count how many chunks each algo won */
    int algo_wins[9] = {0};
    for (int c = 0; c < NUM_CHUNKS; c++)
        algo_wins[results[c].best_algo]++;

    printf("\n  Exhaustive winner distribution:\n");
    for (int a = 1; a <= 8; a++) {
        if (algo_wins[a] > 0)
            printf("    %-10s: %3d/%d chunks\n", ALGO_NAMES[a], algo_wins[a], NUM_CHUNKS);
    }

    cudaFree(d_comp);
}

/* ── NN-RL pass via VOL ─────────────────────────────────────── */

static int run_nn_rl(float* d_data, float* d_readback, const ExhResult* exh)
{
    gpucompress_reset_chunk_history();
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.3f, 0.20f, 0.20f);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_k(31);           /* try all 31 alternatives */
    gpucompress_set_exploration_threshold(0.10); /* trigger at 10% MAPE */

    /* Write */
    hid_t fapl = make_vol_fapl();
    hid_t fid  = H5Fcreate(HDF5_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);

    hsize_t dims[1]  = { TOTAL_FLOATS };
    hsize_t chunk[1] = { CHUNK_FLOATS };
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = GPUCOMPRESS_ALGO_AUTO;
    cd[1] = 0;
    cd[2] = 0;
    pack_double(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                            space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    herr_t wr = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, d_data);
    H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);

    if (wr < 0) { fprintf(stderr, "H5Dwrite failed\n"); return 1; }

    /* Print per-chunk diagnostics */
    int n_hist = gpucompress_get_chunk_history_count();
    printf("  chunk | region         | NN orig             | final action         | ratio  | pred   | MAPE    | expl | SGD | oracle              | match\n");
    printf("  ------+----------------+---------------------+----------------------+--------+--------+---------+------+-----+---------------------+------\n");

    int algo_correct = 0, algo_wrong = 0;
    int bitcomp_picks = 0, zstd_picks = 0, other_picks = 0;
    int first_half_bitcomp = 0, second_half_zstd = 0;
    int exploration_count = 0;
    double mape_sum = 0; int mape_n = 0;

    /* Print every chunk for small datasets */
    auto should_print = [](int i) -> bool {
        (void)i;
        return true;
    };
    int last_printed = -2;

    for (int i = 0; i < n_hist && i < NUM_CHUNKS; i++) {
        gpucompress_chunk_diag_t d;
        gpucompress_get_chunk_diag(i, &d);

        /* Final action (what was actually used) */
        int final_action = d.nn_action;
        int algo_id = (final_action % 8) + 1;
        int shuf    = (final_action / 16) % 2;
        const char* aname = (algo_id >= 1 && algo_id <= 8) ? ALGO_NAMES[algo_id] : "???";
        char final_str[32];
        snprintf(final_str, sizeof(final_str), "%s%s", aname, shuf ? "+shuf" : "");

        /* Original NN action (before exploration) */
        int orig_action = d.nn_original_action;
        int orig_algo = (orig_action % 8) + 1;
        int orig_shuf = (orig_action / 16) % 2;
        const char* orig_aname = (orig_algo >= 1 && orig_algo <= 8) ? ALGO_NAMES[orig_algo] : "???";
        char orig_str[32];
        snprintf(orig_str, sizeof(orig_str), "%s%s", orig_aname, orig_shuf ? "+shuf" : "");

        /* Oracle from exhaustive */
        char oracle_str[32];
        snprintf(oracle_str, sizeof(oracle_str), "%s%s",
                 ALGO_NAMES[exh[i].best_algo], exh[i].best_shuf ? "+shuf" : "");

        double mape = (d.actual_ratio > 0)
            ? fabs((double)d.predicted_ratio - (double)d.actual_ratio)
              / (double)d.actual_ratio * 100.0
            : 0.0;
        mape_sum += mape; mape_n++;

        const char* region = (i < HALF) ? "bitcomp-region" : "zstd-region   ";

        /* Check if final action matches oracle algo */
        bool algo_match = (algo_id == exh[i].best_algo);
        if (algo_match) algo_correct++; else algo_wrong++;

        if (algo_id == 8) bitcomp_picks++;
        else if (algo_id == 5) zstd_picks++;
        else other_picks++;

        if (i < HALF && algo_id == 8) first_half_bitcomp++;
        if (i >= HALF && algo_id == 5) second_half_zstd++;
        if (d.exploration_triggered) exploration_count++;

        if (should_print(i)) {
            if (last_printed < i - 1)
                printf("        |                |                     |          ...         |        |        |         |      |     |                     |\n");
            printf("  %5d | %s | %-19s | %-20s | %5.2fx | %5.2fx | %6.1f%% | %s | %s | %-19s | %s\n",
                   i, region, orig_str, final_str,
                   (double)d.actual_ratio, (double)d.predicted_ratio,
                   mape,
                   d.exploration_triggered ? "yes " : "  - ",
                   d.sgd_fired ? "yes" : "  -",
                   oracle_str, algo_match ? "YES" : "no");
            last_printed = i;
        }
    }

    printf("\n  NN Algorithm Selection Summary:\n");
    printf("    Bitcomp picks: %d/%d  (oracle says bitcomp for first %d)\n",
           bitcomp_picks, n_hist, HALF);
    printf("    Zstd picks:    %d/%d  (oracle says zstd for last %d)\n",
           zstd_picks, n_hist, HALF);
    printf("    Other picks:   %d/%d\n", other_picks, n_hist);
    printf("    Algo match vs oracle: %d/%d (%.1f%%)\n",
           algo_correct, n_hist, 100.0 * algo_correct / n_hist);
    printf("    First half picked bitcomp:  %d/%d\n", first_half_bitcomp, HALF);
    printf("    Second half picked zstd:    %d/%d\n", second_half_zstd, HALF);
    printf("    Exploration triggered: %d/%d chunks\n", exploration_count, n_hist);
    printf("    Overall MAPE: %.1f%%\n", mape_sum / mape_n);

    /* Read back and verify */
    fapl = make_vol_fapl();
    fid  = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(fid, "data", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
    H5Dclose(dset); H5Fclose(fid);

    unsigned long long* d_mm;
    CUDA_CHECK(cudaMalloc(&d_mm, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_mm, 0, sizeof(unsigned long long)));
    int threads = 256;
    int blocks  = ((int)TOTAL_FLOATS + threads - 1) / threads;
    compare_kernel<<<blocks, threads>>>(d_data, d_readback, TOTAL_FLOATS, d_mm);
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long mm = 0;
    cudaMemcpy(&mm, d_mm, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_mm);

    printf("\n  Lossless verification: %llu mismatches -> %s\n",
           mm, mm == 0 ? "PASS" : "FAIL");

    gpucompress_disable_online_learning();
    return (mm > 0) ? 1 : 0;
}

/* ── Main ────────────────────────────────────────────────────── */

int main(int argc, char** argv)
{
    const char* weights = (argc > 1) ? argv[1] : "neural_net/weights/model.nnwt";
    if (argc <= 1) {
        FILE* f = fopen(weights, "rb");
        if (f) fclose(f);
        else weights = "../neural_net/weights/model.nnwt";
    }

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    printf("================================================================\n");
    printf("  NN Bitcomp Selection Test\n");
    printf("================================================================\n");
    printf("  Layout: %d chunks x %d floats = %.1f MB\n",
           NUM_CHUNKS, CHUNK_FLOATS, (double)DATA_BYTES / (1 << 20));
    printf("  Chunks 0-%d:   uint32 range [0,1]      (bitcomp should win)\n", HALF - 1);
    printf("  Chunks %d-%d: smooth ramp + noise      (zstd+shuf should win)\n", HALF, NUM_CHUNKS - 1);
    printf("================================================================\n\n");

    /* Init */
    gpucompress_error_t rc = gpucompress_init(weights);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %s\n", gpucompress_error_string(rc));
        return 1;
    }

    /* Allocate and fill */
    float *d_data = NULL, *d_readback = NULL;
    CUDA_CHECK(cudaMalloc(&d_data, DATA_BYTES));
    CUDA_CHECK(cudaMalloc(&d_readback, DATA_BYTES));

    size_t half_floats = (size_t)HALF * CHUNK_FLOATS;
    size_t half_bytes  = half_floats * sizeof(float);
    int threads = 256;

    /* First half: bitcomp-friendly (uint32 range [0,1]) — fill on CPU, copy to GPU */
    uint32_t* h_bitcomp = (uint32_t*)malloc(half_bytes);
    fill_bitcomp_cpu(h_bitcomp, half_floats);
    CUDA_CHECK(cudaMemcpy(d_data, h_bitcomp, half_bytes, cudaMemcpyHostToDevice));
    free(h_bitcomp);

    /* Second half: zstd+shuf-friendly (smooth ramp + noise) — fill on GPU */
    int blocks2 = ((int)half_floats + threads - 1) / threads;
    fill_zstd_region<<<blocks2, threads>>>(d_data, half_floats, half_floats);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ── Phase 1: Exhaustive ─────────────────────────────────── */
    printf("-- Phase 1: Exhaustive Search (ground truth) ----------------\n");
    ExhResult exh[NUM_CHUNKS];
    run_exhaustive(d_data, exh);
    printf("\n");

    /* ── Phase 2: NN-RL ──────────────────────────────────────── */
    printf("-- Phase 2: NN-RL + Exploration (K=31, threshold=10%%, SGD LR=0.3) --\n");
    int fail = run_nn_rl(d_data, d_readback, exh);
    printf("\n");

    /* Cleanup */
    cudaFree(d_data);
    cudaFree(d_readback);
    gpucompress_cleanup();
    remove(HDF5_FILE);

    return fail;
}
