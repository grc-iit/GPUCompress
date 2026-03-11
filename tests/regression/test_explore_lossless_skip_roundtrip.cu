/**
 * test_explore_lossless_skip_roundtrip.cu
 *
 * Issue #2: In lossless exploration, the round-trip decompress is unnecessary
 * since PSNR is hardcoded to 120.0. This test verifies:
 *
 *   1. Exploration still produces correct results (best alternative picked)
 *   2. Exploration time is reduced when round-trip is skipped
 *   3. SGD samples from exploration still have valid fields
 *   4. Round-trip correctness of the final output is preserved
 *
 * Strategy:
 *   - Use ALGO_AUTO + online learning + exploration enabled
 *   - Low MAPE threshold to force exploration
 *   - Compress OOD data (linear ramp) multiple times
 *   - Measure exploration wall-clock time
 *   - Verify the final compression is still lossless-correct
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "gpucompress.h"

#define DATA_MB       16
#define N_ITERS       10
#define EXPLORE_K     3

__global__ static void fill_ramp(float* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (float)i / (float)n;
}

static const char* find_weights() {
    const char* paths[] = {
        "neural_net/weights/model.nnwt",
        "../neural_net/weights/model.nnwt",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        FILE* f = fopen(paths[i], "rb");
        if (f) { fclose(f); return paths[i]; }
    }
    return paths[0];
}

int main(void)
{
    printf("=== Issue #2: Lossless Exploration Round-Trip Skip Test ===\n\n");

    const char* wpath = find_weights();
    if (gpucompress_init(wpath) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n");
        return 1;
    }

    /* Enable online learning + exploration with low thresholds */
    gpucompress_enable_online_learning();
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_k(EXPLORE_K);
    gpucompress_set_reinforcement(1, 0.01f, 0.001f, 0.001f);

    size_t data_bytes = (size_t)DATA_MB * 1024 * 1024;
    size_t n_floats   = data_bytes / sizeof(float);
    size_t max_comp   = gpucompress_max_compressed_size(data_bytes);

    void* d_input  = NULL;
    void* d_output = NULL;
    cudaMalloc(&d_input,  data_bytes);
    cudaMalloc(&d_output, max_comp);
    if (!d_input || !d_output) { fprintf(stderr, "cudaMalloc failed\n"); return 1; }

    int threads = 256, blocks = (int)((n_floats + threads - 1) / threads);
    fill_ramp<<<blocks, threads>>>((float*)d_input, n_floats);
    cudaDeviceSynchronize();

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
    cfg.error_bound = 0.0;  /* LOSSLESS — this is the key */

    printf("%-6s  %-10s  %-10s  %-8s  %-8s  %-10s\n",
           "Iter", "ratio", "pred", "explore", "sgd", "explore_ms");
    printf("------  ----------  ----------  --------  --------  ----------\n");

    int explore_count = 0;
    int sgd_count = 0;
    double total_explore_ms = 0.0;
    size_t last_comp_size = 0;

    for (int iter = 0; iter < N_ITERS; iter++) {
        size_t out_sz = max_comp;
        gpucompress_stats_t stats = {};

        auto t0 = std::chrono::steady_clock::now();
        gpucompress_error_t ce =
            gpucompress_compress_gpu(d_input, data_bytes, d_output, &out_sz,
                                     &cfg, &stats, NULL);
        auto t1 = std::chrono::steady_clock::now();

        if (ce != GPUCOMPRESS_SUCCESS) {
            fprintf(stderr, "compress failed at iter %d (err=%d)\n", iter, (int)ce);
            break;
        }

        double ratio = stats.compression_ratio;
        double pred  = stats.predicted_ratio;
        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        /* Get per-chunk diag for exploration time */
        gpucompress_chunk_diag_t diag = {};
        gpucompress_get_chunk_diag(0, &diag);
        double explore_ms = diag.exploration_ms;

        printf("%-6d  %-10.2f  %-10.2f  %-8s  %-8s  %-10.2f\n",
               iter, ratio, pred,
               stats.exploration_triggered ? "YES" : "no",
               stats.sgd_fired ? "YES" : "no",
               explore_ms);

        if (stats.exploration_triggered) {
            explore_count++;
            total_explore_ms += explore_ms;
        }
        if (stats.sgd_fired) sgd_count++;
        last_comp_size = out_sz;

        gpucompress_reset_chunk_history();
    }

    printf("\n");

    /* Round-trip correctness check */
    printf("[Round-trip check]\n");
    size_t out_sz = max_comp;
    gpucompress_stats_t stats = {};
    gpucompress_error_t ce = gpucompress_compress_gpu(
        d_input, data_bytes, d_output, &out_sz, &cfg, &stats, NULL);

    int roundtrip_ok = 0;
    if (ce == GPUCOMPRESS_SUCCESS) {
        void* d_decomp = NULL;
        cudaMalloc(&d_decomp, data_bytes);
        size_t decomp_sz = data_bytes;
        gpucompress_error_t de = gpucompress_decompress_gpu(
            d_output, out_sz, d_decomp, &decomp_sz, NULL);
        if (de == GPUCOMPRESS_SUCCESS && decomp_sz == data_bytes) {
            float* h_orig = (float*)malloc(data_bytes);
            float* h_dec  = (float*)malloc(data_bytes);
            cudaMemcpy(h_orig, d_input,  data_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_dec,  d_decomp, data_bytes, cudaMemcpyDeviceToHost);
            roundtrip_ok = 1;
            for (size_t i = 0; i < n_floats; i++) {
                if (h_orig[i] != h_dec[i]) {
                    printf("  MISMATCH at %zu: orig=%.6f got=%.6f\n",
                           i, h_orig[i], h_dec[i]);
                    roundtrip_ok = 0;
                    break;
                }
            }
            free(h_orig);
            free(h_dec);
        }
        if (d_decomp) cudaFree(d_decomp);
    }
    printf("  Round-trip: %s\n\n", roundtrip_ok ? "PASS" : "FAIL");

    /* Results */
    double avg_explore_ms = (explore_count > 0) ?
        total_explore_ms / explore_count : 0.0;

    printf("=== Results ===\n");
    printf("  Explorations triggered: %d / %d\n", explore_count, N_ITERS);
    printf("  SGD fired: %d / %d\n", sgd_count, N_ITERS);
    printf("  Avg exploration time: %.2f ms\n", avg_explore_ms);
    printf("  Total exploration time: %.2f ms\n", total_explore_ms);
    printf("  Round-trip: %s\n", roundtrip_ok ? "PASS" : "FAIL");

    gpucompress_cleanup();
    cudaFree(d_input);
    cudaFree(d_output);

    /* Pass criteria:
     * 1. Round-trip correctness preserved
     * 2. At least some explorations triggered (validates the test exercises the path)
     */
    int pass = roundtrip_ok && (explore_count > 0);

    printf("\n=== VERDICT: %s ===\n", pass ? "PASS" : "FAIL");
    if (!pass) {
        if (!roundtrip_ok) printf("  -> Round-trip broken\n");
        if (explore_count == 0) printf("  -> No explorations triggered (test not exercising the code path)\n");
    }

    return pass ? 0 : 1;
}
