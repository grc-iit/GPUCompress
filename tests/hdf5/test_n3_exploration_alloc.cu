/**
 * test_n3_exploration_alloc.cu
 *
 * N3: Exploration buffer alloc per-alternative.
 *
 * Measures exploration overhead with forced exploration (K=3).
 * Each alternative does up to 6 cudaMalloc/cudaFree cycles for
 * preprocessing + compression + round-trip buffers.
 *
 * Pre-fix: per-alt alloc overhead ~1ms × 6 buffers × 3 alts = ~18ms/chunk.
 * Post-fix: scratch buffers reused across alternatives.
 *
 * PASS criteria: exploration time < 120ms on 8 MB data with K=3.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>

#include "gpucompress.h"
#include <cuda_runtime.h>
#include <chrono>

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

static const char* WEIGHTS_PATH = "neural_net/weights/model.nnwt";

int main(void) {
    printf("=== N3: Exploration buffer allocation overhead ===\n\n");

    gpucompress_error_t gerr = gpucompress_init(WEIGHTS_PATH);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        printf("SKIP: gpucompress_init failed (need NN weights)\n");
        return 1;
    }
    PASS("init succeeded");

    /* Force exploration on every chunk */
    gpucompress_enable_online_learning();
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);
    gpucompress_set_exploration_k(3);
    gpucompress_set_reinforcement(1, 0.01f, 0.0f, 0.0f);

    const size_t N = 2 * 1024 * 1024;  /* 8 MB (2M floats) */
    float* d_data = NULL;
    if (cudaMalloc(&d_data, N * sizeof(float)) != cudaSuccess) {
        FAIL("cudaMalloc"); goto cleanup;
    }

    {
        float* h_data = (float*)malloc(N * sizeof(float));
        for (size_t i = 0; i < N; i++)
            h_data[i] = sinf((float)i * 0.01f) * 100.0f + (float)(i % 256);
        cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
        free(h_data);
    }

    {
        size_t out_size = gpucompress_max_compressed_size(N * sizeof(float));
        void* d_output = NULL;
        if (cudaMalloc(&d_output, out_size) != cudaSuccess) {
            FAIL("cudaMalloc output"); goto cleanup;
        }

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

        /* Warmup (2 runs) */
        for (int w = 0; w < 2; w++) {
            size_t sz = out_size;
            gpucompress_compress_gpu(d_data, N * sizeof(float), d_output, &sz,
                                     &cfg, NULL, NULL);
        }

        printf("\n--- Exploration alloc overhead (K=3, 8 MB data, 10 runs) ---\n");

        const int N_RUNS = 10;
        double explore_ms[N_RUNS];
        double total_ms[N_RUNS];

        for (int r = 0; r < N_RUNS; r++) {
            size_t sz = out_size;
            cudaDeviceSynchronize();

            auto t0 = std::chrono::steady_clock::now();
            gpucompress_compress_gpu(d_data, N * sizeof(float), d_output, &sz,
                                     &cfg, NULL, NULL);
            auto t1 = std::chrono::steady_clock::now();

            total_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();

            int chunk_count = gpucompress_get_chunk_history_count();
            gpucompress_chunk_diag_t diag = {};
            if (chunk_count > 0)
                gpucompress_get_chunk_diag(chunk_count - 1, &diag);

            explore_ms[r] = diag.exploration_ms;
            printf("  Run %2d: total=%6.2f ms  explore=%6.2f ms  compress=%5.2f ms\n",
                   r + 1, total_ms[r], explore_ms[r], diag.compression_ms);
        }

        /* Stats */
        double sum_explore = 0, min_explore = 1e9;
        int explore_count = 0;
        for (int r = 0; r < N_RUNS; r++) {
            if (explore_ms[r] > 0) {
                sum_explore += explore_ms[r];
                if (explore_ms[r] < min_explore) min_explore = explore_ms[r];
                explore_count++;
            }
        }

        if (explore_count > 0) {
            double avg_explore = sum_explore / explore_count;
            printf("\n  Exploration triggered: %d/%d\n", explore_count, N_RUNS);
            printf("  Avg exploration: %.2f ms\n", avg_explore);
            printf("  Min exploration: %.2f ms\n", min_explore);

            if (min_explore < 120.0) {
                PASS("exploration overhead < 120ms (alloc not dominant)");
            } else {
                char buf[128];
                snprintf(buf, sizeof(buf),
                         "exploration min %.1fms >= 120ms (alloc overhead too high)", min_explore);
                FAIL(buf);
            }
        } else {
            printf("\n  Exploration did not trigger\n");
            FAIL("exploration should have triggered with threshold=0");
        }

        cudaFree(d_output);
    }

cleanup:
    if (d_data) cudaFree(d_data);
    gpucompress_cleanup();

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("OVERALL: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
