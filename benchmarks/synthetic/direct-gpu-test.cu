/**
 * @file direct-gpu-test.cu
 * @brief Calls gpucompress_compress_gpu() directly (no VOL) to test
 *        whether the NN ever picks shuffle/quant preprocessing.
 *
 * Usage:
 *   ./build/direct_gpu_test model.nnwt [--error-bound 0.01]
 *
 * Set GC_TRACE=1 to see per-chunk DECODE/QUANT/SHUFFLE traces.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include "gpucompress.h"

/* ── GPU kernel: 5 data patterns in separate chunks ────────────── */

__global__ void fill_patterns(float *data, size_t chunk_elems, int n_chunks,
                              unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (int)(chunk_elems * n_chunks);
    if (idx >= total) return;

    int chunk_id = idx / (int)chunk_elems;
    int local    = idx % (int)chunk_elems;
    int pattern  = chunk_id % 5;

    unsigned int h = seed ^ (chunk_id * 2654435761u);
    float val = 0.0f;

    switch (pattern) {
    case 0: { /* Random noise */
        curandState rng;
        curand_init(h + idx, 0, 0, &rng);
        val = curand_uniform(&rng) * 1000.0f;
        break;
    }
    case 1: /* Constant */
        val = 3.14159f;
        break;
    case 2: /* Smooth gradient (small ints) */
        val = (float)(local % 256);
        break;
    case 3: /* Periodic */
        val = (float)((local % 32) * 100);
        break;
    case 4: { /* Sparse spikes */
        curandState rng;
        curand_init(h + idx, 0, 0, &rng);
        val = (curand_uniform(&rng) > 0.99f) ? 99999.0f : 0.0f;
        break;
    }
    }
    data[idx] = val;
}

static const char* action_str(int action, char *buf, size_t sz) {
    static const char *algos[] = {"lz4","snappy","deflate","gdeflate",
                                   "zstd","ans","cascaded","bitcomp"};
    if (action < 0) { snprintf(buf, sz, "none"); return buf; }
    int algo  = action % 8;
    int quant = (action / 8) % 2;
    int shuf  = (action / 16) % 2;
    snprintf(buf, sz, "%s%s%s", algos[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
    return buf;
}

int main(int argc, char **argv)
{
    const char *weights_path = NULL;
    double error_bound = 0.0;

    for (int i = 1; i < argc; i++) {
        if (!weights_path && argv[i][0] != '-')
            weights_path = argv[i];
        else if (strcmp(argv[i], "--error-bound") == 0 && i + 1 < argc)
            error_bound = atof(argv[++i]);
    }
    if (!weights_path) {
        fprintf(stderr, "Usage: %s model.nnwt [--error-bound 0.01]\n", argv[0]);
        return 1;
    }

    /* Init */
    if (gpucompress_init(weights_path) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    /* Enable exploration so we can see what alternatives exist */
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.5f, 0.10f, 0.10f);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);
    gpucompress_set_exploration_k(16);

    /* Allocate: 10 chunks of 256KB each */
    const int N_CHUNKS = 10;
    const size_t CHUNK_ELEMS = 64 * 1024;  /* 256 KB per chunk */
    const size_t CHUNK_BYTES = CHUNK_ELEMS * sizeof(float);
    const size_t TOTAL_ELEMS = CHUNK_ELEMS * N_CHUNKS;
    const size_t TOTAL_BYTES = TOTAL_ELEMS * sizeof(float);

    float *d_data = NULL;
    cudaMalloc(&d_data, TOTAL_BYTES);

    int threads = 256;
    int blocks = ((int)TOTAL_ELEMS + threads - 1) / threads;
    fill_patterns<<<blocks, threads>>>(d_data, CHUNK_ELEMS, N_CHUNKS, 42);
    cudaDeviceSynchronize();

    /* Compressed output buffer (per-chunk) */
    size_t max_comp = gpucompress_max_compressed_size(CHUNK_BYTES);
    uint8_t *d_comp = NULL;
    cudaMalloc(&d_comp, max_comp);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Direct gpucompress_compress_gpu() Preprocessing Test   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("  Chunks     : %d x %zu KB\n", N_CHUNKS, CHUNK_BYTES / 1024);
    printf("  Error bound: %.6e%s\n", error_bound,
           error_bound > 0 ? " (lossy)" : " (lossless)");
    printf("  Weights    : %s\n\n", weights_path);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
    cfg.error_bound = error_bound;

    const char *patterns[] = {"noise", "constant", "gradient", "periodic", "sparse"};

    printf("  %-5s  %-10s  %-22s  %-8s  %-10s  %-10s  %-3s  %-3s\n",
           "Chunk", "Pattern", "Action", "Ratio", "PredRatio", "CompMs", "SGD", "EXP");
    printf("  -----  ----------  ----------------------  --------  ----------  ----------  ---  ---\n");

    for (int c = 0; c < N_CHUNKS; c++) {
        const uint8_t *chunk_ptr = (const uint8_t*)d_data + c * CHUNK_BYTES;
        size_t comp_sz = max_comp;
        gpucompress_stats_t stats = {};

        gpucompress_reset_chunk_history();

        gpucompress_error_t err = gpucompress_compress_gpu(
            chunk_ptr, CHUNK_BYTES, d_comp, &comp_sz, &cfg, &stats, NULL);

        if (err != GPUCOMPRESS_SUCCESS) {
            printf("  %-5d  %-10s  FAILED (err=%d)\n", c, patterns[c % 5], (int)err);
            continue;
        }

        double ratio = (double)CHUNK_BYTES / (double)comp_sz;

        /* Get diagnostics */
        gpucompress_chunk_diag_t diag = {};
        gpucompress_get_chunk_diag(0, &diag);

        char astr[40];
        action_str(diag.nn_action, astr, sizeof(astr));

        printf("  %-5d  %-10s  %-22s  %6.2fx   %8.2f    %8.3f    %s  %s\n",
               c, patterns[c % 5], astr, ratio,
               (double)diag.predicted_ratio,
               (double)diag.compression_ms,
               diag.sgd_fired ? "Y" : ".",
               diag.exploration_triggered ? "Y" : ".");
    }

    printf("\n=== Direct GPU test complete ===\n");

    cudaFree(d_data);
    cudaFree(d_comp);
    gpucompress_cleanup();
    return 0;
}
