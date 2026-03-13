/**
 * Test: Can the NN adapt via online SGD to pick LZ4 over zstd
 * when repeatedly seeing repeat_blk64 data (where lz4 = 201x, zstd = 37x)?
 *
 * Setup: 1GB of repeat_blk64 data, 64KB chunks = 16384 chunks.
 * Online SGD enabled (LR=0.3, MAPE threshold=20%).
 * Track per-chunk: what the NN picks, predicted ratio, actual ratio.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

static const char* algo_names[] = {
    "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
};

static void gen_repeat_blk64(float* d, size_t n) {
    float blk[64];
    for (int i = 0; i < 64; i++) blk[i] = (float)(i * i);
    for (size_t i = 0; i < n; i++) d[i] = blk[i % 64];
}

static const char* find_weights() {
    static char buf[512];
    snprintf(buf, sizeof(buf), "%s/GPUCompress/neural_net/weights/model.nnwt",
             getenv("HOME") ? getenv("HOME") : ".");
    FILE* f = fopen(buf, "rb");
    if (f) { fclose(f); return buf; }
    f = fopen("neural_net/weights/model.nnwt", "rb");
    if (f) { fclose(f); return "neural_net/weights/model.nnwt"; }
    return NULL;
}

int main() {
    const char* w = find_weights();
    if (!w) { fprintf(stderr, "No weights\n"); return 1; }

    gpucompress_error_t err = gpucompress_init(w);
    if (err != GPUCOMPRESS_SUCCESS) { fprintf(stderr, "init failed\n"); return 1; }

    // Enable online learning + SGD + exploration
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.9f, 0.20f, 0.50f);
    gpucompress_set_exploration(0);
    // gpucompress_set_exploration_k();  // try 3 alternative configs per chunk

    size_t chunk_size = 128 * 1024;  // 64KB chunks
    size_t total_size = (size_t)1024 * 1024 * 1024;  // 1GB
    int n_chunks = (int)(total_size / chunk_size);  // 16384

    size_t n_floats = chunk_size / sizeof(float);
    float* data = (float*)malloc(chunk_size);
    size_t max_out = gpucompress_max_compressed_size(chunk_size);
    void* output = malloc(max_out);

    // Generate the data once (same pattern every chunk)
    gen_repeat_blk64(data, n_floats);

    // Track stats
    int algo_counts[8] = {0};
    int lz4_streak = 0, max_lz4_streak = 0;
    int first_lz4 = -1;
    int sgd_fires = 0;

    printf("%-7s  %-10s  %8s  %8s  %7s  %s\n",
           "Chunk", "NN_PICK", "PRED", "ACTUAL", "MAPE%", "SGD");
    printf("-------  ----------  --------  --------  -------  ---\n");

    for (int c = 0; c < n_chunks; c++) {
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
        cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;

        gpucompress_stats_t stats;
        size_t sz = max_out;
        err = gpucompress_compress(data, chunk_size, output, &sz, &cfg, &stats);
        if (err != GPUCOMPRESS_SUCCESS) {
            printf("Chunk %d: compress failed\n", c);
            continue;
        }

        int action = stats.nn_final_action;
        int algo_idx = action % 8;
        algo_counts[algo_idx]++;

        double mape = (stats.compression_ratio > 0) ?
            fabs(stats.predicted_ratio - stats.compression_ratio) /
            stats.compression_ratio * 100.0 : 0.0;

        int sgd = stats.sgd_fired;
        if (sgd) sgd_fires++;

        if (algo_idx == 0) {  // lz4
            lz4_streak++;
            if (lz4_streak > max_lz4_streak) max_lz4_streak = lz4_streak;
            if (first_lz4 < 0) first_lz4 = c;
        } else {
            lz4_streak = 0;
        }

        // Print every chunk for first 50, then every 100, then every 1000
        if (c < 50 || (c < 500 && c % 50 == 0) || (c < 5000 && c % 500 == 0) || c % 2000 == 0 || c == n_chunks - 1) {
            printf("%5d    %-10s  %8.1f  %8.1f  %6.1f%%  %s\n",
                   c, algo_names[algo_idx],
                   stats.predicted_ratio, stats.compression_ratio,
                   mape, sgd ? "YES" : "");
        }
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  SUMMARY: %d chunks of repeat_blk64 (64KB each)      ║\n", n_chunks);
    printf("╠═══════════════════════════════════════════════════════╣\n");
    printf("║  Algorithm picks:                                     ║\n");
    for (int a = 0; a < 8; a++) {
        if (algo_counts[a] > 0) {
            printf("║    %-10s  %5d  (%5.1f%%)                          ║\n",
                   algo_names[a], algo_counts[a],
                   100.0 * algo_counts[a] / n_chunks);
        }
    }
    printf("║                                                       ║\n");
    printf("║  SGD fired: %d/%d chunks                              ║\n", sgd_fires, n_chunks);
    printf("║  First LZ4 pick: chunk %d                             ║\n", first_lz4);
    printf("║  Max LZ4 streak: %d consecutive                      ║\n", max_lz4_streak);
    printf("║                                                       ║\n");
    printf("║  Ground truth: lz4+shuf = 201x, zstd+shuf = 37.7x    ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");

    free(data);
    free(output);
    gpucompress_cleanup();
    return 0;
}
