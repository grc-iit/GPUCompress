/**
 * Test various data patterns to find what Snappy compresses best
 * (relative to other algorithms).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

static const size_t DATA_SIZE = 1 << 20;  // 1 MB

struct Pattern {
    const char* name;
    void (*generate)(float* data, size_t n);
};

static void gen_plateau(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = (float)(i / 256);
}

static void gen_small_vocab(float* d, size_t n) {
    float v[] = {1.0f, 2.0f, 3.0f, 5.0f, 8.0f};
    for (size_t i = 0; i < n; i++) d[i] = v[i % 5];
}

static void gen_periodic_short(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = sinf((float)(i % 64) * 0.1f);
}

static void gen_periodic_long(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = sinf((float)(i % 1024) * 0.01f);
}

static void gen_constant(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = 42.0f;
}

static void gen_two_values(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = (i % 3 == 0) ? 1.0f : 0.0f;
}

static void gen_int_ramp(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = (float)(i % 128);
}

static void gen_repeated_block(float* d, size_t n) {
    // 64-float block repeated
    float block[64];
    for (int i = 0; i < 64; i++) block[i] = (float)(i * i);
    for (size_t i = 0; i < n; i++) d[i] = block[i % 64];
}

static void gen_sparse(float* d, size_t n) {
    // Mostly zeros with occasional 1.0
    for (size_t i = 0; i < n; i++) d[i] = (i % 32 == 0) ? 1.0f : 0.0f;
}

static void gen_quantized_sine(float* d, size_t n) {
    // Sine quantized to 16 levels
    for (size_t i = 0; i < n; i++) {
        float v = sinf((float)i * 0.01f);
        d[i] = roundf(v * 8.0f) / 8.0f;
    }
}

static void gen_run_length(float* d, size_t n) {
    // Random run lengths of same value
    srand(42);
    float val = 1.0f;
    int run = 0;
    for (size_t i = 0; i < n; i++) {
        if (run <= 0) {
            val = (float)(rand() % 10);
            run = 50 + rand() % 200;
        }
        d[i] = val;
        run--;
    }
}

static void gen_repeated_4byte(float* d, size_t n) {
    // Just two float values alternating in blocks of 512
    for (size_t i = 0; i < n; i++) {
        d[i] = ((i / 512) % 2 == 0) ? 100.0f : 200.0f;
    }
}

int main() {
    const char* algo_names[] = {
        "auto", "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
    };

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    Pattern patterns[] = {
        {"plateau_256",     gen_plateau},
        {"small_vocab_5",   gen_small_vocab},
        {"periodic_64",     gen_periodic_short},
        {"periodic_1024",   gen_periodic_long},
        {"constant",        gen_constant},
        {"two_values",      gen_two_values},
        {"int_ramp_128",    gen_int_ramp},
        {"repeated_blk64",  gen_repeated_block},
        {"sparse_32",       gen_sparse},
        {"quant_sine_16",   gen_quantized_sine},
        {"run_length",      gen_run_length},
        {"repeat_blk512",   gen_repeated_4byte},
    };
    int n_patterns = sizeof(patterns) / sizeof(patterns[0]);

    float* data = (float*)malloc(DATA_SIZE);
    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* output = malloc(max_out);
    size_t n_floats = DATA_SIZE / sizeof(float);

    printf("%-18s", "Pattern");
    for (int a = 1; a <= 8; a++) printf(" %10s", algo_names[a]);
    printf("  BEST_ALGO    SNAPPY_RANK\n");
    printf("%-18s", "-------");
    for (int a = 1; a <= 8; a++) printf(" %10s", "------");
    printf("  ---------    -----------\n");

    for (int p = 0; p < n_patterns; p++) {
        patterns[p].generate(data, n_floats);

        double ratios[9] = {0};
        for (int a = 1; a <= 8; a++) {
            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = (gpucompress_algorithm_t)a;
            gpucompress_stats_t stats;
            size_t out_sz = max_out;
            err = gpucompress_compress(data, DATA_SIZE, output, &out_sz, &cfg, &stats);
            ratios[a] = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0.0;
        }

        // Find best and snappy rank
        int best_algo = 1;
        for (int a = 2; a <= 8; a++) {
            if (ratios[a] > ratios[best_algo]) best_algo = a;
        }
        // Rank snappy (algo 2)
        int snappy_rank = 1;
        for (int a = 1; a <= 8; a++) {
            if (a != 2 && ratios[a] > ratios[2]) snappy_rank++;
        }

        printf("%-18s", patterns[p].name);
        for (int a = 1; a <= 8; a++) {
            printf(" %10.2f", ratios[a]);
        }
        printf("  %-10s  %d/8\n", algo_names[best_algo], snappy_rank);
    }

    free(data);
    free(output);
    gpucompress_cleanup();
    return 0;
}
