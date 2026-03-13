/**
 * Find data patterns where Bitcomp wins, then test if NN picks it.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

static const size_t DATA_SIZE = 1 << 20;  // 1 MB
static const char* algo_names[] = {
    "auto", "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
};

struct Pattern {
    const char* name;
    void (*gen)(float*, size_t);
};

// Bitcomp is a bitwise compressor - it works best on data with
// redundancy in bit patterns (leading zeros, common bit prefixes)

static void gen_sparse(float* d, size_t n) {
    // Mostly zeros, occasional spikes
    for (size_t i = 0; i < n; i++) d[i] = (i % 32 == 0) ? 1.0f : 0.0f;
}

static void gen_very_sparse(float* d, size_t n) {
    // 99% zeros
    for (size_t i = 0; i < n; i++) d[i] = (i % 100 == 0) ? 1.0f : 0.0f;
}

static void gen_tiny_ints(float* d, size_t n) {
    // Small integers (few significant bits)
    for (size_t i = 0; i < n; i++) d[i] = (float)(i % 4);
}

static void gen_binary(float* d, size_t n) {
    // Just 0 and 1
    srand(42);
    for (size_t i = 0; i < n; i++) d[i] = (rand() % 2) ? 1.0f : 0.0f;
}

static void gen_powers_of_2(float* d, size_t n) {
    // Powers of 2 (clean bit patterns)
    for (size_t i = 0; i < n; i++) d[i] = powf(2.0f, (float)(i % 8));
}

static void gen_smooth_sine(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = sinf((float)i * 0.001f) * 1000.0f;
}

static void gen_noisy(float* d, size_t n) {
    srand(123);
    for (size_t i = 0; i < n; i++) d[i] = (float)rand() / (float)RAND_MAX;
}

static void gen_constant(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = 42.0f;
}

static void gen_near_zero(float* d, size_t n) {
    // Very small values (denormals / small exponents)
    for (size_t i = 0; i < n; i++) d[i] = (float)i * 1e-30f;
}

static void gen_zeros(float* d, size_t n) {
    memset(d, 0, n * sizeof(float));
}

static void gen_int_small_range(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = (float)(i % 8);
}

static void gen_mostly_same(float* d, size_t n) {
    // 95% one value, 5% another
    for (size_t i = 0; i < n; i++) d[i] = (i % 20 == 0) ? 99.0f : 1.0f;
}

static void gen_delta_small(float* d, size_t n) {
    // Values with small differences (adjacent bits similar after shuffle)
    d[0] = 100.0f;
    for (size_t i = 1; i < n; i++) d[i] = d[i-1] + 0.001f;
}

int main() {
    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) { fprintf(stderr, "init failed\n"); return 1; }

    Pattern patterns[] = {
        {"zeros",         gen_zeros},
        {"constant",      gen_constant},
        {"sparse_1/32",   gen_sparse},
        {"sparse_1/100",  gen_very_sparse},
        {"tiny_ints_0-3", gen_tiny_ints},
        {"binary_0_1",    gen_binary},
        {"powers_of_2",   gen_powers_of_2},
        {"smooth_sine",   gen_smooth_sine},
        {"noisy_random",  gen_noisy},
        {"near_zero",     gen_near_zero},
        {"int_0-7",       gen_int_small_range},
        {"mostly_same",   gen_mostly_same},
        {"delta_small",   gen_delta_small},
    };
    int np = sizeof(patterns) / sizeof(patterns[0]);

    float* data = (float*)malloc(DATA_SIZE);
    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* output = malloc(max_out);
    size_t n = DATA_SIZE / sizeof(float);

    for (int shuf = 0; shuf <= 1; shuf++) {
        printf("\n=== %s ===\n", shuf ? "WITH SHUFFLE" : "NO SHUFFLE");
        printf("%-16s", "Pattern");
        for (int a = 1; a <= 8; a++) printf(" %9s", algo_names[a]);
        printf("   BEST      BC_RANK\n");

        for (int p = 0; p < np; p++) {
            patterns[p].gen(data, n);
            double ratios[9] = {};
            for (int a = 1; a <= 8; a++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)a;
                if (shuf) cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
                gpucompress_stats_t stats;
                size_t sz = max_out;
                err = gpucompress_compress(data, DATA_SIZE, output, &sz, &cfg, &stats);
                ratios[a] = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0.0;
            }

            int best = 1;
            for (int a = 2; a <= 8; a++) if (ratios[a] > ratios[best]) best = a;
            int bc_rank = 1;
            for (int a = 1; a <= 8; a++) if (a != 8 && ratios[a] > ratios[8]) bc_rank++;

            printf("%-16s", patterns[p].name);
            for (int a = 1; a <= 8; a++) {
                const char* mark = (a == best) ? "*" : " ";
                printf(" %8.2f%s", ratios[a], mark);
            }
            printf("  %-8s  %d/8\n", algo_names[best], bc_rank);
        }
    }

    free(data);
    free(output);
    gpucompress_cleanup();
    return 0;
}
