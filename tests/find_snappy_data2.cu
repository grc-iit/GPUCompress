/**
 * Test where Snappy wins: include shuffle preprocessing and measure time.
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

static void gen_plateau(float* d, size_t n)     { for (size_t i = 0; i < n; i++) d[i] = (float)(i / 256); }
static void gen_small_vocab(float* d, size_t n) { float v[]={1,2,3,5,8}; for (size_t i=0;i<n;i++) d[i]=v[i%5]; }
static void gen_constant(float* d, size_t n)    { for (size_t i = 0; i < n; i++) d[i] = 42.0f; }
static void gen_sparse(float* d, size_t n)      { for (size_t i = 0; i < n; i++) d[i] = (i%32==0)?1.0f:0.0f; }
static void gen_run_length(float* d, size_t n) {
    srand(42); float val=1; int run=0;
    for (size_t i=0;i<n;i++) { if(run<=0){val=(float)(rand()%10);run=50+rand()%200;} d[i]=val; run--; }
}
static void gen_smooth_sine(float* d, size_t n) {
    for (size_t i = 0; i < n; i++) d[i] = sinf((float)i * 0.001f) * 1000.0f;
}
static void gen_noisy(float* d, size_t n) {
    srand(123);
    for (size_t i = 0; i < n; i++) d[i] = (float)rand() / (float)RAND_MAX;
}

int main() {
    const char* algo_names[] = {
        "auto", "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
    };

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) { fprintf(stderr, "init failed\n"); return 1; }

    Pattern patterns[] = {
        {"plateau_256",   gen_plateau},
        {"small_vocab_5", gen_small_vocab},
        {"constant",      gen_constant},
        {"sparse_32",     gen_sparse},
        {"run_length",    gen_run_length},
        {"smooth_sine",   gen_smooth_sine},
        {"noisy_random",  gen_noisy},
    };
    int n_patterns = sizeof(patterns) / sizeof(patterns[0]);

    float* data = (float*)malloc(DATA_SIZE);
    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* output = malloc(max_out);
    size_t n_floats = DATA_SIZE / sizeof(float);

    // Test both with and without shuffle
    for (int use_shuffle = 0; use_shuffle <= 1; use_shuffle++) {
        printf("\n=== %s ===\n", use_shuffle ? "WITH BYTE SHUFFLE" : "NO SHUFFLE (raw)");
        printf("%-16s", "Pattern");
        for (int a = 1; a <= 8; a++) printf("  %8s", algo_names[a]);
        printf("  BEST       SNP_RANK\n");

        for (int p = 0; p < n_patterns; p++) {
            patterns[p].generate(data, n_floats);

            double ratios[9] = {0};
            double times[9] = {0};
            for (int a = 1; a <= 8; a++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)a;
                if (use_shuffle) cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
                gpucompress_stats_t stats;
                size_t out_sz = max_out;
                err = gpucompress_compress(data, DATA_SIZE, output, &out_sz, &cfg, &stats);
                ratios[a] = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0.0;
                times[a] = (err == GPUCOMPRESS_SUCCESS) ? stats.actual_comp_time_ms : 999.0;
            }

            int best = 1;
            for (int a = 2; a <= 8; a++) if (ratios[a] > ratios[best]) best = a;
            int snp_rank = 1;
            for (int a = 1; a <= 8; a++) if (a!=2 && ratios[a] > ratios[2]) snp_rank++;

            printf("%-16s", patterns[p].name);
            for (int a = 1; a <= 8; a++) printf("  %8.2f", ratios[a]);
            printf("  %-8s  %d/8\n", algo_names[best], snp_rank);
        }

        // Now show compression times
        printf("\nCompression times (ms):\n");
        printf("%-16s", "Pattern");
        for (int a = 1; a <= 8; a++) printf("  %8s", algo_names[a]);
        printf("  FASTEST\n");

        for (int p = 0; p < n_patterns; p++) {
            patterns[p].generate(data, n_floats);

            double times[9] = {0};
            for (int a = 1; a <= 8; a++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)a;
                if (use_shuffle) cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
                gpucompress_stats_t stats;
                size_t out_sz = max_out;
                // Warm up
                gpucompress_compress(data, DATA_SIZE, output, &out_sz, &cfg, &stats);
                // Measure
                out_sz = max_out;
                err = gpucompress_compress(data, DATA_SIZE, output, &out_sz, &cfg, &stats);
                times[a] = (err == GPUCOMPRESS_SUCCESS) ? stats.actual_comp_time_ms : 999.0;
            }

            int fastest = 1;
            for (int a = 2; a <= 8; a++) if (times[a] < times[fastest]) fastest = a;

            printf("%-16s", patterns[p].name);
            for (int a = 1; a <= 8; a++) printf("  %8.3f", times[a]);
            printf("  %s\n", algo_names[fastest]);
        }
    }

    free(data);
    free(output);
    gpucompress_cleanup();
    return 0;
}
