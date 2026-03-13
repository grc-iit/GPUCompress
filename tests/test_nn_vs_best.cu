/**
 * For each data pattern: run all 8 algorithms, then ask NN what it picks.
 * Show side-by-side: actual best vs NN choice.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

static const char* algo_names[] = {
    "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
};

struct Pattern {
    const char* name;
    void (*gen)(float*, size_t);
};

static void gen_zeros(float* d, size_t n)        { memset(d, 0, n * sizeof(float)); }
static void gen_constant(float* d, size_t n)      { for (size_t i=0;i<n;i++) d[i]=42.0f; }
static void gen_sparse32(float* d, size_t n)      { for (size_t i=0;i<n;i++) d[i]=(i%32==0)?1.0f:0.0f; }
static void gen_sparse100(float* d, size_t n)     { for (size_t i=0;i<n;i++) d[i]=(i%100==0)?1.0f:0.0f; }
static void gen_tiny_ints(float* d, size_t n)     { for (size_t i=0;i<n;i++) d[i]=(float)(i%4); }
static void gen_binary(float* d, size_t n)        { srand(42); for(size_t i=0;i<n;i++) d[i]=(rand()%2)?1.0f:0.0f; }
static void gen_powers2(float* d, size_t n)       { for (size_t i=0;i<n;i++) d[i]=powf(2.0f,(float)(i%8)); }
static void gen_sine(float* d, size_t n)          { for (size_t i=0;i<n;i++) d[i]=sinf((float)i*0.001f)*1000.0f; }
static void gen_noisy(float* d, size_t n)         { srand(123); for(size_t i=0;i<n;i++) d[i]=(float)rand()/(float)RAND_MAX; }
static void gen_near_zero(float* d, size_t n)     { for (size_t i=0;i<n;i++) d[i]=(float)i*1e-30f; }
static void gen_int8(float* d, size_t n)          { for (size_t i=0;i<n;i++) d[i]=(float)(i%8); }
static void gen_mostly_same(float* d, size_t n)   { for (size_t i=0;i<n;i++) d[i]=(i%20==0)?99.0f:1.0f; }
static void gen_delta(float* d, size_t n)         { d[0]=100.0f; for(size_t i=1;i<n;i++) d[i]=d[i-1]+0.001f; }
static void gen_ramp(float* d, size_t n)          { for (size_t i=0;i<n;i++) d[i]=(float)i; }
static void gen_plateau(float* d, size_t n)       { for (size_t i=0;i<n;i++) d[i]=(float)(i/256); }
static void gen_vocab5(float* d, size_t n)        { float v[]={1,2,3,5,8}; for(size_t i=0;i<n;i++) d[i]=v[i%5]; }
static void gen_periodic64(float* d, size_t n)    { for (size_t i=0;i<n;i++) d[i]=sinf((float)(i%64)*0.1f); }
static void gen_step(float* d, size_t n)          { for (size_t i=0;i<n;i++) d[i]=(i<n/2)?0.0f:1000.0f; }
static void gen_run_length(float* d, size_t n) {
    srand(42); float val=1; int run=0;
    for (size_t i=0;i<n;i++) { if(run<=0){val=(float)(rand()%10);run=50+rand()%200;} d[i]=val; run--; }
}
static void gen_quant_sine(float* d, size_t n) {
    for (size_t i=0;i<n;i++) { float v=sinf((float)i*0.01f); d[i]=roundf(v*8.0f)/8.0f; }
}
static void gen_repeated_blk(float* d, size_t n) {
    float blk[64]; for(int i=0;i<64;i++) blk[i]=(float)(i*i);
    for(size_t i=0;i<n;i++) d[i]=blk[i%64];
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
    if (!w) { fprintf(stderr, "No weights file found\n"); return 1; }

    gpucompress_error_t err = gpucompress_init(w);
    if (err != GPUCOMPRESS_SUCCESS) { fprintf(stderr, "init failed\n"); return 1; }

    // Enable online learning + exploration (5 alternative configs)
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.3f, 0.20f, 0.50f);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_k(5);

    Pattern patterns[] = {
        {"zeros",          gen_zeros},
        {"constant",       gen_constant},
        {"sparse_1/32",    gen_sparse32},
        {"sparse_1/100",   gen_sparse100},
        {"tiny_ints_0-3",  gen_tiny_ints},
        {"binary_0_1",     gen_binary},
        {"powers_of_2",    gen_powers2},
        {"smooth_sine",    gen_sine},
        {"noisy_random",   gen_noisy},
        {"near_zero",      gen_near_zero},
        {"int_0-7",        gen_int8},
        {"mostly_same",    gen_mostly_same},
        {"delta_small",    gen_delta},
        {"linear_ramp",    gen_ramp},
        {"plateau_256",    gen_plateau},
        {"small_vocab_5",  gen_vocab5},
        {"periodic_64",    gen_periodic64},
        {"step_func",      gen_step},
        {"run_length",     gen_run_length},
        {"quant_sine_16",  gen_quant_sine},
        {"repeated_blk64", gen_repeated_blk},
    };
    int np = sizeof(patterns) / sizeof(patterns[0]);

    size_t sizes[] = {4096, 65536, 1<<20};
    const char* sz_names[] = {"4KB", "64KB", "1MB"};

    for (int si = 0; si < 3; si++) {
        size_t dsz = sizes[si];
        size_t n = dsz / sizeof(float);
        float* data = (float*)malloc(dsz);
        size_t max_out = gpucompress_max_compressed_size(dsz);
        void* output = malloc(max_out);

        printf("\n");
        printf("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
        printf("║  DATA SIZE: %-6s                                                                                 ║\n", sz_names[si]);
        printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║ %-15s │ %7s %7s %7s %7s %7s %7s %7s %7s │ %-10s │ %-10s │ %-5s ║\n",
               "Pattern", "lz4", "snappy", "deflat", "gdefl", "zstd", "ans", "cascad", "bitcomp",
               "BEST", "NN_PICK", "MATCH");
        printf("╠═════════════════╪═════════════════════════════════════════════════════════════════╪════════════╪════════════╪═══════╣\n");

        int correct = 0, total = 0;

        for (int p = 0; p < np; p++) {
            patterns[p].gen(data, n);

            // 1) Run all 8 algorithms
            double ratios[8] = {0};
            for (int a = 0; a < 8; a++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)(a + 1);  // 1-indexed
                cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
                gpucompress_stats_t stats;
                size_t sz = max_out;
                err = gpucompress_compress(data, dsz, output, &sz, &cfg, &stats);
                ratios[a] = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0.0;
            }

            // Find best
            int best = 0;
            for (int a = 1; a < 8; a++) if (ratios[a] > ratios[best]) best = a;

            // 2) Ask NN (ALGO_AUTO)
            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
            cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
            gpucompress_stats_t stats;
            size_t sz = max_out;
            err = gpucompress_compress(data, dsz, output, &sz, &cfg, &stats);

            const char* nn_algo = "???";
            int nn_algo_idx = -1;
            if (err == GPUCOMPRESS_SUCCESS) {
                int action = stats.nn_final_action;
                nn_algo_idx = action % 8;
                nn_algo = algo_names[nn_algo_idx];
            }

            int match = (nn_algo_idx == best);
            if (err == GPUCOMPRESS_SUCCESS) { total++; if (match) correct++; }

            // Check if NN pick is within 5% of best (close enough)
            int close = 0;
            if (nn_algo_idx >= 0 && ratios[best] > 0) {
                close = (ratios[nn_algo_idx] >= ratios[best] * 0.95);
            }

            printf("║ %-15s │", patterns[p].name);
            for (int a = 0; a < 8; a++) {
                if (a == best)
                    printf(" %6.2f*", ratios[a]);
                else
                    printf(" %7.2f", ratios[a]);
            }
            printf(" │ %-10s │ %-10s │ %s ║\n",
                   algo_names[best], nn_algo,
                   match ? "YES" : (close ? "~95%" : "NO"));
        }

        printf("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
        printf("║ NN accuracy: %d/%d exact match (%.0f%%)                                                              ║\n",
               correct, total, total > 0 ? 100.0*correct/total : 0.0);
        printf("╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");

        free(data);
        free(output);
    }

    gpucompress_cleanup();
    return 0;
}
