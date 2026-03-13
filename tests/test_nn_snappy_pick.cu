/**
 * Test if the NN ever picks Snappy for any data pattern.
 * Uses ALGO_AUTO to see what the NN actually chooses.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "gpucompress.h"

static const size_t DATA_SIZE = 1 << 20;  // 1 MB

static const char* algo_names[] = {
    "auto", "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
};

static const char* find_weights() {
    static const char* paths[] = {
        "neural_net/weights/model.nnwt",
        "../neural_net/weights/model.nnwt",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        FILE* f = fopen(paths[i], "rb");
        if (f) { fclose(f); return paths[i]; }
    }
    static char buf[512];
    snprintf(buf, sizeof(buf), "%s/GPUCompress/neural_net/weights/model.nnwt",
             getenv("HOME") ? getenv("HOME") : ".");
    return buf;
}

struct TestCase {
    const char* name;
    void (*gen)(float*, size_t);
};

static void gen_plateau(float* d, size_t n)   { for (size_t i=0;i<n;i++) d[i]=(float)(i/256); }
static void gen_vocab5(float* d, size_t n)    { float v[]={1,2,3,5,8}; for(size_t i=0;i<n;i++) d[i]=v[i%5]; }
static void gen_const(float* d, size_t n)     { for (size_t i=0;i<n;i++) d[i]=42.0f; }
static void gen_sparse(float* d, size_t n)    { for (size_t i=0;i<n;i++) d[i]=(i%32==0)?1.0f:0.0f; }
static void gen_sine(float* d, size_t n)      { for (size_t i=0;i<n;i++) d[i]=sinf((float)i*0.001f)*1000.0f; }
static void gen_noisy(float* d, size_t n)     { srand(123); for(size_t i=0;i<n;i++) d[i]=(float)rand()/(float)RAND_MAX; }
static void gen_ramp(float* d, size_t n)      { for (size_t i=0;i<n;i++) d[i]=(float)i; }
static void gen_periodic(float* d, size_t n)  { for (size_t i=0;i<n;i++) d[i]=sinf((float)(i%64)*0.1f); }
static void gen_step(float* d, size_t n)      { for (size_t i=0;i<n;i++) d[i]=(i<n/2)?0.0f:1000.0f; }
static void gen_mixed(float* d, size_t n) {
    for (size_t i=0;i<n;i++) {
        if (i < n/3) d[i] = 0.0f;
        else if (i < 2*n/3) d[i] = sinf((float)i * 0.1f);
        else d[i] = (float)(rand() % 100);
    }
}

int main() {
    const char* weights = find_weights();
    printf("Using weights: %s\n", weights);

    gpucompress_error_t err = gpucompress_init(weights);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "init failed: %s\n", gpucompress_error_string(err));
        return 1;
    }

    TestCase tests[] = {
        {"plateau",    gen_plateau},
        {"small_vocab",gen_vocab5},
        {"constant",   gen_const},
        {"sparse",     gen_sparse},
        {"smooth_sine",gen_sine},
        {"noisy",      gen_noisy},
        {"linear_ramp",gen_ramp},
        {"periodic_64",gen_periodic},
        {"step_func",  gen_step},
        {"mixed",      gen_mixed},
    };

    float* data = (float*)malloc(DATA_SIZE);
    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* output = malloc(max_out);
    size_t n = DATA_SIZE / sizeof(float);

    printf("\n%-14s  NN_CHOICE          PRED_RATIO  ACTUAL_RATIO  ACTION  (algo+quant+shuf)\n", "Pattern");
    printf("%-14s  ---------          ----------  ------------  ------  -----------------\n", "-------");

    for (int t = 0; t < (int)(sizeof(tests)/sizeof(tests[0])); t++) {
        tests[t].gen(data, n);

        // Test with different error bounds (lossless + lossy)
        double ebs[] = {0.0, 1e-5, 1e-3};
        for (int e = 0; e < 3; e++) {
            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
            cfg.error_bound = ebs[e];
            if (ebs[e] > 0) cfg.preprocessing = GPUCOMPRESS_PREPROC_QUANTIZE | GPUCOMPRESS_PREPROC_SHUFFLE_4;
            else cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;

            gpucompress_stats_t stats;
            size_t out_sz = max_out;
            err = gpucompress_compress(data, DATA_SIZE, output, &out_sz, &cfg, &stats);
            if (err != GPUCOMPRESS_SUCCESS) continue;

            int action = stats.nn_final_action;
            int algo_idx = action % 8;
            int quant = (action / 8) % 2;
            int shuffle = (action / 16) % 2;

            char label[64];
            snprintf(label, sizeof(label), "%s(eb=%.0e)", tests[t].name, ebs[e]);

            printf("%-28s  %-18s  %8.2f    %8.2f      %3d   (%s%s%s)\n",
                   label,
                   algo_names[algo_idx + 1],
                   stats.predicted_ratio,
                   stats.compression_ratio,
                   action,
                   algo_names[algo_idx + 1],
                   quant ? "+quant" : "",
                   shuffle ? "+shuf" : "");
        }
    }

    free(data);
    free(output);
    gpucompress_cleanup();
    return 0;
}
