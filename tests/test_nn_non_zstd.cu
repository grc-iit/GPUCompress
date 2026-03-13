/**
 * Test NN bias: does it correctly pick non-zstd algorithms
 * when they actually win?
 *
 * 10 patterns where non-zstd algorithms win, tested both
 * with and without byte shuffle.
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
    const char* expected_winner;
    int use_shuffle;
    void (*gen)(float*, size_t);
};

// === LZ4 winners (with shuffle) ===

// Repeated block of 64 floats: lz4 201x vs zstd 37x
static void gen_repeat_blk64(float* d, size_t n) {
    float blk[64]; for(int i=0;i<64;i++) blk[i]=(float)(i*i);
    for(size_t i=0;i<n;i++) d[i]=blk[i%64];
}

// Quantized sine (16 levels): lz4 86x vs zstd 73x
static void gen_quant_sine16(float* d, size_t n) {
    for(size_t i=0;i<n;i++) { float v=sinf((float)i*0.01f); d[i]=roundf(v*8.0f)/8.0f; }
}

// === Cascaded winner (no shuffle) ===

// Linear ramp: cascaded 69x vs zstd 1.5x (delta encoding)
static void gen_linear_ramp(float* d, size_t n) {
    for(size_t i=0;i<n;i++) d[i]=(float)i;
}

// === Bitcomp winners (no shuffle) ===

// Random uniform [0,1]: bitcomp 1.22x vs zstd 1.07x
static void gen_random_uniform(float* d, size_t n) {
    srand(123);
    for(size_t i=0;i<n;i++) d[i]=(float)rand()/(float)RAND_MAX;
}

// Same exponent range [1,2): bitcomp 1.4x vs zstd 1.2x
static void gen_same_exponent(float* d, size_t n) {
    srand(90);
    for(size_t i=0;i<n;i++) d[i]=1.0f+(float)rand()/(float)RAND_MAX;
}

// Integers as float [0,255]: bitcomp 3.0x vs zstd 2.5x
static void gen_int_as_float(float* d, size_t n) {
    srand(100);
    for(size_t i=0;i<n;i++) d[i]=(float)(rand()%256);
}

// Structured noise (rand%1024)/1024: bitcomp 2.6x vs zstd 2.0x
static void gen_struct_noise(float* d, size_t n) {
    srand(110);
    for(size_t i=0;i<n;i++) d[i]=(float)(rand()%1024)/1024.0f;
}

// Chirp signal: bitcomp 1.5x vs zstd 1.1x
static void gen_chirp(float* d, size_t n) {
    for(size_t i=0;i<n;i++) {
        double t=(double)i/(double)n;
        d[i]=(float)sin(2.0*3.14159*(10.0*t+50.0*t*t));
    }
}

// Exponential distribution: bitcomp 1.2x vs zstd 1.1x
static void gen_exponential(float* d, size_t n) {
    srand(70);
    for(size_t i=0;i<n;i++) {
        double u=((double)rand()+1.0)/((double)RAND_MAX+1.0);
        d[i]=(float)(-log(u)*0.1);
    }
}

// Random walk: bitcomp 1.5x vs zstd 1.2x
static void gen_random_walk(float* d, size_t n) {
    srand(130);
    d[0]=0.0f;
    for(size_t i=1;i<n;i++) d[i]=d[i-1]+0.001f*((float)rand()/(float)RAND_MAX-0.5f);
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

    Pattern patterns[] = {
        // LZ4 winners (with shuffle)
        {"repeat_blk64",   "lz4",      1, gen_repeat_blk64},
        {"quant_sine16",   "lz4",      1, gen_quant_sine16},
        // Cascaded winner (no shuffle)
        {"linear_ramp",    "cascaded",  0, gen_linear_ramp},
        // Bitcomp winners (no shuffle)
        {"random_uniform", "bitcomp",   0, gen_random_uniform},
        {"same_exponent",  "bitcomp",   0, gen_same_exponent},
        {"int_as_float",   "bitcomp",   0, gen_int_as_float},
        {"struct_noise",   "bitcomp",   0, gen_struct_noise},
        {"chirp",          "bitcomp",   0, gen_chirp},
        {"exponential",    "bitcomp",   0, gen_exponential},
        {"random_walk",    "bitcomp",   0, gen_random_walk},
    };
    int np = sizeof(patterns) / sizeof(patterns[0]);

    size_t dsz = 1 << 20;  // 1MB
    size_t n = dsz / sizeof(float);
    float* data = (float*)malloc(dsz);
    size_t max_out = gpucompress_max_compressed_size(dsz);
    void* output = malloc(max_out);

    printf("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  NN BIAS TEST: 10 patterns where non-zstd algorithms win (1MB)                                                             ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ %-15s │ SHUF │ %7s %7s %7s %7s %7s %7s %7s %7s │ %-9s │ %-9s │ %-5s ║\n",
           "Pattern", "lz4", "snappy", "deflat", "gdefl", "zstd", "ans", "cascad", "bitcomp",
           "BEST", "NN_PICK", "MATCH");
    printf("╠═════════════════╪══════╪═════════════════════════════════════════════════════════════════╪═══════════╪═══════════╪═══════╣\n");

    int correct = 0, close = 0, zstd_picks = 0, total = 0;

    for (int p = 0; p < np; p++) {
        patterns[p].gen(data, n);

        // Run all 8 algorithms
        double ratios[8] = {0};
        for (int a = 0; a < 8; a++) {
            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = (gpucompress_algorithm_t)(a + 1);
            if (patterns[p].use_shuffle)
                cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
            gpucompress_stats_t stats;
            size_t sz = max_out;
            err = gpucompress_compress(data, dsz, output, &sz, &cfg, &stats);
            ratios[a] = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0.0;
        }

        int best = 0;
        for (int a = 1; a < 8; a++) if (ratios[a] > ratios[best]) best = a;

        // Ask NN
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
        if (patterns[p].use_shuffle)
            cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
        gpucompress_stats_t stats;
        size_t sz = max_out;
        err = gpucompress_compress(data, dsz, output, &sz, &cfg, &stats);

        int nn_algo_idx = -1;
        const char* nn_algo = "???";
        if (err == GPUCOMPRESS_SUCCESS) {
            nn_algo_idx = stats.nn_final_action % 8;
            nn_algo = algo_names[nn_algo_idx];
        }

        int is_match = (nn_algo_idx == best);
        int is_close = 0;
        if (nn_algo_idx >= 0 && ratios[best] > 0)
            is_close = (ratios[nn_algo_idx] >= ratios[best] * 0.95);

        total++;
        if (is_match) correct++;
        else if (is_close) close++;
        if (nn_algo_idx == 4) zstd_picks++;  // zstd = index 4

        // Compute ratio loss from NN pick vs best
        double nn_ratio = (nn_algo_idx >= 0) ? ratios[nn_algo_idx] : 0;
        double loss_pct = (ratios[best] > 0) ? (1.0 - nn_ratio/ratios[best]) * 100.0 : 0;

        printf("║ %-15s │  %s   │", patterns[p].name,
               patterns[p].use_shuffle ? "Y" : "N");
        for (int a = 0; a < 8; a++) {
            if (a == best)
                printf(" %6.1f*", ratios[a]);
            else
                printf(" %7.1f", ratios[a]);
        }
        printf(" │ %-9s │ %-9s │ %-5s ║\n",
               algo_names[best], nn_algo,
               is_match ? "YES" : (is_close ? "~95%" : "NO"));
    }

    printf("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                                                                             ║\n");
    printf("║  RESULTS: Exact match: %d/%d (%.0f%%)  |  Within 5%%: %d/%d  |  Total good: %d/%d (%.0f%%)                                          ║\n",
           correct, total, 100.0*correct/total,
           close, total,
           correct+close, total, 100.0*(correct+close)/total);
    printf("║  NN picked zstd: %d/%d times (%.0f%%) — on data where zstd is NOT the best                                                    ║\n",
           zstd_picks, total, 100.0*zstd_picks/total);
    printf("║                                                                                                                             ║\n");

    // Show what NN should have picked vs what it did
    printf("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  DETAIL: Ratio comparison when NN picks wrong                                                                               ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");

    for (int p = 0; p < np; p++) {
        patterns[p].gen(data, n);

        double ratios[8] = {0};
        for (int a = 0; a < 8; a++) {
            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = (gpucompress_algorithm_t)(a + 1);
            if (patterns[p].use_shuffle)
                cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
            gpucompress_stats_t stats;
            size_t sz = max_out;
            err = gpucompress_compress(data, dsz, output, &sz, &cfg, &stats);
            ratios[a] = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0.0;
        }

        int best = 0;
        for (int a = 1; a < 8; a++) if (ratios[a] > ratios[best]) best = a;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
        if (patterns[p].use_shuffle)
            cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
        gpucompress_stats_t stats;
        size_t sz = max_out;
        err = gpucompress_compress(data, dsz, output, &sz, &cfg, &stats);

        int nn_algo_idx = (err == GPUCOMPRESS_SUCCESS) ? stats.nn_final_action % 8 : -1;
        if (nn_algo_idx != best && nn_algo_idx >= 0) {
            double loss = (1.0 - ratios[nn_algo_idx] / ratios[best]) * 100.0;
            printf("║  %-15s  Best=%-9s (%.1fx)  NN=%-9s (%.1fx)  Loss: %.1f%%  Pred: %.1fx                              ║\n",
                   patterns[p].name, algo_names[best], ratios[best],
                   algo_names[nn_algo_idx], ratios[nn_algo_idx], loss,
                   stats.predicted_ratio);
        }
    }

    printf("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");

    free(data);
    free(output);
    gpucompress_cleanup();
    return 0;
}
