/**
 * Find data patterns where algorithms other than zstd win.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "gpucompress.h"

static const char* algo_names[] = {
    "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
};

static const size_t DATA_SIZE = 1 << 20;  // 1MB

struct Pattern {
    const char* name;
    void (*gen)(float*, size_t);
};

// === LZ4-favoring patterns ===
// LZ4 excels at finding exact byte matches at short offsets.
// After byte shuffle, we need patterns that create long literal runs in each byte lane.

// Repeated block of 64 floats - after shuffle, creates periodic byte patterns
static void gen_repeat_blk64(float* d, size_t n) {
    float blk[64]; for(int i=0;i<64;i++) blk[i]=(float)(i*i);
    for(size_t i=0;i<n;i++) d[i]=blk[i%64];
}

// Repeated block of 128 floats
static void gen_repeat_blk128(float* d, size_t n) {
    srand(10);
    float blk[128]; for(int i=0;i<128;i++) blk[i]=(float)rand()/(float)RAND_MAX*100.0f;
    for(size_t i=0;i<n;i++) d[i]=blk[i%128];
}

// Repeated block of 256 floats
static void gen_repeat_blk256(float* d, size_t n) {
    srand(20);
    float blk[256]; for(int i=0;i<256;i++) blk[i]=(float)rand()/(float)RAND_MAX*1000.0f;
    for(size_t i=0;i<n;i++) d[i]=blk[i%256];
}

// Repeated block of 512 floats
static void gen_repeat_blk512(float* d, size_t n) {
    srand(30);
    float blk[512]; for(int i=0;i<512;i++) blk[i]=sinf((float)i*0.05f)*500.0f;
    for(size_t i=0;i<n;i++) d[i]=blk[i%512];
}

// Repeated block of 1024 floats
static void gen_repeat_blk1k(float* d, size_t n) {
    srand(40);
    float blk[1024]; for(int i=0;i<1024;i++) blk[i]=(float)rand()/(float)RAND_MAX*50.0f;
    for(size_t i=0;i<n;i++) d[i]=blk[i%1024];
}

// Repeated block of 4096 floats
static void gen_repeat_blk4k(float* d, size_t n) {
    srand(50);
    float blk[4096]; for(int i=0;i<4096;i++) blk[i]=(float)rand()/(float)RAND_MAX*200.0f;
    for(size_t i=0;i<n;i++) d[i]=blk[i%4096];
}

// Quantized sine - few unique values, periodic
static void gen_quant_sine16(float* d, size_t n) {
    for(size_t i=0;i<n;i++) { float v=sinf((float)i*0.01f); d[i]=roundf(v*8.0f)/8.0f; }
}

// Quantized sine with more levels
static void gen_quant_sine64(float* d, size_t n) {
    for(size_t i=0;i<n;i++) { float v=sinf((float)i*0.01f); d[i]=roundf(v*32.0f)/32.0f; }
}

// Quantized sine with 256 levels
static void gen_quant_sine256(float* d, size_t n) {
    for(size_t i=0;i<n;i++) { float v=sinf((float)i*0.01f); d[i]=roundf(v*128.0f)/128.0f; }
}

// === Cascaded-favoring patterns ===
// Cascaded = RLE + delta + bitpacking. Needs runs or small deltas.

// Linear ramp (perfect delta encoding)
static void gen_linear_ramp(float* d, size_t n) {
    for(size_t i=0;i<n;i++) d[i]=(float)i;
}

// Step function with long runs
static void gen_long_runs(float* d, size_t n) {
    for(size_t i=0;i<n;i++) d[i]=(float)(i/1024);
}

// Small integer values 0-3 (2-bit values, great for bitpacking)
static void gen_2bit_vals(float* d, size_t n) {
    for(size_t i=0;i<n;i++) d[i]=(float)(i%4);
}

// Integer ramp mod 16 (4-bit values)
static void gen_4bit_vals(float* d, size_t n) {
    for(size_t i=0;i<n;i++) d[i]=(float)(i%16);
}

// Constant with rare spikes (great for RLE)
static void gen_rle_friendly(float* d, size_t n) {
    for(size_t i=0;i<n;i++) d[i]=(i%10000==0)?1.0f:0.0f;
}

// === ANS-favoring patterns ===
// ANS (entropy coding) should win when there's statistical redundancy
// but no positional patterns for LZ to exploit

// Skewed distribution (mostly one value, rarely others)
static void gen_skewed(float* d, size_t n) {
    srand(60);
    float vals[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    // 90% zeros, 5% ones, 3% twos, 1.5% threes, 0.5% fours
    for(size_t i=0;i<n;i++) {
        int r = rand() % 200;
        if (r < 180) d[i] = vals[0];
        else if (r < 190) d[i] = vals[1];
        else if (r < 196) d[i] = vals[2];
        else if (r < 199) d[i] = vals[3];
        else d[i] = vals[4];
    }
}

// Exponential distribution (many small, few large)
static void gen_exponential(float* d, size_t n) {
    srand(70);
    for(size_t i=0;i<n;i++) {
        double u = ((double)rand()+1.0)/((double)RAND_MAX+1.0);
        d[i] = (float)(-log(u) * 0.1);
    }
}

// === Bitcomp-favoring patterns ===
// Bitcomp: bitwise compression. Common bit prefixes, leading zeros.

// Random data (barely compressible, bitcomp sometimes wins here)
static void gen_random(float* d, size_t n) {
    srand(123);
    for(size_t i=0;i<n;i++) d[i]=(float)rand()/(float)RAND_MAX;
}

// Random with limited exponent range
static void gen_random_small_exp(float* d, size_t n) {
    srand(80);
    for(size_t i=0;i<n;i++) d[i]=(float)rand()/(float)RAND_MAX * 1e-5f;
}

// IEEE floats with common high bytes (same sign+exponent, varying mantissa)
static void gen_same_exponent(float* d, size_t n) {
    srand(90);
    for(size_t i=0;i<n;i++) {
        // All values between 1.0 and 2.0 (same exponent 127)
        d[i] = 1.0f + (float)rand()/(float)RAND_MAX;
    }
}

// Integers stored as float (clean bit patterns)
static void gen_int_as_float(float* d, size_t n) {
    srand(100);
    for(size_t i=0;i<n;i++) d[i]=(float)(rand()%256);
}

// === Deflate/GDeflate-favoring patterns ===
// Better Huffman + LZ77. Should win on moderate-complexity data where
// statistical coding helps but data is too complex for cascaded/bitcomp

// Mixed periodic with varying amplitude
static void gen_varying_periodic(float* d, size_t n) {
    for(size_t i=0;i<n;i++) {
        float amp = 1.0f + 0.5f * sinf((float)i * 0.0001f);
        d[i] = amp * sinf((float)i * 0.1f);
    }
}

// Structured noise: random but each value quantized to 1/1024
static void gen_structured_noise(float* d, size_t n) {
    srand(110);
    for(size_t i=0;i<n;i++) {
        d[i] = (float)(rand() % 1024) / 1024.0f;
    }
}

// Chirp signal (increasing frequency)
static void gen_chirp(float* d, size_t n) {
    for(size_t i=0;i<n;i++) {
        double t = (double)i / (double)n;
        d[i] = (float)sin(2.0 * 3.14159 * (10.0 * t + 50.0 * t * t));
    }
}

// Sawtooth wave quantized
static void gen_sawtooth_quant(float* d, size_t n) {
    for(size_t i=0;i<n;i++) {
        float v = (float)(i % 100) / 100.0f;
        d[i] = roundf(v * 16.0f) / 16.0f;
    }
}

// Two interleaved sequences
static void gen_interleaved_seq(float* d, size_t n) {
    for(size_t i=0;i<n;i++) {
        if (i % 2 == 0) d[i] = (float)(i/2 % 64);
        else d[i] = sinf((float)(i/2) * 0.01f) * 100.0f;
    }
}

// Short repeated pattern (period 3)
static void gen_period3(float* d, size_t n) {
    float vals[] = {1.0f, 2.0f, 3.0f};
    for(size_t i=0;i<n;i++) d[i] = vals[i%3];
}

// Short repeated pattern (period 7)
static void gen_period7(float* d, size_t n) {
    float vals[] = {1.0f, 0.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f};
    for(size_t i=0;i<n;i++) d[i] = vals[i%7];
}

// Huffman-friendly: few symbols with specific probabilities
static void gen_huffman_friendly(float* d, size_t n) {
    srand(120);
    // 8 symbols with power-of-2-ish frequencies
    float syms[] = {0,1,2,3,4,5,6,7};
    int weights[] = {128, 64, 32, 16, 8, 4, 2, 2}; // sum=256
    int cum[9]; cum[0]=0;
    for(int i=0;i<8;i++) cum[i+1]=cum[i]+weights[i];
    for(size_t i=0;i<n;i++) {
        int r=rand()%256;
        for(int s=0;s<8;s++) { if(r<cum[s+1]) { d[i]=syms[s]; break; } }
    }
}

// === Mixed / edge cases ===

// Random walk (cumulative sum of small steps)
static void gen_random_walk(float* d, size_t n) {
    srand(130);
    d[0] = 0.0f;
    for(size_t i=1;i<n;i++) d[i] = d[i-1] + 0.001f*((float)rand()/(float)RAND_MAX - 0.5f);
}

// Piecewise constant with random levels
static void gen_piecewise_const(float* d, size_t n) {
    srand(140);
    float level = (float)(rand()%100);
    int run_left = 500 + rand()%2000;
    for(size_t i=0;i<n;i++) {
        d[i] = level;
        if (--run_left <= 0) {
            level = (float)(rand()%100);
            run_left = 500 + rand()%2000;
        }
    }
}

// Only two float values (but random ordering)
static void gen_two_val_random(float* d, size_t n) {
    srand(150);
    for(size_t i=0;i<n;i++) d[i] = (rand()%2) ? 42.0f : 99.0f;
}

// Integer ramp (linear, perfect for delta+bitpack)
static void gen_int_ramp(float* d, size_t n) {
    for(size_t i=0;i<n;i++) d[i]=(float)(i%128);
}

// Fibonacci-like sequence mod 256
static void gen_fib_mod(float* d, size_t n) {
    d[0]=1; d[1]=1;
    for(size_t i=2;i<n;i++) d[i]=(float)(((int)d[i-1]+(int)d[i-2])%256);
}

// All same byte pattern (0x42424242 = 48.25f repeating)
static void gen_same_bytes(float* d, size_t n) {
    unsigned int pattern = 0x42424242;
    float val; memcpy(&val, &pattern, 4);
    for(size_t i=0;i<n;i++) d[i]=val;
}

int main() {
    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) { fprintf(stderr, "init failed\n"); return 1; }

    Pattern patterns[] = {
        {"repeat_blk64", gen_repeat_blk64},
        {"repeat_blk128", gen_repeat_blk128},
        {"repeat_blk256", gen_repeat_blk256},
        {"repeat_blk512", gen_repeat_blk512},
        {"repeat_blk1k", gen_repeat_blk1k},
        {"repeat_blk4k", gen_repeat_blk4k},
        {"quant_sine16", gen_quant_sine16},
        {"quant_sine64", gen_quant_sine64},
        {"quant_sine256", gen_quant_sine256},
        {"linear_ramp", gen_linear_ramp},
        {"long_runs_1k", gen_long_runs},
        {"2bit_vals", gen_2bit_vals},
        {"4bit_vals", gen_4bit_vals},
        {"rle_friendly", gen_rle_friendly},
        {"skewed_90pct", gen_skewed},
        {"exponential", gen_exponential},
        {"random_uniform", gen_random},
        {"random_small_e", gen_random_small_exp},
        {"same_exponent", gen_same_exponent},
        {"int_as_float", gen_int_as_float},
        {"varying_period", gen_varying_periodic},
        {"struct_noise", gen_structured_noise},
        {"chirp", gen_chirp},
        {"sawtooth_q16", gen_sawtooth_quant},
        {"interleaved", gen_interleaved_seq},
        {"period3", gen_period3},
        {"period7", gen_period7},
        {"huffman_frdly", gen_huffman_friendly},
        {"random_walk", gen_random_walk},
        {"piecewise_c", gen_piecewise_const},
        {"two_val_rand", gen_two_val_random},
        {"int_ramp128", gen_int_ramp},
        {"fib_mod256", gen_fib_mod},
        {"same_bytes", gen_same_bytes},
    };
    int np = sizeof(patterns)/sizeof(patterns[0]);

    float* data = (float*)malloc(DATA_SIZE);
    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* output = malloc(max_out);
    size_t n = DATA_SIZE / sizeof(float);

    // Test with shuffle (typical mode)
    for (int use_shuffle = 0; use_shuffle <= 1; use_shuffle++) {
        printf("\n=== %s (1MB) ===\n", use_shuffle ? "WITH SHUFFLE" : "NO SHUFFLE");
        printf("%-16s", "Pattern");
        for (int a = 0; a < 8; a++) printf(" %9s", algo_names[a]);
        printf("   BEST       WINNER_NOT_ZSTD\n");

        for (int p = 0; p < np; p++) {
            patterns[p].gen(data, n);

            double ratios[8] = {0};
            for (int a = 0; a < 8; a++) {
                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = (gpucompress_algorithm_t)(a + 1);
                if (use_shuffle) cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
                gpucompress_stats_t stats;
                size_t sz = max_out;
                err = gpucompress_compress(data, DATA_SIZE, output, &sz, &cfg, &stats);
                ratios[a] = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0.0;
            }

            int best = 0;
            for (int a = 1; a < 8; a++) if (ratios[a] > ratios[best]) best = a;

            // Only print if best is NOT zstd (index 4)
            if (best != 4) {
                printf("%-16s", patterns[p].name);
                for (int a = 0; a < 8; a++) {
                    if (a == best) printf(" %8.1f*", ratios[a]);
                    else printf(" %9.1f", ratios[a]);
                }
                printf("   %-10s  <<<\n", algo_names[best]);
            }
        }
    }

    free(data);
    free(output);
    gpucompress_cleanup();
    return 0;
}
