/**
 * @file explore_algo_patterns.cu
 * @brief Explore GPU data patterns to find where non-zstd algorithms beat zstd.
 *
 * Tests 30+ synthetic data patterns (1 MB each = 262144 floats) against all
 * 8 nvcomp algorithms with and without byte shuffle (16 configs total).
 * Prints per-pattern winners and a summary of which non-zstd algorithms win.
 *
 * Usage: ./explore_algo_patterns [weights_path]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "gpucompress.h"

#define CHUNK_FLOATS  262144   /* 1 MB */
#define CHUNK_BYTES   (CHUNK_FLOATS * sizeof(float))
#define NUM_ALGOS     8
#define NUM_CONFIGS   16       /* 8 algos x 2 shuffle */

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while (0)

static const char* ALGO_NAMES[] = {
    "auto", "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

/* ── Result tracking ─────────────────────────────────────────── */

struct PatternResult {
    const char* name;
    int    best_algo;       /* 1-8 */
    int    best_shuf;
    double best_ratio;
    int    zstd_best_algo;  /* best zstd config (5) */
    double zstd_best_ratio; /* best of zstd / zstd+shuf */
    double margin;          /* best_ratio - zstd_best_ratio */
};

static PatternResult results[256];
static int n_results = 0;
static int print_enabled = 1; /* set to 0 to suppress per-pattern output */

/* algo win counters (non-zstd only, indexed by algo 1-8) */
static int algo_wins[9] = {0};
static int algo_wins_shuf[9] = {0};

/* ── Host data generation ────────────────────────────────────── */

static float* h_buf = NULL;

static void alloc_host() {
    if (!h_buf) h_buf = (float*)malloc(CHUNK_BYTES);
}

/* Simple xorshift PRNG */
static uint32_t xor_state = 12345;
static uint32_t xorshift() {
    xor_state ^= xor_state << 13;
    xor_state ^= xor_state >> 17;
    xor_state ^= xor_state << 5;
    return xor_state;
}
static float rand_float() {
    return (float)(xorshift() & 0xFFFFFF) / (float)0xFFFFFF;
}

/* Pattern generators - each fills h_buf with CHUNK_FLOATS values */

static void gen_all_zeros() {
    for (int i = 0; i < CHUNK_FLOATS; i++) h_buf[i] = 0.0f;
}

static void gen_constant_one() {
    for (int i = 0; i < CHUNK_FLOATS; i++) h_buf[i] = 1.0f;
}

static void gen_repeated_small_ints() {
    for (int i = 0; i < CHUNK_FLOATS; i++) h_buf[i] = (float)(i % 4);
}

static void gen_random_uniform() {
    xor_state = 42;
    for (int i = 0; i < CHUNK_FLOATS; i++) h_buf[i] = rand_float();
}

static void gen_random_int_0_255() {
    xor_state = 77;
    for (int i = 0; i < CHUNK_FLOATS; i++) h_buf[i] = (float)(xorshift() % 256);
}

static void gen_monotonic_inc() {
    for (int i = 0; i < CHUNK_FLOATS; i++) h_buf[i] = (float)i;
}

static void gen_monotonic_noise() {
    xor_state = 99;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = (float)i + (rand_float() - 0.5f) * 0.01f;
}

static void gen_sine_low_freq() {
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = sinf((float)i * 2.0f * 3.14159265f / 1024.0f);
}

static void gen_sine_high_freq() {
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = sinf((float)i * 2.0f * 3.14159265f / 8.0f);
}

static void gen_gaussian() {
    xor_state = 123;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        /* Box-Muller */
        float u1 = rand_float() * 0.999f + 0.001f;
        float u2 = rand_float();
        h_buf[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    }
}

static void gen_sparse_95() {
    xor_state = 200;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = (xorshift() % 100 < 5) ? rand_float() * 100.0f : 0.0f;
}

static void gen_sparse_99() {
    xor_state = 300;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = (xorshift() % 100 < 1) ? rand_float() * 100.0f : 0.0f;
}

static void gen_run_length() {
    xor_state = 400;
    float val = rand_float() * 10.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        if (i % 1024 == 0) val = rand_float() * 10.0f;
        h_buf[i] = val;
    }
}

static void gen_constant_delta() {
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = 1.5f + (float)i * 0.003f;
}

static void gen_bit_packed_8() {
    /* Only lower 8 bits used in 32-bit int, reinterpreted as float */
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        uint32_t v = (uint32_t)(i % 256);
        memcpy(&h_buf[i], &v, sizeof(float));
    }
}

static void gen_alternating_two() {
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = (i % 2 == 0) ? 3.14f : 2.71f;
}

static void gen_exponential_growth() {
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = expf((float)i / (float)CHUNK_FLOATS * 10.0f);
}

static void gen_log_normal() {
    xor_state = 500;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        float u1 = rand_float() * 0.999f + 0.001f;
        float u2 = rand_float();
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        h_buf[i] = expf(z * 0.5f + 1.0f);
    }
}

static void gen_poisson_like() {
    xor_state = 600;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        /* Mostly small values, rare large */
        float r = rand_float();
        if (r < 0.7f) h_buf[i] = 0.0f;
        else if (r < 0.9f) h_buf[i] = 1.0f;
        else if (r < 0.97f) h_buf[i] = 2.0f;
        else if (r < 0.99f) h_buf[i] = (float)(xorshift() % 10 + 3);
        else h_buf[i] = (float)(xorshift() % 1000);
    }
}

static void gen_repeated_block_64() {
    xor_state = 700;
    float block[64];
    for (int j = 0; j < 64; j++) block[j] = rand_float() * 100.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = block[i % 64];
}

static void gen_power_law() {
    /* Huffman-friendly: power-law frequency distribution */
    xor_state = 800;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        float r = rand_float();
        /* Zipf-like: value k with probability ~ 1/k */
        h_buf[i] = floorf(1.0f / (r * 0.999f + 0.001f));
        if (h_buf[i] > 1000.0f) h_buf[i] = 1000.0f;
    }
}

static void gen_binary() {
    xor_state = 900;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = (xorshift() % 2 == 0) ? 0.0f : 1.0f;
}

static void gen_quantized_001() {
    xor_state = 1000;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = floorf(rand_float() * 100.0f) * 0.01f;
}

static void gen_slowly_varying() {
    xor_state = 1100;
    float val = 50.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        val += (rand_float() - 0.5f) * 0.001f;
        h_buf[i] = val;
    }
}

static void gen_sawtooth() {
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = (float)(i % 256) / 256.0f;
}

static void gen_step_function() {
    xor_state = 1200;
    float val = rand_float() * 100.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        if (i % 4096 == 0) val = rand_float() * 100.0f;
        h_buf[i] = val;
    }
}

static void gen_interleaved_channels() {
    /* RGBRGB-like: 3 channels with different characteristics */
    xor_state = 1300;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        int ch = i % 3;
        if (ch == 0) h_buf[i] = (float)(i / 3) / (float)CHUNK_FLOATS * 255.0f;
        else if (ch == 1) h_buf[i] = 128.0f + sinf((float)i * 0.01f) * 50.0f;
        else h_buf[i] = rand_float() * 255.0f;
    }
}

static void gen_fibonacci_mod() {
    int a = 0, b = 1;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        h_buf[i] = (float)(a % 97);
        int c = (a + b) % 997;
        a = b; b = c;
    }
}

static void gen_geometric() {
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = powf(1.001f, (float)(i % 10000));
}

static void gen_repeat_every_4() {
    xor_state = 1400;
    float block[4];
    for (int j = 0; j < 4; j++) block[j] = rand_float() * 100.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = block[i % 4];
}

static void gen_repeat_every_16() {
    xor_state = 1500;
    float block[16];
    for (int j = 0; j < 16; j++) block[j] = rand_float() * 100.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = block[i % 16];
}

static void gen_repeat_every_256() {
    xor_state = 1600;
    float block[256];
    for (int j = 0; j < 256; j++) block[j] = rand_float() * 100.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = block[i % 256];
}

static void gen_xor_pattern() {
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        uint32_t v = (uint32_t)i ^ ((uint32_t)i >> 1);
        memcpy(&h_buf[i], &v, sizeof(float));
    }
}

/* ── Targeted patterns (round 2) ─────────────────────────────── */

/* --- Bitcomp targets: bit-plane correlation --- */

static void gen_bitcomp_toggle_lsb() {
    /* Consecutive integers toggling only lowest 1-2 bits */
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = 1000 + (i & 1);  /* 1000, 1001, 1000, 1001, ... */
}

static void gen_bitcomp_const_exponent() {
    /* IEEE 754 floats: constant exponent, slowly changing mantissa */
    uint32_t* u = (uint32_t*)h_buf;
    uint32_t base = 0x3F800000u; /* float 1.0 */
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = base + (uint32_t)(i & 0x7FF); /* exponent fixed, mantissa LSBs change */
}

static void gen_bitcomp_small_xor_delta() {
    /* XOR of consecutive elements has very few bits set (1-2 bits) */
    uint32_t* u = (uint32_t*)h_buf;
    u[0] = 0xDEADBEEFu;
    xor_state = 2222;
    for (int i = 1; i < CHUNK_FLOATS; i++) {
        int bit = xorshift() % 32;
        u[i] = u[i-1] ^ (1u << bit);
    }
}

static void gen_bitcomp_float_one_epsilon() {
    /* All values in [1.0, 1.0+epsilon] range (uint32 0x3F800000..0x3F800001) */
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = 0x3F800000u + (uint32_t)(i & 1);
}

static void gen_bitcomp_drift_int() {
    /* Slowly drifting integers: delta of +1 or -1 each step */
    uint32_t* u = (uint32_t*)h_buf;
    u[0] = 0x00100000u;
    xor_state = 3333;
    for (int i = 1; i < CHUNK_FLOATS; i++) {
        if (xorshift() & 1)
            u[i] = u[i-1] + 1;
        else
            u[i] = u[i-1] - 1;
    }
}

/* --- Cascaded targets: run-length + delta encoding --- */

static void gen_cascaded_long_runs() {
    /* Very long runs (10000+ identical values) with occasional changes */
    xor_state = 4444;
    float val = 42.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        if (i % 10000 == 0) val = (float)(xorshift() % 100);
        h_buf[i] = val;
    }
}

static void gen_cascaded_perfect_arith() {
    /* Perfect arithmetic sequence: constant delta, zero noise */
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = (uint32_t)(1000 + i * 3);
}

static void gen_cascaded_block_const() {
    /* Constant within blocks of 1024, different between blocks */
    xor_state = 5555;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        if (i % 1024 == 0) xor_state = 5555 + (uint32_t)(i / 1024);
        if (i % 1024 == 0) h_buf[i] = (float)(xorshift() % 256);
        else h_buf[i] = h_buf[i - (i % 1024)];
    }
}

static void gen_cascaded_staircase() {
    /* Value increases by 1 every 256 elements */
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = (uint32_t)(i / 256);
}

/* --- LZ4/Snappy targets: byte-level exact matching --- */

static void gen_lz4_repeat_4byte() {
    /* Exact repetition of a 4-byte pattern (1 float) */
    float pat = 3.14159f;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = pat;
}

static void gen_lz4_repeat_16byte() {
    /* Exact repetition of a 16-byte (4-float) pattern */
    float pat[4] = {1.0f, 2.5f, -3.7f, 0.001f};
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = pat[i % 4];
}

static void gen_lz4_repeat_64byte() {
    /* Exact repetition of a 64-byte (16-float) pattern */
    xor_state = 6666;
    float pat[16];
    for (int j = 0; j < 16; j++) pat[j] = rand_float() * 50.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = pat[i % 16];
}

static void gen_lz4_lookup_table() {
    /* Lookup table-like: lots of exact value reuse from 32 entries */
    xor_state = 7777;
    float table[32];
    for (int j = 0; j < 32; j++) table[j] = rand_float() * 1000.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = table[xorshift() % 32];
}

static void gen_lz4_tile_rows() {
    /* 2D tile: each row is identical, 512 cols x 512 rows */
    xor_state = 8888;
    float row[512];
    for (int j = 0; j < 512; j++) row[j] = rand_float() * 100.0f;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        h_buf[i] = row[i % 512];
}

/* --- ANS targets: entropy coding --- */

static void gen_ans_skewed_99() {
    /* 99% one byte value, 1% others - raw uint32 */
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 9999;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        if (xorshift() % 100 == 0)
            u[i] = xorshift();
        else
            u[i] = 0x42u;
    }
}

static void gen_ans_markov() {
    /* Markov chain data: next byte predicted by previous */
    uint8_t* bytes = (uint8_t*)h_buf;
    int total_bytes = CHUNK_BYTES;
    xor_state = 10101;
    bytes[0] = 0;
    for (int i = 1; i < total_bytes; i++) {
        uint32_t r = xorshift() % 100;
        if (r < 80)
            bytes[i] = bytes[i-1];          /* 80% stay same */
        else if (r < 95)
            bytes[i] = bytes[i-1] + 1;      /* 15% increment */
        else
            bytes[i] = (uint8_t)(xorshift() % 256); /* 5% random */
    }
}

/* --- Deflate/GDeflate targets: LZ77 sliding window --- */

static void gen_deflate_backward_refs() {
    /* LZ77-friendly: copy from varying distances back */
    uint8_t* bytes = (uint8_t*)h_buf;
    int total_bytes = CHUNK_BYTES;
    xor_state = 11111;
    /* Fill first 256 bytes with random */
    for (int i = 0; i < 256; i++) bytes[i] = (uint8_t)(xorshift() % 256);
    /* Rest: copy from random distances back */
    for (int i = 256; i < total_bytes; i++) {
        int dist = (xorshift() % 255) + 1;
        bytes[i] = bytes[i - dist];
    }
}

static void gen_deflate_sliding_window() {
    /* Sliding window friendly: repeated strings at varying offsets */
    uint8_t* bytes = (uint8_t*)h_buf;
    int total_bytes = CHUNK_BYTES;
    /* Create repeated 8-byte "words" scattered throughout */
    xor_state = 12121;
    uint8_t words[16][8];
    for (int w = 0; w < 16; w++)
        for (int b = 0; b < 8; b++)
            words[w][b] = (uint8_t)(xorshift() % 256);
    for (int i = 0; i < total_bytes; i += 8) {
        int w = xorshift() % 16;
        int remain = total_bytes - i;
        int len = remain < 8 ? remain : 8;
        memcpy(&bytes[i], words[w], len);
    }
}

/* --- Integer patterns stored as uint32_t --- */

static void gen_uint32_monotonic() {
    /* Monotonic uint32 with constant delta of 1 */
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = (uint32_t)i;
}

static void gen_uint32_rle() {
    /* RLE-style uint32: runs of 512 identical values */
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = (uint32_t)(i / 512);
}

static void gen_uint32_all_zeros() {
    /* All-zero uint32 buffer */
    memset(h_buf, 0, CHUNK_BYTES);
}

static void gen_uint32_small_range() {
    /* uint32 values in [0, 7] range - highly compressible */
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 13131;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = xorshift() % 8;
}

/* ── Round 3: Deep exploration generators ────────────────────── */

/* --- Bitcomp variations: small-range uint32 --- */

static void gen_uint32_range_1bit() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20001;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = xorshift() % 2;
}

static void gen_uint32_range_2bit() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20002;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = xorshift() % 4;
}

static void gen_uint32_range_4bit() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20003;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = xorshift() % 16;
}

static void gen_uint32_range_5bit() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20004;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = xorshift() % 32;
}

static void gen_uint32_range_6bit() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20005;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = xorshift() % 64;
}

static void gen_uint32_range_7bit() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20006;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = xorshift() % 128;
}

static void gen_uint32_range_8bit() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20007;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = xorshift() % 256;
}

static void gen_uint32_range_10bit() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20008;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = xorshift() % 1024;
}

static void gen_uint32_range_12bit() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20009;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = xorshift() % 4096;
}

static void gen_uint32_range_16bit() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20010;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = xorshift() % 65536;
}

static void gen_uint32_powers_of_2() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20011;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = 1u << (xorshift() % 32);
}

static void gen_uint32_const_upper24_rand8() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20012;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = 0xABCDEF00u | (xorshift() % 256);
}

static void gen_uint32_const_upper16_rand16() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20013;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = 0xDEAD0000u | (xorshift() % 65536);
}

static void gen_uint32_monotonic_delta1() {
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = (uint32_t)i;
}

static void gen_uint32_monotonic_reset() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20014;
    uint32_t val = 0;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        u[i] = val;
        val++;
        if (xorshift() % 256 == 0) val = 0; /* occasional reset */
    }
}

static void gen_int32_neg1_0_1() {
    int32_t* s = (int32_t*)h_buf;
    xor_state = 20015;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        s[i] = (int32_t)(xorshift() % 3) - 1; /* -1, 0, or 1 */
}

static void gen_uint32_gray_code() {
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = (uint32_t)i ^ ((uint32_t)i >> 1); /* standard Gray code */
}

static void gen_uint32_xor_1bit_chain() {
    uint32_t* u = (uint32_t*)h_buf;
    u[0] = 0x12345678u;
    xor_state = 20016;
    for (int i = 1; i < CHUNK_FLOATS; i++) {
        int bit = xorshift() % 32;
        u[i] = u[i-1] ^ (1u << bit);
    }
}

static void gen_uint32_counter_mod16() {
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = (uint32_t)(i % 16);
}

static void gen_uint32_packed_4x8_range3() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 20017;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        uint32_t b0 = xorshift() % 4;
        uint32_t b1 = xorshift() % 4;
        uint32_t b2 = xorshift() % 4;
        uint32_t b3 = xorshift() % 4;
        u[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}

/* --- Cascaded deep exploration --- */

static void gen_uint32_perfect_constant() {
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = 0x42424242u;
}

static void gen_uint32_two_alternating() {
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = (i % 2 == 0) ? 0x11111111u : 0x99999999u;
}

static void gen_uint32_long_runs_3vals() {
    uint32_t* u = (uint32_t*)h_buf;
    uint32_t vals[3] = {100, 200, 300};
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = vals[(i / 256) % 3];
}

static void gen_uint32_delta1_exact() {
    uint32_t* u = (uint32_t*)h_buf;
    for (int i = 0; i < CHUNK_FLOATS; i++) u[i] = 1000 + (uint32_t)i;
}

static void gen_uint32_quadratic() {
    /* Triangular numbers: 0,1,3,6,10,15,... (double delta = 0) */
    uint32_t* u = (uint32_t*)h_buf;
    uint32_t val = 0;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        u[i] = val;
        val += (uint32_t)i + 1;
    }
}

/* --- ANS deep exploration --- */

static void gen_uint32_geometric_p50() {
    /* Geometric dist with p=0.5: P(x) = 0.5^(x+1) */
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 30001;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        uint32_t val = 0;
        while ((xorshift() & 1) == 0 && val < 31) val++;
        u[i] = val;
    }
}

static void gen_uint32_geometric_p90() {
    /* Geometric dist with p=0.9: very skewed toward 0 (90% are 0) */
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 30002;
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        uint32_t val = 0;
        /* Each trial: 10% chance to continue, 90% chance to stop */
        while ((xorshift() % 10) == 0 && val < 31) val++;
        u[i] = val;
    }
}

static void gen_uint32_zipf() {
    /* Zipf distribution: freq ~ 1/rank, 256 unique values */
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 30003;
    /* Inverse CDF method: harmonic sum for 256 values */
    for (int i = 0; i < CHUNK_FLOATS; i++) {
        float r = (float)(xorshift() & 0xFFFFFF) / (float)0xFFFFFF;
        /* Approximate Zipf: rank = floor(N^r) for r in [0,1) doesn't work well,
           use rejection-free: pick rank k with prob 1/k / H_N */
        float cumsum = 0.0f;
        float H = 0.0f;
        for (int k = 1; k <= 256; k++) H += 1.0f / (float)k;
        float target = r * H;
        cumsum = 0.0f;
        uint32_t val = 0;
        for (int k = 1; k <= 256; k++) {
            cumsum += 1.0f / (float)k;
            if (cumsum >= target) { val = (uint32_t)(k - 1); break; }
        }
        u[i] = val;
    }
}

/* --- Mixed --- */

static void gen_uint32_bitmap_0_or_ff() {
    uint32_t* u = (uint32_t*)h_buf;
    xor_state = 40001;
    for (int i = 0; i < CHUNK_FLOATS; i++)
        u[i] = (xorshift() & 1) ? 0xFFFFFFFFu : 0x00000000u;
}

static void gen_uint32_half_zero_half_ff() {
    uint32_t* u = (uint32_t*)h_buf;
    int half = CHUNK_FLOATS / 2;
    for (int i = 0; i < half; i++) u[i] = 0x00000000u;
    for (int i = half; i < CHUNK_FLOATS; i++) u[i] = 0xFFFFFFFFu;
}

/* ── Compression test ────────────────────────────────────────── */

static void test_pattern(const char* name, void (*gen_fn)(), void* d_data, void* d_comp) {
    gen_fn();
    CUDA_CHECK(cudaMemcpy(d_data, h_buf, CHUNK_BYTES, cudaMemcpyHostToDevice));

    size_t out_buf_size = CHUNK_BYTES + 65536;
    double ratios[NUM_CONFIGS];
    int    config_algo[NUM_CONFIGS];
    int    config_shuf[NUM_CONFIGS];

    double best_ratio = 0;
    int    best_algo = 1, best_shuf = 0;
    double zstd_best = 0;

    int ci = 0;
    for (int algo = 1; algo <= NUM_ALGOS; algo++) {
        for (int shuf = 0; shuf <= 1; shuf++, ci++) {
            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = (gpucompress_algorithm_t)algo;
            cfg.preprocessing = shuf ? GPUCOMPRESS_PREPROC_SHUFFLE_4 : 0;
            cfg.error_bound = 0.0;

            size_t out_size = out_buf_size;
            gpucompress_stats_t stats;
            gpucompress_error_t err = gpucompress_compress_gpu(
                d_data, CHUNK_BYTES, d_comp, &out_size, &cfg, &stats, NULL);

            double ratio = (err == GPUCOMPRESS_SUCCESS) ? stats.compression_ratio : 0.0;
            ratios[ci] = ratio;
            config_algo[ci] = algo;
            config_shuf[ci] = shuf;

            if (ratio > best_ratio) {
                best_ratio = ratio;
                best_algo  = algo;
                best_shuf  = shuf;
            }
            if (algo == 5 && ratio > zstd_best) {
                zstd_best = ratio;
            }
        }
    }

    double margin = best_ratio - zstd_best;
    int non_zstd_wins = (best_algo != 5);

    /* Record result */
    PatternResult& r = results[n_results++];
    r.name = name;
    r.best_algo = best_algo;
    r.best_shuf = best_shuf;
    r.best_ratio = best_ratio;
    r.zstd_best_algo = 5;
    r.zstd_best_ratio = zstd_best;
    r.margin = margin;

    if (non_zstd_wins && margin > 0.001) {
        algo_wins[best_algo]++;
        if (best_shuf) algo_wins_shuf[best_algo]++;
    }

    /* Print per-pattern detail */
    char best_str[32];
    snprintf(best_str, sizeof(best_str), "%s%s",
             ALGO_NAMES[best_algo], best_shuf ? "+shuf" : "");

    const char* flag = (non_zstd_wins && margin > 0.001) ? " <<<" :
                       (non_zstd_wins ? " (tie)" : "");
    if (print_enabled) {
        printf("  %-35s | %-16s %7.3fx | zstd-best %7.3fx | margin %+.4fx%s\n",
               name, best_str, best_ratio, zstd_best, margin, flag);

        /* If non-zstd wins, print all ratios */
        if (non_zstd_wins) {
            printf("    All ratios:");
            ci = 0;
            for (int algo = 1; algo <= NUM_ALGOS; algo++) {
                for (int shuf = 0; shuf <= 1; shuf++, ci++) {
                    if (ratios[ci] > 0.01)
                        printf(" %s%s=%.2f", ALGO_NAMES[algo],
                               shuf ? "+s" : "", ratios[ci]);
                }
            }
            printf("\n");
        }
    }
}

/* ── Main ────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    const char* weights = (argc > 1) ? argv[1] : "neural_net/weights/model.nnwt";

    printf("================================================================\n");
    printf("  Algo Pattern Explorer: Round 3 Deep Exploration\n");
    printf("================================================================\n");
    printf("  Data size: %d uint32 = %.1f MB per pattern\n",
           CHUNK_FLOATS, (double)CHUNK_BYTES / (1 << 20));
    printf("  Configs: 8 algos x 2 shuffle = 16 per pattern\n");
    printf("  (Patterns 1-55 run silently, only new 56-85 printed)\n");
    printf("================================================================\n\n");

    gpucompress_error_t rc = gpucompress_init(weights);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %s\n", gpucompress_error_string(rc));
        return 1;
    }

    alloc_host();

    void *d_data = NULL, *d_comp = NULL;
    size_t out_buf_size = CHUNK_BYTES + 65536;
    CUDA_CHECK(cudaMalloc(&d_data, CHUNK_BYTES));
    CUDA_CHECK(cudaMalloc(&d_comp, out_buf_size));

    /* Skip printing for first 55 patterns (already explored) */
    print_enabled = 0;

    /* Run all patterns */
    test_pattern("1. All zeros",                    gen_all_zeros,          d_data, d_comp);
    test_pattern("2. Constant float (1.0)",         gen_constant_one,       d_data, d_comp);
    test_pattern("3. Repeated small ints (0-3)",    gen_repeated_small_ints,d_data, d_comp);
    test_pattern("4. Random uniform [0,1]",         gen_random_uniform,     d_data, d_comp);
    test_pattern("5. Random int 0-255 as float",    gen_random_int_0_255,   d_data, d_comp);
    test_pattern("6. Monotonic increasing",         gen_monotonic_inc,      d_data, d_comp);
    test_pattern("7. Monotonic + noise",            gen_monotonic_noise,    d_data, d_comp);
    test_pattern("8. Sine wave (low freq)",         gen_sine_low_freq,      d_data, d_comp);
    test_pattern("9. Sine wave (high freq)",        gen_sine_high_freq,     d_data, d_comp);
    test_pattern("10. Gaussian noise",              gen_gaussian,           d_data, d_comp);
    test_pattern("11. Sparse 95% zeros",            gen_sparse_95,          d_data, d_comp);
    test_pattern("12. Sparse 99% zeros",            gen_sparse_99,          d_data, d_comp);
    test_pattern("13. Run-length (1024 blocks)",    gen_run_length,         d_data, d_comp);
    test_pattern("14. Constant delta",              gen_constant_delta,     d_data, d_comp);
    test_pattern("15. Bit-packed 8-in-32",          gen_bit_packed_8,       d_data, d_comp);
    test_pattern("16. Alternating two values",      gen_alternating_two,    d_data, d_comp);
    test_pattern("17. Exponential growth",          gen_exponential_growth, d_data, d_comp);
    test_pattern("18. Log-normal",                  gen_log_normal,         d_data, d_comp);
    test_pattern("19. Poisson-like",                gen_poisson_like,       d_data, d_comp);
    test_pattern("20. Repeated block (64 floats)",  gen_repeated_block_64,  d_data, d_comp);
    test_pattern("21. Power-law (Huffman-friendly)", gen_power_law,         d_data, d_comp);
    test_pattern("22. Binary (0.0 / 1.0)",          gen_binary,             d_data, d_comp);
    test_pattern("23. Quantized to 0.01",           gen_quantized_001,      d_data, d_comp);
    test_pattern("24. Slowly varying (<0.001)",     gen_slowly_varying,     d_data, d_comp);
    test_pattern("25. Sawtooth wave",               gen_sawtooth,           d_data, d_comp);
    test_pattern("26. Step function",               gen_step_function,      d_data, d_comp);
    test_pattern("27. Interleaved channels (RGB)",  gen_interleaved_channels, d_data, d_comp);
    test_pattern("28. Fibonacci mod 97",            gen_fibonacci_mod,      d_data, d_comp);
    test_pattern("29. Geometric sequence",          gen_geometric,          d_data, d_comp);
    test_pattern("30. Repeat every 4",              gen_repeat_every_4,     d_data, d_comp);
    test_pattern("31. Repeat every 16",             gen_repeat_every_16,    d_data, d_comp);
    test_pattern("32. Repeat every 256",            gen_repeat_every_256,   d_data, d_comp);
    test_pattern("33. XOR pattern",                 gen_xor_pattern,        d_data, d_comp);

    /* ── Round 2: Targeted patterns (output suppressed) ────── */

    /* Bitcomp targets */
    test_pattern("34. Bitcomp: toggle LSB",          gen_bitcomp_toggle_lsb,       d_data, d_comp);
    test_pattern("35. Bitcomp: const exponent",      gen_bitcomp_const_exponent,   d_data, d_comp);
    test_pattern("36. Bitcomp: small XOR delta",     gen_bitcomp_small_xor_delta,  d_data, d_comp);
    test_pattern("37. Bitcomp: float 1.0±eps",       gen_bitcomp_float_one_epsilon,d_data, d_comp);
    test_pattern("38. Bitcomp: drift ±1 int",        gen_bitcomp_drift_int,        d_data, d_comp);

    /* Cascaded targets */
    test_pattern("39. Cascaded: 10k runs",           gen_cascaded_long_runs,       d_data, d_comp);
    test_pattern("40. Cascaded: perfect arith seq",  gen_cascaded_perfect_arith,   d_data, d_comp);
    test_pattern("41. Cascaded: block-constant 1024",gen_cascaded_block_const,     d_data, d_comp);
    test_pattern("42. Cascaded: staircase /256",     gen_cascaded_staircase,       d_data, d_comp);

    /* LZ4/Snappy targets */
    test_pattern("43. LZ4: repeat 4-byte",           gen_lz4_repeat_4byte,        d_data, d_comp);
    test_pattern("44. LZ4: repeat 16-byte",          gen_lz4_repeat_16byte,       d_data, d_comp);
    test_pattern("45. LZ4: repeat 64-byte",          gen_lz4_repeat_64byte,       d_data, d_comp);
    test_pattern("46. LZ4: lookup table 32",         gen_lz4_lookup_table,        d_data, d_comp);
    test_pattern("47. LZ4: tile rows 512",           gen_lz4_tile_rows,           d_data, d_comp);

    /* ANS targets */
    test_pattern("48. ANS: 99% same byte",           gen_ans_skewed_99,           d_data, d_comp);
    test_pattern("49. ANS: Markov chain",            gen_ans_markov,              d_data, d_comp);

    /* Deflate/GDeflate targets */
    test_pattern("50. Deflate: backward refs",       gen_deflate_backward_refs,   d_data, d_comp);
    test_pattern("51. Deflate: sliding window",      gen_deflate_sliding_window,  d_data, d_comp);

    /* Integer patterns as uint32 */
    test_pattern("52. uint32: monotonic",            gen_uint32_monotonic,        d_data, d_comp);
    test_pattern("53. uint32: RLE (runs of 512)",    gen_uint32_rle,              d_data, d_comp);
    test_pattern("54. uint32: all zeros",            gen_uint32_all_zeros,        d_data, d_comp);
    test_pattern("55. uint32: small range [0,7]",    gen_uint32_small_range,      d_data, d_comp);

    /* ── Round 3: Deep exploration (new patterns) ─────────────── */
    print_enabled = 1;
    printf("\n  --- Round 3: Deep exploration ---\n\n");
    printf("  %-35s | %-16s %8s | %-18s | %s\n",
           "Pattern", "Winner", "Ratio", "Zstd-best", "Margin");
    printf("  -----------------------------------+-------------------------"
           "+--------------------+-----------\n");

    /* Bitcomp variations */
    test_pattern("56. uint32 range [0,1] (1-bit)",    gen_uint32_range_1bit,       d_data, d_comp);
    test_pattern("57. uint32 range [0,3] (2-bit)",    gen_uint32_range_2bit,       d_data, d_comp);
    test_pattern("58. uint32 range [0,15] (4-bit)",   gen_uint32_range_4bit,       d_data, d_comp);
    test_pattern("59. uint32 range [0,31] (5-bit)",   gen_uint32_range_5bit,       d_data, d_comp);
    test_pattern("60. uint32 range [0,63] (6-bit)",   gen_uint32_range_6bit,       d_data, d_comp);
    test_pattern("61. uint32 range [0,127] (7-bit)",  gen_uint32_range_7bit,       d_data, d_comp);
    test_pattern("62. uint32 range [0,255] (8-bit)",  gen_uint32_range_8bit,       d_data, d_comp);
    test_pattern("63. uint32 range [0,1023] (10-bit)",gen_uint32_range_10bit,      d_data, d_comp);
    test_pattern("64. uint32 range [0,4095] (12-bit)",gen_uint32_range_12bit,      d_data, d_comp);
    test_pattern("65. uint32 range [0,65535] (16-bit)",gen_uint32_range_16bit,     d_data, d_comp);
    test_pattern("66. uint32 powers of 2 only",       gen_uint32_powers_of_2,      d_data, d_comp);
    test_pattern("67. uint32 const upper24, rand8",   gen_uint32_const_upper24_rand8, d_data, d_comp);
    test_pattern("68. uint32 const upper16, rand16",  gen_uint32_const_upper16_rand16, d_data, d_comp);
    test_pattern("69. uint32 monotonic delta=1",      gen_uint32_monotonic_delta1, d_data, d_comp);
    test_pattern("70. uint32 monotonic+reset",        gen_uint32_monotonic_reset,  d_data, d_comp);
    test_pattern("71. int32 {-1,0,1} random",         gen_int32_neg1_0_1,          d_data, d_comp);
    test_pattern("72. uint32 Gray code",              gen_uint32_gray_code,        d_data, d_comp);
    test_pattern("73. uint32 XOR 1-bit chain",        gen_uint32_xor_1bit_chain,   d_data, d_comp);
    test_pattern("74. uint32 counter mod 16",         gen_uint32_counter_mod16,    d_data, d_comp);
    test_pattern("75. uint32 packed 4x8 range[0,3]",  gen_uint32_packed_4x8_range3, d_data, d_comp);

    /* Cascaded deep exploration */
    test_pattern("76. Perfect constant uint32",       gen_uint32_perfect_constant, d_data, d_comp);
    test_pattern("77. Two alternating uint32 vals",   gen_uint32_two_alternating,  d_data, d_comp);
    test_pattern("78. Runs 256: 3 values cycling",    gen_uint32_long_runs_3vals,  d_data, d_comp);
    test_pattern("79. Delta=1 exact (1000+i)",        gen_uint32_delta1_exact,     d_data, d_comp);
    test_pattern("80. Quadratic (triangular nums)",   gen_uint32_quadratic,        d_data, d_comp);

    /* ANS deep exploration */
    test_pattern("81. Geometric dist p=0.5",          gen_uint32_geometric_p50,    d_data, d_comp);
    test_pattern("82. Geometric dist p=0.9 (skewed)", gen_uint32_geometric_p90,    d_data, d_comp);
    test_pattern("83. Zipf distribution (256 vals)",  gen_uint32_zipf,             d_data, d_comp);

    /* Mixed */
    test_pattern("84. Bitmap: 0x0 or 0xFFFFFFFF rand",gen_uint32_bitmap_0_or_ff,  d_data, d_comp);
    test_pattern("85. Half 0x0, half 0xFFFFFFFF",     gen_uint32_half_zero_half_ff, d_data, d_comp);

    /* ── Summary ─────────────────────────────────────────────── */
    printf("\n================================================================\n");
    printf("  SUMMARY: Non-zstd winners\n");
    printf("================================================================\n");

    int total_non_zstd = 0;
    for (int a = 1; a <= 8; a++) {
        if (a == 5) continue; /* skip zstd */
        total_non_zstd += algo_wins[a];
    }

    printf("  Total patterns tested: %d\n", n_results);
    printf("  Patterns where non-zstd wins: %d\n", total_non_zstd);
    printf("\n  Algorithm win counts (sorted):\n");

    /* Sort by win count */
    int sorted[9];
    for (int i = 0; i < 9; i++) sorted[i] = i;
    for (int i = 1; i <= 8; i++)
        for (int j = i + 1; j <= 8; j++)
            if (algo_wins[sorted[j]] > algo_wins[sorted[i]]) {
                int t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
            }

    for (int k = 1; k <= 8; k++) {
        int a = sorted[k];
        if (a == 0) continue;
        printf("    %-12s: %d wins", ALGO_NAMES[a], algo_wins[a]);
        if (algo_wins[a] > 0)
            printf(" (%d with shuffle)", algo_wins_shuf[a]);
        printf("\n");
    }

    /* Print detailed non-zstd winners */
    printf("\n  Detailed non-zstd winners:\n");
    printf("  %-35s | %-16s | ratio   | zstd    | margin\n", "Pattern", "Winner");
    printf("  -----------------------------------+------------------+---------+---------+--------\n");
    for (int i = 0; i < n_results; i++) {
        PatternResult& r = results[i];
        if (r.best_algo == 5) continue;
        char best_str[32];
        snprintf(best_str, sizeof(best_str), "%s%s",
                 ALGO_NAMES[r.best_algo], r.best_shuf ? "+shuf" : "");
        printf("  %-35s | %-16s | %6.2fx | %6.2fx | %+.2fx\n",
               r.name, best_str, r.best_ratio, r.zstd_best_ratio, r.margin);
    }

    /* Cleanup */
    cudaFree(d_data);
    cudaFree(d_comp);
    free(h_buf);
    gpucompress_cleanup();

    printf("\nDone.\n");
    return 0;
}
