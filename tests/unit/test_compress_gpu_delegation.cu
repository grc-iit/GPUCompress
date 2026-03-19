/**
 * @file test_compress_gpu_delegation.cu
 * @brief Tests that compress_gpu correctly delegates to _with_action_gpu.
 *
 * After refactoring, compress_gpu is a thin wrapper:
 *   1. Non-AUTO: encode algo as action → delegate
 *   2. ALGO_AUTO: infer_gpu → delegate with pre-computed stats
 *
 * This test verifies:
 *   - Round-trip correctness for all 8 algorithms via delegation
 *   - Shuffle + explicit algo delegation
 *   - ALGO_AUTO delegation (if NN weights available)
 *   - Stats struct populated correctly after delegation
 *   - No data corruption in the delegation chain
 */

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;

#define TEST(name) printf("  [TEST] %s ... ", name)
#define PASS() do { printf("PASS\n"); g_pass++; } while (0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); g_fail++; } while (0)

/* ============================================================
 * Helper: GPU round-trip with given config
 * ============================================================ */
static bool roundtrip_gpu(const float* h_orig, size_t N,
                          gpucompress_config_t* cfg,
                          gpucompress_stats_t* out_stats) {
    size_t data_bytes = N * sizeof(float);

    void *d_in = nullptr, *d_comp = nullptr, *d_decomp = nullptr;
    cudaMalloc(&d_in, data_bytes);
    cudaMalloc(&d_comp, data_bytes * 4);
    cudaMalloc(&d_decomp, data_bytes);
    cudaMemcpy(d_in, h_orig, data_bytes, cudaMemcpyHostToDevice);

    size_t comp_sz = data_bytes * 4;
    gpucompress_stats_t stats = {};
    gpucompress_error_t err = gpucompress_compress_gpu(
        d_in, data_bytes, d_comp, &comp_sz, cfg, &stats, nullptr);

    if (err != GPUCOMPRESS_SUCCESS) {
        cudaFree(d_in); cudaFree(d_comp); cudaFree(d_decomp);
        return false;
    }

    size_t decomp_sz = data_bytes;
    err = gpucompress_decompress_gpu(d_comp, comp_sz, d_decomp, &decomp_sz, nullptr);
    if (err != GPUCOMPRESS_SUCCESS) {
        cudaFree(d_in); cudaFree(d_comp); cudaFree(d_decomp);
        return false;
    }

    std::vector<float> h_decomp(N);
    cudaMemcpy(h_decomp.data(), d_decomp, data_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_comp); cudaFree(d_decomp);

    if (out_stats) *out_stats = stats;

    for (size_t i = 0; i < N; i++) {
        if (h_orig[i] != h_decomp[i]) return false;
    }
    return true;
}

/* ============================================================
 * Test 1: All 8 algorithms via non-AUTO delegation
 * ============================================================ */
static void test_all_algos_delegation() {
    const size_t N = 64 * 1024;
    std::vector<float> h_data(N);
    for (size_t i = 0; i < N; i++) h_data[i] = sinf((float)i * 0.01f);

    const char* names[] = {"", "LZ4", "Snappy", "Deflate", "GDeflate",
                           "Zstd", "ANS", "Cascaded", "Bitcomp"};

    for (int algo = 1; algo <= 8; algo++) {
        char label[64];
        snprintf(label, sizeof(label), "Non-AUTO delegation: %s round-trip", names[algo]);
        TEST(label);

        gpucompress_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.algorithm = static_cast<gpucompress_algorithm_t>(algo);

        gpucompress_stats_t stats = {};
        bool ok = roundtrip_gpu(h_data.data(), N, &cfg, &stats);

        if (!ok) { FAIL("round-trip mismatch"); continue; }
        if (stats.algorithm_used != algo) {
            char buf[128];
            snprintf(buf, sizeof(buf), "algorithm_used=%d, expected=%d",
                     stats.algorithm_used, algo);
            FAIL(buf); continue;
        }
        PASS();
    }
}

/* ============================================================
 * Test 2: Explicit algo + shuffle delegation
 * ============================================================ */
static void test_shuffle_delegation() {
    TEST("Non-AUTO delegation: Zstd+Shuffle round-trip");

    const size_t N = 64 * 1024;
    std::vector<float> h_data(N);
    for (size_t i = 0; i < N; i++) h_data[i] = (float)(i % 256);

    gpucompress_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.algorithm = GPUCOMPRESS_ALGO_ZSTD;
    cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;

    gpucompress_stats_t stats = {};
    bool ok = roundtrip_gpu(h_data.data(), N, &cfg, &stats);

    if (!ok) { FAIL("round-trip mismatch"); return; }
    if (stats.algorithm_used != GPUCOMPRESS_ALGO_ZSTD) {
        FAIL("algorithm_used != ZSTD"); return;
    }
    if (!(stats.preprocessing_used & GPUCOMPRESS_PREPROC_SHUFFLE_4)) {
        FAIL("shuffle not reported in preprocessing_used"); return;
    }
    PASS();
}

/* ============================================================
 * Test 3: Stats struct populated after delegation
 * ============================================================ */
static void test_stats_populated() {
    TEST("Stats struct: ratio and throughput populated after delegation");

    const size_t N = 64 * 1024;
    std::vector<float> h_data(N, 3.14f);  /* constant → high ratio */

    gpucompress_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

    gpucompress_stats_t stats = {};
    bool ok = roundtrip_gpu(h_data.data(), N, &cfg, &stats);

    if (!ok) { FAIL("round-trip failed"); return; }
    if (stats.compression_ratio < 2.0) {
        char buf[128];
        snprintf(buf, sizeof(buf), "ratio=%.2f (expected > 2.0 for constant data)",
                 stats.compression_ratio);
        FAIL(buf); return;
    }
    if (stats.original_size != N * sizeof(float)) {
        FAIL("original_size mismatch"); return;
    }
    if (stats.compressed_size == 0 || stats.compressed_size >= stats.original_size) {
        FAIL("compressed_size unexpected"); return;
    }
    PASS();
}

/* ============================================================
 * Test 4: ALGO_AUTO delegation (requires NN weights)
 * ============================================================ */
static void test_auto_delegation(const char* weights_path) {
    TEST("ALGO_AUTO delegation: infer_gpu → _with_action_gpu");

    if (!weights_path) {
        printf("SKIP (no weights path)\n"); g_pass++;
        return;
    }

    gpucompress_cleanup();
    gpucompress_error_t err = gpucompress_init(weights_path);
    if (err != GPUCOMPRESS_SUCCESS) {
        FAIL("gpucompress_init with weights failed"); return;
    }
    if (!gpucompress_nn_is_loaded()) {
        FAIL("NN not loaded"); gpucompress_cleanup(); return;
    }

    const size_t N = 64 * 1024;
    std::vector<float> h_data(N);
    for (size_t i = 0; i < N; i++) h_data[i] = sinf((float)i * 0.01f);

    gpucompress_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    gpucompress_stats_t stats = {};
    bool ok = roundtrip_gpu(h_data.data(), N, &cfg, &stats);

    gpucompress_cleanup();

    if (!ok) { FAIL("ALGO_AUTO round-trip mismatch"); return; }
    if (stats.algorithm_used < 1 || stats.algorithm_used > 8) {
        char buf[64];
        snprintf(buf, sizeof(buf), "algorithm_used=%d (expected 1-8)", stats.algorithm_used);
        FAIL(buf); return;
    }
    if (stats.nn_original_action < 0 || stats.nn_original_action > 31) {
        FAIL("nn_original_action out of range"); return;
    }
    PASS();
}

/* ============================================================
 * Test 5: Multiple algos produce different compressed sizes
 * (ensures delegation actually uses the specified algorithm)
 * ============================================================ */
static void test_different_algos_different_sizes() {
    TEST("Different algos produce different compressed sizes");

    const size_t N = 64 * 1024;
    std::vector<float> h_data(N);
    for (size_t i = 0; i < N; i++) h_data[i] = sinf((float)i * 0.001f);

    size_t sizes[8] = {};
    bool all_ok = true;

    for (int algo = 1; algo <= 8; algo++) {
        gpucompress_config_t cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.algorithm = static_cast<gpucompress_algorithm_t>(algo);

        void *d_in = nullptr, *d_comp = nullptr;
        size_t data_bytes = N * sizeof(float);
        cudaMalloc(&d_in, data_bytes);
        cudaMalloc(&d_comp, data_bytes * 4);
        cudaMemcpy(d_in, h_data.data(), data_bytes, cudaMemcpyHostToDevice);

        size_t comp_sz = data_bytes * 4;
        gpucompress_error_t err = gpucompress_compress_gpu(
            d_in, data_bytes, d_comp, &comp_sz, &cfg, nullptr, nullptr);

        cudaFree(d_in); cudaFree(d_comp);

        if (err != GPUCOMPRESS_SUCCESS) { all_ok = false; break; }
        sizes[algo - 1] = comp_sz;
    }

    if (!all_ok) { FAIL("compression failed for some algo"); return; }

    /* At least some algos should produce different sizes */
    int unique = 0;
    for (int i = 0; i < 8; i++) {
        bool is_unique = true;
        for (int j = 0; j < i; j++) {
            if (sizes[i] == sizes[j]) { is_unique = false; break; }
        }
        if (is_unique) unique++;
    }

    if (unique < 3) {
        char buf[128];
        snprintf(buf, sizeof(buf), "only %d unique sizes across 8 algos (expected >= 3)", unique);
        FAIL(buf); return;
    }
    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char** argv) {
    printf("=== test_compress_gpu_delegation ===\n");
    printf("Tests for compress_gpu → _with_action_gpu delegation\n\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found - skipping\n");
        return 1;
    }

    gpucompress_error_t err = gpucompress_init(nullptr);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("gpucompress_init failed\n");
        return 1;
    }

    test_all_algos_delegation();
    test_shuffle_delegation();
    test_stats_populated();
    test_different_algos_different_sizes();

    gpucompress_cleanup();

    /* ALGO_AUTO test needs weights path as argv[1] */
    const char* weights = (argc > 1) ? argv[1] : nullptr;
    test_auto_delegation(weights);

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
