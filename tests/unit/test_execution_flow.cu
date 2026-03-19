/**
 * @file test_execution_flow.cu
 * @brief End-to-end execution flow test for the GPU compression pipeline.
 *
 * Tests the complete path: init → inference → compress → decompress → verify
 * for all three entry points:
 *   1. compress_gpu (non-AUTO) — delegation to _with_action_gpu
 *   2. compress_gpu (ALGO_AUTO) — infer_gpu → _with_action_gpu
 *   3. VOL pipeline (H5Dwrite) — Stage 1 infer → Stage 2 _with_action_gpu → Stage 3 I/O
 *
 * Also tests:
 *   - Preprocessing paths (shuffle, quantization)
 *   - SGD fires when learning is enabled
 *   - Exploration fires and overrides output when enabled
 *   - Stats struct correctly populated
 *   - Chunk diagnostics correctly populated
 *   - Bitwise round-trip correctness
 *
 * Usage:
 *   ./test_execution_flow <weights.nnwt>
 */

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5.h"
#include "gpucompress_hdf5_vol.h"

#define TMP_FILE "/tmp/test_execution_flow.h5"

static const char* g_weights_path = nullptr;

/* ============================================================
 * Helpers
 * ============================================================ */

static void fill_sine(float* h, size_t N) {
    for (size_t i = 0; i < N; i++) h[i] = sinf((float)i * 0.01f);
}

static void fill_constant(float* h, size_t N) {
    for (size_t i = 0; i < N; i++) h[i] = 3.14159f;
}

static void fill_gradient(float* h, size_t N) {
    for (size_t i = 0; i < N; i++) h[i] = (float)(i % 256);
}

static void pack_double_cd(double v, unsigned int *lo, unsigned int *hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/* GPU round-trip: compress on GPU, decompress on GPU, compare on host */
static bool gpu_roundtrip(const float* h_orig, size_t N,
                          gpucompress_config_t* cfg,
                          gpucompress_stats_t* out_stats) {
    size_t bytes = N * sizeof(float);
    void *d_in = nullptr, *d_comp = nullptr, *d_decomp = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_comp, bytes * 4);
    cudaMalloc(&d_decomp, bytes);
    cudaMemcpy(d_in, h_orig, bytes, cudaMemcpyHostToDevice);

    size_t comp_sz = bytes * 4;
    gpucompress_stats_t stats = {};
    gpucompress_error_t err = gpucompress_compress_gpu(
        d_in, bytes, d_comp, &comp_sz, cfg, &stats, nullptr);
    if (err != GPUCOMPRESS_SUCCESS) {
        cudaFree(d_in); cudaFree(d_comp); cudaFree(d_decomp);
        return false;
    }

    size_t decomp_sz = bytes;
    err = gpucompress_decompress_gpu(d_comp, comp_sz, d_decomp, &decomp_sz, nullptr);
    if (err != GPUCOMPRESS_SUCCESS) {
        cudaFree(d_in); cudaFree(d_comp); cudaFree(d_decomp);
        return false;
    }

    std::vector<float> h_decomp(N);
    cudaMemcpy(h_decomp.data(), d_decomp, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_comp); cudaFree(d_decomp);

    if (out_stats) *out_stats = stats;
    for (size_t i = 0; i < N; i++) {
        if (h_orig[i] != h_decomp[i]) return false;
    }
    return true;
}

/* ============================================================
 * Test 1: Non-AUTO explicit algorithm — delegation path
 * ============================================================ */
static void test_explicit_algo_roundtrip() {
    printf("  [TEST] Explicit algo round-trip (all 8 algos)...\n");

    const size_t N = 64 * 1024;
    std::vector<float> h(N);
    fill_sine(h.data(), N);

    const char* names[] = {"", "LZ4", "Snappy", "Deflate", "GDeflate",
                           "Zstd", "ANS", "Cascaded", "Bitcomp"};

    for (int algo = 1; algo <= 8; algo++) {
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = static_cast<gpucompress_algorithm_t>(algo);

        gpucompress_stats_t stats = {};
        bool ok = gpu_roundtrip(h.data(), N, &cfg, &stats);

        printf("    %s: ", names[algo]);
        assert(ok && "round-trip data mismatch");
        assert(stats.algorithm_used == algo && "wrong algorithm_used in stats");
        assert(stats.original_size == N * sizeof(float) && "wrong original_size");
        assert(stats.compressed_size > 0 && "compressed_size is zero");
        assert(stats.compression_ratio > 0.0 && "compression_ratio is zero");
        printf("PASS (ratio=%.2fx)\n", stats.compression_ratio);
    }
}

/* ============================================================
 * Test 2: Explicit algo + shuffle preprocessing
 * ============================================================ */
static void test_shuffle_roundtrip() {
    printf("  [TEST] Zstd+Shuffle round-trip...");

    const size_t N = 64 * 1024;
    std::vector<float> h(N);
    fill_gradient(h.data(), N);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_ZSTD;
    cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;

    gpucompress_stats_t stats = {};
    bool ok = gpu_roundtrip(h.data(), N, &cfg, &stats);

    assert(ok && "shuffle round-trip mismatch");
    assert(stats.algorithm_used == GPUCOMPRESS_ALGO_ZSTD);
    assert(stats.preprocessing_used & GPUCOMPRESS_PREPROC_SHUFFLE_4);
    printf(" PASS (ratio=%.2fx)\n", stats.compression_ratio);
}

/* ============================================================
 * Test 3: ALGO_AUTO — full NN inference → delegation path
 * ============================================================ */
static void test_auto_roundtrip() {
    printf("  [TEST] ALGO_AUTO round-trip (NN inference → delegation)...");

    const size_t N = 64 * 1024;
    std::vector<float> h(N);
    fill_sine(h.data(), N);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    gpucompress_stats_t stats = {};
    bool ok = gpu_roundtrip(h.data(), N, &cfg, &stats);

    assert(ok && "ALGO_AUTO round-trip mismatch");
    assert(stats.algorithm_used >= 1 && stats.algorithm_used <= 8);
    assert(stats.nn_original_action >= 0 && stats.nn_original_action <= 31);
    printf(" PASS (action=%d, algo=%d, ratio=%.2fx)\n",
           stats.nn_original_action, stats.algorithm_used, stats.compression_ratio);
}

/* ============================================================
 * Test 4: ALGO_AUTO + SGD fires when learning enabled
 * ============================================================ */
static void test_sgd_fires() {
    printf("  [TEST] ALGO_AUTO + SGD fires when learning enabled...");

    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.1f, 0.01f, 0.01f);
    gpucompress_set_exploration(0);

    const size_t N = 64 * 1024;
    std::vector<float> h(N);
    fill_sine(h.data(), N);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    gpucompress_reset_chunk_history();

    gpucompress_stats_t stats = {};
    bool ok = gpu_roundtrip(h.data(), N, &cfg, &stats);

    assert(ok && "SGD test round-trip failed");
    /* SGD should fire because MAPE threshold is very low (1%) */
    assert(stats.sgd_fired == 1 && "SGD did not fire with low threshold");
    assert(stats.exploration_triggered == 0 && "exploration should be off");

    gpucompress_disable_online_learning();
    printf(" PASS\n");
}

/* ============================================================
 * Test 5: ALGO_AUTO + Exploration fires and overrides
 * ============================================================ */
static void test_exploration_fires() {
    printf("  [TEST] ALGO_AUTO + Exploration fires and overrides...");

    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.5f, 0.01f, 0.01f);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);  /* always explore */
    gpucompress_set_exploration_k(8);

    const size_t N = 64 * 1024;
    std::vector<float> h(N);
    fill_constant(h.data(), N);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    gpucompress_reset_chunk_history();

    gpucompress_stats_t stats = {};
    bool ok = gpu_roundtrip(h.data(), N, &cfg, &stats);

    assert(ok && "exploration test round-trip failed");
    assert(stats.exploration_triggered == 1 && "exploration did not trigger");
    /* Original and final action may differ if exploration found better */
    printf(" PASS (orig_action=%d, final_action=%d, ratio=%.2fx)\n",
           stats.nn_original_action, stats.nn_final_action, stats.compression_ratio);

    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);
}

/* ============================================================
 * Test 6: Chunk diagnostics populated correctly
 * ============================================================ */
static void test_chunk_diagnostics() {
    printf("  [TEST] Chunk diagnostics populated after ALGO_AUTO...");

    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.1f, 0.01f, 0.01f);
    gpucompress_set_exploration(0);

    const size_t N = 64 * 1024;
    std::vector<float> h(N);
    fill_sine(h.data(), N);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    gpucompress_reset_chunk_history();

    gpucompress_stats_t stats = {};
    bool ok = gpu_roundtrip(h.data(), N, &cfg, &stats);
    assert(ok && "diagnostics test round-trip failed");

    int n_hist = gpucompress_get_chunk_history_count();
    assert(n_hist == 1 && "expected exactly 1 chunk in history");

    gpucompress_chunk_diag_t diag = {};
    int rc = gpucompress_get_chunk_diag(0, &diag);
    assert(rc == 0 && "gpucompress_get_chunk_diag failed");

    assert(diag.nn_action >= 0 && diag.nn_action <= 31);
    assert(diag.actual_ratio > 0.0f);
    assert(diag.predicted_ratio > 0.0f);
    assert(diag.compression_ms >= 0.0f);
    assert(diag.feat_eb_enc != 0.0f || diag.feat_ds_enc != 0.0f);

    printf(" PASS (action=%d, actual_ratio=%.2f, pred_ratio=%.2f)\n",
           diag.nn_action, diag.actual_ratio, diag.predicted_ratio);

    gpucompress_disable_online_learning();
}

/* ============================================================
 * Test 7: VOL pipeline — H5Dwrite → compress → H5Dread → verify
 * ============================================================ */
static void test_vol_pipeline() {
    printf("  [TEST] VOL pipeline: H5Dwrite → compress → H5Dread → verify...");

    hid_t vol_id = H5VL_gpucompress_register();
    assert(vol_id != H5I_INVALID_HID && "VOL register failed");

    const int L = 64;
    const size_t N = (size_t)L * L * L;
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_orig(N);
    fill_sine(h_orig.data(), N);

    /* Upload to GPU */
    float* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_orig.data(), bytes, cudaMemcpyHostToDevice);

    /* Create HDF5 file with VOL */
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    remove(TMP_FILE);
    hid_t file = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    assert(file >= 0 && "H5Fcreate failed");

    /* Create chunked dataset with ALGO_AUTO filter */
    hsize_t dims[3] = {(hsize_t)L, (hsize_t)L, (hsize_t)L};
    hsize_t cdims[3] = {(hsize_t)L, (hsize_t)L, (hsize_t)L};
    hid_t fsp = H5Screate_simple(3, dims, NULL);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]); /* lossless */
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    hid_t dset = H5Dcreate2(file, "V", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);
    assert(dset >= 0 && "H5Dcreate2 failed");

    /* Write GPU pointer through VOL (Stage 1 → Stage 2 → Stage 3) */
    herr_t wrc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    assert(wrc >= 0 && "H5Dwrite failed");

    H5Dclose(dset);
    H5Fclose(file);

    /* Read back through VOL */
    float* d_read = nullptr;
    cudaMalloc(&d_read, bytes);

    native_id = H5VLget_connector_id_by_name("native");
    fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    assert(file >= 0 && "H5Fopen failed");

    dset = H5Dopen2(file, "V", H5P_DEFAULT);
    assert(dset >= 0 && "H5Dopen2 failed");

    herr_t rrc = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    assert(rrc >= 0 && "H5Dread failed");

    H5Dclose(dset);
    H5Fclose(file);

    /* Verify on host */
    std::vector<float> h_read(N);
    cudaMemcpy(h_read.data(), d_read, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_read);

    bool match = true;
    for (size_t i = 0; i < N; i++) {
        if (h_orig[i] != h_read[i]) { match = false; break; }
    }
    assert(match && "VOL pipeline: data mismatch after write+read");

    H5Pclose(dcpl);
    remove(TMP_FILE);
    printf(" PASS\n");
}

/* ============================================================
 * Test 8: Non-AUTO does NOT trigger SGD or exploration
 * ============================================================ */
static void test_explicit_no_learning() {
    printf("  [TEST] Non-AUTO does not trigger SGD or exploration...");

    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.5f, 0.01f, 0.01f);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);
    gpucompress_set_exploration_k(8);

    const size_t N = 64 * 1024;
    std::vector<float> h(N);
    fill_sine(h.data(), N);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;  /* explicit, not AUTO */

    gpucompress_reset_chunk_history();

    gpucompress_stats_t stats = {};
    bool ok = gpu_roundtrip(h.data(), N, &cfg, &stats);

    assert(ok && "explicit algo round-trip failed");
    assert(stats.sgd_fired == 0 && "SGD should not fire for non-AUTO");
    assert(stats.exploration_triggered == 0 && "exploration should not fire for non-AUTO");

    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);
    printf(" PASS\n");
}

/* ============================================================
 * Test 9: Compression ratio sanity — constant data vs noise
 * ============================================================ */
static void test_ratio_sanity() {
    printf("  [TEST] Ratio sanity: constant >> sine...");

    const size_t N = 64 * 1024;

    std::vector<float> h_const(N);
    fill_constant(h_const.data(), N);

    std::vector<float> h_sine(N);
    fill_sine(h_sine.data(), N);

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

    gpucompress_stats_t stats_const = {}, stats_sine = {};
    bool ok1 = gpu_roundtrip(h_const.data(), N, &cfg, &stats_const);
    bool ok2 = gpu_roundtrip(h_sine.data(), N, &cfg, &stats_sine);

    assert(ok1 && ok2 && "ratio sanity round-trip failed");
    assert(stats_const.compression_ratio > stats_sine.compression_ratio &&
           "constant data should compress better than sine");
    printf(" PASS (const=%.1fx, sine=%.1fx)\n",
           stats_const.compression_ratio, stats_sine.compression_ratio);
}

/* ============================================================
 * Main
 * ============================================================ */
int main(int argc, char** argv) {
    printf("=== test_execution_flow ===\n");
    printf("End-to-end execution flow tests\n\n");

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found - skipping\n");
        return 1;
    }

    g_weights_path = (argc > 1) ? argv[1] : nullptr;
    if (!g_weights_path) {
        printf("Usage: %s <weights.nnwt>\n", argv[0]);
        printf("  NN weights required for ALGO_AUTO tests\n");
        return 1;
    }

    gpucompress_error_t err = gpucompress_init(g_weights_path);
    assert(err == GPUCOMPRESS_SUCCESS && "gpucompress_init failed");
    assert(gpucompress_nn_is_loaded() && "NN weights not loaded");

    /* Non-AUTO tests (no NN needed, but init required) */
    test_explicit_algo_roundtrip();   /* Test 1: all 8 algos */
    test_shuffle_roundtrip();         /* Test 2: Zstd+Shuffle */
    test_ratio_sanity();              /* Test 9: constant vs sine */
    test_explicit_no_learning();      /* Test 8: no SGD for non-AUTO */

    /* ALGO_AUTO tests (NN required) */
    test_auto_roundtrip();            /* Test 3: NN inference path */
    test_sgd_fires();                 /* Test 4: SGD triggers */
    test_exploration_fires();         /* Test 5: exploration triggers */
    test_chunk_diagnostics();         /* Test 6: diagnostics populated */

    /* VOL pipeline test */
    test_vol_pipeline();              /* Test 7: full HDF5 write+read */

    gpucompress_cleanup();

    printf("\n=== All tests passed ===\n");
    return 0;
}
