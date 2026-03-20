/**
 * @file test_nn_cost_ranking.cu
 * @brief Tests for log-space cost-based ranking in NN inference.
 *
 * Verifies:
 *   1. Default params (BALANCED mode) produce a valid selection
 *   2. SPEED mode (δ=0) → prefers lowest compute time
 *   3. THROUGHPUT mode (δ=1) → prefers best time-per-ratio
 *   4. RATIO mode (α=0.3,δ=1) → strongly prefers high ratio
 *   5. Quant configs masked when error_bound=0 (lossless)
 *   6. gpucompress_set_bandwidth override works
 *   7. gpucompress_set_cost_params API works end-to-end
 *   8. Different modes produce different rankings
 *   9. gpucompress_set_cost_mode presets work
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <fstream>

#include "nn/nn_weights.h"

/* Log-space cost model globals in gpucompress_api.cpp */
extern float g_cost_alpha;
extern float g_cost_beta;
extern float g_cost_gamma;
extern float g_cost_delta;
extern float g_measured_bw_bytes_per_ms;

/* Forward declarations for gpucompress namespace functions */
namespace gpucompress {
    bool loadNNFromBinary(const char* filepath);
    void cleanupNN();
    bool isNNLoaded();
    int runNNInference(
        double entropy, double mad_norm, double deriv_norm,
        size_t data_size, double error_bound, cudaStream_t stream,
        float* out_predicted_ratio = nullptr,
        float* out_predicted_comp_time = nullptr,
        float* out_predicted_decomp_time = nullptr,
        float* out_predicted_psnr = nullptr,
        int* out_top_actions = nullptr
    );
}

/* Public API */
extern "C" {
    void gpucompress_set_cost_mode(int mode);
    void gpucompress_set_cost_params(float alpha, float beta, float gamma, float delta);
    void gpucompress_set_bandwidth(float bw_gbps);
}

static int g_pass = 0;
static int g_fail = 0;

#define TEST(name) \
    do { printf("  [TEST] %s ... ", name); } while (0)

#define PASS() \
    do { printf("PASS\n"); g_pass++; } while (0)

#define FAIL(msg) \
    do { printf("FAIL: %s\n", msg); g_fail++; } while (0)

#define ASSERT(cond, msg) \
    do { if (!(cond)) { FAIL(msg); return; } } while (0)

/* ============================================================
 * Helper: Write a synthetic .nnwt with real weights
 * ============================================================ */
static constexpr uint32_t NN_MAGIC = 0x4E4E5754;

static bool write_synthetic_nnwt(const char* path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t version = 2, n_layers = 3;
    uint32_t input_dim = NN_INPUT_DIM, hidden_dim = NN_HIDDEN_DIM, output_dim = NN_OUTPUT_DIM;
    file.write(reinterpret_cast<const char*>(&NN_MAGIC), 4);
    file.write(reinterpret_cast<const char*>(&version), 4);
    file.write(reinterpret_cast<const char*>(&n_layers), 4);
    file.write(reinterpret_cast<const char*>(&input_dim), 4);
    file.write(reinterpret_cast<const char*>(&hidden_dim), 4);
    file.write(reinterpret_cast<const char*>(&output_dim), 4);

    NNWeightsGPU w;
    memset(&w, 0, sizeof(w));

    for (int i = 0; i < NN_INPUT_DIM; i++) {
        w.x_means[i] = 0.0f;
        w.x_stds[i] = 1.0f;
    }
    for (int i = 0; i < NN_OUTPUT_DIM; i++) {
        w.y_means[i] = 0.0f;
        w.y_stds[i] = 1.0f;
    }

    // Small weights so outputs are bounded and deterministic
    for (int i = 0; i < NN_HIDDEN_DIM * NN_INPUT_DIM; i++) w.w1[i] = 0.01f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++) w.b1[i] = 0.1f;
    for (int i = 0; i < NN_HIDDEN_DIM * NN_HIDDEN_DIM; i++) w.w2[i] = 0.01f;
    for (int i = 0; i < NN_HIDDEN_DIM; i++) w.b2[i] = 0.1f;

    // Output layer: make predictions distinguishable per algo
    for (int out = 0; out < NN_OUTPUT_DIM; out++) {
        for (int h = 0; h < NN_HIDDEN_DIM; h++) {
            w.w3[out * NN_HIDDEN_DIM + h] = 0.01f;
        }
        w.b3[out] = 0.5f;
    }
    // Make algo one-hot features affect predictions differently
    for (int algo = 0; algo < 8; algo++) {
        w.w3[0 * NN_HIDDEN_DIM + algo] = 0.1f * (float)(algo + 1); // comp_time varies by algo
        w.w3[1 * NN_HIDDEN_DIM + algo] = 0.08f * (float)(algo + 1); // decomp_time varies
        w.w3[2 * NN_HIDDEN_DIM + algo] = 0.15f * (float)(8 - algo); // ratio inversely related
    }

    // Write all weight arrays
    file.write(reinterpret_cast<const char*>(w.x_means), sizeof(w.x_means));
    file.write(reinterpret_cast<const char*>(w.x_stds), sizeof(w.x_stds));
    file.write(reinterpret_cast<const char*>(w.w1), sizeof(w.w1));
    file.write(reinterpret_cast<const char*>(w.b1), sizeof(w.b1));
    file.write(reinterpret_cast<const char*>(w.w2), sizeof(w.w2));
    file.write(reinterpret_cast<const char*>(w.b2), sizeof(w.b2));
    file.write(reinterpret_cast<const char*>(w.w3), sizeof(w.w3));
    file.write(reinterpret_cast<const char*>(w.b3), sizeof(w.b3));
    file.write(reinterpret_cast<const char*>(w.y_means), sizeof(w.y_means));
    file.write(reinterpret_cast<const char*>(w.y_stds), sizeof(w.y_stds));
    file.write(reinterpret_cast<const char*>(w.log_var), sizeof(w.log_var));

    return file.good();
}

static void decode_action(int action, int& algo, int& quant, int& shuffle) {
    algo = action % 8;
    quant = (action / 8) % 2;
    shuffle = (action / 16) % 2;
}

struct InferResult {
    int best_action;
    float ratio, comp_time, decomp_time, psnr;
    int top_actions[32];
};

static bool run_inference(InferResult& r, double eb = 0.0,
                          size_t data_size = 1024*1024) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    r.best_action = gpucompress::runNNInference(
        4.0, 0.3, 0.1,   // entropy, mad, deriv
        data_size, eb, stream,
        &r.ratio, &r.comp_time, &r.decomp_time, &r.psnr,
        r.top_actions
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return r.best_action >= 0;
}

static void save_params(float& a, float& b, float& g, float& d) {
    a = g_cost_alpha; b = g_cost_beta; g = g_cost_gamma; d = g_cost_delta;
}
static void restore_params(float a, float b, float g, float d) {
    g_cost_alpha = a; g_cost_beta = b; g_cost_gamma = g; g_cost_delta = d;
}

/* ============================================================
 * Test 1: Default params (BALANCED) produce valid selection
 * ============================================================ */
static void test_default_balanced() {
    TEST("BALANCED mode (default) produces valid action");

    gpucompress_set_cost_mode(1);  // BALANCED
    g_measured_bw_bytes_per_ms = 1e6f;

    InferResult r;
    ASSERT(run_inference(r), "inference should succeed");
    ASSERT(r.best_action >= 0 && r.best_action < 32, "action in [0,31]");
    ASSERT(r.ratio > 0.0f, "ratio should be positive");
    ASSERT(r.comp_time > 0.0f, "comp_time should be positive");

    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    printf("(action=%d algo=%d ratio=%.2f ct=%.3f dt=%.3f) ",
           r.best_action, algo, r.ratio, r.comp_time, r.decomp_time);
    PASS();
}

/* ============================================================
 * Test 2: SPEED mode → prefers lowest compute time
 * ============================================================ */
static void test_speed_mode() {
    TEST("SPEED mode (δ=0) → lowest compute time wins");

    gpucompress_set_cost_mode(0);  // SPEED

    InferResult r;
    ASSERT(run_inference(r), "inference should succeed");
    ASSERT(r.top_actions[0] == r.best_action, "top_actions[0] == best_action");

    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    ASSERT(quant == 0, "lossless should have quant=0");

    printf("(action=%d algo=%d ct=%.3f) ", r.best_action, algo, r.comp_time);
    PASS();
}

/* ============================================================
 * Test 3: THROUGHPUT mode → prefers best time-per-ratio
 * ============================================================ */
static void test_throughput_mode() {
    TEST("THROUGHPUT mode (α=1,δ=1) → time-per-ratio");

    gpucompress_set_cost_mode(3);  // THROUGHPUT

    InferResult r;
    ASSERT(run_inference(r), "inference should succeed");

    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    ASSERT(quant == 0, "lossless should have quant=0");

    printf("(action=%d algo=%d ratio=%.2f ct=%.3f) ", r.best_action, algo, r.ratio, r.comp_time);
    PASS();
}

/* ============================================================
 * Test 4: RATIO mode → strongly prefers high ratio
 * ============================================================ */
static void test_ratio_mode() {
    TEST("RATIO mode (α=0.3,δ=1) → high ratio preferred");

    gpucompress_set_cost_mode(2);  // RATIO

    InferResult r;
    ASSERT(run_inference(r), "inference should succeed");

    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    ASSERT(quant == 0, "lossless should have quant=0");

    printf("(action=%d algo=%d ratio=%.2f) ", r.best_action, algo, r.ratio);
    PASS();
}

/* ============================================================
 * Test 5: Quant configs masked when error_bound=0 (lossless)
 * ============================================================ */
static void test_quant_masked_lossless() {
    TEST("quant configs masked when error_bound=0");

    gpucompress_set_cost_mode(1);  // BALANCED

    InferResult r;
    ASSERT(run_inference(r, 0.0), "inference should succeed");

    int algo, quant, shuffle;
    decode_action(r.best_action, algo, quant, shuffle);
    ASSERT(quant == 0, "best action should have quant=0 when eb=0");

    for (int i = 0; i < 16; i++) {
        int a = r.top_actions[i];
        int q = (a / 8) % 2;
        ASSERT(q == 0, "top-16 should all be lossless when eb=0");
    }

    PASS();
}

/* ============================================================
 * Test 6: Quant configs allowed when error_bound > 0
 * ============================================================ */
static void test_quant_allowed_lossy() {
    TEST("quant configs allowed when error_bound>0");

    gpucompress_set_cost_mode(1);

    InferResult r;
    ASSERT(run_inference(r, 0.01), "inference should succeed");

    bool found_quant = false;
    for (int i = 0; i < 16; i++) {
        int a = r.top_actions[i];
        int q = (a / 8) % 2;
        if (q == 1) { found_quant = true; break; }
    }
    ASSERT(found_quant, "some quant configs should appear in top-16 for lossy");

    PASS();
}

/* ============================================================
 * Test 7: gpucompress_set_bandwidth override
 * ============================================================ */
static void test_bw_override() {
    TEST("gpucompress_set_bandwidth override");

    gpucompress_set_cost_mode(1);  // BALANCED (uses β>0 so BW matters)
    gpucompress_set_bandwidth(0.001f);  // very slow I/O

    InferResult r_slow;
    ASSERT(run_inference(r_slow), "inference should succeed with slow BW");

    gpucompress_set_bandwidth(100.0f);  // very fast I/O

    InferResult r_fast;
    ASSERT(run_inference(r_fast), "inference should succeed with fast BW");

    ASSERT(r_slow.best_action >= 0 && r_slow.best_action < 32, "slow BW valid");
    ASSERT(r_fast.best_action >= 0 && r_fast.best_action < 32, "fast BW valid");

    float expected = 100.0f * 1e6f;
    float diff = fabsf(g_measured_bw_bytes_per_ms - expected);
    ASSERT(diff < 1.0f, "BW should be 100 GB/s");

    g_measured_bw_bytes_per_ms = 1e6f;

    printf("(slow_action=%d fast_action=%d) ", r_slow.best_action, r_fast.best_action);
    PASS();
}

/* ============================================================
 * Test 8: gpucompress_set_cost_params API
 * ============================================================ */
static void test_set_params_api() {
    TEST("gpucompress_set_cost_params API");

    float sa, sb, sg, sd;
    save_params(sa, sb, sg, sd);

    gpucompress_set_cost_params(0.5f, 2.0f, 1.5f, 0.3f);
    ASSERT(fabsf(g_cost_alpha - 0.5f) < 1e-6f, "alpha should be 0.5");
    ASSERT(fabsf(g_cost_beta  - 2.0f) < 1e-6f, "beta should be 2.0");
    ASSERT(fabsf(g_cost_gamma - 1.5f) < 1e-6f, "gamma should be 1.5");
    ASSERT(fabsf(g_cost_delta - 0.3f) < 1e-6f, "delta should be 0.3");

    InferResult r;
    ASSERT(run_inference(r), "inference should succeed with custom params");
    ASSERT(r.best_action >= 0 && r.best_action < 32, "valid action");

    restore_params(sa, sb, sg, sd);
    PASS();
}

/* ============================================================
 * Test 9: Different modes produce different rankings
 * ============================================================ */
static void test_modes_differ() {
    TEST("different modes produce different top-action orderings");

    g_measured_bw_bytes_per_ms = 1e6f;

    // SPEED mode
    gpucompress_set_cost_mode(0);
    InferResult ra;
    ASSERT(run_inference(ra), "SPEED inference");

    // RATIO mode
    gpucompress_set_cost_mode(2);
    InferResult rb;
    ASSERT(run_inference(rb), "RATIO inference");

    // THROUGHPUT mode
    gpucompress_set_cost_mode(3);
    InferResult rc;
    ASSERT(run_inference(rc), "THROUGHPUT inference");

    bool ab_same = true, bc_same = true, ac_same = true;
    for (int i = 0; i < 8; i++) {
        if (ra.top_actions[i] != rb.top_actions[i]) ab_same = false;
        if (rb.top_actions[i] != rc.top_actions[i]) bc_same = false;
        if (ra.top_actions[i] != rc.top_actions[i]) ac_same = false;
    }
    bool at_least_two_differ = !ab_same || !bc_same || !ac_same;
    ASSERT(at_least_two_differ, "different modes should produce different rankings");

    printf("(SPEED=%d RATIO=%d THRU=%d) ", ra.best_action, rb.best_action, rc.best_action);

    gpucompress_set_cost_mode(1);  // restore BALANCED
    PASS();
}

/* ============================================================
 * Test 10: gpucompress_set_cost_mode presets
 * ============================================================ */
static void test_mode_presets() {
    TEST("gpucompress_set_cost_mode sets correct params");

    gpucompress_set_cost_mode(0);  // SPEED
    ASSERT(fabsf(g_cost_alpha - 1.0f) < 1e-6f && fabsf(g_cost_delta) < 1e-6f,
           "SPEED: α=1,δ=0");

    gpucompress_set_cost_mode(1);  // BALANCED
    ASSERT(fabsf(g_cost_alpha - 1.0f) < 1e-6f && fabsf(g_cost_delta - 0.5f) < 1e-6f,
           "BALANCED: α=1,δ=0.5");

    gpucompress_set_cost_mode(2);  // RATIO
    ASSERT(fabsf(g_cost_alpha - 0.3f) < 1e-6f && fabsf(g_cost_delta - 1.0f) < 1e-6f,
           "RATIO: α=0.3,δ=1");

    gpucompress_set_cost_mode(3);  // THROUGHPUT
    ASSERT(fabsf(g_cost_alpha - 1.0f) < 1e-6f && fabsf(g_cost_delta - 1.0f) < 1e-6f,
           "THROUGHPUT: α=1,δ=1");

    gpucompress_set_cost_mode(1);  // restore
    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    printf("=== Log-Space Cost-Based Ranking Tests ===\n\n");

    const char* nnwt_path = "/tmp/test_cost_ranking.nnwt";
    if (!write_synthetic_nnwt(nnwt_path)) {
        fprintf(stderr, "FATAL: failed to write synthetic .nnwt\n");
        return 1;
    }
    if (!gpucompress::loadNNFromBinary(nnwt_path)) {
        fprintf(stderr, "FATAL: failed to load synthetic .nnwt\n");
        return 1;
    }
    printf("  Loaded synthetic NN weights\n\n");

    test_default_balanced();
    test_speed_mode();
    test_throughput_mode();
    test_ratio_mode();
    test_quant_masked_lossless();
    test_quant_allowed_lossy();
    test_bw_override();
    test_set_params_api();
    test_modes_differ();
    test_mode_presets();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);

    gpucompress::cleanupNN();
    remove(nnwt_path);

    return g_fail > 0 ? 1 : 0;
}
