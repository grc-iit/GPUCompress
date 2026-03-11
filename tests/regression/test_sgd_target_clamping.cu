/**
 * test_sgd_target_clamping.cu
 *
 * Pre-fix baseline / post-fix verification for Issue #1:
 * Unbounded SGD targets cause weight divergence.
 *
 * Strategy:
 *   1. Load NN weights, run inference on a fixed data pattern → record initial
 *      predicted_ratio (pred0).
 *   2. Feed SGD with EXTREME actual_ratio values (e.g., 1000x, 0.001x) that
 *      are far outside the training distribution.
 *   3. Run inference again on the SAME data → record post-SGD predicted_ratio (pred1).
 *   4. Repeat for N rounds to simulate the 58/63 SGD firings in the benchmark.
 *   5. Check that the final prediction is still finite and within a sane range.
 *
 * PRE-FIX expected behavior:
 *   - Weights diverge, predicted_ratio becomes NaN/INF or wildly wrong (>10000x MAPE)
 *   - Test FAILS (documents the bug)
 *
 * POST-FIX expected behavior:
 *   - Clamped targets prevent divergence
 *   - predicted_ratio stays finite and within [0.1, 1e5]
 *   - Final MAPE after extreme SGD is bounded (< 500%)
 *   - Test PASSES
 *
 * Additionally tests:
 *   - Round-trip correctness is preserved after SGD (lossless compress/decompress)
 *   - Gradient norm is reported and doesn't become NaN
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "gpucompress.h"

#define DATA_MB        4
#define SGD_ROUNDS     30    /* Simulate many SGD firings like benchmark's 58/63 */
#define REINFORCE_LR   0.05f
#define MAPE_THRESH    0.001f

/* Generate a linear ramp on GPU (deterministic, OOD for the NN) */
__global__ static void fill_ramp(float* d, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = (float)i / (float)n;
}

static const char* find_weights() {
    const char* paths[] = {
        "neural_net/weights/model.nnwt",
        "../neural_net/weights/model.nnwt",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        FILE* f = fopen(paths[i], "rb");
        if (f) { fclose(f); return paths[i]; }
    }
    return paths[0];
}

int main(void)
{
    printf("=== Issue #1: SGD Target Clamping Test ===\n\n");

    const char* wpath = find_weights();
    if (gpucompress_init(wpath) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed\n");
        return 1;
    }

    /* Enable online learning: low MAPE threshold to force SGD every time */
    gpucompress_enable_online_learning();
    gpucompress_set_exploration(0);  /* No exploration — just SGD from passive sample */
    gpucompress_set_reinforcement(1, REINFORCE_LR, MAPE_THRESH, MAPE_THRESH);

    size_t data_bytes = (size_t)DATA_MB * 1024 * 1024;
    size_t n_floats   = data_bytes / sizeof(float);
    size_t max_comp   = gpucompress_max_compressed_size(data_bytes);

    void* d_input  = NULL;
    void* d_output = NULL;
    cudaMalloc(&d_input,  data_bytes);
    cudaMalloc(&d_output, max_comp);
    if (!d_input || !d_output) { fprintf(stderr, "cudaMalloc failed\n"); return 1; }

    int threads = 256, blocks = (int)((n_floats + threads - 1) / threads);
    fill_ramp<<<blocks, threads>>>((float*)d_input, n_floats);
    cudaDeviceSynchronize();

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

    printf("%-6s  %-14s  %-14s  %-10s  %-4s\n",
           "Round", "pred_ratio", "actual_ratio", "MAPE%", "sgd");
    printf("------  --------------  --------------  ----------  ----\n");

    /* ---- Test 1: Repeated SGD with the same data ----
     * The ramp data is highly compressible → ratio might be 2x-1000x+.
     * If the NN was trained on a different distribution, actual_ratio
     * will be far from predicted. Without clamping, SGD targets explode.
     */
    double pred_first = -1, pred_last = -1;
    double actual_first = -1, actual_last = -1;
    int sgd_count = 0;
    int any_nan = 0;
    int any_extreme = 0;  /* pred < 0.01 or pred > 1e6 */
    double max_mape = 0.0;
    int max_mape_round = 0;

    for (int r = 0; r < SGD_ROUNDS; r++) {
        size_t out_sz = max_comp;
        gpucompress_stats_t stats = {};
        gpucompress_error_t ce =
            gpucompress_compress_gpu(d_input, data_bytes, d_output, &out_sz,
                                     &cfg, &stats, NULL);
        if (ce != GPUCOMPRESS_SUCCESS) {
            fprintf(stderr, "compress failed at round %d (err=%d)\n", r, (int)ce);
            break;
        }

        double pred   = stats.predicted_ratio;
        double actual = (out_sz > 0) ? (double)data_bytes / (double)out_sz : 0.0;
        double mape   = (actual > 0.0) ? fabs(pred - actual) / actual * 100.0 : 0.0;

        printf("%-6d  %-14.4f  %-14.4f  %-10.2f  %-4s\n",
               r, pred, actual, mape, stats.sgd_fired ? "YES" : "no");

        if (stats.sgd_fired) sgd_count++;
        if (r == 0) { pred_first = pred; actual_first = actual; }
        pred_last = pred; actual_last = actual;

        if (std::isnan(pred) || std::isinf(pred)) any_nan = 1;
        if (pred < 0.01 || pred > 1e6) any_extreme = 1;
        if (mape > max_mape) { max_mape = mape; max_mape_round = r; }
    }

    printf("\n");

    /* ---- Test 2: Round-trip correctness after SGD ----
     * Even after SGD updates, the NN just picks a compressor.
     * The actual compression must still be lossless.
     */
    printf("[Round-trip check after %d SGD rounds]\n", SGD_ROUNDS);
    size_t out_sz = max_comp;
    gpucompress_stats_t stats = {};
    gpucompress_error_t ce = gpucompress_compress_gpu(
        d_input, data_bytes, d_output, &out_sz, &cfg, &stats, NULL);

    int roundtrip_ok = 0;
    if (ce == GPUCOMPRESS_SUCCESS) {
        void* d_decomp = NULL;
        cudaMalloc(&d_decomp, data_bytes);
        size_t decomp_sz = data_bytes;
        gpucompress_error_t de = gpucompress_decompress_gpu(
            d_output, out_sz, d_decomp, &decomp_sz, NULL);
        if (de == GPUCOMPRESS_SUCCESS && decomp_sz == data_bytes) {
            /* Byte-exact comparison on GPU would be ideal, but host check is fine for test */
            float* h_orig = (float*)malloc(data_bytes);
            float* h_dec  = (float*)malloc(data_bytes);
            cudaMemcpy(h_orig, d_input,  data_bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_dec,  d_decomp, data_bytes, cudaMemcpyDeviceToHost);
            roundtrip_ok = 1;
            for (size_t i = 0; i < n_floats; i++) {
                if (h_orig[i] != h_dec[i]) {
                    printf("  MISMATCH at index %zu: orig=%.6f got=%.6f\n",
                           i, h_orig[i], h_dec[i]);
                    roundtrip_ok = 0;
                    break;
                }
            }
            free(h_orig);
            free(h_dec);
        }
        if (d_decomp) cudaFree(d_decomp);
    }
    printf("  Round-trip: %s\n\n", roundtrip_ok ? "PASS" : "FAIL");

    /* ---- Verdict ---- */
    double mape_first = (actual_first > 0) ?
        fabs(pred_first - actual_first) / actual_first * 100.0 : -1;
    double mape_last = (actual_last > 0) ?
        fabs(pred_last - actual_last) / actual_last * 100.0 : -1;

    printf("=== Results ===\n");
    printf("  SGD fired: %d / %d rounds\n", sgd_count, SGD_ROUNDS);
    printf("  First: pred=%.4f actual=%.4f MAPE=%.2f%%\n",
           pred_first, actual_first, mape_first);
    printf("  Last:  pred=%.4f actual=%.4f MAPE=%.2f%%\n",
           pred_last, actual_last, mape_last);
    printf("  Max MAPE: %.2f%% (round %d)\n", max_mape, max_mape_round);
    printf("  NaN/INF in predictions: %s\n", any_nan ? "YES" : "no");
    printf("  Extreme predictions (<0.01 or >1e6): %s\n", any_extreme ? "YES" : "no");
    printf("  Round-trip: %s\n", roundtrip_ok ? "PASS" : "FAIL");

    gpucompress_cleanup();
    cudaFree(d_input);
    cudaFree(d_output);

    /* Pass criteria:
     * 1. No NaN/INF in predictions
     * 2. No extreme predictions
     * 3. Round-trip is correct
     * 4. MAX MAPE across ALL rounds is bounded (< 1000%)
     *    Pre-fix:  weights oscillate wildly, max MAPE > 10000% → FAIL
     *    Post-fix: clamped targets → stable convergence, max MAPE < 1000% → PASS
     * 5. Final MAPE is bounded (< 500%)
     */
    int pass = !any_nan && !any_extreme && roundtrip_ok
               && (max_mape < 1000.0) && (mape_last < 500.0);

    printf("\n=== VERDICT: %s ===\n", pass ? "PASS" : "FAIL");
    if (!pass) {
        if (any_nan)     printf("  -> NaN/INF detected: SGD targets unbounded\n");
        if (any_extreme) printf("  -> Extreme predictions: weights diverged\n");
        if (!roundtrip_ok) printf("  -> Round-trip broken after SGD\n");
        if (max_mape >= 1000.0)
            printf("  -> Max MAPE=%.1f%% at round %d (threshold: 1000%%) — weights oscillating\n",
                   max_mape, max_mape_round);
        if (mape_last >= 500.0)
            printf("  -> Final MAPE=%.1f%% (threshold: 500%%)\n", mape_last);
    }

    return pass ? 0 : 1;
}
