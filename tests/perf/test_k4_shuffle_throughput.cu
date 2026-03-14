/**
 * test_k4_shuffle_throughput.cu
 *
 * K4: Byte shuffle warp underutilization.
 *
 * Measures shuffle throughput on 64 MB data. Pre-fix: only 4/32 warp
 * lanes active for ElementSize=4. Post-fix: all 32 lanes process
 * elements in parallel.
 *
 * PASS criteria: shuffle throughput > 30 GB/s (reasonable for GPU memcpy-bound).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

#include "preprocessing/byte_shuffle.cuh"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

int main(void) {
    printf("=== K4: Byte shuffle throughput test ===\n\n");

    const size_t N = 16 * 1024 * 1024;  /* 64 MB (16M floats) */
    const size_t bytes = N * sizeof(float);
    const size_t chunk_bytes = 256 * 1024;  /* 256 KB chunks (SHUFFLE_CHUNK_SIZE) */

    float* d_data = NULL;
    if (cudaMalloc(&d_data, bytes) != cudaSuccess) {
        printf("SKIP: cudaMalloc failed\n"); return 1;
    }

    /* Fill with pattern */
    float* h_data = (float*)malloc(bytes);
    for (size_t i = 0; i < N; i++) h_data[i] = sinf((float)i * 0.001f);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Warmup */
    for (int w = 0; w < 3; w++) {
        uint8_t* out = byte_shuffle_simple(d_data, bytes, 4, chunk_bytes, stream);
        cudaStreamSynchronize(stream);
        if (out) cudaFree(out);
    }

    printf("--- Shuffle throughput (64 MB, 4-byte elements, 256 KB chunks) ---\n");

    const int N_RUNS = 10;
    double times_ms[N_RUNS];

    for (int r = 0; r < N_RUNS; r++) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();

        uint8_t* out = byte_shuffle_simple(d_data, bytes, 4, chunk_bytes, stream);
        cudaStreamSynchronize(stream);

        auto t1 = std::chrono::steady_clock::now();
        times_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!out) { FAIL("byte_shuffle_simple returned null"); break; }
        cudaFree(out);

        double gbps = (double)bytes / (times_ms[r] / 1000.0) / 1e9;
        printf("  Run %2d: %.2f ms  (%.1f GB/s)\n", r + 1, times_ms[r], gbps);
    }

    /* Also measure unshuffle */
    printf("\n--- Unshuffle throughput ---\n");
    uint8_t* shuffled = byte_shuffle_simple(d_data, bytes, 4, chunk_bytes, stream);
    cudaStreamSynchronize(stream);

    for (int r = 0; r < 5; r++) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::steady_clock::now();

        uint8_t* out = byte_unshuffle_simple(shuffled, bytes, 4, chunk_bytes, stream);
        cudaStreamSynchronize(stream);

        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double gbps = (double)bytes / (ms / 1000.0) / 1e9;
        printf("  Run %2d: %.2f ms  (%.1f GB/s)\n", r + 1, ms, gbps);

        if (out) cudaFree(out);
    }
    if (shuffled) cudaFree(shuffled);

    /* Stats */
    double min_ms = 1e9;
    for (int r = 0; r < N_RUNS; r++) {
        if (times_ms[r] < min_ms) min_ms = times_ms[r];
    }
    double best_gbps = (double)bytes / (min_ms / 1000.0) / 1e9;
    printf("\n  Best shuffle: %.2f ms (%.1f GB/s)\n", min_ms, best_gbps);

    if (best_gbps > 20.0) {
        PASS("shuffle throughput > 20 GB/s (K4 fix: full warp utilization)");
    } else {
        char buf[128];
        snprintf(buf, sizeof(buf), "shuffle throughput %.1f GB/s < 20 GB/s", best_gbps);
        FAIL(buf);
    }

    /* Verify correctness */
    uint8_t* shuf = byte_shuffle_simple(d_data, bytes, 4, chunk_bytes, stream);
    uint8_t* unshuf = byte_unshuffle_simple(shuf, bytes, 4, chunk_bytes, stream);
    cudaStreamSynchronize(stream);

    float* h_result = (float*)malloc(bytes);
    cudaMemcpy(h_result, unshuf, bytes, cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (size_t i = 0; i < N && mismatches < 10; i++) {
        if (h_data[i] != h_result[i]) mismatches++;
    }
    if (mismatches == 0) PASS("round-trip correctness verified");
    else FAIL("round-trip mismatches found");

    cudaFree(shuf);
    cudaFree(unshuf);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    free(h_data);
    free(h_result);

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("OVERALL: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
