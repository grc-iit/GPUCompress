/**
 * test_m13_compute_range_errors.cu
 *
 * M-13: compute_data_range ignores cudaMemcpyAsync/kernel errors —
 *       silent wrong quantization. Function returns 0 unconditionally.
 *
 * Test: verify quantization produces correct results (range was computed right).
 *
 * Run: ./test_m13_compute_range_errors
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

int main(void) {
    printf("=== M-13: compute_data_range error handling ===\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* ---- Test 1: Compress/decompress round-trip (exercises compute_data_range) ---- */
    printf("--- Test 1: Round-trip verifies correct data range computation ---\n");
    {
        const size_t N = 2048;
        const size_t DATA_SIZE = N * sizeof(float);
        float* h_data = (float*)malloc(DATA_SIZE);

        /* Data with known range: [-100.0, 100.0] */
        for (size_t i = 0; i < N; i++) {
            h_data[i] = -100.0f + 200.0f * ((float)i / (float)(N - 1));
        }

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);
        size_t compressed_size = max_out;

        err = gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, NULL, NULL);
        if (err != GPUCOMPRESS_SUCCESS) {
            printf("  SKIP: compression failed (%d)\n", err);
            free(h_data);
            free(h_compressed);
            gpucompress_cleanup();
            return 1;
        }

        float* h_decompressed = (float*)malloc(DATA_SIZE);
        size_t decompressed_size = DATA_SIZE;
        err = gpucompress_decompress(h_compressed, compressed_size,
                                     h_decompressed, &decompressed_size);

        if (err == GPUCOMPRESS_SUCCESS && decompressed_size == DATA_SIZE) {
            /* Check max absolute error */
            float max_err = 0.0f;
            for (size_t i = 0; i < N; i++) {
                float e = fabsf(h_data[i] - h_decompressed[i]);
                if (e > max_err) max_err = e;
            }
            printf("  Max absolute error: %e\n", max_err);

            /* If compute_data_range failed silently, min/max would be
               FLT_MAX/-FLT_MAX, causing terrible quantization error */
            if (max_err < 10.0f) {
                PASS("round-trip error reasonable (data range was correct)");
            } else {
                FAIL("excessive error — compute_data_range may have returned wrong range");
            }
        } else {
            printf("  decompression err=%d, size=%zu\n", err, decompressed_size);
            FAIL("decompression failed");
        }

        free(h_data);
        free(h_compressed);
        free(h_decompressed);
    }

    /* ---- Test 2: Edge case — uniform data (min == max) ---- */
    printf("\n--- Test 2: Uniform data (range = 0) ---\n");
    {
        const size_t N = 1024;
        const size_t DATA_SIZE = N * sizeof(float);
        float* h_data = (float*)malloc(DATA_SIZE);
        for (size_t i = 0; i < N; i++) {
            h_data[i] = 42.0f;
        }

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);
        size_t compressed_size = max_out;

        err = gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, NULL, NULL);
        if (err == GPUCOMPRESS_SUCCESS) {
            PASS("uniform data compressed without crash (range=0 handled)");
        } else {
            FAIL("uniform data compression failed");
        }

        free(h_data);
        free(h_compressed);
    }

    gpucompress_cleanup();

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
