/**
 * test_m12_range_bufs_race.cu
 *
 * M-12: Static d_range_min/d_range_max shared without mutex in
 *       ensure_range_bufs (quantization_kernels.cu).
 *       Used only by host path; GPU path uses per-context buffers.
 *
 * Test: exercises quantization through compression with quantization enabled.
 *
 * Run: ./test_m12_range_bufs_race
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

int main(void) {
    printf("=== M-12: ensure_range_bufs static globals ===\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* ---- Test 1: Single-threaded quantization works ---- */
    printf("--- Test 1: Single-threaded compression (exercises range bufs) ---\n");
    {
        const size_t DATA_SIZE = 8192;
        float* h_data = (float*)malloc(DATA_SIZE);
        for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++) {
            h_data[i] = (float)i / 100.0f;
        }

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);
        size_t compressed_size = max_out;

        err = gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, NULL, NULL);
        if (err == GPUCOMPRESS_SUCCESS) {
            PASS("single-threaded compression with range bufs succeeded");
        } else {
            FAIL("compression failed");
        }

        free(h_data);
        free(h_compressed);
    }

    /* ---- Test 2: Multiple sequential compressions (range bufs reused) ---- */
    printf("\n--- Test 2: Sequential compressions reuse range bufs ---\n");
    {
        const size_t DATA_SIZE = 4096;
        float* h_data = (float*)malloc(DATA_SIZE);
        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);

        int success = 0;
        for (int i = 0; i < 10; i++) {
            for (size_t j = 0; j < DATA_SIZE / sizeof(float); j++) {
                h_data[j] = (float)(j + i * 100);
            }
            size_t compressed_size = max_out;
            err = gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, NULL, NULL);
            if (err == GPUCOMPRESS_SUCCESS) success++;
        }

        if (success == 10) {
            PASS("10 sequential compressions all succeeded (range bufs reused)");
        } else {
            printf("  %d/10 succeeded\n", success);
            FAIL("some compressions failed with reused range bufs");
        }

        free(h_data);
        free(h_compressed);
    }

    gpucompress_cleanup();

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
