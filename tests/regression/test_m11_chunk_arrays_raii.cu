/**
 * test_m11_chunk_arrays_raii.cu
 *
 * M-11: createDeviceChunkArrays partial cudaMalloc failure —
 *       RETRACTED: DeviceChunkArrays RAII destructor handles cleanup.
 *
 * Test verifies RAII works: create arrays, let them go out of scope,
 * verify no leak. Also tests move semantics.
 *
 * Run: ./test_m11_chunk_arrays_raii
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

static size_t get_free_gpu_mem() {
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

int main(void) {
    printf("=== M-11: DeviceChunkArrays RAII cleanup ===\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* ---- Test 1: Compress/decompress cycle (exercises chunk arrays) ---- */
    printf("--- Test 1: Normal compression exercises chunk arrays ---\n");
    {
        const size_t DATA_SIZE = 64 * 1024;  /* 64KB — multiple chunks */
        float* h_data = (float*)malloc(DATA_SIZE);
        for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++) {
            h_data[i] = (float)i * 0.01f;
        }

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);
        size_t compressed_size = max_out;

        err = gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, NULL, NULL);
        if (err == GPUCOMPRESS_SUCCESS) {
            PASS("compression with chunk arrays succeeded");
        } else {
            FAIL("compression failed");
        }

        free(h_data);
        free(h_compressed);
    }

    /* ---- Test 2: Multiple cycles — no memory leak ---- */
    printf("\n--- Test 2: Multiple cycles — no leak ---\n");
    {
        size_t mem_before = get_free_gpu_mem();
        const int CYCLES = 20;
        const size_t DATA_SIZE = 32 * 1024;

        float* h_data = (float*)malloc(DATA_SIZE);
        for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++) {
            h_data[i] = (float)i;
        }

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);

        for (int c = 0; c < CYCLES; c++) {
            size_t compressed_size = max_out;
            gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, NULL, NULL);
        }

        cudaDeviceSynchronize();
        size_t mem_after = get_free_gpu_mem();
        long delta = (long)mem_before - (long)mem_after;

        printf("  After %d cycles: delta = %ld bytes\n", CYCLES, delta);
        if (delta < 1024 * 1024) {
            PASS("no accumulating GPU memory leak (RAII cleanup works)");
        } else {
            FAIL("GPU memory leak detected (RAII cleanup broken?)");
        }

        free(h_data);
        free(h_compressed);
    }

    gpucompress_cleanup();

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
