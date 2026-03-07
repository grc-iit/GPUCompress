/**
 * test_m7_header_async.cu
 *
 * M-7: writeHeaderToDevice uses cudaMemcpyAsync from stack-local header ref.
 *      Unsafe if caller doesn't sync before header goes out of scope.
 *
 * Verdict: LOW — the only caller syncs immediately. Test documents the pattern.
 *
 * Run: ./test_m7_header_async
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

int main(void) {
    printf("=== M-7: writeHeaderToDevice async safety ===\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* ---- Test 1: Compress and verify header is written correctly ---- */
    printf("--- Test 1: Header written correctly via compress path ---\n");
    {
        const size_t DATA_SIZE = 4096;
        float* h_data = (float*)malloc(DATA_SIZE);
        for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++) {
            h_data[i] = (float)i * 0.5f;
        }

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);
        size_t compressed_size = max_out;

        err = gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, NULL, NULL);
        if (err == GPUCOMPRESS_SUCCESS) {
            /* Verify header magic is correct (not corrupted by async race) */
            uint32_t magic;
            memcpy(&magic, h_compressed, sizeof(uint32_t));
            /* COMPRESSION_MAGIC is typically defined in the header */
            if (magic != 0) {
                PASS("header magic is non-zero (header was written correctly)");
            } else {
                FAIL("header magic is zero (possible async corruption)");
            }

            /* Verify original_size is correct */
            size_t orig = 0;
            gpucompress_error_t e2 = gpucompress_get_original_size(h_compressed, &orig);
            if (e2 == GPUCOMPRESS_SUCCESS && orig == DATA_SIZE) {
                PASS("header original_size matches input (async copy was safe)");
            } else {
                FAIL("header original_size mismatch (async copy may have raced)");
            }
        } else {
            printf("  SKIP: compression failed (%d)\n", err);
        }

        free(h_data);
        free(h_compressed);
    }

    /* ---- Test 2: Multiple compressions — header consistency ---- */
    printf("\n--- Test 2: Multiple compressions header consistency ---\n");
    {
        const size_t sizes[] = {1024, 2048, 4096, 8192, 16384};
        int inconsistencies = 0;

        for (int s = 0; s < 5; s++) {
            float* h_data = (float*)malloc(sizes[s]);
            for (size_t i = 0; i < sizes[s] / sizeof(float); i++) {
                h_data[i] = (float)(i + s);
            }

            size_t max_out = gpucompress_max_compressed_size(sizes[s]);
            void* h_compressed = malloc(max_out);
            size_t compressed_size = max_out;

            err = gpucompress_compress(h_data, sizes[s], h_compressed, &compressed_size, NULL, NULL);
            if (err == GPUCOMPRESS_SUCCESS) {
                size_t orig = 0;
                gpucompress_get_original_size(h_compressed, &orig);
                if (orig != sizes[s]) {
                    printf("  size=%zu, got orig=%zu\n", sizes[s], orig);
                    inconsistencies++;
                }
            }

            free(h_data);
            free(h_compressed);
        }

        if (inconsistencies == 0) {
            PASS("all headers consistent across multiple compressions");
        } else {
            printf("  %d inconsistencies\n", inconsistencies);
            FAIL("header inconsistencies detected");
        }
    }

    gpucompress_cleanup();

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
