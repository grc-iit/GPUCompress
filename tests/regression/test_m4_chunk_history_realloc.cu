/**
 * test_m4_chunk_history_realloc.cu
 *
 * M-4: Chunk history realloc failure causes silent data loss —
 *      g_chunk_history_count incremented but record dropped.
 *
 * Test strategy:
 *   1. Compress many chunks and verify history count matches
 *   2. Verify history records contain valid data
 *   3. Reset history and verify it clears properly
 *   4. Stress test with many compressions to trigger realloc growth
 *
 * Run: ./test_m4_chunk_history_realloc
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
    printf("=== M-4: Chunk history realloc / silent data loss ===\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* Prepare test data */
    const size_t DATA_SIZE = 4096;
    float* h_data = (float*)malloc(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++) {
        h_data[i] = (float)i * 0.1f;
    }

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* h_output = malloc(max_out);

    /* ---- Test 1: History grows with compressions ---- */
    printf("--- Test 1: History count matches compression count ---\n");
    {
        gpucompress_reset_chunk_history();
        const int NUM_COMPRESS = 50;
        int success_count = 0;

        for (int i = 0; i < NUM_COMPRESS; i++) {
            size_t compressed_size = max_out;
            err = gpucompress_compress(h_data, DATA_SIZE, h_output, &compressed_size, NULL, NULL);
            if (err == GPUCOMPRESS_SUCCESS) {
                success_count++;
            }
        }

        int history_count = gpucompress_get_chunk_history_count();

        printf("  Compressions succeeded: %d, history count: %d\n",
               success_count, history_count);

        if (history_count == success_count) {
            PASS("history count matches compression count");
        } else if (history_count < success_count) {
            printf("  FAIL: %d records lost (realloc may have failed)\n",
                   success_count - history_count);
            g_fail++;
        } else {
            FAIL("history count exceeds compression count (unexpected)");
        }
    }

    /* ---- Test 2: History records contain valid nn_action values ---- */
    printf("\n--- Test 2: History records contain valid data ---\n");
    {
        int count = gpucompress_get_chunk_history_count();

        int invalid = 0;
        for (int i = 0; i < count; i++) {
            gpucompress_chunk_diag_t diag;
            if (gpucompress_get_chunk_diag(i, &diag) == 0) {
                /* nn_action: -1 = NN not loaded, 0-31 = valid action */
                if (diag.nn_action < -1 || diag.nn_action > 31) {
                    invalid++;
                }
            }
        }

        if (invalid == 0) {
            PASS("all history records have valid nn_action");
        } else {
            printf("  %d/%d records have invalid nn_action\n", invalid, count);
            FAIL("invalid nn_action values in history");
        }
    }

    /* ---- Test 3: Reset clears history ---- */
    printf("\n--- Test 3: History reset ---\n");
    {
        gpucompress_reset_chunk_history();
        int count = gpucompress_get_chunk_history_count();

        if (count == 0) {
            PASS("history cleared after reset");
        } else {
            FAIL("history not cleared after reset");
        }
    }

    /* ---- Test 4: Stress test — trigger realloc growth ---- */
    printf("\n--- Test 4: Stress test — exceed initial capacity (4096) ---\n");
    {
        gpucompress_reset_chunk_history();
        const int NUM_COMPRESS = 5000;  /* > 4096 initial capacity */
        int success_count = 0;

        for (int i = 0; i < NUM_COMPRESS; i++) {
            size_t compressed_size = max_out;
            err = gpucompress_compress(h_data, DATA_SIZE, h_output, &compressed_size, NULL, NULL);
            if (err == GPUCOMPRESS_SUCCESS) {
                success_count++;
            }
        }

        int history_count = gpucompress_get_chunk_history_count();

        printf("  Compressed %d chunks (initial cap=4096), history has %d records\n",
               success_count, history_count);

        if (history_count == success_count) {
            PASS("realloc growth succeeded — all records preserved");
        } else {
            printf("  Lost %d records during realloc growth\n",
                   success_count - history_count);
            FAIL("records lost during realloc (M-4 bug)");
        }
    }

    free(h_data);
    free(h_output);
    gpucompress_cleanup();

    /* ---- Summary ---- */
    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
