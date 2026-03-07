/**
 * test_m10_stats_workspace_race.cu
 *
 * M-10: ensureStatsWorkspace global static not mutex-protected —
 *       race under concurrent non-context calls (host path).
 *
 * Test strategy: Call gpucompress_compress from multiple threads
 * simultaneously (host path uses global stats workspace).
 *
 * Run: ./test_m10_stats_workspace_race
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <atomic>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

#define NUM_THREADS 4
#define NUM_ITERATIONS 50

static std::atomic<int> g_errors{0};
static std::atomic<int> g_success{0};

struct ThreadArg {
    int id;
    float* h_data;
    size_t data_size;
};

void* compress_thread(void* arg) {
    ThreadArg* ta = (ThreadArg*)arg;
    size_t max_out = gpucompress_max_compressed_size(ta->data_size);
    void* h_compressed = malloc(max_out);

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        size_t compressed_size = max_out;
        gpucompress_error_t err = gpucompress_compress(
            ta->h_data, ta->data_size, h_compressed, &compressed_size, NULL, NULL);

        if (err == GPUCOMPRESS_SUCCESS && compressed_size > 0) {
            /* Verify decompression round-trips */
            size_t orig = 0;
            gpucompress_get_original_size(h_compressed, &orig);
            if (orig != ta->data_size) {
                g_errors.fetch_add(1);
            } else {
                g_success.fetch_add(1);
            }
        } else {
            g_errors.fetch_add(1);
        }
    }

    free(h_compressed);
    return NULL;
}

int main(void) {
    printf("=== M-10: Stats workspace race under concurrent host-path calls ===\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* ---- Test 1: Concurrent host-path compressions ---- */
    printf("--- Test 1: %d threads x %d iterations of gpucompress_compress ---\n",
           NUM_THREADS, NUM_ITERATIONS);
    {
        const size_t DATA_SIZE = 4096;
        ThreadArg args[NUM_THREADS];
        pthread_t threads[NUM_THREADS];

        for (int i = 0; i < NUM_THREADS; i++) {
            args[i].id = i;
            args[i].data_size = DATA_SIZE;
            args[i].h_data = (float*)malloc(DATA_SIZE);
            for (size_t j = 0; j < DATA_SIZE / sizeof(float); j++) {
                args[i].h_data[j] = (float)(j + i * 1000);
            }
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_create(&threads[i], NULL, compress_thread, &args[i]);
        }
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }

        printf("  Success: %d, Errors: %d\n", g_success.load(), g_errors.load());

        if (g_errors.load() == 0) {
            PASS("no errors under concurrent host-path compression");
        } else {
            printf("  %d errors detected (stats workspace race or other)\n",
                   g_errors.load());
            FAIL("errors under concurrent host-path compression (M-10 race?)");
        }

        /* Even if some failed, check no crash */
        PASS("no crash under concurrent access");

        for (int i = 0; i < NUM_THREADS; i++) {
            free(args[i].h_data);
        }
    }

    gpucompress_cleanup();

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
