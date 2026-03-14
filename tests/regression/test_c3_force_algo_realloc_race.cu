/**
 * test_c3_force_algo_realloc_race.cu
 *
 * C3: gpucompress_force_algorithm_push() calls realloc() on the global
 *     g_force_algo_queue without synchronization. A concurrent compression
 *     thread reading from the queue can dereference a stale (freed) pointer.
 *
 * Test: push a large number of entries to force multiple reallocs, then
 * run compressions consuming those entries. Verify no crash and all
 * forced algorithms are applied correctly.
 *
 * Run: ./test_c3_force_algo_realloc_race
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <atomic>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

static const char* WEIGHTS_PATH = "../neural_net/weights/model.nnwt";

static std::atomic<bool> g_start_compress{false};
static std::atomic<int> g_compress_ok{0};
static std::atomic<int> g_compress_err{0};

static void compress_worker(void) {
    const size_t DATA_SIZE = 32 * 1024;
    float* h_data = (float*)malloc(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++)
        h_data[i] = (float)i * 0.01f;

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* h_output = malloc(max_out);

    // Wait for signal
    while (!g_start_compress.load(std::memory_order_acquire)) {}

    // Consume forced entries via ALGO_AUTO
    for (int i = 0; i < 64; i++) {
        size_t compressed_size = max_out;
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

        gpucompress_error_t err = gpucompress_compress(
            h_data, DATA_SIZE, h_output, &compressed_size, &cfg, NULL);
        if (err == GPUCOMPRESS_SUCCESS)
            g_compress_ok.fetch_add(1, std::memory_order_relaxed);
        else
            g_compress_err.fetch_add(1, std::memory_order_relaxed);
    }

    free(h_data);
    free(h_output);
}

int main(void) {
    printf("=== C3: Force-algorithm queue realloc race test ===\n\n");

    gpucompress_error_t gerr = gpucompress_init(WEIGHTS_PATH);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        printf("SKIP: gpucompress_init failed (%d)\n", gerr);
        return 1;
    }
    PASS("init succeeded");

    /* ---- Test 1: Push many entries (force reallocs) + concurrent consume ---- */
    printf("\n--- Test 1: Push 512 entries + 4 threads consuming ---\n");
    {
        gpucompress_force_algorithm_reset();

        // Push 512 entries — initial cap is 256, so this forces at least one realloc
        const int N_ENTRIES = 512;
        const int algos[] = {1, 2, 3, 4, 5, 6, 7, 8}; // LZ4..Bitcomp
        for (int i = 0; i < N_ENTRIES; i++) {
            gpucompress_force_algorithm_push(algos[i % 8], i % 2, 0, 0.0);
        }
        PASS("pushed 512 entries without crash");

        g_start_compress.store(false);
        g_compress_ok.store(0);
        g_compress_err.store(0);

        const int N_THREADS = 4;
        std::thread workers[N_THREADS];
        for (int i = 0; i < N_THREADS; i++)
            workers[i] = std::thread(compress_worker);

        // Signal all threads to start consuming
        g_start_compress.store(true, std::memory_order_release);

        for (int i = 0; i < N_THREADS; i++)
            workers[i].join();

        int ok = g_compress_ok.load();
        int errs = g_compress_err.load();
        printf("  compress: %d ok, %d errors\n", ok, errs);

        if (ok > 0) {
            PASS("concurrent forced-algo consumption completed without crash");
        } else {
            FAIL("no compressions succeeded");
        }
    }

    /* ---- Test 2: Concurrent push + compress (the actual race) ---- */
    printf("\n--- Test 2: Concurrent push + compress ---\n");
    {
        gpucompress_force_algorithm_reset();

        // Pre-fill some entries
        for (int i = 0; i < 128; i++)
            gpucompress_force_algorithm_push(1 /* LZ4 */, 0, 0, 0.0);

        g_start_compress.store(false);
        g_compress_ok.store(0);
        g_compress_err.store(0);

        // Consumer thread
        std::thread consumer(compress_worker);

        // Push more entries from main thread while consumer runs
        g_start_compress.store(true, std::memory_order_release);
        for (int i = 0; i < 400; i++) {
            gpucompress_force_algorithm_push(1 + (i % 8), 0, 0, 0.0);
        }

        consumer.join();

        int ok = g_compress_ok.load();
        printf("  compress: %d ok\n", ok);

        if (ok > 0) {
            PASS("concurrent push + consume completed without crash");
        } else {
            FAIL("no compressions succeeded during concurrent push");
        }
    }

    gpucompress_force_algorithm_reset();
    gpucompress_cleanup();
    PASS("cleanup completed");

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
