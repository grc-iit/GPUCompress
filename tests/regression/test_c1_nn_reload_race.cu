/**
 * test_c1_nn_reload_race.cu
 *
 * C1: NN weight pointer swap not atomic during hot-reload.
 *
 * gpucompress_reload_nn() calls cleanupNN() (which frees d_nn_weights)
 * then loadNNFromBinary() (which allocates new). A concurrent inference
 * thread that read the old pointer before the free can use-after-free.
 *
 * Test: spawn threads doing concurrent compressions with ALGO_AUTO
 * (which triggers NN inference), while the main thread repeatedly
 * calls gpucompress_reload_nn(). If the race exists, this will
 * crash or produce errors. After the fix, all operations complete
 * cleanly.
 *
 * Run: ./test_c1_nn_reload_race
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

static std::atomic<bool> g_stop{false};
static std::atomic<int> g_compress_ok{0};
static std::atomic<int> g_compress_err{0};

/**
 * Worker thread: repeatedly compresses small data with ALGO_AUTO
 * to exercise the NN inference path.
 */
static void compress_worker(int id) {
    const size_t DATA_SIZE = 32 * 1024;  // 32 KB
    float* h_data = (float*)malloc(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++)
        h_data[i] = (float)(i + id) * 0.01f;

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* h_output = malloc(max_out);

    while (!g_stop.load(std::memory_order_relaxed)) {
        size_t compressed_size = max_out;
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;  // triggers NN inference

        gpucompress_error_t err = gpucompress_compress(
            h_data, DATA_SIZE, h_output, &compressed_size, &cfg, NULL);

        if (err == GPUCOMPRESS_SUCCESS) {
            g_compress_ok.fetch_add(1, std::memory_order_relaxed);
        } else {
            // NN not loaded momentarily during reload is acceptable
            // (returns GPUCOMPRESS_ERROR_NN_NOT_LOADED), but a crash is not
            g_compress_err.fetch_add(1, std::memory_order_relaxed);
        }
    }

    free(h_data);
    free(h_output);
}

int main(void) {
    printf("=== C1: NN weight pointer swap race test ===\n\n");

    gpucompress_error_t gerr = gpucompress_init(WEIGHTS_PATH);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        printf("SKIP: gpucompress_init failed (%d)\n", gerr);
        return 1;
    }

    if (!gpucompress_nn_is_loaded()) {
        printf("SKIP: NN weights not loaded\n");
        gpucompress_cleanup();
        return 1;
    }
    PASS("init + NN loaded");

    /* ---- Test 1: Concurrent compress + reload ---- */
    printf("\n--- Test 1: 4 compress threads + 10 reloads ---\n");
    {
        const int N_THREADS = 4;
        const int N_RELOADS = 10;

        g_stop.store(false);
        g_compress_ok.store(0);
        g_compress_err.store(0);

        // Launch worker threads
        std::thread workers[N_THREADS];
        for (int i = 0; i < N_THREADS; i++)
            workers[i] = std::thread(compress_worker, i);

        // Main thread: reload NN weights repeatedly
        int reload_ok = 0;
        for (int i = 0; i < N_RELOADS; i++) {
            gpucompress_error_t err = gpucompress_reload_nn(WEIGHTS_PATH);
            if (err == GPUCOMPRESS_SUCCESS) {
                reload_ok++;
            } else {
                printf("  reload %d failed: %d\n", i, err);
            }
        }

        // Signal threads to stop
        g_stop.store(true);
        for (int i = 0; i < N_THREADS; i++)
            workers[i].join();

        int ok = g_compress_ok.load();
        int errs = g_compress_err.load();
        printf("  compress: %d ok, %d transient errors\n", ok, errs);
        printf("  reload:   %d/%d ok\n", reload_ok, N_RELOADS);

        if (reload_ok == N_RELOADS && ok > 0) {
            PASS("concurrent reload + compress completed without crash");
        } else {
            FAIL("concurrent reload + compress had issues");
        }
    }

    /* ---- Test 2: Reload preserves NN functionality ---- */
    printf("\n--- Test 2: Post-reload compression works ---\n");
    {
        const size_t DATA_SIZE = 64 * 1024;
        float* h_data = (float*)malloc(DATA_SIZE);
        for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++)
            h_data[i] = (float)i * 0.01f;

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_output = malloc(max_out);
        size_t compressed_size = max_out;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;

        gpucompress_error_t err = gpucompress_compress(
            h_data, DATA_SIZE, h_output, &compressed_size, &cfg, NULL);
        if (err == GPUCOMPRESS_SUCCESS && compressed_size > 0) {
            PASS("post-reload ALGO_AUTO compression works");
        } else {
            FAIL("post-reload compression failed");
        }

        free(h_data);
        free(h_output);
    }

    gpucompress_cleanup();
    PASS("cleanup completed");

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
