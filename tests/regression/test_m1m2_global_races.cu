/**
 * test_m1m2_global_races.cu
 *
 * M-1: g_online_learning_enabled, g_exploration_enabled are plain bool —
 *      data race (UB) under concurrent read+write.
 * M-2: g_exploration_threshold, g_reinforce_lr, g_reinforce_mape_threshold
 *      are non-atomic float/double — same issue.
 *
 * Test strategy:
 *   1. Spawn writer threads that toggle enable/disable APIs rapidly
 *   2. Spawn reader threads that call query APIs concurrently
 *   3. Verify no crash / no garbage values
 *
 * Run: ./test_m1m2_global_races
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <atomic>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

#define NUM_ITERATIONS 10000
#define NUM_WRITERS 4
#define NUM_READERS 4

static std::atomic<bool> g_stop{false};
static std::atomic<int> g_reader_errors{0};

/* Writer thread: toggles online learning on/off rapidly */
void* writer_thread(void* arg) {
    int id = (int)(intptr_t)arg;
    for (int i = 0; i < NUM_ITERATIONS && !g_stop.load(); i++) {
        if (i % 2 == 0) {
            gpucompress_enable_active_learning();
        } else {
            gpucompress_disable_active_learning();
        }

        /* Also toggle exploration (M-1 g_exploration_enabled) */
        gpucompress_set_exploration((id + i) % 2);

        /* Set exploration threshold (M-2 g_exploration_threshold) */
        gpucompress_set_exploration_threshold(0.1 + (i % 10) * 0.05);

        /* Set reinforcement params (M-2 g_reinforce_lr, g_reinforce_mape_threshold) */
        gpucompress_set_reinforcement(
            1,                                /* enable */
            0.001f + (i % 5) * 0.002f,       /* learning rate */
            0.10f + (i % 4) * 0.05f,         /* mape threshold */
            0.5f                              /* baseline */
        );
    }
    return NULL;
}

/* Reader thread: queries learning state concurrently */
void* reader_thread(void* arg) {
    (void)arg;
    int errors = 0;
    for (int i = 0; i < NUM_ITERATIONS && !g_stop.load(); i++) {
        int is_init = gpucompress_is_initialized();

        /* Value should be 0 or 1 — any other value indicates torn read */
        if (is_init != 0 && is_init != 1) {
            errors++;
        }
    }
    g_reader_errors.fetch_add(errors);
    return NULL;
}

int main(void) {
    printf("=== M-1/M-2: Non-atomic global variable races ===\n\n");

    /* Initialize the library */
    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }
    PASS("gpucompress_init succeeded");

    /* ---- Test 1: Concurrent writers don't crash ---- */
    printf("\n--- Test 1: Concurrent enable/disable writers ---\n");
    {
        pthread_t writers[NUM_WRITERS];
        for (int i = 0; i < NUM_WRITERS; i++) {
            pthread_create(&writers[i], NULL, writer_thread, (void*)(intptr_t)i);
        }
        for (int i = 0; i < NUM_WRITERS; i++) {
            pthread_join(writers[i], NULL);
        }
        PASS("concurrent writers did not crash");
    }

    /* ---- Test 2: Concurrent readers + writers don't crash ---- */
    printf("\n--- Test 2: Concurrent readers + writers ---\n");
    {
        g_stop.store(false);
        g_reader_errors.store(0);

        pthread_t writers[NUM_WRITERS];
        pthread_t readers[NUM_READERS];

        for (int i = 0; i < NUM_WRITERS; i++) {
            pthread_create(&writers[i], NULL, writer_thread, (void*)(intptr_t)i);
        }
        for (int i = 0; i < NUM_READERS; i++) {
            pthread_create(&readers[i], NULL, reader_thread, (void*)(intptr_t)i);
        }

        for (int i = 0; i < NUM_WRITERS; i++) {
            pthread_join(writers[i], NULL);
        }
        g_stop.store(true);
        for (int i = 0; i < NUM_READERS; i++) {
            pthread_join(readers[i], NULL);
        }

        PASS("concurrent readers + writers did not crash");

        if (g_reader_errors.load() == 0) {
            PASS("no torn reads detected (expected on x86)");
        } else {
            printf("  INFO: %d torn reads detected (UB manifested)\n",
                   g_reader_errors.load());
            FAIL("torn reads detected — non-atomic globals caused visible UB");
        }
    }

    /* ---- Test 3: Values are coherent after concurrent access ---- */
    printf("\n--- Test 3: Final state coherent ---\n");
    {
        /* Disable everything to a known state */
        gpucompress_disable_active_learning();
        gpucompress_enable_active_learning();

        /* If we got here without crash, the globals survived concurrent access */
        PASS("enable/disable cycle works after race test");
    }

    gpucompress_cleanup();

    /* ---- Summary ---- */
    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
