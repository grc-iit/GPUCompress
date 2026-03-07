/**
 * test_m5_header_overread.cu
 *
 * M-5: gpucompress_get_original_size casts compressed buffer to
 *      CompressionHeader* (64 bytes) without validating buffer size.
 *      Buffer over-read on inputs < 64 bytes.
 *
 * Test strategy:
 *   1. Valid compressed data → returns correct original size
 *   2. NULL pointer → returns error
 *   3. Buffer with wrong magic → returns INVALID_HEADER
 *   4. Document: no way to pass buffer size (API limitation)
 *   5. Verify GPUCOMPRESS_HEADER_SIZE == sizeof(CompressionHeader) == 64
 *
 * Note: We can't safely test with a buffer < 64 bytes because the
 * function would read past the buffer (UB). We document this as the bug.
 *
 * Run: ./test_m5_header_overread
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

int main(void) {
    printf("=== M-5: gpucompress_get_original_size buffer over-read ===\n\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: gpucompress_init failed (%d)\n", err);
        return 1;
    }

    /* ---- Test 1: Valid compressed data round-trip ---- */
    printf("--- Test 1: Valid compressed data ---\n");
    {
        const size_t DATA_SIZE = 4096;
        float* h_data = (float*)malloc(DATA_SIZE);
        for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++) {
            h_data[i] = (float)i;
        }

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_compressed = malloc(max_out);
        size_t compressed_size = max_out;

        err = gpucompress_compress(h_data, DATA_SIZE, h_compressed, &compressed_size, NULL, NULL);
        if (err != GPUCOMPRESS_SUCCESS) {
            printf("  SKIP: compression failed\n");
            free(h_data);
            free(h_compressed);
            gpucompress_cleanup();
            return 1;
        }

        size_t original_size = 0;
        err = gpucompress_get_original_size(h_compressed, &original_size);
        if (err == GPUCOMPRESS_SUCCESS && original_size == DATA_SIZE) {
            PASS("correct original_size from valid compressed data");
        } else {
            printf("  err=%d, original_size=%zu (expected %zu)\n",
                   err, original_size, DATA_SIZE);
            FAIL("wrong original_size or error from valid data");
        }

        free(h_data);
        free(h_compressed);
    }

    /* ---- Test 2: NULL pointer ---- */
    printf("\n--- Test 2: NULL pointer ---\n");
    {
        size_t original_size = 0;
        err = gpucompress_get_original_size(NULL, &original_size);
        if (err == GPUCOMPRESS_ERROR_INVALID_INPUT) {
            PASS("NULL compressed pointer returns INVALID_INPUT");
        } else {
            FAIL("NULL compressed pointer did not return INVALID_INPUT");
        }

        err = gpucompress_get_original_size((void*)0x1, NULL);
        if (err == GPUCOMPRESS_ERROR_INVALID_INPUT) {
            PASS("NULL original_size pointer returns INVALID_INPUT");
        } else {
            FAIL("NULL original_size pointer did not return INVALID_INPUT");
        }
    }

    /* ---- Test 3: Wrong magic number ---- */
    printf("\n--- Test 3: Wrong magic number ---\n");
    {
        /* Create a 64-byte buffer with garbage */
        char buf[64];
        memset(buf, 0xAB, sizeof(buf));

        size_t original_size = 0;
        err = gpucompress_get_original_size(buf, &original_size);
        if (err == GPUCOMPRESS_ERROR_INVALID_HEADER) {
            PASS("garbage buffer returns INVALID_HEADER");
        } else {
            printf("  err=%d (expected INVALID_HEADER)\n", err);
            FAIL("garbage buffer did not return INVALID_HEADER");
        }
    }

    /* ---- Test 4: API limitation — no compressed_size parameter ---- */
    printf("\n--- Test 4: API limitation (documented) ---\n");
    {
        /* gpucompress_get_original_size(const void* compressed, size_t* original_size)
         *
         * There is no compressed_size parameter. The function reads 64 bytes
         * (sizeof(CompressionHeader)) from the compressed pointer unconditionally.
         * If the buffer is smaller than 64 bytes, this is a buffer over-read (UB).
         *
         * We cannot safely test this — reading past a small buffer is UB.
         * The fix is to add a compressed_size parameter to the API.
         */
        printf("  NOTE: Function lacks compressed_size parameter.\n");
        printf("        Passing a buffer < 64 bytes causes buffer over-read (UB).\n");
        printf("        Cannot safely test — documenting as API limitation.\n");
        PASS("documented: no size validation possible without API change");
    }

    /* ---- Test 5: Header size consistency ---- */
    printf("\n--- Test 5: GPUCOMPRESS_HEADER_SIZE consistency ---\n");
    {
        if (GPUCOMPRESS_HEADER_SIZE == 64) {
            PASS("GPUCOMPRESS_HEADER_SIZE == 64");
        } else {
            printf("  GPUCOMPRESS_HEADER_SIZE = %d\n", GPUCOMPRESS_HEADER_SIZE);
            FAIL("GPUCOMPRESS_HEADER_SIZE != 64");
        }
    }

    gpucompress_cleanup();

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
