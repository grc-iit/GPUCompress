/**
 * test_h6_size_overflow.cu
 *
 * H6: Unchecked integer overflow in size calculations.
 *
 * gpucompress_max_compressed_size() computes:
 *   HEADER_SIZE + input_size + (input_size / 8) + 1024
 * which can overflow for large input_size values near SIZE_MAX.
 *
 * Similarly, header_size + max_compressed_size in the compress paths
 * can wrap around, causing a small allocation followed by a large write.
 *
 * Test: verify that extreme input sizes are rejected gracefully
 * (return error, not crash or wrap-around allocation).
 *
 * Run: ./test_h6_size_overflow
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

int main(void) {
    printf("=== H6: Integer overflow in size calculations ===\n\n");

    gpucompress_error_t gerr = gpucompress_init(NULL);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        printf("SKIP: gpucompress_init failed (%d)\n", gerr);
        return 1;
    }
    PASS("init succeeded");

    /* ---- Test 1: gpucompress_max_compressed_size overflow ---- */
    printf("\n--- Test 1: max_compressed_size with near-SIZE_MAX input ---\n");
    {
        // SIZE_MAX would overflow: HEADER + SIZE_MAX + SIZE_MAX/8 + 1024
        size_t result = gpucompress_max_compressed_size(SIZE_MAX);

        // After fix: should return 0 or SIZE_MAX to signal overflow
        // Before fix: wraps around to a small number
        size_t expected_min = SIZE_MAX / 2;  // any sane result must be huge
        if (result == 0) {
            PASS("max_compressed_size(SIZE_MAX) returned 0 (overflow detected)");
        } else if (result < expected_min) {
            printf("  result = %zu (wrapped around!)\n", result);
            FAIL("max_compressed_size(SIZE_MAX) wrapped around");
        } else {
            // Returned a huge value — technically correct (no overflow occurred)
            // but caller would fail on allocation anyway
            PASS("max_compressed_size(SIZE_MAX) returned large value (no wrap)");
        }
    }

    /* ---- Test 2: Compress with huge claimed size should fail, not crash ---- */
    printf("\n--- Test 2: compress with absurd input_size ---\n");
    {
        // Pass a small buffer but claim it's huge — should be rejected
        // before any allocation attempt
        float small_buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        size_t huge_size = SIZE_MAX - 100;

        size_t out_size = 1024;
        char out_buf[1024];

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

        // This should fail gracefully (error return), not crash
        gpucompress_error_t err = gpucompress_compress(
            small_buf, huge_size, out_buf, &out_size, &cfg, NULL);

        if (err != GPUCOMPRESS_SUCCESS) {
            PASS("compress with huge input_size correctly rejected");
        } else {
            FAIL("compress with huge input_size should have failed");
        }
    }

    /* ---- Test 3: Decompress with overflow in header sizes ---- */
    printf("\n--- Test 3: decompress with crafted overflow header ---\n");
    {
        // Create a buffer that looks like a valid header but has
        // compressed_size near SIZE_MAX to trigger overflow in
        // the validation: input_size < HEADER_SIZE + compressed_size
        // If HEADER_SIZE + compressed_size overflows, the check passes
        // when it shouldn't.
        unsigned char fake[128];
        memset(fake, 0, sizeof(fake));

        // Set magic bytes (0x43555047 = "GPUC" in little-endian)
        fake[0] = 0x47; fake[1] = 0x55; fake[2] = 0x50; fake[3] = 0x43;
        // Set version = 1
        fake[4] = 1;
        // Set compressed_size to SIZE_MAX - 32 (would overflow with HEADER_SIZE=64)
        size_t crafted_size = SIZE_MAX - 32;
        memcpy(fake + 16, &crafted_size, sizeof(size_t));

        size_t out_size = 1024;
        char out_buf[1024];

        gpucompress_error_t err = gpucompress_decompress(
            fake, sizeof(fake), out_buf, &out_size);

        if (err != GPUCOMPRESS_SUCCESS) {
            PASS("decompress with overflow header correctly rejected");
        } else {
            FAIL("decompress with overflow header should have failed");
        }
    }

    gpucompress_cleanup();
    PASS("cleanup completed");

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
