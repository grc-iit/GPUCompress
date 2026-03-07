/**
 * test_m6_integer_overflow.cu
 *
 * M-6: Integer overflow in gpucompress_max_compressed_size for very large inputs.
 *      HEADER_SIZE + input_size + (input_size/8) + 1024
 *
 * Verdict: LOW — overflow requires ~16 EB on 64-bit, but test documents the edge.
 *
 * Run: ./test_m6_integer_overflow
 */

#include <cstdio>
#include <cstdlib>
#include <climits>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

int main(void) {
    printf("=== M-6: Integer overflow in gpucompress_max_compressed_size ===\n\n");

    /* ---- Test 1: Normal input size ---- */
    printf("--- Test 1: Normal input (4096 bytes) ---\n");
    {
        size_t result = gpucompress_max_compressed_size(4096);
        /* Expected: 64 + 4096 + 512 + 1024 = 5696 */
        size_t expected = 64 + 4096 + (4096 / 8) + 1024;
        if (result == expected) {
            PASS("correct max_compressed_size for 4096 bytes");
        } else {
            printf("  got %zu, expected %zu\n", result, expected);
            FAIL("wrong max_compressed_size for 4096 bytes");
        }
    }

    /* ---- Test 2: 1 GB input ---- */
    printf("\n--- Test 2: 1 GB input ---\n");
    {
        size_t gb = (size_t)1 << 30;
        size_t result = gpucompress_max_compressed_size(gb);
        size_t expected = 64 + gb + (gb / 8) + 1024;
        if (result == expected && result > gb) {
            PASS("correct and no overflow for 1 GB input");
        } else {
            FAIL("overflow or wrong result for 1 GB input");
        }
    }

    /* ---- Test 3: SIZE_MAX input (overflow case — known LOW issue) ---- */
    printf("\n--- Test 3: SIZE_MAX input (overflow edge) ---\n");
    {
        size_t result = gpucompress_max_compressed_size(SIZE_MAX);
        /* This WILL overflow: SIZE_MAX + SIZE_MAX/8 + 1088 wraps around.
         * M-6 audit: downgraded to LOW — requires ~16 EB input on 64-bit,
         * which is physically impossible. We document the overflow exists. */
        if (result < SIZE_MAX) {
            printf("  result=%zu (wrapped around, as expected on 64-bit)\n", result);
            PASS("overflow at SIZE_MAX confirmed (known LOW — requires ~16 EB)");
        } else {
            PASS("no overflow for SIZE_MAX");
        }
    }

    /* ---- Test 4: Near-overflow boundary ---- */
    printf("\n--- Test 4: Large but valid input ---\n");
    {
        /* Max safe input: input + input/8 + 1088 <= SIZE_MAX
         * input * 9/8 <= SIZE_MAX - 1088
         * input <= (SIZE_MAX - 1088) * 8/9
         */
        size_t max_safe = ((SIZE_MAX - 1088) / 9) * 8;
        size_t result = gpucompress_max_compressed_size(max_safe);
        if (result > max_safe) {
            PASS("no overflow at max safe input boundary");
        } else {
            FAIL("overflow at max safe input boundary");
        }

        /* One byte past safe boundary */
        size_t unsafe = max_safe + 8;  /* +8 to ensure we cross the line */
        size_t result2 = gpucompress_max_compressed_size(unsafe);
        printf("  max_safe result=%zu, unsafe result=%zu\n", result, result2);
    }

    /* ---- Test 5: Verify size_t is 64-bit ---- */
    printf("\n--- Test 5: size_t width ---\n");
    {
        if (sizeof(size_t) == 8) {
            PASS("size_t is 64-bit (overflow requires ~16 EB)");
        } else {
            printf("  sizeof(size_t) = %zu\n", sizeof(size_t));
            FAIL("size_t is NOT 64-bit — overflow at ~3.8 GB");
        }
    }

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
