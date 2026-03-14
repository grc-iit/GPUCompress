/**
 * test_h5_configure_compression_check.cu
 *
 * H5: configure_compression() result not validated.
 *
 * Verifies that compression works correctly for all 8 algorithms and that
 * the library handles the configure_compression result properly. Each
 * algorithm produces a different max_compressed_buffer_size — if the
 * library blindly uses an invalid size, the compression or allocation
 * would fail.
 *
 * Run: ./test_h5_configure_compression_check
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

static const char* WEIGHTS_PATH = "../neural_net/weights/model.nnwt";

static const char* algo_names[] = {
    "LZ4", "Snappy", "Deflate", "GDeflate", "Zstd", "ANS", "Cascaded", "Bitcomp"
};

int main(void) {
    printf("=== H5: configure_compression result validation ===\n\n");

    gpucompress_error_t gerr = gpucompress_init(WEIGHTS_PATH);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        printf("SKIP: gpucompress_init failed (%d)\n", gerr);
        return 1;
    }
    PASS("init succeeded");

    /* ---- Test 1: All 8 algorithms produce valid compression ---- */
    printf("\n--- Test 1: All 8 algorithms compress successfully ---\n");
    {
        const size_t DATA_SIZE = 64 * 1024;  // 64 KB
        float* h_data = (float*)malloc(DATA_SIZE);
        for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++)
            h_data[i] = (float)i * 0.01f;

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_output = malloc(max_out);

        for (int algo = 1; algo <= 8; algo++) {
            size_t compressed_size = max_out;
            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = (gpucompress_algorithm_t)algo;

            gpucompress_error_t err = gpucompress_compress(
                h_data, DATA_SIZE, h_output, &compressed_size, &cfg, NULL);

            char msg[64];
            if (err == GPUCOMPRESS_SUCCESS && compressed_size > 0) {
                snprintf(msg, sizeof(msg), "%s: compressed %zu -> %zu bytes",
                         algo_names[algo-1], DATA_SIZE, compressed_size);
                PASS(msg);
            } else {
                snprintf(msg, sizeof(msg), "%s: failed (err=%d, size=%zu)",
                         algo_names[algo-1], err, compressed_size);
                FAIL(msg);
            }
        }

        free(h_data);
        free(h_output);
    }

    /* ---- Test 2: Tiny input (edge case for configure_compression) ---- */
    printf("\n--- Test 2: Tiny input (16 bytes) ---\n");
    {
        const size_t DATA_SIZE = 16;
        float h_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

        size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
        void* h_output = malloc(max_out);
        size_t compressed_size = max_out;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

        gpucompress_error_t err = gpucompress_compress(
            h_data, DATA_SIZE, h_output, &compressed_size, &cfg, NULL);

        if (err == GPUCOMPRESS_SUCCESS && compressed_size > 0) {
            PASS("tiny input compressed successfully");

            // Verify round-trip
            size_t decomp_size = DATA_SIZE;
            float h_decomp[4] = {0};
            err = gpucompress_decompress(h_output, compressed_size,
                                          h_decomp, &decomp_size);
            if (err == GPUCOMPRESS_SUCCESS && decomp_size == DATA_SIZE &&
                memcmp(h_data, h_decomp, DATA_SIZE) == 0) {
                PASS("tiny input round-trip verified");
            } else {
                FAIL("tiny input round-trip failed");
            }
        } else {
            FAIL("tiny input compression failed");
        }

        free(h_output);
    }

    /* ---- Test 3: Zero-size input (should return error, not crash) ---- */
    printf("\n--- Test 3: Zero-size input ---\n");
    {
        float dummy = 0.0f;
        size_t max_out = 1024;
        void* h_output = malloc(max_out);
        size_t compressed_size = max_out;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

        gpucompress_error_t err = gpucompress_compress(
            &dummy, 0, h_output, &compressed_size, &cfg, NULL);

        if (err != GPUCOMPRESS_SUCCESS) {
            PASS("zero-size input correctly rejected");
        } else {
            FAIL("zero-size input should have been rejected");
        }

        free(h_output);
    }

    gpucompress_cleanup();
    PASS("cleanup completed");

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
