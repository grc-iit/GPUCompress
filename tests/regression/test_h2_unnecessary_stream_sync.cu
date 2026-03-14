/**
 * test_h2_unnecessary_stream_sync.cu
 *
 * H2: Unnecessary cudaStreamSynchronize() calls.
 *
 * Several stream syncs in the compress pipeline serialize operations
 * that could overlap (e.g., preprocessing sync before compression on
 * the same stream, byte_shuffle internal sync).
 *
 * Test: Verify that removing unnecessary syncs doesn't break
 * correctness by exercising compress + decompress round-trips
 * with all preprocessing options (quantize, shuffle, both).
 *
 * Run: ./test_h2_unnecessary_stream_sync
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

static const char* WEIGHTS_PATH = "../neural_net/weights/model.nnwt";

int main(void) {
    printf("=== H2: Unnecessary stream sync removal ===\n\n");

    gpucompress_error_t gerr = gpucompress_init(WEIGHTS_PATH);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        gerr = gpucompress_init(NULL);
        if (gerr != GPUCOMPRESS_SUCCESS) {
            printf("SKIP: gpucompress_init failed (%d)\n", gerr);
            return 1;
        }
    }
    PASS("init succeeded");

    const size_t DATA_SIZE = 64 * 1024;  // 64 KB
    float* h_data = (float*)malloc(DATA_SIZE);
    for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++)
        h_data[i] = sinf((float)i * 0.001f) * 1000.0f;

    size_t max_out = gpucompress_max_compressed_size(DATA_SIZE);
    void* h_compressed = malloc(max_out);
    float* h_decompressed = (float*)malloc(DATA_SIZE);

    /* ---- Test 1: Compress with no preprocessing (baseline) ---- */
    printf("\n--- Test 1: No preprocessing ---\n");
    {
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;
        cfg.preprocessing = 0;

        size_t comp_size = max_out;
        gpucompress_error_t err = gpucompress_compress(
            h_data, DATA_SIZE, h_compressed, &comp_size, &cfg, NULL);

        if (err != GPUCOMPRESS_SUCCESS) {
            FAIL("no-preproc compress failed");
        } else {
            size_t decomp_size = DATA_SIZE;
            err = gpucompress_decompress(h_compressed, comp_size,
                                          h_decompressed, &decomp_size);
            if (err == GPUCOMPRESS_SUCCESS && decomp_size == DATA_SIZE &&
                memcmp(h_data, h_decompressed, DATA_SIZE) == 0) {
                PASS("no-preproc round-trip OK");
            } else {
                FAIL("no-preproc round-trip mismatch");
            }
        }
    }

    /* ---- Test 2: Compress with byte shuffle ---- */
    printf("\n--- Test 2: Byte shuffle (4-byte) ---\n");
    {
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;
        cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;

        size_t comp_size = max_out;
        gpucompress_error_t err = gpucompress_compress(
            h_data, DATA_SIZE, h_compressed, &comp_size, &cfg, NULL);

        if (err != GPUCOMPRESS_SUCCESS) {
            FAIL("shuffle compress failed");
        } else {
            size_t decomp_size = DATA_SIZE;
            err = gpucompress_decompress(h_compressed, comp_size,
                                          h_decompressed, &decomp_size);
            if (err == GPUCOMPRESS_SUCCESS && decomp_size == DATA_SIZE &&
                memcmp(h_data, h_decompressed, DATA_SIZE) == 0) {
                PASS("shuffle round-trip OK");
            } else {
                FAIL("shuffle round-trip mismatch");
            }
        }
    }

    /* ---- Test 3: Compress with quantization ---- */
    printf("\n--- Test 3: Quantization (error_bound=1e-3) ---\n");
    {
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;
        cfg.preprocessing = GPUCOMPRESS_PREPROC_QUANTIZE;
        cfg.error_bound = 1e-3;

        size_t comp_size = max_out;
        gpucompress_error_t err = gpucompress_compress(
            h_data, DATA_SIZE, h_compressed, &comp_size, &cfg, NULL);

        if (err != GPUCOMPRESS_SUCCESS) {
            FAIL("quantize compress failed");
        } else {
            size_t decomp_size = DATA_SIZE;
            err = gpucompress_decompress(h_compressed, comp_size,
                                          h_decompressed, &decomp_size);
            if (err == GPUCOMPRESS_SUCCESS && decomp_size == DATA_SIZE) {
                /* Lossy: check within error bound */
                int ok = 1;
                for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++) {
                    if (fabsf(h_decompressed[i] - h_data[i]) > cfg.error_bound * 1.01f) {
                        ok = 0;
                        break;
                    }
                }
                if (ok) PASS("quantize round-trip within error bound");
                else FAIL("quantize round-trip exceeds error bound");
            } else {
                FAIL("quantize decompress failed");
            }
        }
    }

    /* ---- Test 4: Compress with quantization + shuffle ---- */
    printf("\n--- Test 4: Quantization + shuffle ---\n");
    {
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;
        cfg.preprocessing = GPUCOMPRESS_PREPROC_QUANTIZE | GPUCOMPRESS_PREPROC_SHUFFLE_4;
        cfg.error_bound = 1e-3;

        size_t comp_size = max_out;
        gpucompress_error_t err = gpucompress_compress(
            h_data, DATA_SIZE, h_compressed, &comp_size, &cfg, NULL);

        if (err != GPUCOMPRESS_SUCCESS) {
            FAIL("quant+shuffle compress failed");
        } else {
            size_t decomp_size = DATA_SIZE;
            err = gpucompress_decompress(h_compressed, comp_size,
                                          h_decompressed, &decomp_size);
            if (err == GPUCOMPRESS_SUCCESS && decomp_size == DATA_SIZE) {
                int ok = 1;
                for (size_t i = 0; i < DATA_SIZE / sizeof(float); i++) {
                    if (fabsf(h_decompressed[i] - h_data[i]) > cfg.error_bound * 1.01f) {
                        ok = 0;
                        break;
                    }
                }
                if (ok) PASS("quant+shuffle round-trip within error bound");
                else FAIL("quant+shuffle round-trip exceeds error bound");
            } else {
                FAIL("quant+shuffle decompress failed");
            }
        }
    }

    /* ---- Test 5: Multiple algorithms with shuffle ---- */
    printf("\n--- Test 5: Multiple algorithms with shuffle ---\n");
    {
        static const char* algo_names[] = {
            "LZ4", "Snappy", "Deflate", "GDeflate", "Zstd", "ANS", "Cascaded", "Bitcomp"
        };
        for (int algo = 1; algo <= 8; algo++) {
            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = (gpucompress_algorithm_t)algo;
            cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;

            size_t comp_size = max_out;
            gpucompress_error_t err = gpucompress_compress(
                h_data, DATA_SIZE, h_compressed, &comp_size, &cfg, NULL);

            char msg[128];
            if (err == GPUCOMPRESS_SUCCESS) {
                size_t decomp_size = DATA_SIZE;
                err = gpucompress_decompress(h_compressed, comp_size,
                                              h_decompressed, &decomp_size);
                if (err == GPUCOMPRESS_SUCCESS && decomp_size == DATA_SIZE &&
                    memcmp(h_data, h_decompressed, DATA_SIZE) == 0) {
                    snprintf(msg, sizeof(msg), "%s + shuffle round-trip OK", algo_names[algo-1]);
                    PASS(msg);
                } else {
                    snprintf(msg, sizeof(msg), "%s + shuffle round-trip failed", algo_names[algo-1]);
                    FAIL(msg);
                }
            } else {
                snprintf(msg, sizeof(msg), "%s + shuffle compress failed (err=%d)",
                         algo_names[algo-1], err);
                FAIL(msg);
            }
        }
    }

    free(h_data);
    free(h_compressed);
    free(h_decompressed);

    gpucompress_cleanup();
    PASS("cleanup completed");

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
