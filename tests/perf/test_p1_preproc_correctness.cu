/**
 * test_p1_preproc_correctness.cu
 *
 * Correctness test for P1: pre-allocated preprocessing buffers.
 * Verifies roundtrip correctness with AUTO compression.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "gpucompress.h"

static int g_pass = 0, g_fail = 0;
#define PASS(msg) do { fprintf(stderr, "  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { fprintf(stderr, "  FAIL: %s\n", msg); g_fail++; } while(0)

int main() {
    fprintf(stderr, "=== P1: Preprocessing Pre-allocation Correctness ===\n\n");

    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) weights = "neural_net/weights/model.nnwt";
    if (gpucompress_init(weights) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "ERROR: gpucompress_init failed\n");
        return 1;
    }

    size_t data_size = 4UL << 20;
    size_t n_floats = data_size / sizeof(float);

    void *d_in = nullptr, *d_out = nullptr, *d_decomp = nullptr;
    cudaMalloc(&d_in, data_size);
    cudaMalloc(&d_out, data_size * 2);
    cudaMalloc(&d_decomp, data_size);

    /* Test 1: Lossless AUTO roundtrip */
    fprintf(stderr, "--- Test 1: Lossless AUTO roundtrip (4MB) ---\n");
    {
        std::vector<float> h_orig(n_floats);
        for (size_t i = 0; i < n_floats; i++)
            h_orig[i] = sinf((float)i * 0.007f) + 2.0f * cosf((float)i * 0.0003f);
        cudaMemcpy(d_in, h_orig.data(), data_size, cudaMemcpyHostToDevice);

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
        gpucompress_stats_t stats = {};
        size_t comp_size = data_size * 2;

        gpucompress_error_t err = gpucompress_compress_gpu(
            d_in, data_size, d_out, &comp_size, &cfg, &stats, nullptr);
        fprintf(stderr, "    compress: err=%d algo=%d ratio=%.2f\n", err, stats.algorithm_used, stats.compression_ratio);

        if (err != GPUCOMPRESS_SUCCESS) {
            FAIL("compress_gpu failed");
        } else {
            size_t decomp_size = data_size;
            err = gpucompress_decompress_gpu(d_out, comp_size, d_decomp, &decomp_size, nullptr);
            if (err != GPUCOMPRESS_SUCCESS) {
                FAIL("decompress_gpu failed");
            } else {
                std::vector<float> h_decomp(n_floats);
                cudaMemcpy(h_decomp.data(), d_decomp, data_size, cudaMemcpyDeviceToHost);
                bool match = true;
                for (size_t i = 0; i < n_floats && match; i++)
                    if (h_orig[i] != h_decomp[i]) match = false;
                if (match) PASS("lossless roundtrip exact match");
                else FAIL("lossless roundtrip data mismatch");
            }
        }
    }

    /* Test 2: 16 sequential chunks (buffer reuse) */
    fprintf(stderr, "\n--- Test 2: 16 sequential AUTO compressions ---\n");
    {
        int ok = 0;
        for (int i = 0; i < 16; i++) {
            std::vector<float> h_data(n_floats);
            for (size_t j = 0; j < n_floats; j++)
                h_data[j] = sinf((float)(j + i * 1000) * 0.013f) * (float)(i + 1);
            cudaMemcpy(d_in, h_data.data(), data_size, cudaMemcpyHostToDevice);

            gpucompress_config_t cfg = gpucompress_default_config();
            cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
            size_t comp_size = data_size * 2;
            if (gpucompress_compress_gpu(d_in, data_size, d_out, &comp_size, &cfg, nullptr, nullptr) != GPUCOMPRESS_SUCCESS) continue;
            size_t decomp_size = data_size;
            if (gpucompress_decompress_gpu(d_out, comp_size, d_decomp, &decomp_size, nullptr) != GPUCOMPRESS_SUCCESS) continue;
            std::vector<float> h_out(n_floats);
            cudaMemcpy(h_out.data(), d_decomp, data_size, cudaMemcpyDeviceToHost);
            bool match = true;
            for (size_t j = 0; j < n_floats && match; j++)
                if (h_data[j] != h_out[j]) match = false;
            if (match) ok++;
        }
        if (ok == 16) PASS("16/16 sequential roundtrips exact match");
        else { fprintf(stderr, "  %d/16 succeeded\n", ok); FAIL("some roundtrips failed"); }
    }

    /* Test 3: Fixed LZ4+shuffle roundtrip */
    fprintf(stderr, "\n--- Test 3: Fixed LZ4+shuffle roundtrip ---\n");
    {
        std::vector<float> h_orig(n_floats);
        for (size_t i = 0; i < n_floats; i++) h_orig[i] = (float)(i % 256) / 256.0f;
        cudaMemcpy(d_in, h_orig.data(), data_size, cudaMemcpyHostToDevice);

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;
        cfg.preprocessing = GPUCOMPRESS_PREPROC_SHUFFLE_4;
        size_t comp_size = data_size * 2;
        gpucompress_error_t e1 = gpucompress_compress_gpu(d_in, data_size, d_out, &comp_size, &cfg, nullptr, nullptr);
        if (e1 != GPUCOMPRESS_SUCCESS) {
            FAIL("LZ4+shuffle compress failed");
        } else {
            size_t decomp_size = data_size;
            gpucompress_error_t e2 = gpucompress_decompress_gpu(d_out, comp_size, d_decomp, &decomp_size, nullptr);
            if (e2 != GPUCOMPRESS_SUCCESS) { FAIL("LZ4+shuffle decompress failed"); }
            else {
                std::vector<float> h_out(n_floats);
                cudaMemcpy(h_out.data(), d_decomp, data_size, cudaMemcpyDeviceToHost);
                bool match = true;
                for (size_t i = 0; i < n_floats && match; i++)
                    if (h_orig[i] != h_out[i]) match = false;
                if (match) PASS("LZ4+shuffle exact roundtrip");
                else FAIL("LZ4+shuffle data mismatch");
            }
        }
    }

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_decomp);

    fprintf(stderr, "\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    fprintf(stderr, "%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
