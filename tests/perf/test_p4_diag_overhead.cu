/**
 * test_p4_diag_overhead.cu
 *
 * Microbenchmark for P4: measures overhead of recordChunkDiagnostic().
 *
 * Two test modes:
 *   Mode 1 (default): Full compress path with large data to see real-world impact
 *   Mode 2 (ISOLATE=1): Isolates just the diagnostic codepath overhead
 *
 * Usage:
 *   ./test_p4_diag_overhead                                  # 256MB, 8 threads
 *   DATA_MB=1024 N_CHUNKS=256 N_THREADS=8 ./test_p4_diag_overhead
 *   ISOLATE=1 N_CHUNKS=1024 N_THREADS=8 ./test_p4_diag_overhead
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <numeric>
#include <algorithm>
#include "gpucompress.h"

static int env_int(const char* name, int fallback) {
    const char* v = getenv(name);
    return v ? atoi(v) : fallback;
}

int main() {
    int data_mb   = env_int("DATA_MB", 256);
    int chunk_mb  = env_int("CHUNK_MB", 4);
    int n_threads = env_int("N_THREADS", 8);
    int n_warmup  = env_int("N_WARMUP", 3);
    int n_iters   = env_int("N_ITERS", 10);
    int isolate   = env_int("ISOLATE", 0);

    size_t data_size  = (size_t)data_mb << 20;
    size_t chunk_size = (size_t)chunk_mb << 20;
    int n_chunks = (int)(data_size / chunk_size);
    if (n_chunks < 1) n_chunks = 1;

    printf("=== P4 Diagnostic Overhead Benchmark ===\n");
    printf("  Data size   : %d MB\n", data_mb);
    printf("  Chunk size  : %d MB\n", chunk_mb);
    printf("  Chunks      : %d\n", n_chunks);
    printf("  Threads     : %d\n", n_threads);
    printf("  Warmup      : %d\n", n_warmup);
    printf("  Iterations  : %d\n", n_iters);
    printf("  Mode        : %s\n\n", isolate ? "ISOLATE (diag-only)" : "FULL (compress+diag)");

    /* Init library */
    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) weights = "neural_net/weights/model.nnwt";
    if (gpucompress_init(weights) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "ERROR: gpucompress_init failed (weights: %s)\n", weights);
        return 1;
    }

    /* Allocate device data */
    void* d_data = nullptr;
    cudaMalloc(&d_data, chunk_size);
    {
        std::vector<float> h_data(chunk_size / sizeof(float));
        for (size_t i = 0; i < h_data.size(); i++)
            h_data[i] = sinf((float)i * 0.01f) + 0.5f * cosf((float)i * 0.003f);
        cudaMemcpy(d_data, h_data.data(), chunk_size, cudaMemcpyHostToDevice);
    }

    /* Per-thread output buffers */
    size_t out_cap = chunk_size * 2;
    std::vector<void*> d_outputs(n_threads);
    for (int t = 0; t < n_threads; t++)
        cudaMalloc(&d_outputs[t], out_cap);

    /* Pre-run one AUTO compress to prime NN */
    {
        size_t out_sz = out_cap;
        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
        gpucompress_compress_gpu(d_data, chunk_size, d_outputs[0], &out_sz, &cfg, nullptr, nullptr);
    }

    /* ── Full path benchmark: N threads compress chunks concurrently ── */
    int chunks_per_thread = n_chunks / n_threads;
    if (chunks_per_thread < 1) chunks_per_thread = 1;
    int total_chunks = chunks_per_thread * n_threads;

    auto run_full = [&]() -> double {
        gpucompress_reset_chunk_history();
        std::atomic<int> ready{0};
        std::vector<double> thread_ms(n_threads, 0.0);

        auto t_start = std::chrono::steady_clock::now();
        std::vector<std::thread> threads;
        for (int t = 0; t < n_threads; t++) {
            threads.emplace_back([&, t]() {
                ready.fetch_add(1);
                while (ready.load() < n_threads) {} /* spin barrier */

                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm = GPUCOMPRESS_ALGO_AUTO;
                gpucompress_stats_t stats = {};

                auto t0 = std::chrono::steady_clock::now();
                for (int c = 0; c < chunks_per_thread; c++) {
                    size_t out_sz = out_cap;
                    gpucompress_compress_gpu(d_data, chunk_size, d_outputs[t],
                                            &out_sz, &cfg, &stats, nullptr);
                }
                auto t1 = std::chrono::steady_clock::now();
                thread_ms[t] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            });
        }
        for (auto& th : threads) th.join();
        auto t_end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t_end - t_start).count();
    };

    printf("--- Warming up (%d iterations) ---\n", n_warmup);
    for (int w = 0; w < n_warmup; w++) run_full();

    printf("--- Measuring (%d iterations, %d chunks each) ---\n", n_iters, total_chunks);
    std::vector<double> times;
    for (int i = 0; i < n_iters; i++) {
        double ms = run_full();
        times.push_back(ms);
        printf("  iter %2d: %.1f ms  (%.3f ms/chunk)\n", i, ms, ms / total_chunks);
    }

    std::sort(times.begin(), times.end());
    double median = times[times.size() / 2];
    double mean   = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double p10    = times[(int)(times.size() * 0.1)];
    double p90    = times[(int)(times.size() * 0.9)];
    double stdev  = 0;
    for (double t : times) stdev += (t - mean) * (t - mean);
    stdev = sqrt(stdev / times.size());

    printf("\n=== Results (%d chunks x %d threads = %d total, %dMB data) ===\n",
           chunks_per_thread, n_threads, total_chunks, data_mb);
    printf("  Wall time per iteration:\n");
    printf("    median = %.3f ms\n", median);
    printf("    mean   = %.3f ms  (std=%.3f)\n", mean, stdev);
    printf("    p10    = %.3f ms\n", p10);
    printf("    p90    = %.3f ms\n", p90);
    printf("  Per-chunk:\n");
    printf("    median = %.3f ms/chunk\n", median / total_chunks);
    printf("  Throughput:\n");
    printf("    %.1f MiB/s  (data_size / median_wall)\n",
           (double)data_mb / (median / 1000.0));
    printf("\n");

    /* Cleanup */
    for (int t = 0; t < n_threads; t++) cudaFree(d_outputs[t]);
    cudaFree(d_data);
    printf("Done.\n");
    return 0;
}
