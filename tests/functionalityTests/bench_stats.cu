/**
 * @file bench_stats.cu
 * @brief CPU vs GPU stats performance benchmark
 *
 * Measures wall-clock time for computeStatsCPU vs runStatsOnlyPipeline
 * across multiple data sizes and patterns.
 *
 * GPU is measured two ways:
 *   1. "GPU (incl. transfer)" — includes cudaMemcpy H→D
 *   2. "GPU (on-device)"      — data already resident on GPU
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>

#include "stats/stats_cpu.h"

namespace gpucompress {
int runStatsOnlyPipeline(
    const void* d_input, size_t input_size, cudaStream_t stream,
    double* out_entropy, double* out_mad, double* out_deriv);
}

/* ============================================================
 * Timing helpers (CUDA events for GPU, clock_gettime for CPU)
 * ============================================================ */

static double wall_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================
 * Data patterns
 * ============================================================ */

static void fill_ramp(float* buf, size_t n) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (float)i / (float)n;
}

static void fill_random(float* buf, size_t n) {
    srand(42);
    for (size_t i = 0; i < n; i++)
        buf[i] = (float)rand() / (float)RAND_MAX * 200.0f - 100.0f;
}

static void fill_sparse(float* buf, size_t n) {
    srand(0xCAFE);
    for (size_t i = 0; i < n; i++)
        buf[i] = ((rand() % 100) == 0) ? (float)rand() / RAND_MAX * 1000.0f : 0.0f;
}

struct Pattern {
    const char* name;
    void (*fill)(float*, size_t);
};

static Pattern patterns[] = {
    {"ramp",   fill_ramp},
    {"random", fill_random},
    {"sparse", fill_sparse},
};
static const int N_PATTERNS = sizeof(patterns) / sizeof(patterns[0]);

/* ============================================================
 * Benchmark sizes
 * ============================================================ */

struct BenchSize {
    const char* label;
    size_t n_floats;
};

static BenchSize sizes[] = {
    {"64 KB",    64 * 1024 / 4},
    {"256 KB",   256 * 1024 / 4},
    {"1 MB",     1024 * 1024 / 4},
    {"4 MB",     4 * 1024 * 1024 / 4},
    {"16 MB",    16 * 1024 * 1024 / 4},
    {"64 MB",    64 * 1024 * 1024 / 4},
};
static const int N_SIZES = sizeof(sizes) / sizeof(sizes[0]);

/* ============================================================
 * Benchmark core
 * ============================================================ */

static const int WARMUP = 2;
static const int ITERS  = 10;

static double bench_cpu(const float* h_data, size_t bytes) {
    double ent, mad, deriv;
    // Warmup
    for (int i = 0; i < WARMUP; i++)
        gpucompress::computeStatsCPU(h_data, bytes, &ent, &mad, &deriv);

    double t0 = wall_ms();
    for (int i = 0; i < ITERS; i++)
        gpucompress::computeStatsCPU(h_data, bytes, &ent, &mad, &deriv);
    double t1 = wall_ms();
    return (t1 - t0) / ITERS;
}

static double bench_gpu_with_transfer(const float* h_data, size_t bytes) {
    double ent, mad, deriv;
    void* d_data = nullptr;
    cudaMalloc(&d_data, bytes);

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
        gpucompress::runStatsOnlyPipeline(d_data, bytes, 0, &ent, &mad, &deriv);
        cudaDeviceSynchronize();
    }

    double t0 = wall_ms();
    for (int i = 0; i < ITERS; i++) {
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
        gpucompress::runStatsOnlyPipeline(d_data, bytes, 0, &ent, &mad, &deriv);
        cudaDeviceSynchronize();
    }
    double t1 = wall_ms();

    cudaFree(d_data);
    return (t1 - t0) / ITERS;
}

static double bench_gpu_on_device(const float* h_data, size_t bytes) {
    double ent, mad, deriv;
    void* d_data = nullptr;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        gpucompress::runStatsOnlyPipeline(d_data, bytes, 0, &ent, &mad, &deriv);
        cudaDeviceSynchronize();
    }

    double t0 = wall_ms();
    for (int i = 0; i < ITERS; i++) {
        gpucompress::runStatsOnlyPipeline(d_data, bytes, 0, &ent, &mad, &deriv);
        cudaDeviceSynchronize();
    }
    double t1 = wall_ms();

    cudaFree(d_data);
    return (t1 - t0) / ITERS;
}

/* ============================================================
 * Main
 * ============================================================ */

int main() {
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== Stats Benchmark: CPU vs GPU ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Warmup: %d, Iterations: %d\n\n", WARMUP, ITERS);

    // Allocate max size
    size_t max_floats = sizes[N_SIZES - 1].n_floats;
    float* h_data = (float*)malloc(max_floats * sizeof(float));
    if (!h_data) { fprintf(stderr, "malloc failed\n"); return 1; }

    for (int p = 0; p < N_PATTERNS; p++) {
        printf("--- Pattern: %s ---\n", patterns[p].name);
        printf("%-10s | %10s  %10s  %10s | %8s  %8s\n",
               "Size", "CPU (ms)", "GPU+xfer", "GPU only",
               "CPU/GPU", "Throughput");
        printf("%-10s-+-%10s--%10s--%10s-+-%8s--%8s\n",
               "----------", "----------", "----------", "----------",
               "--------", "--------");

        patterns[p].fill(h_data, max_floats);

        for (int s = 0; s < N_SIZES; s++) {
            size_t nf = sizes[s].n_floats;
            size_t bytes = nf * sizeof(float);

            // Re-fill for exact size (patterns may depend on n)
            patterns[p].fill(h_data, nf);

            double cpu_ms  = bench_cpu(h_data, bytes);
            double gpu_xf  = bench_gpu_with_transfer(h_data, bytes);
            double gpu_dev = bench_gpu_on_device(h_data, bytes);

            double speedup = cpu_ms / gpu_dev;
            double tp_mbps = (bytes / (1024.0 * 1024.0)) / (cpu_ms / 1000.0);

            printf("%-10s | %8.3f ms  %8.3f ms  %8.3f ms | %7.2fx  %6.0f MB/s\n",
                   sizes[s].label, cpu_ms, gpu_xf, gpu_dev,
                   speedup, tp_mbps);
        }
        printf("\n");
    }

    free(h_data);
    printf("Done.\n");
    return 0;
}
