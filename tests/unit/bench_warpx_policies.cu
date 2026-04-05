/**
 * @file bench_warpx_policies.cu
 * @brief WarpX compression benchmark: fixed algorithms vs NN policies
 *
 * Compares all 8 fixed algorithms against NN AUTO under:
 *   - Ratio policy   (w0=0, w1=0, w2=1) — maximize compression ratio
 *   - Balanced policy (w0=1, w1=1, w2=1) — balance speed and ratio
 *   - Lossless mode   (error_bound=0.0)
 *   - Lossy mode      (error_bound=0.01, 0.1)
 *
 * Tests multiple WarpX-representative data patterns:
 *   - Smooth E-field (sinusoidal)
 *   - Noisy B-field (sine + high-freq noise)
 *   - Scalar rho (smooth, single component)
 *   - Particle data (mixed smooth positions + oscillatory momenta)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include "gpucompress.h"
#include "gpucompress_warpx.h"

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                    \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* ============================================================
 * Data generators
 * ============================================================ */

/** Smooth sinusoidal field — typical EM field on structured grid */
__global__ void gen_smooth_field(float* data, int n, int ncomp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * ncomp) return;
    int elem = idx / ncomp;
    int var  = idx % ncomp;
    float x = (float)elem / (float)n;
    data[idx] = sinf(x * 6.2831853f * (var + 1)) * 100.0f
              + cosf(x * 3.1415927f * (var + 2)) * 50.0f;
}

/** Noisy field — smooth base + high-frequency perturbation */
__global__ void gen_noisy_field(float* data, int n, int ncomp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * ncomp) return;
    int elem = idx / ncomp;
    int var  = idx % ncomp;
    float x = (float)elem / (float)n;
    /* smooth base */
    float base = sinf(x * 6.2831853f * (var + 1)) * 100.0f;
    /* high-freq noise via integer hash */
    unsigned int h = (unsigned int)(elem * 2654435761u + var * 40503u);
    float noise = ((float)(h & 0xFFFF) / 65535.0f - 0.5f) * 2.0f;
    data[idx] = base + noise;
}

/** Scalar rho — smooth density profile */
__global__ void gen_scalar_rho(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = (float)idx / (float)n;
    data[idx] = 1.0f + 0.5f * sinf(x * 12.5663706f)
                      + 0.1f * cosf(x * 62.8318530f);
}

/** Particle data: positions (smooth), momenta (wide range), weight */
__global__ void gen_particles(float* data, int np)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= np * 7) return;
    int p   = idx / 7;
    int var = idx % 7;
    float t = (float)p / (float)np;
    if (var < 3) {
        data[idx] = t + 0.01f * sinf(t * 100.0f * (var + 1));
    } else if (var < 6) {
        data[idx] = sinf(t * 20.0f * (var - 2)) * 1e6f;
    } else {
        data[idx] = 1.0f + 0.001f * sinf(t * 50.0f);
    }
}

/* ============================================================
 * Benchmark infrastructure
 * ============================================================ */

struct BenchResult {
    const char* algo_name;
    const char* policy;
    const char* mode;
    size_t original;
    size_t compressed;
    double ratio;
    double comp_ms;
    double throughput_mbps;
    double max_error;
    int    nn_action;
};

static const int MAX_RESULTS = 256;
static BenchResult g_results[MAX_RESULTS];
static int g_n_results = 0;

static void record(const char* algo, const char* policy, const char* mode,
                   size_t orig, size_t comp, double ratio, double ms,
                   double tp, double maxerr, int action)
{
    if (g_n_results >= MAX_RESULTS) return;
    BenchResult* r = &g_results[g_n_results++];
    r->algo_name     = algo;
    r->policy        = policy;
    r->mode          = mode;
    r->original      = orig;
    r->compressed    = comp;
    r->ratio         = ratio;
    r->comp_ms       = ms;
    r->throughput_mbps = tp;
    r->max_error     = maxerr;
    r->nn_action     = action;
}

/** Run one compress + decompress cycle, record stats */
static int run_one(const void* d_data, size_t nbytes, size_t total_floats,
                   gpucompress_algorithm_t algo, unsigned int preproc,
                   double error_bound,
                   const char* algo_name, const char* policy, const char* mode)
{
    size_t max_comp = gpucompress_max_compressed_size(nbytes);
    void* d_comp = NULL;
    CHECK_CUDA(cudaMalloc(&d_comp, max_comp));
    size_t comp_size = max_comp;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm     = algo;
    cfg.preprocessing = preproc;
    cfg.error_bound   = error_bound;

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));

    gpucompress_error_t err = gpucompress_compress_gpu(
        d_data, nbytes, d_comp, &comp_size, &cfg, &stats, NULL);

    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "    compress FAILED for %s/%s/%s (err=%d)\n",
                algo_name, policy, mode, err);
        CHECK_CUDA(cudaFree(d_comp));
        return -1;
    }

    /* Decompress + verify */
    void* d_dec = NULL;
    CHECK_CUDA(cudaMalloc(&d_dec, nbytes));
    size_t dec_size = nbytes;
    err = gpucompress_decompress_gpu(d_comp, comp_size, d_dec, &dec_size, NULL);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "    decompress FAILED for %s/%s/%s\n", algo_name, policy, mode);
        CHECK_CUDA(cudaFree(d_comp));
        CHECK_CUDA(cudaFree(d_dec));
        return -1;
    }

    /* Compute max error on host */
    double max_err = 0.0;
    if (total_floats > 0) {
        std::vector<float> h_orig(total_floats), h_dec(total_floats);
        CHECK_CUDA(cudaMemcpy(h_orig.data(), d_data, nbytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_dec.data(), d_dec, nbytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < total_floats; i++) {
            double diff = fabs((double)h_orig[i] - (double)h_dec[i]);
            if (diff > max_err) max_err = diff;
        }

        /* For lossless, verify bitwise */
        if (error_bound == 0.0) {
            int mismatches = 0;
            for (size_t i = 0; i < total_floats; i++) {
                if (h_orig[i] != h_dec[i]) mismatches++;
            }
            if (mismatches > 0) {
                fprintf(stderr, "    LOSSLESS MISMATCH: %d/%zu for %s/%s/%s\n",
                        mismatches, total_floats, algo_name, policy, mode);
            }
        }
    }

    record(algo_name, policy, mode, nbytes, comp_size,
           stats.compression_ratio, stats.actual_comp_time_ms,
           stats.throughput_mbps, max_err, stats.nn_final_action);

    CHECK_CUDA(cudaFree(d_comp));
    CHECK_CUDA(cudaFree(d_dec));
    return 0;
}

/* ============================================================
 * Main benchmark
 * ============================================================ */

int main()
{
    printf("=== WarpX Compression Policy Benchmark ===\n\n");

    /* ---- Dataset setup ---- */
    const size_t N = 256 * 1024;  /* 256K cells — ~1 MB per component */

    struct Dataset {
        const char* name;
        warpx_data_type_t type;
        int ncomp;
        size_t n_elem;
        float* d_data;
        size_t nbytes;
        size_t total_floats;
    };

    Dataset datasets[4];
    datasets[0] = { "Smooth E-field",  WARPX_DATA_EFIELD,    3, N, NULL, 0, 0 };
    datasets[1] = { "Noisy B-field",   WARPX_DATA_BFIELD,    3, N, NULL, 0, 0 };
    datasets[2] = { "Scalar rho",      WARPX_DATA_RHO,       1, N, NULL, 0, 0 };
    datasets[3] = { "Particle data",   WARPX_DATA_PARTICLES, 7, N/4, NULL, 0, 0 };

    for (int d = 0; d < 4; d++) {
        datasets[d].total_floats = datasets[d].n_elem * datasets[d].ncomp;
        datasets[d].nbytes = datasets[d].total_floats * sizeof(float);
        CHECK_CUDA(cudaMalloc(&datasets[d].d_data, datasets[d].nbytes));
        int grid = ((int)datasets[d].total_floats + 255) / 256;
        switch (d) {
            case 0: gen_smooth_field<<<grid, 256>>>(datasets[d].d_data, (int)datasets[d].n_elem, 3); break;
            case 1: gen_noisy_field<<<grid, 256>>>(datasets[d].d_data, (int)datasets[d].n_elem, 3); break;
            case 2: gen_scalar_rho<<<grid, 256>>>(datasets[d].d_data, (int)datasets[d].n_elem); break;
            case 3: gen_particles<<<grid, 256>>>(datasets[d].d_data, (int)datasets[d].n_elem); break;
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        printf("Dataset: %-20s  %zu cells x %d comp = %.1f MB\n",
               datasets[d].name, datasets[d].n_elem, datasets[d].ncomp,
               datasets[d].nbytes / (1024.0 * 1024.0));
    }
    printf("\n");

    /* ---- Fixed algorithms ---- */
    gpucompress_algorithm_t fixed_algos[] = {
        GPUCOMPRESS_ALGO_LZ4, GPUCOMPRESS_ALGO_SNAPPY,
        GPUCOMPRESS_ALGO_DEFLATE, GPUCOMPRESS_ALGO_GDEFLATE,
        GPUCOMPRESS_ALGO_ZSTD, GPUCOMPRESS_ALGO_ANS,
        GPUCOMPRESS_ALGO_CASCADED, GPUCOMPRESS_ALGO_BITCOMP,
    };
    const char* fixed_names[] = {
        "LZ4", "Snappy", "Deflate", "Gdeflate",
        "Zstd", "ANS", "Cascaded", "Bitcomp",
    };

    /* Modes: lossless, lossy-0.01, lossy-0.1 */
    struct Mode {
        const char* name;
        double error_bound;
        unsigned int preproc;
    };
    Mode modes[] = {
        { "lossless",   0.0,  GPUCOMPRESS_PREPROC_NONE },
        { "lossy-0.01", 0.01, GPUCOMPRESS_PREPROC_QUANTIZE },
        { "lossy-0.1",  0.1,  GPUCOMPRESS_PREPROC_QUANTIZE },
    };

    /* ============================================================
     * Phase 1: Fixed algorithms (no NN needed)
     * ============================================================ */
    printf("======================================\n");
    printf("PHASE 1: Fixed Algorithms\n");
    printf("======================================\n\n");

    gpucompress_error_t gerr = gpucompress_init(NULL);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init(NULL) failed: %d\n", gerr);
        return 1;
    }

    for (int d = 0; d < 4; d++) {
        for (int m = 0; m < 3; m++) {
            printf("--- %s / %s ---\n", datasets[d].name, modes[m].name);
            printf("  %-12s %8s %8s %10s %8s\n",
                   "Algorithm", "Comp(KB)", "Ratio", "Speed(MB/s)", "MaxErr");

            for (int a = 0; a < 8; a++) {
                unsigned int pp = modes[m].preproc;
                /* Add shuffle for lossless float data */
                if (modes[m].error_bound == 0.0)
                    pp |= GPUCOMPRESS_PREPROC_SHUFFLE_4;

                int prev = g_n_results;
                run_one(datasets[d].d_data, datasets[d].nbytes,
                        datasets[d].total_floats,
                        fixed_algos[a], pp, modes[m].error_bound,
                        fixed_names[a], "fixed", modes[m].name);

                if (g_n_results > prev) {
                    BenchResult* r = &g_results[g_n_results - 1];
                    printf("  %-12s %8.1f %8.2f %10.1f %8.4f\n",
                           r->algo_name,
                           r->compressed / 1024.0,
                           r->ratio,
                           r->throughput_mbps,
                           r->max_error);
                }
            }
            printf("\n");
        }
    }

    gpucompress_cleanup();

    /* ============================================================
     * Phase 2: NN AUTO — Ratio policy (w0=0, w1=0, w2=1)
     * ============================================================ */
    printf("======================================\n");
    printf("PHASE 2: NN AUTO — Ratio Policy\n");
    printf("  (w0=0, w1=0, w2=1: maximize ratio)\n");
    printf("======================================\n\n");

    gerr = gpucompress_init("neural_net/weights/model.nnwt");
    if (gerr != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "NN init failed, skipping Phase 2\n");
    } else {
        gpucompress_set_ranking_weights(0.0f, 0.0f, 1.0f);

        for (int d = 0; d < 4; d++) {
            for (int m = 0; m < 3; m++) {
                printf("--- %s / %s ---\n", datasets[d].name, modes[m].name);
                printf("  %-12s %8s %8s %10s %8s %6s\n",
                       "Policy", "Comp(KB)", "Ratio", "Speed(MB/s)", "MaxErr", "Action");

                unsigned int pp = modes[m].preproc;
                if (modes[m].error_bound == 0.0)
                    pp |= GPUCOMPRESS_PREPROC_SHUFFLE_4;

                int prev = g_n_results;
                run_one(datasets[d].d_data, datasets[d].nbytes,
                        datasets[d].total_floats,
                        GPUCOMPRESS_ALGO_AUTO, pp, modes[m].error_bound,
                        "NN-AUTO", "ratio", modes[m].name);

                if (g_n_results > prev) {
                    BenchResult* r = &g_results[g_n_results - 1];
                    int action = r->nn_action;
                    int algo_id = action % 8;
                    int quant   = (action / 8) % 2;
                    int shuf    = (action / 16) % 2;
                    printf("  %-12s %8.1f %8.2f %10.1f %8.4f  a%d(%s%s%s)\n",
                           "ratio",
                           r->compressed / 1024.0,
                           r->ratio,
                           r->throughput_mbps,
                           r->max_error,
                           action,
                           gpucompress_algorithm_name((gpucompress_algorithm_t)(algo_id + 1)),
                           quant ? "+Q" : "",
                           shuf  ? "+S" : "");
                }
                printf("\n");
            }
        }
        gpucompress_cleanup();
    }

    /* ============================================================
     * Phase 3: NN AUTO — Balanced policy (w0=1, w1=1, w2=1)
     * ============================================================ */
    printf("======================================\n");
    printf("PHASE 3: NN AUTO — Balanced Policy\n");
    printf("  (w0=1, w1=1, w2=1: speed + ratio)\n");
    printf("======================================\n\n");

    gerr = gpucompress_init("neural_net/weights/model.nnwt");
    if (gerr != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "NN init failed, skipping Phase 3\n");
    } else {
        gpucompress_set_ranking_weights(1.0f, 1.0f, 1.0f);

        for (int d = 0; d < 4; d++) {
            for (int m = 0; m < 3; m++) {
                printf("--- %s / %s ---\n", datasets[d].name, modes[m].name);
                printf("  %-12s %8s %8s %10s %8s %6s\n",
                       "Policy", "Comp(KB)", "Ratio", "Speed(MB/s)", "MaxErr", "Action");

                unsigned int pp = modes[m].preproc;
                if (modes[m].error_bound == 0.0)
                    pp |= GPUCOMPRESS_PREPROC_SHUFFLE_4;

                int prev = g_n_results;
                run_one(datasets[d].d_data, datasets[d].nbytes,
                        datasets[d].total_floats,
                        GPUCOMPRESS_ALGO_AUTO, pp, modes[m].error_bound,
                        "NN-AUTO", "balanced", modes[m].name);

                if (g_n_results > prev) {
                    BenchResult* r = &g_results[g_n_results - 1];
                    int action = r->nn_action;
                    int algo_id = action % 8;
                    int quant   = (action / 8) % 2;
                    int shuf    = (action / 16) % 2;
                    printf("  %-12s %8.1f %8.2f %10.1f %8.4f  a%d(%s%s%s)\n",
                           "balanced",
                           r->compressed / 1024.0,
                           r->ratio,
                           r->throughput_mbps,
                           r->max_error,
                           action,
                           gpucompress_algorithm_name((gpucompress_algorithm_t)(algo_id + 1)),
                           quant ? "+Q" : "",
                           shuf  ? "+S" : "");
                }
                printf("\n");
            }
        }
        gpucompress_cleanup();
    }

    /* ============================================================
     * Summary table
     * ============================================================ */
    printf("======================================\n");
    printf("FULL RESULTS TABLE\n");
    printf("======================================\n\n");
    printf("%-20s %-12s %-10s %-12s %8s %8s %10s %8s %6s\n",
           "Dataset", "Algorithm", "Policy", "Mode",
           "Comp(KB)", "Ratio", "Speed(MB/s)", "MaxErr", "Action");
    printf("%-20s %-12s %-10s %-12s %8s %8s %10s %8s %6s\n",
           "--------------------", "------------", "----------", "------------",
           "--------", "--------", "----------", "--------", "------");

    for (int i = 0; i < g_n_results; i++) {
        BenchResult* r = &g_results[i];
        printf("%-20s %-12s %-10s %-12s %8.1f %8.2f %10.1f %8.4f %6d\n",
               /* pick dataset name from context — use algo+policy+mode as key */
               "",  /* will be printed in per-phase output */
               r->algo_name, r->policy, r->mode,
               r->compressed / 1024.0,
               r->ratio,
               r->throughput_mbps,
               r->max_error,
               r->nn_action);
    }

    /* Cleanup datasets */
    for (int d = 0; d < 4; d++) {
        CHECK_CUDA(cudaFree(datasets[d].d_data));
    }

    printf("\nBenchmark complete: %d configurations tested.\n", g_n_results);
    return 0;
}
