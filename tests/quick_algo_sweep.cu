/**
 * quick_algo_sweep.cu
 *
 * For each of 8 GPU-generated data patterns, sweeps all 8 compression
 * algorithms x 2 shuffle modes and reports ratio + throughput.
 * Uses gpucompress_compress_gpu directly (no HDF5), so each combo is fast.
 *
 * Usage:
 *   ./build/quick_algo_sweep [--chunk-mb N]   (default 16)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "gpucompress.h"

/* ── GPU fill kernel (same 8 patterns as benchmark_vol_gpu) ── */

__device__ static float elem_rand(unsigned long long s)
{
    s ^= s >> 33; s *= 0xff51afd7ed558ccdULL;
    s ^= s >> 33; s *= 0xc4ceb9fe1a85ec53ULL;
    s ^= s >> 33;
    return (float)(s & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ static void fill_kernel(float *buf, size_t n, int pat)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        unsigned long long s = (unsigned long long)i * 6364136223846793005ULL;
        float r = elem_rand(s), v = 0.f;
        switch (pat) {
        case 0: v = 42.f; break;
        case 1: v = 1000.f * sinf(2.f*3.14159265f*(float)i/(float)n); break;
        case 2: v = (float)i/(float)n; break;
        case 3: v = (r < .01f) ? r*10000.f : 0.f; break;
        case 4: v = (float)(i/(n/8u))*100.f; break;
        case 5: v = 1000.f * expf(-5.f*(float)i/(float)n); break;
        case 6: { size_t p=n/16u; v=(float)(i%p)/(float)p*1000.f; break; }
        case 7: v = ((i%1024u)<4u) ? 5000.f+(r-.5f)*200.f : 0.f; break;
        }
        buf[i] = v;
    }
}

static const char *PNAMES[] = {
    "constant","smooth_sine","ramp","sparse",
    "step","exp_decay","sawtooth","impulse_train"
};
static const char *ANAMES[] = {
    "lz4","snappy","deflate","gdeflate","zstd","ans","cascaded","bitcomp"
};
/* gpucompress_algorithm_t enum: AUTO=0, LZ4=1..BITCOMP=8 */
static const gpucompress_algorithm_t ALGOS[] = {
    GPUCOMPRESS_ALGO_LZ4,
    GPUCOMPRESS_ALGO_SNAPPY,
    GPUCOMPRESS_ALGO_DEFLATE,
    GPUCOMPRESS_ALGO_GDEFLATE,
    GPUCOMPRESS_ALGO_ZSTD,
    GPUCOMPRESS_ALGO_ANS,
    GPUCOMPRESS_ALGO_CASCADED,
    GPUCOMPRESS_ALGO_BITCOMP,
};
#define N_ALGOS   8
#define N_PATS    8
#define N_SHUFFLE 2

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
}

typedef struct {
    int    algo;
    int    shuffle;
    double ratio;
    double mbps;
    int    ok;
} Combo;

int main(int argc, char **argv)
{
    int chunk_mb = 16;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i],"--chunk-mb") && i+1<argc) chunk_mb = atoi(argv[++i]);

    size_t n_floats   = (size_t)chunk_mb * 1024 * 1024 / sizeof(float);
    size_t data_bytes = n_floats * sizeof(float);
    size_t max_comp   = gpucompress_max_compressed_size(data_bytes);

    printf("Quick Algorithm Sweep — %d MB chunk, %d patterns x %d algos x %d shuffle\n\n",
           chunk_mb, N_PATS, N_ALGOS, N_SHUFFLE);

    /* Init library (weights not needed for fixed-algo compression) */
    gpucompress_init(NULL);

    /* Allocate GPU buffers once */
    void *d_in = NULL, *d_comp = NULL;
    cudaMalloc(&d_in,   data_bytes);
    cudaMalloc(&d_comp, max_comp);
    if (!d_in || !d_comp) { fprintf(stderr, "cudaMalloc failed\n"); return 1; }

    /* Results [pattern][algo][shuffle] */
    Combo results[N_PATS][N_ALGOS][N_SHUFFLE];

    for (int p = 0; p < N_PATS; p++) {
        /* Fill pattern */
        fill_kernel<<<512,256>>>((float*)d_in, n_floats, p);
        cudaDeviceSynchronize();

        for (int a = 0; a < N_ALGOS; a++) {
            for (int sh = 0; sh < N_SHUFFLE; sh++) {
                Combo *c = &results[p][a][sh];
                c->algo = a; c->shuffle = sh; c->ok = 0;
                c->ratio = 0; c->mbps = 0;

                gpucompress_config_t cfg = gpucompress_default_config();
                cfg.algorithm   = ALGOS[a];
                cfg.error_bound = 0.0;  /* lossless */
                cfg.preprocessing = sh ? GPUCOMPRESS_PREPROC_SHUFFLE_4 : GPUCOMPRESS_PREPROC_NONE;

                size_t comp_sz = max_comp;
                gpucompress_stats_t st = {};

                double t0 = now_ms();
                gpucompress_error_t e =
                    gpucompress_compress_gpu(d_in, data_bytes, d_comp,
                                            &comp_sz, &cfg, &st, NULL);
                double t1 = now_ms();

                if (e == GPUCOMPRESS_SUCCESS && comp_sz > 0) {
                    c->ratio = (double)data_bytes / (double)comp_sz;
                    c->mbps  = (double)data_bytes / (1<<20) / ((t1-t0)/1000.0);
                    c->ok    = 1;
                }
            }
        }
    }

    cudaFree(d_in);
    cudaFree(d_comp);
    gpucompress_cleanup();

    /* ── Print per-pattern table ── */
    for (int p = 0; p < N_PATS; p++) {
        printf("━━━ Pattern: %-14s (%d MB) ━━━━━━━━━━━━━━━━━━━━━━━\n",
               PNAMES[p], chunk_mb);
        printf("  %-10s  %-7s  %7s  %8s\n",
               "Algorithm", "Shuffle", "Ratio", "MB/s");
        printf("  %-10s  %-7s  %7s  %8s\n",
               "----------", "-------", "-------", "--------");

        /* Collect and sort by ratio descending */
        typedef struct { int a; int sh; double ratio; double mbps; } Row;
        Row rows[N_ALGOS*N_SHUFFLE];
        int nr = 0;
        for (int a = 0; a < N_ALGOS; a++)
            for (int sh = 0; sh < N_SHUFFLE; sh++)
                if (results[p][a][sh].ok) {
                    rows[nr].a     = a;
                    rows[nr].sh    = sh;
                    rows[nr].ratio = results[p][a][sh].ratio;
                    rows[nr].mbps  = results[p][a][sh].mbps;
                    nr++;
                }
        /* bubble sort */
        for (int i=0;i<nr-1;i++) for (int j=i+1;j<nr;j++)
            if (rows[j].ratio > rows[i].ratio) { Row t=rows[i]; rows[i]=rows[j]; rows[j]=t; }

        for (int i = 0; i < nr; i++) {
            const char *best = (i == 0) ? " ← best ratio" : "";
            printf("  %-10s  %-7s  %7.2fx  %8.0f%s\n",
                   ANAMES[rows[i].a],
                   rows[i].sh ? "+shuffle" : "none",
                   rows[i].ratio, rows[i].mbps, best);
        }
        printf("\n");
    }

    /* ── Summary: best algo per pattern ── */
    printf("━━━ Best Algorithm Per Pattern ━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  %-14s  %-14s  %-9s  %7s  %8s\n",
           "Pattern", "Best Algo", "Shuffle", "Ratio", "MB/s");
    printf("  %-14s  %-14s  %-9s  %7s  %8s\n",
           "--------------","-------------- ","-------","-------","--------");
    for (int p = 0; p < N_PATS; p++) {
        double best_r = 0; int ba = 0, bsh = 0;
        for (int a = 0; a < N_ALGOS; a++)
            for (int sh = 0; sh < N_SHUFFLE; sh++)
                if (results[p][a][sh].ok && results[p][a][sh].ratio > best_r) {
                    best_r = results[p][a][sh].ratio;
                    ba = a; bsh = sh;
                }
        printf("  %-14s  %-14s  %-9s  %7.2fx  %8.0f\n",
               PNAMES[p], ANAMES[ba],
               bsh ? "+shuffle" : "none",
               best_r, results[p][ba][bsh].mbps);
    }

    return 0;
}
