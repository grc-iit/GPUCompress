/**
 * @file synthetic-workloads-benchmark.cu
 * @brief Synthetic scientific workload benchmark for NN algorithm selection.
 *
 * Generates 20+ distinct data patterns that mimic real scientific workloads,
 * each producing very different (entropy, MAD, second_derivative) feature
 * vectors.  The goal is to stress-test whether the NN can differentiate
 * between data types and select different compression configs, or whether
 * it always collapses to a single choice.
 *
 * Data originates on the GPU, goes through the HDF5 VOL connector (same
 * path as production workloads).  Three NN phases are supported:
 *   - nn          : inference only (no learning)
 *   - nn-rl       : inference + online SGD
 *   - nn-rl+exp   : inference + SGD + Level-2 exploration
 *
 * Usage:
 *   ./build/synthetic_workloads_benchmark model.nnwt \
 *       [--L 256] [--chunk-mb 16] [--timesteps 10] [--runs 3] \
 *       [--phase nn] [--phase nn-rl] [--phase nn-rl+exp] \
 *       [--mode lossless] [--mode lossy] [--mode both] \
 *       [--error-bound 0.001] [--lr 0.1] \
 *       [--out-dir results/synthetic] [--verbose-chunks]
 *
 * Dataset size reference (--L):
 *   --L 128 →   8 MB      --L 640  → 1 GB
 *   --L 256 →  64 MB      --L 1000 → 4 GB
 *   --L 512 → 512 MB
 *
 * Patterns are grouped by which NN feature they stress:
 *
 * Group A — Entropy extremes:
 *    0: All zeros                   (ent≈0, MAD=0, deriv=0)
 *    1: Constant pi                 (ent≈2 bits, MAD=0, deriv=0)
 *    2: Small integers 0-7          (ent=low, MAD=low)
 *    3: High-entropy random         (ent=high, MAD=high, deriv=high)
 *
 * Group B — MAD extremes:
 *    4: Tight Gaussian(1000,0.001)  (MAD≈0)
 *    5: Wide uniform [0, 1e8]       (MAD=very high)
 *    6: Bimodal 0 / 1e6             (MAD=extreme)
 *    7: Sparse 1% spikes at 1e7     (MAD=low, range=huge)
 *
 * Group C — Second derivative extremes:
 *    8: Linear ramp                 (deriv2=0)
 *    9: Smooth parabola             (deriv2=constant)
 *   10: Alternating ±1e5            (deriv2=maximum)
 *   11: High-freq sine (500 cycles) (deriv=very high)
 *
 * Group D — Scientific workload mimics:
 *   12: CFD pressure field          (smooth + boundary layer spikes)
 *   13: Turbulence                  (multi-freq sum, high entropy)
 *   14: MD velocity                 (Maxwell-Boltzmann distribution)
 *   15: Climate SST                 (smooth gradient + tiny noise)
 *   16: Seismic wavelet             (sparse high-amplitude events)
 *
 * Group E — Compressibility spectrum:
 *   17: Repeated 16-float block     (trivial for LZ family)
 *   18: Monotonic integers          (great for delta/cascaded)
 *   19: Exponential decay           (high dynamic range)
 *   20: Log-normal                  (heavy tail, wide range)
 *   21: Power law                   (extreme outliers)
 *   22: Checkerboard 3D             (alternating 0/1e5)
 *   23: Mixed smooth + noise        (half smooth, half random)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <curand_kernel.h>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5.h"
#include "gpucompress_hdf5_vol.h"

/* ============================================================
 * Constants
 * ============================================================ */

#define N_PATTERNS 24

static const char *PATTERN_NAMES[N_PATTERNS] = {
    "all_zeros",           /*  0: ent≈0, MAD=0, deriv=0      */
    "constant_pi",         /*  1: ent≈2, MAD=0, deriv=0      */
    "small_int_0_7",       /*  2: ent≈low, MAD=low           */
    "high_ent_random",     /*  3: ent=high, MAD=high, deriv=high */
    "tight_gaussian",      /*  4: ent=med, MAD≈0             */
    "wide_uniform_1e8",    /*  5: ent=high, MAD=very high    */
    "bimodal_0_1e6",       /*  6: ent=low, MAD=extreme       */
    "sparse_1pct_1e7",     /*  7: ent≈0, MAD=low, range=huge */
    "linear_ramp",         /*  8: deriv2=0                    */
    "smooth_parabola",     /*  9: deriv2=const                */
    "alternating_1e5",     /* 10: deriv2=max                  */
    "hfreq_sine_500",      /* 11: deriv=very high             */
    "cfd_pressure",        /* 12: smooth+boundary spikes      */
    "turbulence",          /* 13: multi-freq, high entropy    */
    "md_velocity",         /* 14: Maxwell-Boltzmann           */
    "climate_sst",         /* 15: smooth gradient+noise       */
    "seismic_wavelet",     /* 16: sparse wavelets             */
    "repeated_block_16",   /* 17: trivial for LZ family       */
    "monotonic_int",       /* 18: great for delta/cascaded    */
    "exponential_decay",   /* 19: high dynamic range          */
    "log_normal",          /* 20: heavy tail                  */
    "power_law",           /* 21: extreme outliers            */
    "checkerboard_3d",     /* 22: alternating 0/1e5           */
    "mixed_smooth_noise",  /* 23: half smooth, half random    */
};

#define DEFAULT_L           256
#define DEFAULT_CHUNK_MB    16
#define DEFAULT_TIMESTEPS   5
#define DEFAULT_RUNS        1
#define DEFAULT_LR          0.1f
#define DEFAULT_SGD_MAPE    0.10f
#define DEFAULT_EXPL_MAPE   0.20f
#define DEFAULT_EXPL_K      4
#define DEFAULT_OUT_DIR     "benchmarks/synthetic/results"

#define TMP_FILE_PREFIX     "/tmp/bm_synth_"

/* HDF5 filter constants */
#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* Phase bit flags */
enum { P_NN = 1, P_NNRL = 2, P_NNRLEXP = 4 };

/* Compression mode */
enum CompMode { MODE_LOSSLESS = 0, MODE_LOSSY = 1, MODE_BOTH = 2 };

/* ============================================================
 * Helpers
 * ============================================================ */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static size_t file_size_bytes(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    off_t sz = lseek(fd, 0, SEEK_END);
    close(fd);
    return (sz < 0) ? 0 : (size_t)sz;
}

static void drop_pagecache(const char *path) {
    int fd = open(path, O_RDWR);
    if (fd < 0) return;
    fdatasync(fd);
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
}

static void mkdirs(const char *path) {
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

static void pack_double_cd(double v, unsigned int *lo, unsigned int *hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static const char *ACTION_ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate",
    "zstd", "ans", "cascaded", "bitcomp"
};

static void action_to_str(int action, char *buf, size_t bufsz) {
    if (action < 0) { snprintf(buf, bufsz, "none"); return; }
    int algo  = action % 8;
    int quant = (action / 8) % 2;
    int shuf  = (action / 16) % 2;
    snprintf(buf, bufsz, "%s%s%s", ACTION_ALGO_NAMES[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
}

/* ============================================================
 * HDF5 helpers
 * ============================================================ */

static hid_t make_vol_fapl(void) {
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

static hid_t make_dcpl_auto(int L, int chunk_z, double eb) {
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(eb, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

static hid_t make_dcpl_nocomp(int L, int chunk_z) {
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);
    return dcpl;
}

/* ============================================================
 * GPU kernel: generate scientific data patterns
 *
 * IMPORTANT: HDF5 chunks along the LAST dimension (dim 2) when
 * cdims = {L, L, chunk_z}.  So pattern assignment must use
 * dim2 = i % L, NOT dim0 = i / (L*L).  Otherwise each HDF5
 * chunk contains ALL patterns mixed together, producing
 * identical stats for every chunk.
 *
 * Each chunk (determined by dim2 / chunk_z) gets a pattern
 * based on (chunk_id % N_PATTERNS).  The seed shifts each
 * timestep to evolve the data while preserving pattern diversity.
 *
 * Patterns are designed to produce WIDE ranges of byte-level
 * entropy, MAD, and second derivative — the 3 data-dependent
 * features the NN uses for algorithm selection.
 * ============================================================ */

__device__ float gpu_hash_float(unsigned long long seed, size_t idx) {
    unsigned long long h = (seed ^ (idx * 6364136223846793005ULL)) + 1442695040888963407ULL;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL; h ^= h >> 33;
    return (float)(h & 0xFFFFFF) / (float)0xFFFFFF;  /* [0, 1) */
}

/* Box-Muller: two uniform -> one Gaussian */
__device__ float gpu_gaussian(unsigned long long seed, size_t idx) {
    float u1 = gpu_hash_float(seed, idx * 2 + 0);
    float u2 = gpu_hash_float(seed, idx * 2 + 1);
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/* ============================================================
 * Merged-pattern mode: 5 extreme sub-patterns within each chunk.
 *
 * Each chunk is divided into 5 bands by position.  The dominant
 * band (60% of the chunk) rotates per chunk_id, so different
 * chunks get different blended (entropy, MAD, deriv) stats.
 *
 * Sub-patterns chosen for maximum feature-space separation:
 *   A: all zeros        (ent≈0, MAD=0, deriv=0)
 *   B: random [0,1e8]   (ent≈7, MAD=high, deriv=high)
 *   C: bimodal 0/1e6    (ent≈1.5, MAD=0.5, deriv=1)
 *   D: smooth sine      (ent≈7, MAD≈0.3, deriv≈0)
 *   E: alternating ±1e5 (ent≈2.3, MAD=0.5, deriv=2)
 * ============================================================ */
#define MERGED_N_SUB 5

__device__ float merged_sub_value(int sub, float rnd, float gauss, float t) {
    switch (sub) {
    case 0: return 0.0f;                                         /* zeros */
    case 1: return rnd * 1.0e8f;                                 /* wide uniform */
    case 2: return (rnd < 0.5f) ? 0.0f : 1.0e6f;                /* bimodal */
    case 3: return sinf(t * 2.0f * 3.14159265f * 5.0f) * 5000.0f; /* smooth sine */
    case 4: return ((int)(t * 1000.0f) & 1) ? 1.0e5f : -1.0e5f; /* alternating */
    default: return 0.0f;
    }
}

__global__ void generate_patterns_kernel(
    float *data, int L, int chunk_z, int n_chunks,
    unsigned long long seed, int pattern_offset,
    int merged_mode)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)L * L * L;

    for (size_t i = idx; i < total; i += (size_t)gridDim.x * blockDim.x) {
        /* ── Chunk ID from dim 2 (last dimension, which HDF5 chunks along) ── */
        int dim2 = (int)(i % L);
        int chunk_id = dim2 / chunk_z;
        int pattern = (chunk_id + pattern_offset) % N_PATTERNS;

        /* Local position within the HDF5 chunk for smooth gradients.
         * HDF5 chunk k covers dim2 in [k*chunk_z, (k+1)*chunk_z) for all dim0,dim1.
         * We use a simple normalized position based on the element index within
         * the chunk's linear extent. */
        int dim0 = (int)(i / ((size_t)L * L));
        int dim1 = (int)((i / L) % L);
        int local_z = dim2 - chunk_id * chunk_z;
        size_t local = (size_t)dim0 * L * chunk_z + (size_t)dim1 * chunk_z + local_z;
        size_t chunk_elems = (size_t)L * L * chunk_z;

        /* Normalized position [0, 1) within chunk */
        float t = (float)local / (float)chunk_elems;

        /* Per-element randomness */
        float rnd   = gpu_hash_float(seed, i);
        float gauss = gpu_gaussian(seed + 104729ULL, i);

        float val = 0.0f;

        /* ── Merged mode: 5 sub-patterns per chunk ──
         * The dominant sub-pattern (60% of chunk) rotates by chunk_id.
         * Remaining 40% is split among the other 4 sub-patterns (10% each).
         * This produces different blended stats per chunk. */
        if (merged_mode) {
            int dominant = chunk_id % MERGED_N_SUB;
            /* Map t to band: [0, 0.6) = dominant, then 4 x [0.6+k*0.1, 0.6+(k+1)*0.1) */
            int sub;
            if (t < 0.6f) {
                sub = dominant;
            } else {
                int minor_idx = (int)((t - 0.6f) / 0.1f);
                if (minor_idx > 3) minor_idx = 3;
                /* Skip dominant in the minor rotation */
                sub = minor_idx;
                if (sub >= dominant) sub++;
                if (sub >= MERGED_N_SUB) sub = 0;
            }
            val = merged_sub_value(sub, rnd, gauss, t);
            data[i] = val;
            continue;
        }

        switch (pattern) {

        /* ── Group A: Extreme entropy differences ──────────────── */

        case 0:  /* ALL ZEROS: entropy≈0, MAD=0, deriv=0 */
            val = 0.0f;
            break;

        case 1:  /* CONSTANT nonzero: entropy≈2 bits (4 unique bytes), MAD=0 */
            val = 3.14159265f;
            break;

        case 2:  /* SMALL INTEGERS 0-7: low entropy (few byte patterns), low MAD */
        {
            int v = (int)(rnd * 8.0f);
            if (v > 7) v = 7;
            val = (float)v;
            break;
        }

        case 3:  /* HIGH-ENTROPY RANDOM: varied byte patterns, high MAD, high deriv */
        {
            /* Use wide-range random floats across many orders of magnitude
             * to maximize byte-level entropy while staying finite.
             * rnd in [0,1) → exponent range via pow gives diverse byte patterns. */
            float sign = (rnd < 0.5f) ? 1.0f : -1.0f;
            float mag = gpu_hash_float(seed + 55555ULL, i);
            val = sign * powf(10.0f, mag * 8.0f - 4.0f);  /* range: [1e-4, 1e4] both signs */
            break;
        }

        /* ── Group B: MAD extremes ─────────────────────────────── */

        case 4:  /* TIGHT CLUSTER: Gaussian(1000, 0.001) — near-zero MAD */
            val = 1000.0f + gauss * 0.001f;
            break;

        case 5:  /* WIDE UNIFORM: [0, 1e8] — very high MAD */
            val = rnd * 1.0e8f;
            break;

        case 6:  /* BIMODAL: half at 0, half at 1e6 — extreme MAD */
            val = (rnd < 0.5f) ? 0.0f : 1.0e6f;
            break;

        case 7:  /* SPARSE SPIKES: 99% zeros, 1% at 1e7 — low MAD, extreme range */
            val = (rnd < 0.01f) ? 1.0e7f : 0.0f;
            break;

        /* ── Group C: Second derivative extremes ───────────────── */

        case 8:  /* LINEAR RAMP: constant first deriv, zero second deriv */
            val = t * 50000.0f;
            break;

        case 9:  /* SMOOTH PARABOLA: constant second deriv */
            val = t * t * 50000.0f;
            break;

        case 10: /* ALTERNATING ±1e5: maximum second derivative */
            val = ((local & 1) == 0) ? 1.0e5f : -1.0e5f;
            break;

        case 11: /* HIGH-FREQ SINE: 500 cycles, high deriv */
            val = sinf(t * 2.0f * 3.14159265f * 500.0f) * 10000.0f;
            break;

        /* ── Group D: Scientific workload mimics ───────────────── */

        case 12: /* CFD PRESSURE: smooth + boundary layer spikes */
        {
            float r = fabsf(t - 0.5f) * 2.0f;
            float smooth = sinf(t * 3.14159265f) * 1000.0f;
            float boundary = (r > 0.95f) ? (r - 0.95f) * 200000.0f : 0.0f;
            val = smooth + boundary;
            break;
        }

        case 13: /* TURBULENCE: multi-frequency sum — high entropy, high deriv */
        {
            float pi2 = 2.0f * 3.14159265f;
            val = sinf(t * pi2 * 3.0f) * 5000.0f
                + sinf(t * pi2 * 17.0f) * 3000.0f
                + sinf(t * pi2 * 71.0f) * 1500.0f
                + sinf(t * pi2 * 307.0f) * 800.0f
                + gauss * 200.0f;
            break;
        }

        case 14: /* MD VELOCITY: Maxwell-Boltzmann, always positive */
        {
            float g1 = gpu_gaussian(seed + 200003ULL, i);
            float g2 = gpu_gaussian(seed + 300007ULL, i);
            float g3 = gpu_gaussian(seed + 400009ULL, i);
            val = sqrtf(g1*g1 + g2*g2 + g3*g3) * 500.0f;
            break;
        }

        case 15: /* CLIMATE SST: smooth gradient + tiny noise */
            val = 28.0f - 25.0f * t * t + gauss * 0.1f;
            break;

        case 16: /* SEISMIC: background ≈0, rare large wavelets */
        {
            float frac = fmodf(t, 0.05f) / 0.05f - 0.5f;
            float sigma = 0.08f;
            float ricker = (1.0f - frac*frac/(sigma*sigma))
                         * expf(-frac*frac/(2.0f*sigma*sigma));
            val = (fabsf(frac) < 0.25f) ? ricker * 50000.0f : gauss * 0.5f;
            break;
        }

        /* ── Group E: Compressibility spectrum ─────────────────── */

        case 17: /* REPEATED 64-BYTE BLOCK: trivial for LZ family */
        {
            int blk = (int)(local % 16);  /* 16 distinct floats, tiled */
            val = (float)(blk * blk) * 100.0f;
            break;
        }

        case 18: /* MONOTONIC INTEGERS: great for delta/cascaded */
            val = (float)local;
            break;

        case 19: /* EXPONENTIAL DECAY: high dynamic range */
            val = expf(-8.0f * t) * 1.0e6f;
            break;

        case 20: /* LOG-NORMAL: heavy tail, wide range */
        {
            float ln_val = expf(gauss * 2.0f + 5.0f);
            val = fminf(ln_val, 1.0e8f);
            break;
        }

        case 21: /* POWER LAW: extreme outliers */
        {
            float u = fmaxf(rnd, 1.0e-6f);
            val = powf(u, -2.0f);
            val = fminf(val, 1.0e8f);
            break;
        }

        case 22: /* CHECKERBOARD 3D: alternating 0/1e5, per 4×4 blocks */
        {
            int cell = ((dim0 / 4) + (dim1 / 4) + (local_z / 4)) & 1;
            val = cell ? 1.0e5f : 0.0f;
            break;
        }

        case 23: /* MIXED SMOOTH+NOISE: half chunk smooth, half random */
        {
            if (t < 0.5f) {
                val = sinf(t * 2.0f * 3.14159265f * 3.0f) * 500.0f + 500.0f;
            } else {
                val = rnd * 100000.0f;
            }
            break;
        }
        }

        data[i] = val;
    }
}

/* ============================================================
 * GPU-side byte-exact comparison
 * ============================================================ */

__global__ void count_mismatches_kernel(const float * __restrict__ a,
                                        const float * __restrict__ b,
                                        size_t n,
                                        unsigned long long *count)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long local_cnt = 0;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (a[i] != b[i]) local_cnt++;
    atomicAdd(count, local_cnt);
}

static unsigned long long gpu_compare(const float *a, const float *b,
                                      size_t n_floats,
                                      unsigned long long *d_count)
{
    cudaMemset(d_count, 0, sizeof(unsigned long long));
    count_mismatches_kernel<<<512, 256>>>(a, b, n_floats, d_count);
    cudaDeviceSynchronize();
    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(h_count), cudaMemcpyDeviceToHost);
    return h_count;
}

/* ============================================================
 * Result struct
 * ============================================================ */

typedef struct {
    char   phase[20];
    char   mode[12];           /* "lossless" or "lossy" */
    double error_bound;
    double write_ms;
    double read_ms;
    size_t file_bytes;
    size_t orig_bytes;
    double ratio;
    double write_mbps;
    double read_mbps;
    unsigned long long mismatches;
    int    sgd_fires;
    int    explorations;
    int    n_chunks;
    double smape_ratio_pct;
    double smape_comp_pct;
    double smape_decomp_pct;
    double mape_ratio_pct;
    double mape_comp_pct;
    double mape_decomp_pct;
    double nn_ms;
    double stats_ms;
    double preproc_ms;
    double comp_ms;
    double decomp_ms;
    double explore_ms;
    double sgd_ms;
    double comp_gbps;
    double decomp_gbps;
    int    n_unique_configs;  /* how many distinct configs chosen */
} PhaseRunResult;

/* ============================================================
 * Per-run collection logic (single write+read+verify)
 * ============================================================ */

static int run_phase(float *d_data, float *d_read,
                     unsigned long long *d_count,
                     size_t n_floats, int L, int chunk_z,
                     const char *phase_name, const char *mode_name,
                     double error_bound, hid_t dcpl,
                     const char *tmp_file,
                     PhaseRunResult *r)
{
    memset(r, 0, sizeof(*r));
    size_t total_bytes = n_floats * sizeof(float);
    int n_chunks = (L + chunk_z - 1) / chunk_z;
    hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };

    /* Write */
    gpucompress_reset_chunk_history();
    H5VL_gpucompress_reset_stats();
    remove(tmp_file);

    hid_t fapl = make_vol_fapl();
    hid_t file = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "H5Fcreate failed\n"); return 1; }

    hid_t fsp  = H5Screate_simple(3, dims, NULL);
    hid_t dset = H5Dcreate2(file, "V", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);

    double t0 = now_ms();
    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                            H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(dset); H5Fclose(file);
    double t1 = now_ms();
    if (wret < 0) { fprintf(stderr, "H5Dwrite failed\n"); return 1; }

    drop_pagecache(tmp_file);

    /* Read */
    fapl = make_vol_fapl();
    file = H5Fopen(tmp_file, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "V", H5P_DEFAULT);

    double t2 = now_ms();
    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(dset); H5Fclose(file);
    double t3 = now_ms();
    if (rret < 0) { fprintf(stderr, "H5Dread failed\n"); return 1; }

    /* Verify */
    unsigned long long mm = 0;
    if (error_bound == 0.0) {
        mm = gpu_compare(d_data, d_read, n_floats, d_count);
    }
    /* For lossy mode, mismatches are expected — skip comparison */

    /* Collect per-chunk diagnostics */
    int sgd_fires = 0, explorations = 0;
    double smape_r = 0, smape_c = 0, smape_d = 0;
    double mape_r = 0, mape_c = 0, mape_d = 0;
    int sr_cnt = 0, sc_cnt = 0, sd_cnt = 0;
    int mr_cnt = 0, mc_cnt = 0, md_cnt = 0;
    double total_nn_ms = 0, total_stats_ms = 0, total_preproc_ms = 0;
    double total_comp_ms = 0, total_decomp_ms = 0;
    double total_explore_ms = 0, total_sgd_ms = 0;

    /* Track unique configs */
    int config_seen[32];
    memset(config_seen, 0, sizeof(config_seen));

    int n_hist = gpucompress_get_chunk_history_count();
    for (int ci = 0; ci < n_hist; ci++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(ci, &d) != 0) continue;

        sgd_fires += d.sgd_fired;
        explorations += d.exploration_triggered;
        total_nn_ms += d.nn_inference_ms;
        total_stats_ms += d.stats_ms;
        total_preproc_ms += d.preprocessing_ms;
        total_comp_ms += d.compression_ms;
        total_decomp_ms += d.decompression_ms;
        total_explore_ms += d.exploration_ms;
        total_sgd_ms += d.sgd_update_ms;

        if (d.nn_action >= 0 && d.nn_action < 32)
            config_seen[d.nn_action] = 1;

        /* sMAPE */
        double dr = (fabs(d.actual_ratio) + fabs(d.predicted_ratio)) / 2.0;
        double dc = (fabs(d.compression_ms) + fabs(d.predicted_comp_time)) / 2.0;
        double dd = (fabs(d.decompression_ms) + fabs(d.predicted_decomp_time)) / 2.0;
        if (d.predicted_ratio > 0 && d.actual_ratio > 0 && dr > 0) {
            smape_r += fabs(d.predicted_ratio - d.actual_ratio) / dr;
            sr_cnt++;
        }
        if (d.compression_ms > 0 && d.predicted_comp_time > 0 && dc > 0) {
            smape_c += fabs(d.predicted_comp_time - d.compression_ms) / dc;
            sc_cnt++;
        }
        if (d.decompression_ms > 0 && d.predicted_decomp_time > 0 && dd > 0) {
            smape_d += fabs(d.predicted_decomp_time - d.decompression_ms) / dd;
            sd_cnt++;
        }
        /* MAPE */
        if (d.actual_ratio > 0) {
            mape_r += fabs(d.predicted_ratio - d.actual_ratio) / fabs(d.actual_ratio);
            mr_cnt++;
        }
        if (d.compression_ms > 0) {
            mape_c += fabs(d.predicted_comp_time - d.compression_ms) / fabs(d.compression_ms);
            mc_cnt++;
        }
        if (d.decompression_ms > 0) {
            mape_d += fabs(d.predicted_decomp_time - d.decompression_ms) / fabs(d.decompression_ms);
            md_cnt++;
        }
    }

    int n_unique = 0;
    for (int i = 0; i < 32; i++) n_unique += config_seen[i];

    size_t fbytes = file_size_bytes(tmp_file);
    snprintf(r->phase, sizeof(r->phase), "%s", phase_name);
    snprintf(r->mode, sizeof(r->mode), "%s", mode_name);
    r->error_bound  = error_bound;
    r->write_ms     = t1 - t0;
    r->read_ms      = t3 - t2;
    r->file_bytes   = fbytes;
    r->orig_bytes   = total_bytes;
    r->ratio        = (double)total_bytes / (double)(fbytes ? fbytes : 1);
    r->write_mbps   = (double)total_bytes / (1 << 20) / ((t1 - t0) / 1000.0);
    r->read_mbps    = (double)total_bytes / (1 << 20) / ((t3 - t2) / 1000.0);
    r->mismatches   = mm;
    r->sgd_fires    = sgd_fires;
    r->explorations = explorations;
    r->n_chunks     = n_chunks;
    r->smape_ratio_pct  = sr_cnt ? smape_r / sr_cnt * 100.0 : 0.0;
    r->smape_comp_pct   = sc_cnt ? smape_c / sc_cnt * 100.0 : 0.0;
    r->smape_decomp_pct = sd_cnt ? smape_d / sd_cnt * 100.0 : 0.0;
    r->mape_ratio_pct   = mr_cnt ? mape_r / mr_cnt * 100.0 : 0.0;
    r->mape_comp_pct    = mc_cnt ? mape_c / mc_cnt * 100.0 : 0.0;
    r->mape_decomp_pct  = md_cnt ? mape_d / md_cnt * 100.0 : 0.0;
    r->nn_ms         = total_nn_ms;
    r->stats_ms      = total_stats_ms;
    r->preproc_ms    = total_preproc_ms;
    r->comp_ms       = total_comp_ms;
    r->decomp_ms     = total_decomp_ms;
    r->explore_ms    = total_explore_ms;
    r->sgd_ms        = total_sgd_ms;
    r->comp_gbps     = total_comp_ms > 0 ? (double)total_bytes / total_comp_ms / 1e6 : 0.0;
    r->decomp_gbps   = total_decomp_ms > 0 ? (double)total_bytes / total_decomp_ms / 1e6 : 0.0;
    r->n_unique_configs = n_unique;

    remove(tmp_file);
    return (error_bound > 0.0 || mm == 0) ? 0 : 1;
}

/* ============================================================
 * CSV writers
 * ============================================================ */

static void write_aggregate_csv(const char *path, PhaseRunResult *res, int n,
                                int L, int chunk_z, int timesteps)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }
    fprintf(f, "phase,mode,error_bound,L,chunk_z,n_chunks,timesteps,"
               "write_ms,read_ms,file_mib,orig_mib,ratio,"
               "write_mibps,read_mibps,comp_gbps,decomp_gbps,"
               "nn_ms,stats_ms,preproc_ms,comp_ms,decomp_ms,explore_ms,sgd_ms,"
               "mismatches,sgd_fires,explorations,n_unique_configs,"
               "smape_ratio_pct,smape_comp_pct,smape_decomp_pct,"
               "mape_ratio_pct,mape_comp_pct,mape_decomp_pct\n");
    for (int i = 0; i < n; i++) {
        PhaseRunResult *r = &res[i];
        fprintf(f, "%s,%s,%.6f,%d,%d,%d,%d,"
                   "%.2f,%.2f,%.2f,%.2f,%.4f,"
                   "%.1f,%.1f,%.3f,%.3f,"
                   "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                   "%llu,%d,%d,%d,"
                   "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                r->phase, r->mode, r->error_bound, L, chunk_z, r->n_chunks, timesteps,
                r->write_ms, r->read_ms,
                (double)r->file_bytes / (1 << 20),
                (double)r->orig_bytes / (1 << 20), r->ratio,
                r->write_mbps, r->read_mbps, r->comp_gbps, r->decomp_gbps,
                r->nn_ms, r->stats_ms, r->preproc_ms,
                r->comp_ms, r->decomp_ms, r->explore_ms, r->sgd_ms,
                r->mismatches, r->sgd_fires, r->explorations,
                r->n_unique_configs,
                r->smape_ratio_pct, r->smape_comp_pct, r->smape_decomp_pct,
                r->mape_ratio_pct, r->mape_comp_pct, r->mape_decomp_pct);
    }
    fclose(f);
    printf("\n  Aggregate CSV: %s\n", path);
}

static void write_chunk_csv(const char *path, const char *phase_name,
                            const char *mode_name, int timestep,
                            int n_chunks, bool append)
{
    FILE *f = fopen(path, append ? "a" : "w");
    if (!f) { perror(path); return; }
    if (!append) {
        fprintf(f, "phase,mode,timestep,chunk,pattern,action_final,action_orig,"
                   "actual_ratio,predicted_ratio,smape_ratio,mape_ratio,"
                   "actual_comp_ms,predicted_comp_ms,smape_comp,mape_comp,"
                   "actual_decomp_ms,predicted_decomp_ms,smape_decomp,mape_decomp,"
                   "sgd_fired,exploration_triggered,"
                   "nn_inference_ms,compression_ms,exploration_ms,sgd_update_ms,"
                   "feat_entropy,feat_mad,feat_deriv,feat_eb_enc,feat_ds_enc\n");
    }

    int n_hist = gpucompress_get_chunk_history_count();
    for (int ci = 0; ci < n_hist && ci < n_chunks; ci++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(ci, &d) != 0) continue;

        char final_str[40], orig_str[40];
        action_to_str(d.nn_action, final_str, sizeof(final_str));
        action_to_str(d.nn_original_action, orig_str, sizeof(orig_str));

        /* pattern name for this chunk */
        int pattern_idx = ci % N_PATTERNS;
        const char *pname = PATTERN_NAMES[pattern_idx];

        double dr = (fabs(d.actual_ratio) + fabs(d.predicted_ratio)) / 2.0;
        double dc = (fabs(d.compression_ms) + fabs(d.predicted_comp_time)) / 2.0;
        double dd = (fabs(d.decompression_ms) + fabs(d.predicted_decomp_time)) / 2.0;
        double sr = dr > 0 ? fabs(d.predicted_ratio - d.actual_ratio) / dr * 100.0 : 0.0;
        double sc = dc > 0 ? fabs(d.predicted_comp_time - d.compression_ms) / dc * 100.0 : 0.0;
        double sd = dd > 0 ? fabs(d.predicted_decomp_time - d.decompression_ms) / dd * 100.0 : 0.0;
        double mr = d.actual_ratio > 0
            ? fabs(d.predicted_ratio - d.actual_ratio) / fabs(d.actual_ratio) * 100.0 : 0.0;
        double mc = d.compression_ms > 0
            ? fabs(d.predicted_comp_time - d.compression_ms) / fabs(d.compression_ms) * 100.0 : 0.0;
        double md = d.decompression_ms > 0
            ? fabs(d.predicted_decomp_time - d.decompression_ms) / fabs(d.decompression_ms) * 100.0 : 0.0;

        fprintf(f, "%s,%s,%d,%d,%s,%s,%s,"
                   "%.4f,%.4f,%.1f,%.1f,"
                   "%.3f,%.3f,%.1f,%.1f,"
                   "%.3f,%.3f,%.1f,%.1f,"
                   "%d,%d,"
                   "%.3f,%.3f,%.3f,%.3f,"
                   "%.4f,%.6f,%.6f,%.4f,%.4f\n",
                phase_name, mode_name, timestep, ci, pname, final_str, orig_str,
                (double)d.actual_ratio, (double)d.predicted_ratio, sr, mr,
                (double)d.compression_ms, (double)d.predicted_comp_time, sc, mc,
                (double)d.decompression_ms, (double)d.predicted_decomp_time, sd, md,
                d.sgd_fired, d.exploration_triggered,
                (double)d.nn_inference_ms, (double)d.compression_ms,
                (double)d.exploration_ms, (double)d.sgd_update_ms,
                (double)d.feat_entropy, (double)d.feat_mad, (double)d.feat_deriv,
                (double)d.feat_eb_enc, (double)d.feat_ds_enc);
    }
    fclose(f);
}

static void write_timestep_csv(FILE *f, const char *phase_name,
                               const char *mode_name, int timestep,
                               PhaseRunResult *r)
{
    fprintf(f, "%s,%s,%d,%.2f,%.2f,%.4f,"
               "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
               "%d,%d,%d,%d,%llu\n",
            phase_name, mode_name, timestep,
            r->write_ms, r->read_ms, r->ratio,
            r->smape_ratio_pct, r->smape_comp_pct, r->smape_decomp_pct,
            r->mape_ratio_pct, r->mape_comp_pct, r->mape_decomp_pct,
            r->sgd_fires, r->explorations, r->n_chunks,
            r->n_unique_configs, r->mismatches);
}

/* ============================================================
 * Console summary
 * ============================================================ */

static void print_summary(PhaseRunResult *res, int n, int L, int chunk_z)
{
    double dataset_mb = (double)L * L * L * sizeof(float) / (1 << 20);
    double chunk_mb   = (double)L * L * chunk_z * sizeof(float) / (1024.0 * 1024.0);
    int n_chunks      = (L + chunk_z - 1) / chunk_z;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Synthetic Workloads Benchmark Summary                                              ║\n");
    printf("║  Grid: %d^3 (%.0f MiB)  Chunks: %d x %.1f MiB  Patterns: %d                 \n",
           L, dataset_mb, n_chunks, chunk_mb, N_PATTERNS);
    printf("╠══════════════╦══════════╦═════════╦═══════╦═══════╦═══════╦══════╦═══════╦══════════╣\n");
    printf("║  Phase       ║ Mode     ║Write    ║Read   ║ Comp  ║ Ratio ║ File ║Unique ║ Verify   ║\n");
    printf("║              ║          ║(MiB/s)  ║(MiB/s)║(GB/s) ║       ║(MiB) ║Configs║          ║\n");
    printf("╠══════════════╬══════════╬═════════╬═══════╬═══════╬═══════╬══════╬═══════╬══════════╣\n");
    for (int i = 0; i < n; i++) {
        PhaseRunResult *r = &res[i];
        const char *verdict;
        if (r->error_bound > 0.0)
            verdict = "lossy";
        else
            verdict = (r->mismatches == 0) ? "PASS" : "FAIL";
        printf("║  %-12s║ %-8s ║ %7.0f ║ %5.0f ║ %5.2f ║%5.2fx ║ %4.0f ║  %2d   ║ %-8s ║\n",
               r->phase, r->mode, r->write_mbps, r->read_mbps,
               r->comp_gbps, r->ratio,
               (double)r->file_bytes / (1 << 20),
               r->n_unique_configs, verdict);
    }
    printf("╚══════════════╩══════════╩═════════╩═══════╩═══════╩═══════╩══════╩═══════╩══════════╝\n");

    /* NN detail */
    for (int i = 0; i < n; i++) {
        PhaseRunResult *r = &res[i];
        printf("\n  %-12s [%s] SGD:%d/%d Expl:%d/%d UniqueConfigs:%d\n"
               "    sMAPE: ratio=%.1f%% comp=%.1f%% decomp=%.1f%%\n"
               "     MAPE: ratio=%.1f%% comp=%.1f%% decomp=%.1f%%\n",
               r->phase, r->mode,
               r->sgd_fires, r->n_chunks, r->explorations, r->n_chunks,
               r->n_unique_configs,
               r->smape_ratio_pct, r->smape_comp_pct, r->smape_decomp_pct,
               r->mape_ratio_pct, r->mape_comp_pct, r->mape_decomp_pct);
    }
}

static void print_per_chunk_selection(int n_chunks) {
    int n_hist = gpucompress_get_chunk_history_count();
    int stride = (n_hist > 40) ? n_hist / 20 : 1;
    printf("    ── Per-chunk config selection (%d chunks) ──\n", n_hist);
    for (int ci = 0; ci < n_hist; ci++) {
        if (ci >= 5 && ci < n_hist - 3 && ci % stride != 0) continue;
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(ci, &d) != 0) continue;
        char astr[40];
        action_to_str(d.nn_action, astr, sizeof(astr));
        int pattern = ci % N_PATTERNS;
        printf("      C%3d [%-14s] %-18s  ratio=%5.1fx  pred=%5.1fx  "
               "ent=%.2f mad=%.4f drv=%.4f%s%s\n",
               ci, PATTERN_NAMES[pattern], astr,
               (double)d.actual_ratio, (double)d.predicted_ratio,
               (double)d.feat_entropy, (double)d.feat_mad, (double)d.feat_deriv,
               d.sgd_fired ? " [SGD]" : "",
               d.exploration_triggered ? " [EXP]" : "");
    }
    printf("    ── end ──\n");
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv)
{
    /* Defaults */
    const char *weights_path = NULL;
    int L         = DEFAULT_L;
    int chunk_z   = 0;
    int chunk_mb  = DEFAULT_CHUNK_MB;
    int timesteps = DEFAULT_TIMESTEPS;
    int n_runs    = DEFAULT_RUNS;
    float sgd_lr  = DEFAULT_LR;
    float sgd_mape = DEFAULT_SGD_MAPE;
    float expl_mape = DEFAULT_EXPL_MAPE;
    int   expl_k   = DEFAULT_EXPL_K;
    double error_bound = 0.001;  /* default lossy error bound */
    const char *out_dir_override = NULL;
    int verbose_chunks = 0;
    int merged_mode = 0;
    unsigned int phase_mask = 0;
    CompMode comp_mode = MODE_BOTH;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) {
            L = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chunk-mb") == 0 && i + 1 < argc) {
            chunk_mb = atoi(argv[++i]); chunk_z = 0;
        } else if (strcmp(argv[i], "--chunk-z") == 0 && i + 1 < argc) {
            chunk_z = atoi(argv[++i]); chunk_mb = 0;
        } else if (strcmp(argv[i], "--timesteps") == 0 && i + 1 < argc) {
            timesteps = atoi(argv[++i]);
            if (timesteps < 1) timesteps = 1;
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            n_runs = atoi(argv[++i]);
            if (n_runs < 1) n_runs = 1;
            if (n_runs > 32) { printf("  WARNING: --runs %d capped to 32\n", n_runs); n_runs = 32; }
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            sgd_lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--sgd-mape") == 0 && i + 1 < argc) {
            sgd_mape = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--expl-mape") == 0 && i + 1 < argc) {
            expl_mape = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--expl-k") == 0 && i + 1 < argc) {
            expl_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--error-bound") == 0 && i + 1 < argc) {
            error_bound = atof(argv[++i]);
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            out_dir_override = argv[++i];
        } else if (strcmp(argv[i], "--verbose-chunks") == 0) {
            verbose_chunks = 1;
        } else if (strcmp(argv[i], "--single-merged") == 0) {
            merged_mode = 1;
        } else if (strcmp(argv[i], "--phase") == 0 && i + 1 < argc) {
            const char *p = argv[++i];
            if      (strcmp(p, "nn") == 0)         phase_mask |= P_NN;
            else if (strcmp(p, "nn-rl") == 0)      phase_mask |= P_NNRL;
            else if (strcmp(p, "nn-rl+exp") == 0)  phase_mask |= P_NNRLEXP;
            else {
                fprintf(stderr, "Unknown phase: %s\n"
                        "  Valid: nn, nn-rl, nn-rl+exp\n", p);
                return 1;
            }
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if      (strcmp(m, "lossless") == 0) comp_mode = MODE_LOSSLESS;
            else if (strcmp(m, "lossy") == 0)    comp_mode = MODE_LOSSY;
            else if (strcmp(m, "both") == 0)     comp_mode = MODE_BOTH;
            else {
                fprintf(stderr, "Unknown mode: %s\n"
                        "  Valid: lossless, lossy, both\n", m);
                return 1;
            }
        } else if (argv[i][0] != '-') {
            weights_path = argv[i];
        }
    }

    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr,
            "Usage: %s <weights.nnwt> [options]\n\n"
            "Options:\n"
            "  --L N              Grid side length (default %d)\n"
            "  --chunk-mb N       Chunk size in MB (default %d)\n"
            "  --chunk-z Z        Chunk Z-dimension directly\n"
            "  --timesteps N      Multi-timestep iterations (default %d)\n"
            "  --runs N           Repeat single-shot N times (default %d)\n"
            "  --phase <name>     nn, nn-rl, nn-rl+exp (repeatable, default all)\n"
            "  --mode <m>         lossless, lossy, both (default both)\n"
            "  --error-bound E    Lossy error bound (default %.4f)\n"
            "  --lr F             SGD learning rate (default %.2f)\n"
            "  --sgd-mape F       SGD MAPE threshold (default %.2f)\n"
            "  --expl-mape F      Exploration MAPE threshold (default %.2f)\n"
            "  --expl-k N         Exploration alternatives (default %d)\n"
            "  --out-dir DIR      Output directory\n"
            "  --verbose-chunks   Print per-chunk detail\n"
            "  --single-merged    Merged mode: 5 sub-patterns per chunk,\n"
            "                     dominant rotates by chunk_id\n",
            argv[0], DEFAULT_L, DEFAULT_CHUNK_MB, DEFAULT_TIMESTEPS,
            DEFAULT_RUNS, error_bound, (double)DEFAULT_LR,
            (double)DEFAULT_SGD_MAPE, (double)DEFAULT_EXPL_MAPE, DEFAULT_EXPL_K);
        return 1;
    }

    if (phase_mask == 0) phase_mask = P_NN | P_NNRL | P_NNRLEXP;

    /* Compute chunk_z */
    if (chunk_z <= 0 && chunk_mb > 0) {
        chunk_z = (int)((size_t)chunk_mb * 1024 * 1024 / ((size_t)L * L * sizeof(float)));
        if (chunk_z < 1) chunk_z = 1;
        if (chunk_z > L) chunk_z = L;
    }
    if (chunk_z < 1) chunk_z = L / 4;
    if (chunk_z < 1) chunk_z = 1;

    size_t n_floats    = (size_t)L * L * L;
    size_t total_bytes = n_floats * sizeof(float);
    int    n_chunks    = (L + chunk_z - 1) / chunk_z;
    double dataset_mib = (double)total_bytes / (1 << 20);
    double chunk_mib   = (double)L * L * chunk_z * sizeof(float) / (1024.0 * 1024.0);

    /* Determine which modes to run */
    int n_modes = 0;
    struct { const char *name; double eb; } modes[2];
    if (comp_mode == MODE_LOSSLESS || comp_mode == MODE_BOTH) {
        modes[n_modes].name = "lossless";
        modes[n_modes].eb   = 0.0;
        n_modes++;
    }
    if (comp_mode == MODE_LOSSY || comp_mode == MODE_BOTH) {
        modes[n_modes].name = "lossy";
        modes[n_modes].eb   = error_bound;
        n_modes++;
    }

    /* Determine which phases to run */
    struct { const char *name; unsigned int flag; int sgd; int explore; } phases[3];
    int n_phase_defs = 0;
    if (phase_mask & P_NN) {
        phases[n_phase_defs] = {"nn", P_NN, 0, 0};
        n_phase_defs++;
    }
    if (phase_mask & P_NNRL) {
        phases[n_phase_defs] = {"nn-rl", P_NNRL, 1, 0};
        n_phase_defs++;
    }
    if (phase_mask & P_NNRLEXP) {
        phases[n_phase_defs] = {"nn-rl+exp", P_NNRLEXP, 1, 1};
        n_phase_defs++;
    }

    /* Output paths */
    char out_dir[512], out_agg[512], out_chunks[512], out_tstep[512];
    {
        const char *od = out_dir_override ? out_dir_override : DEFAULT_OUT_DIR;
        snprintf(out_dir, sizeof(out_dir), "%s", od);
        snprintf(out_agg, sizeof(out_agg), "%s/synthetic_workloads_aggregate.csv", od);
        snprintf(out_chunks, sizeof(out_chunks), "%s/synthetic_workloads_chunks.csv", od);
        snprintf(out_tstep, sizeof(out_tstep), "%s/synthetic_workloads_timesteps.csv", od);
        mkdirs(od);
    }

    /* Banner */
    printf("╔══════════════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  Synthetic Scientific Workloads Benchmark                                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════════════╝\n\n");
    printf("  Grid       : %d^3 = %zu floats (%.1f MiB)\n", L, n_floats, dataset_mib);
    printf("  Chunks     : %d x %d x %d  (%d chunks, %.1f MiB each)\n",
           L, L, chunk_z, n_chunks, chunk_mib);
    if (merged_mode)
        printf("  Patterns   : MERGED (5 sub-patterns per chunk, dominant rotates)\n");
    else
        printf("  Patterns   : %d scientific workload patterns per cycle\n", N_PATTERNS);
    printf("  Timesteps  : %d\n", timesteps);
    if (n_runs > 1)
        printf("  Runs       : %d (single-shot mean)\n", n_runs);
    printf("  SGD        : LR=%.4f  MAPE_thresh=%.0f%%\n", sgd_lr, sgd_mape * 100.0);
    printf("  Exploration: MAPE_thresh=%.0f%%  K=%d\n", expl_mape * 100.0, expl_k);
    printf("  Mode       : %s", comp_mode == MODE_BOTH ? "lossless + lossy" :
                                 comp_mode == MODE_LOSSY ? "lossy" : "lossless");
    if (comp_mode != MODE_LOSSLESS)
        printf("  (eb=%.6f)", error_bound);
    printf("\n");
    printf("  Weights    : %s\n", weights_path);
    printf("  Output     : %s\n\n", out_dir);

    /* Pattern legend */
    printf("  Pattern assignments (chunk_id %% %d):\n", N_PATTERNS);
    for (int p = 0; p < N_PATTERNS; p++) {
        printf("    %2d: %s\n", p, PATTERN_NAMES[p]);
    }
    printf("\n");

    /* Init GPUCompress */
    if (gpucompress_init(weights_path) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "FATAL: NN weights not loaded\n");
        gpucompress_cleanup(); return 1;
    }

    /* Register VOL */
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "FATAL: VOL register failed\n");
        gpucompress_cleanup(); return 1;
    }
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* Allocate GPU buffers */
    float *d_data = NULL, *d_read = NULL;
    unsigned long long *d_count = NULL;
    cudaMalloc(&d_data, total_bytes);
    cudaMalloc(&d_read, total_bytes);
    cudaMalloc(&d_count, sizeof(unsigned long long));

    /* Generate initial data */
    printf("── Generating %d-pattern synthetic data on GPU... ", N_PATTERNS);
    fflush(stdout);
    {
        int threads = 256;
        int blocks = (int)((n_floats + threads - 1) / threads);
        if (blocks > 65535) blocks = 65535;
        generate_patterns_kernel<<<blocks, threads>>>(
            d_data, L, chunk_z, n_chunks, 42ULL, 0, merged_mode);
        cudaDeviceSynchronize();
    }
    printf("done\n\n");

    /* ── Single-shot: run each (phase × mode) once ──────────── */

    /* Max results: 3 phases × 2 modes = 6 */
    PhaseRunResult all_results[6];
    int n_results = 0;
    int any_fail = 0;

    char tmp_file[256];

    /* Clear stale chunks CSV */
    remove(out_chunks);

    for (int mi = 0; mi < n_modes; mi++) {
        const char *mode_name = modes[mi].name;
        double eb = modes[mi].eb;
        hid_t dcpl = make_dcpl_auto(L, chunk_z, eb);

        for (int pi = 0; pi < n_phase_defs; pi++) {
            const char *pname = phases[pi].name;
            int do_sgd = phases[pi].sgd;
            int do_expl = phases[pi].explore;

            printf("══════════════════════════════════════════════════════════════\n");
            printf("  Phase: %s  Mode: %s  EB: %.6f\n", pname, mode_name, eb);
            printf("══════════════════════════════════════════════════════════════\n");

            /* Reload weights for fresh start */
            gpucompress_reload_nn(weights_path);

            /* Configure NN mode */
            if (do_sgd) {
                gpucompress_enable_online_learning();
                gpucompress_set_reinforcement(1, sgd_lr, sgd_mape, sgd_mape);
            } else {
                gpucompress_disable_online_learning();
            }
            if (do_expl) {
                gpucompress_set_exploration(1);
                gpucompress_set_exploration_threshold(expl_mape);
                gpucompress_set_exploration_k(expl_k);
            } else {
                gpucompress_set_exploration(0);
            }

            /* Warmup (learning disabled) */
            {
                int save_learning = gpucompress_online_learning_enabled();
                gpucompress_disable_online_learning();

                snprintf(tmp_file, sizeof(tmp_file), "%swarmup_%s_%s.h5",
                         TMP_FILE_PREFIX, pname, mode_name);
                remove(tmp_file);
                hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };
                hid_t wfapl = make_vol_fapl();
                hid_t wfile = H5Fcreate(tmp_file, H5F_ACC_TRUNC, H5P_DEFAULT, wfapl);
                H5Pclose(wfapl);
                if (wfile >= 0) {
                    hid_t wfsp = H5Screate_simple(3, dims, NULL);
                    hid_t wdset = H5Dcreate2(wfile, "V", H5T_NATIVE_FLOAT,
                                              wfsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
                    H5Sclose(wfsp);
                    if (wdset >= 0) {
                        H5Dwrite(wdset, H5T_NATIVE_FLOAT,
                                 H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
                        cudaDeviceSynchronize();
                        H5Dclose(wdset);
                    }
                    H5Fclose(wfile);
                    remove(tmp_file);
                }
                if (save_learning) gpucompress_enable_online_learning();
            }

            /* Timed run */
            snprintf(tmp_file, sizeof(tmp_file), "%s%s_%s.h5",
                     TMP_FILE_PREFIX, pname, mode_name);

            PhaseRunResult r;
            int rc = run_phase(d_data, d_read, d_count,
                               n_floats, L, chunk_z,
                               pname, mode_name, eb, dcpl, tmp_file, &r);
            if (rc) any_fail = 1;

            printf("  [%s/%s] ratio=%.2fx  write=%.0f MiB/s  read=%.0f MiB/s  "
                   "unique_configs=%d  sgd=%d  expl=%d  mm=%llu\n",
                   pname, mode_name, r.ratio, r.write_mbps, r.read_mbps,
                   r.n_unique_configs, r.sgd_fires, r.explorations, r.mismatches);

            if (verbose_chunks)
                print_per_chunk_selection(n_chunks);

            /* Write per-chunk CSV */
            write_chunk_csv(out_chunks, pname, mode_name, -1 /* single-shot */,
                            n_chunks, n_results > 0);

            all_results[n_results++] = r;
            gpucompress_disable_online_learning();
        }
        H5Pclose(dcpl);
    }

    /* ── Multi-timestep mode ─────────────────────────────────── */

    if (timesteps > 1) {
        FILE *ts_csv = fopen(out_tstep, "w");
        if (ts_csv) {
            fprintf(ts_csv, "phase,mode,timestep,write_ms,read_ms,ratio,"
                    "smape_ratio,smape_comp,smape_decomp,"
                    "mape_ratio,mape_comp,mape_decomp,"
                    "sgd_fires,explorations,n_chunks,n_unique_configs,mismatches\n");
        }

        for (int mi = 0; mi < n_modes; mi++) {
            const char *mode_name = modes[mi].name;
            double eb = modes[mi].eb;
            hid_t dcpl = make_dcpl_auto(L, chunk_z, eb);

            for (int pi = 0; pi < n_phase_defs; pi++) {
                /* Only nn-rl and nn-rl+exp benefit from multi-timestep */
                if (!phases[pi].sgd) continue;

                const char *pname = phases[pi].name;
                int do_expl = phases[pi].explore;

                printf("\n══════════════════════════════════════════════════════════════\n");
                printf("  Multi-timestep [%s/%s]: %d iterations\n",
                       pname, mode_name, timesteps);
                printf("══════════════════════════════════════════════════════════════\n\n");

                /* Fresh weights */
                gpucompress_reload_nn(weights_path);
                gpucompress_enable_online_learning();
                gpucompress_set_reinforcement(1, sgd_lr, sgd_mape, sgd_mape);
                if (do_expl) {
                    gpucompress_set_exploration(1);
                    gpucompress_set_exploration_threshold(expl_mape);
                    gpucompress_set_exploration_k(expl_k);
                } else {
                    gpucompress_set_exploration(0);
                }

                printf("  %-4s  %-7s  %-7s  %-7s  %-8s  %-8s  %-6s  %-4s  %-4s  %-3s\n",
                       "T", "WrMs", "RdMs", "Ratio",
                       "MAPE_R", "MAPE_C", "MAPE_D", "SGD", "EXP", "UCfg");
                printf("  ----  -------  -------  -------  "
                       "--------  --------  ------  ----  ----  ---\n");

                for (int t = 0; t < timesteps; t++) {
                    /* Regenerate data with shifted seed — preserves pattern types
                     * but changes values so NN/SGD sees evolving data */
                    {
                        int threads = 256;
                        int blocks = (int)((n_floats + threads - 1) / threads);
                        if (blocks > 65535) blocks = 65535;
                        generate_patterns_kernel<<<blocks, threads>>>(
                            d_data, L, chunk_z, n_chunks,
                            42ULL + (unsigned long long)t * 7919ULL, 0, merged_mode);
                        cudaDeviceSynchronize();
                    }

                    snprintf(tmp_file, sizeof(tmp_file), "%sts_%s_%s.h5",
                             TMP_FILE_PREFIX, pname, mode_name);

                    PhaseRunResult r;
                    int rc = run_phase(d_data, d_read, d_count,
                                       n_floats, L, chunk_z,
                                       pname, mode_name, eb, dcpl, tmp_file, &r);
                    if (rc) any_fail = 1;

                    printf("  %-4d  %6.0f  %6.0f   %5.2fx  %7.1f%%  %7.1f%%  %5.1f%%  %3d  %3d   %2d\n",
                           t, r.write_ms, r.read_ms, r.ratio,
                           r.mape_ratio_pct, r.mape_comp_pct, r.mape_decomp_pct,
                           r.sgd_fires, r.explorations, r.n_unique_configs);

                    if (ts_csv)
                        write_timestep_csv(ts_csv, pname, mode_name, t, &r);

                    /* Per-chunk CSV at milestones */
                    if (t == 0 || t == timesteps / 2 || t == timesteps - 1) {
                        write_chunk_csv(out_chunks, pname, mode_name, t,
                                        n_chunks, true);
                        if (verbose_chunks)
                            print_per_chunk_selection(n_chunks);
                    }
                }

                gpucompress_disable_online_learning();
            }
            H5Pclose(dcpl);
        }

        if (ts_csv) {
            fclose(ts_csv);
            printf("\n  Timestep CSV: %s\n", out_tstep);
        }
    }

    /* ── Summary ─────────────────────────────────────────────── */
    print_summary(all_results, n_results, L, chunk_z);
    write_aggregate_csv(out_agg, all_results, n_results, L, chunk_z, timesteps);
    printf("  Chunk CSV: %s\n", out_chunks);

    /* ── Cleanup ─────────────────────────────────────────────── */
    cudaFree(d_data);
    cudaFree(d_read);
    cudaFree(d_count);
    H5VLclose(vol_id);
    gpucompress_cleanup();

    printf("\n=== Benchmark %s ===\n", any_fail ? "FAILED" : "PASSED");
    return any_fail ? 1 : 0;
}
