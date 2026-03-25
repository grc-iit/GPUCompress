/**
 * test_vol_lossless_stress.cu
 *
 * Stress test: GPU-generated adversarial data patterns → HDF5 VOL write
 * (ALGO_AUTO, NN selects algorithm) → read back → bitwise verify.
 *
 * Patterns designed to challenge the NN algorithm selector:
 *   1. Pure noise (incompressible — ratio ~1.0)
 *   2. Alternating NaN/Inf/denormal (edge-case floats)
 *   3. Bit-pattern adversarial (high entropy per byte, low per float)
 *   4. Sawtooth discontinuities (high derivative, poor for delta coding)
 *   5. Repeated single-chunk pattern (degenerate — all chunks identical)
 *   6. Mixed: half zeros, half high-entropy noise (bimodal distribution)
 *   7. Exponential blowup (values span 1e-38 to 1e+38)
 *   8. Quantization-hostile (values clustered near rounding boundaries)
 *   9. Byte-aligned repetition (4-byte period — shuffle-hostile)
 *  10. Ultra-sparse (99.99% zeros, rare large spikes)
 *
 * Each pattern fills a separate HDF5 dataset. All are lossless.
 * Failure = any bit mismatch between original GPU data and readback.
 *
 * Usage:
 *   ./test_vol_lossless_stress                 # default 4MB per pattern
 *   CHUNK_MB=2 TOTAL_MB=16 ./test_vol_lossless_stress
 */

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include <hdf5.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5
#define TMP_FILE "/tmp/test_vol_lossless_stress.h5"

static int g_pass = 0, g_fail = 0;
#define PASS(msg) do { fprintf(stderr, "  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { fprintf(stderr, "  FAIL: %s\n", msg); g_fail++; } while(0)

static void pack_double_cd(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits; memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/* ============================================================
 * GPU Data Generation Kernels — adversarial patterns
 * ============================================================ */

/* 1. Pure pseudorandom noise (incompressible) */
__global__ void gen_noise(float* out, size_t n, unsigned long long seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long x = (i + 1ULL) * 6364136223846793005ULL + seed;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    /* Interpret as raw float bits — may produce NaN/Inf but that's ok for bitwise test */
    unsigned int bits = (unsigned int)(x & 0x7F7FFFFF); /* clamp exponent to avoid NaN */
    if (x & 0x80000000ULL) bits |= 0x80000000u; /* random sign */
    memcpy(&out[i], &bits, sizeof(float));
}

/* 2. Alternating special floats: 0, -0, FLT_MAX, -FLT_MAX, FLT_MIN, FLT_EPSILON */
__global__ void gen_special_floats(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vals[] = { 0.0f, -0.0f, FLT_MAX, -FLT_MAX, FLT_MIN, FLT_EPSILON,
                     1.0f, -1.0f, 1e-30f, 1e30f, 1e-10f, -1e10f };
    out[i] = vals[i % 12];
}

/* 3. High byte-entropy, low float-entropy (shuffled bytes within each float) */
__global__ void gen_byte_adversarial(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* Each float has bytes [i%256, (i/256)%256, (i*7)%256, (i*13)%256] */
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(i % 256);
    bytes[1] = (unsigned char)((i / 256) % 256);
    bytes[2] = (unsigned char)((i * 7) % 256);
    bytes[3] = (unsigned char)((i * 13) % 251);  /* prime stride */
    /* Clamp exponent to avoid NaN/Inf: force exponent byte to [0x01, 0x7E] */
    bytes[3] = (bytes[3] % 126) + 1;
    memcpy(&out[i], bytes, sizeof(float));
}

/* 4. Sawtooth with sharp discontinuities every 64 elements */
__global__ void gen_sawtooth(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int phase = (int)(i % 64);
    int cycle = (int)(i / 64);
    float base = (float)cycle * 1000.0f;
    out[i] = base + (float)phase * 15.625f; /* ramp 0..999.something then jump */
}

/* 5. All identical values (degenerate — trivially compressible) */
__global__ void gen_constant_pi(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 3.14159265358979f;
}

/* 6. Bimodal: first half zeros, second half noise */
__global__ void gen_bimodal(float* out, size_t n, unsigned long long seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i < n / 2) {
        out[i] = 0.0f;
    } else {
        unsigned long long x = (i + 1ULL) * 2862933555777941757ULL + seed;
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        out[i] = (float)((int)(x % 10000)) / 100.0f - 50.0f; /* [-50, 50) */
    }
}

/* 7. Exponential range: values from 1e-38 to 1e+38 */
__global__ void gen_exp_range(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* Exponent sweeps from -38 to +38 linearly across the array */
    float t = (float)i / fmaxf((float)(n - 1), 1.0f); /* 0..1 */
    float exponent = -38.0f + t * 76.0f; /* -38..+38 */
    out[i] = powf(10.0f, exponent);
    if (i % 3 == 0) out[i] = -out[i]; /* sprinkle negatives */
}

/* 8. Quantization-hostile: values at x.499999 and x.500001 boundaries */
__global__ void gen_quant_hostile(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float base = (float)(i % 1000);
    /* Alternate between just-below and just-above .5 boundaries */
    if (i % 2 == 0)
        out[i] = base + 0.499999f;
    else
        out[i] = base + 0.500001f;
}

/* 9. 4-byte periodic (shuffle-hostile: shuffle produces identical byte planes) */
__global__ void gen_shuffle_hostile(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* Repeat the same 4-byte pattern with tiny variation */
    float base_vals[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    out[i] = base_vals[i % 4] + (float)(i / 4) * 1e-7f;
}

/* 10. Ultra-sparse: 99.99% zeros, rare spikes at ±1e6 */
__global__ void gen_ultra_sparse(float* out, size_t n, unsigned long long seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long x = (i + 1ULL) * 6364136223846793005ULL + seed;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    /* ~1 in 10000 chance of spike */
    if ((x & 0x3FFF) == 0) {
        out[i] = (x & 0x4000) ? 1e6f : -1e6f;
    } else {
        out[i] = 0.0f;
    }
}

/* ============================================================
 * Pattern registry
 * ============================================================ */
struct Pattern {
    const char* name;
    void (*launch)(float* d, size_t n, int blocks);
};

#define LAUNCH(kernel, ...) \
    static void launch_##kernel(float* d, size_t n, int blocks) { \
        kernel<<<blocks, 256>>>(d, n, ##__VA_ARGS__); \
    }

LAUNCH(gen_noise, 0xDEADBEEFULL)
LAUNCH(gen_special_floats)
LAUNCH(gen_byte_adversarial)
LAUNCH(gen_sawtooth)
LAUNCH(gen_constant_pi)
LAUNCH(gen_bimodal, 0xCAFEBABEULL)
LAUNCH(gen_exp_range)
LAUNCH(gen_quant_hostile)
LAUNCH(gen_shuffle_hostile)
LAUNCH(gen_ultra_sparse, 0x1337C0DEULL)

static Pattern PATTERNS[] = {
    { "noise_incompressible",  launch_gen_noise },
    { "special_floats",        launch_gen_special_floats },
    { "byte_adversarial",      launch_gen_byte_adversarial },
    { "sawtooth_discontinuous", launch_gen_sawtooth },
    { "constant_pi",           launch_gen_constant_pi },
    { "bimodal_zero_noise",    launch_gen_bimodal },
    { "exponential_range",     launch_gen_exp_range },
    { "quant_hostile",         launch_gen_quant_hostile },
    { "shuffle_hostile",       launch_gen_shuffle_hostile },
    { "ultra_sparse",          launch_gen_ultra_sparse },
};
static const int N_PATTERNS = sizeof(PATTERNS) / sizeof(PATTERNS[0]);

/* ============================================================
 * Bitwise verify on host
 * ============================================================ */
__global__ void gpu_compare_kernel(const unsigned int* a, const unsigned int* b,
                                    size_t n, unsigned long long* d_count) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (a[i] != b[i]) atomicAdd(d_count, 1ULL);
}

static unsigned long long gpu_bitwise_compare(const float* d_a, const float* d_b, size_t n_floats) {
    unsigned long long* d_count;
    cudaMalloc(&d_count, sizeof(unsigned long long));
    cudaMemset(d_count, 0, sizeof(unsigned long long));
    int blocks = (int)((n_floats + 255) / 256);
    gpu_compare_kernel<<<blocks, 256>>>((const unsigned int*)d_a, (const unsigned int*)d_b,
                                         n_floats, d_count);
    unsigned long long h_count = 0;
    cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_count);
    return h_count;
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    float chunk_mb = 4.0f;
    float total_mb = 0.0f;
    const char* env;
    if ((env = getenv("CHUNK_MB"))) chunk_mb = (float)atof(env);
    if ((env = getenv("TOTAL_MB"))) total_mb = (float)atof(env);

    const size_t CHUNK_FLOATS = (size_t)(chunk_mb * 1024 * 1024 / sizeof(float));
    const size_t CHUNK_BYTES  = CHUNK_FLOATS * sizeof(float);
    const size_t PER_PATTERN  = (total_mb > 0)
        ? (size_t)(total_mb * 1024 * 1024 / N_PATTERNS)
        : CHUNK_BYTES;
    const int CHUNKS_PER_PAT  = (int)(PER_PATTERN / CHUNK_BYTES);
    const size_t PAT_FLOATS   = (size_t)CHUNKS_PER_PAT * CHUNK_FLOATS;
    const size_t PAT_BYTES    = PAT_FLOATS * sizeof(float);

    fprintf(stderr, "=== Lossless VOL Stress Test ===\n");
    fprintf(stderr, "  Chunk: %.0f MiB  |  Per-pattern: %d chunks (%.0f MiB)  |  Patterns: %d\n",
            chunk_mb, CHUNKS_PER_PAT, (double)PAT_BYTES / (1<<20), N_PATTERNS);
    fprintf(stderr, "  Total: %.0f MiB  |  Mode: lossless ALGO_AUTO\n\n",
            (double)PAT_BYTES * N_PATTERNS / (1<<20));

    /* Init */
    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) weights = "neural_net/weights/model.nnwt";
    if (gpucompress_init(weights) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "ERROR: gpucompress_init failed\n"); return 1;
    }
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.25f, 0.20f, 0.20f);

    /* Allocate GPU buffers */
    float *d_orig = NULL, *d_read = NULL;
    cudaMalloc(&d_orig, PAT_BYTES);
    cudaMalloc(&d_read, PAT_BYTES);

    /* HDF5 VOL setup */
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl_w = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl_w, native_id, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t cdims[1] = { (hsize_t)CHUNK_FLOATS };
    H5Pset_chunk(dcpl, 1, cdims);
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; cd[1] = 0; cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]); /* error_bound=0 → lossless */
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    /* Run each pattern */
    for (int p = 0; p < N_PATTERNS; p++) {
        fprintf(stderr, "--- [%d/%d] %s ---\n", p+1, N_PATTERNS, PATTERNS[p].name);

        /* Generate on GPU */
        int blocks = (int)((PAT_FLOATS + 255) / 256);
        PATTERNS[p].launch(d_orig, PAT_FLOATS, blocks);
        cudaDeviceSynchronize();

        /* Reset chunk history for clean diagnostics */
        gpucompress_reset_chunk_history();

        /* Write via VOL */
        remove(TMP_FILE);
        hid_t fid = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_w);
        if (fid < 0) { FAIL("H5Fcreate failed"); continue; }

        hsize_t dims[1] = { (hsize_t)PAT_FLOATS };
        hid_t fsp = H5Screate_simple(1, dims, NULL);
        hid_t dset = H5Dcreate2(fid, PATTERNS[p].name, H5T_NATIVE_FLOAT,
                                 fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Sclose(fsp);

        herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_orig);
        H5Dclose(dset); H5Fclose(fid);
        if (wret < 0) { FAIL("H5Dwrite failed"); continue; }

        /* Read back via VOL */
        cudaMemset(d_read, 0xAA, PAT_BYTES); /* poison readback buffer */

        hid_t fapl_r = H5Pcreate(H5P_FILE_ACCESS);
        hid_t nid2 = H5VLget_connector_id_by_name("native");
        H5Pset_fapl_gpucompress(fapl_r, nid2, NULL);
        H5VLclose(nid2);

        fid = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl_r);
        if (fid < 0) { FAIL("H5Fopen failed"); H5Pclose(fapl_r); continue; }

        dset = H5Dopen2(fid, PATTERNS[p].name, H5P_DEFAULT);
        herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        H5Dclose(dset); H5Fclose(fid); H5Pclose(fapl_r);

        if (rret < 0) { FAIL("H5Dread failed"); continue; }

        /* Bitwise compare on GPU */
        unsigned long long mm = gpu_bitwise_compare(d_orig, d_read, PAT_FLOATS);

        /* Report per-chunk diagnostics */
        int n_diag = gpucompress_get_chunk_history_count();
        int total_sgd = 0;
        for (int c = 0; c < n_diag; c++) {
            gpucompress_chunk_diag_t diag;
            if (gpucompress_get_chunk_diag(c, &diag) == 0)
                total_sgd += diag.sgd_fired;
        }
        fprintf(stderr, "    chunks=%d  sgd=%d  mismatches=%llu\n", n_diag, total_sgd, mm);

        char msg[128];
        snprintf(msg, sizeof(msg), "%s: %llu mismatches across %zu floats",
                 PATTERNS[p].name, mm, PAT_FLOATS);
        if (mm == 0)
            PASS(msg);
        else
            FAIL(msg);
    }

    /* Cleanup */
    H5Pclose(dcpl); H5Pclose(fapl_w); H5VLclose(native_id);
    cudaFree(d_orig); cudaFree(d_read);
    remove(TMP_FILE);

    fprintf(stderr, "\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    fprintf(stderr, "%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
