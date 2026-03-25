/**
 * test_vol_algo_shuffle_verify.cu
 *
 * Exhaustive lossless correctness test: every nvcomp algorithm × shuffle on/off
 * through the HDF5 VOL pipeline. No NN — direct algorithm + preprocessing selection.
 *
 * Matrix: 8 algorithms × 2 shuffle modes × 10 data patterns = 160 test cases.
 * Each: GPU generate → H5Dwrite (VOL compress) → H5Dread (VOL decompress) → bitwise verify.
 *
 * Usage:
 *   ./test_vol_algo_shuffle_verify
 *   CHUNK_MB=2 ./test_vol_algo_shuffle_verify
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
#define TMP_FILE "/tmp/test_vol_algo_shuffle.h5"

static int g_pass = 0, g_fail = 0, g_skip = 0;
#define PASS(msg) do { fprintf(stderr, "    PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { fprintf(stderr, "    FAIL: %s\n", msg); g_fail++; } while(0)
#define SKIP(msg) do { fprintf(stderr, "    SKIP: %s\n", msg); g_skip++; } while(0)

static void pack_double_cd(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits; memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/* ============================================================
 * GPU Data Generation Kernels
 * ============================================================ */

__global__ void gen_noise(float* out, size_t n, unsigned long long seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long x = (i + 1ULL) * 6364136223846793005ULL + seed;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    unsigned int bits = (unsigned int)(x & 0x7F7FFFFF);
    if (x & 0x80000000ULL) bits |= 0x80000000u;
    memcpy(&out[i], &bits, sizeof(float));
}

__global__ void gen_special_floats(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float vals[] = { 0.0f, -0.0f, FLT_MAX, -FLT_MAX, FLT_MIN, FLT_EPSILON,
                     1.0f, -1.0f, 1e-30f, 1e30f, 1e-10f, -1e10f };
    out[i] = vals[i % 12];
}

__global__ void gen_sawtooth(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (float)(i / 64) * 1000.0f + (float)(i % 64) * 15.625f;
}

__global__ void gen_constant(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 3.14159265358979f;
}

__global__ void gen_bimodal(float* out, size_t n, unsigned long long seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i < n / 2) { out[i] = 0.0f; return; }
    unsigned long long x = (i + 1ULL) * 2862933555777941757ULL + seed;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    out[i] = (float)((int)(x % 10000)) / 100.0f - 50.0f;
}

__global__ void gen_exp_range(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float t = (float)i / fmaxf((float)(n - 1), 1.0f);
    out[i] = powf(10.0f, -38.0f + t * 76.0f);
    if (i % 3 == 0) out[i] = -out[i];
}

__global__ void gen_sine_multi(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = (float)i;
    out[i] = sinf(x * 0.001f) * 100.0f + cosf(x * 0.017f) * 30.0f
           + sinf(x * 0.31f) * 5.0f + cosf(x * 3.7f) * 0.1f;
}

__global__ void gen_ultra_sparse(float* out, size_t n, unsigned long long seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long x = (i + 1ULL) * 6364136223846793005ULL + seed;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    out[i] = ((x & 0x3FFF) == 0) ? ((x & 0x4000) ? 1e6f : -1e6f) : 0.0f;
}

__global__ void gen_integer_like(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (float)((int)(i % 65536) - 32768);
}

__global__ void gen_alternating(float* out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    out[i] *= (float)(1 + i % 100);
}

/* ============================================================
 * Pattern registry
 * ============================================================ */
typedef void (*GenFn)(float*, size_t, int);

#define LAUNCH_SEED(kernel, seed) \
    static void launch_##kernel(float* d, size_t n, int bl) { kernel<<<bl,256>>>(d, n, seed); }
#define LAUNCH_PLAIN(kernel) \
    static void launch_##kernel(float* d, size_t n, int bl) { kernel<<<bl,256>>>(d, n); }

LAUNCH_SEED(gen_noise, 0xDEADBEEFULL)
LAUNCH_PLAIN(gen_special_floats)
LAUNCH_PLAIN(gen_sawtooth)
LAUNCH_PLAIN(gen_constant)
LAUNCH_SEED(gen_bimodal, 0xCAFEBABEULL)
LAUNCH_PLAIN(gen_exp_range)
LAUNCH_PLAIN(gen_sine_multi)
LAUNCH_SEED(gen_ultra_sparse, 0x1337C0DEULL)
LAUNCH_PLAIN(gen_integer_like)
LAUNCH_PLAIN(gen_alternating)

struct Pattern { const char* name; GenFn launch; };
static Pattern PATTERNS[] = {
    { "noise",          launch_gen_noise },
    { "special_floats", launch_gen_special_floats },
    { "sawtooth",       launch_gen_sawtooth },
    { "constant",       launch_gen_constant },
    { "bimodal",        launch_gen_bimodal },
    { "exp_range",      launch_gen_exp_range },
    { "sine_multi",     launch_gen_sine_multi },
    { "ultra_sparse",   launch_gen_ultra_sparse },
    { "integer_like",   launch_gen_integer_like },
    { "alternating",    launch_gen_alternating },
};
static const int N_PATTERNS = sizeof(PATTERNS) / sizeof(PATTERNS[0]);

/* ============================================================
 * Algorithm registry
 * ============================================================ */
struct AlgoInfo { const char* name; int algo_id; /* GPUCOMPRESS_ALGO_* */ };
static AlgoInfo ALGOS[] = {
    { "lz4",      GPUCOMPRESS_ALGO_LZ4 },
    { "snappy",   GPUCOMPRESS_ALGO_SNAPPY },
    { "deflate",  GPUCOMPRESS_ALGO_DEFLATE },
    { "gdeflate", GPUCOMPRESS_ALGO_GDEFLATE },
    { "zstd",     GPUCOMPRESS_ALGO_ZSTD },
    { "ans",      GPUCOMPRESS_ALGO_ANS },
    { "cascaded", GPUCOMPRESS_ALGO_CASCADED },
    { "bitcomp",  GPUCOMPRESS_ALGO_BITCOMP },
};
static const int N_ALGOS = sizeof(ALGOS) / sizeof(ALGOS[0]);

/* ============================================================
 * GPU bitwise compare
 * ============================================================ */
__global__ void compare_kernel(const unsigned int* a, const unsigned int* b,
                                size_t n, unsigned long long* d_cnt) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) atomicAdd(d_cnt, 1ULL);
}

static unsigned long long gpu_compare(const float* d_a, const float* d_b, size_t n) {
    unsigned long long *d_cnt;
    cudaMalloc(&d_cnt, sizeof(unsigned long long));
    cudaMemset(d_cnt, 0, sizeof(unsigned long long));
    int bl = (int)((n + 255) / 256);
    compare_kernel<<<bl, 256>>>((const unsigned int*)d_a, (const unsigned int*)d_b, n, d_cnt);
    unsigned long long h = 0;
    cudaMemcpy(&h, d_cnt, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_cnt);
    return h;
}

/* ============================================================
 * Run one test: algo × shuffle × pattern
 * ============================================================ */
static void run_test(const char* algo_name, int algo_id, int shuffle,
                     const char* pat_name, float* d_orig, float* d_read,
                     size_t n_floats, size_t chunk_floats,
                     hid_t fapl_template, hid_t native_id)
{
    size_t total_bytes = n_floats * sizeof(float);
    char label[128];
    snprintf(label, sizeof(label), "%s%s + %s", algo_name,
             shuffle ? "+shuf" : "", pat_name);

    /* Create DCPL with this algo + shuffle */
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t cdims[1] = { (hsize_t)chunk_floats };
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = (unsigned int)algo_id;
    cd[1] = shuffle ? GPUCOMPRESS_PREPROC_SHUFFLE_4 : 0;
    cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]); /* lossless */
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    /* Write */
    remove(TMP_FILE);
    hid_t fid = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_template);
    if (fid < 0) { FAIL(label); H5Pclose(dcpl); return; }

    hsize_t dims[1] = { (hsize_t)n_floats };
    hid_t fsp = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp); H5Pclose(dcpl);

    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_orig);
    H5Dclose(dset); H5Fclose(fid);
    if (wret < 0) { FAIL(label); return; }

    /* Read back */
    cudaMemset(d_read, 0xBB, total_bytes);

    hid_t fapl_r = H5Pcreate(H5P_FILE_ACCESS);
    hid_t nid2 = H5VLget_connector_id_by_name("native");
    H5Pset_fapl_gpucompress(fapl_r, nid2, NULL);
    H5VLclose(nid2);

    fid = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl_r);
    if (fid < 0) { FAIL(label); H5Pclose(fapl_r); return; }

    dset = H5Dopen2(fid, "data", H5P_DEFAULT);
    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    H5Dclose(dset); H5Fclose(fid); H5Pclose(fapl_r);

    if (rret < 0) { FAIL(label); return; }

    /* Bitwise compare */
    unsigned long long mm = gpu_compare(d_orig, d_read, n_floats);
    if (mm == 0)
        PASS(label);
    else {
        char msg[256];
        snprintf(msg, sizeof(msg), "%s — %llu mismatches / %zu floats", label, mm, n_floats);
        FAIL(msg);
    }
}

/* ============================================================
 * Main
 * ============================================================ */
int main() {
    float chunk_mb = 4.0f;
    const char* env;
    if ((env = getenv("CHUNK_MB"))) chunk_mb = (float)atof(env);

    const size_t CHUNK_FLOATS = (size_t)(chunk_mb * 1024 * 1024 / sizeof(float));
    const size_t N_FLOATS     = CHUNK_FLOATS; /* 1 chunk per pattern for speed */
    const size_t TOTAL_BYTES  = N_FLOATS * sizeof(float);
    int total_tests = N_ALGOS * 2 * N_PATTERNS;

    fprintf(stderr, "=== Algorithm × Shuffle × Pattern Lossless Verify ===\n");
    fprintf(stderr, "  Chunk: %.0f MiB  |  Algos: %d  |  Patterns: %d  |  Tests: %d\n",
            chunk_mb, N_ALGOS, N_PATTERNS, total_tests);
    fprintf(stderr, "  Matrix: %d algos × 2 shuffle × %d patterns\n\n", N_ALGOS, N_PATTERNS);

    /* Init */
    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) weights = "neural_net/weights/model.nnwt";
    if (gpucompress_init(weights) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "ERROR: gpucompress_init failed\n"); return 1;
    }

    float *d_orig = NULL, *d_read = NULL;
    cudaMalloc(&d_orig, TOTAL_BYTES);
    cudaMalloc(&d_read, TOTAL_BYTES);

    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl_w = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl_w, native_id, NULL);

    int blocks = (int)((N_FLOATS + 255) / 256);

    for (int ai = 0; ai < N_ALGOS; ai++) {
        for (int shuf = 0; shuf <= 1; shuf++) {
            fprintf(stderr, "--- %s%s ---\n", ALGOS[ai].name, shuf ? "+shuffle" : "");

            for (int pi = 0; pi < N_PATTERNS; pi++) {
                /* Generate fresh data for each test */
                PATTERNS[pi].launch(d_orig, N_FLOATS, blocks);
                cudaDeviceSynchronize();

                gpucompress_reset_chunk_history();

                run_test(ALGOS[ai].name, ALGOS[ai].algo_id, shuf,
                         PATTERNS[pi].name, d_orig, d_read,
                         N_FLOATS, CHUNK_FLOATS,
                         fapl_w, native_id);
            }
        }
    }

    H5Pclose(fapl_w); H5VLclose(native_id);
    cudaFree(d_orig); cudaFree(d_read);
    remove(TMP_FILE);

    fprintf(stderr, "\n=== Summary: %d pass, %d fail, %d skip (of %d) ===\n",
            g_pass, g_fail, g_skip, total_tests);
    fprintf(stderr, "%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
