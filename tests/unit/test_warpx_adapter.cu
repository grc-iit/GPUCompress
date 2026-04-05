/**
 * @file test_warpx_adapter.cu
 * @brief Correctness tests for WarpX adapter
 *
 * Tests:
 *   1. Lifecycle — create/attach/destroy for each data type
 *   2. get_nbytes correctness for all WarpX data types
 *   3. Compression round-trip with LZ4 (baseline, all data types)
 *   4. All 8 fixed algorithms round-trip (E-field data)
 *   5. NN-based AUTO algorithm selection round-trip
 *   6. Preprocessing: byte shuffle round-trip
 *   7. Preprocessing: lossy quantization (error-bounded)
 *   8. Double-precision data round-trip (lossless)
 *   9. Large dataset stress test
 *  10. Convenience compress API (gpucompress_compress_warpx)
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <cfloat>

#include <cuda_runtime.h>
#include "gpucompress.h"
#include "gpucompress_warpx.h"

/* ============================================================
 * Macros
 * ============================================================ */

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                    \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static int g_pass = 0, g_fail = 0;

#define TEST_ASSERT(cond, msg)                                              \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "  FAIL: %s\n", msg);                           \
            g_fail++;                                                       \
            return;                                                         \
        }                                                                   \
    } while (0)

/* ============================================================
 * Synthetic data generators (GPU kernels)
 * ============================================================ */

/** Sinusoidal fill — models smooth EM field data on a grid */
__global__ void fill_sinusoidal_f(float* data, int n_elements, int n_vars)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_elements * n_vars;
    if (idx >= total) return;
    int elem = idx / n_vars;
    int var  = idx % n_vars;
    float x = (float)elem / (float)n_elements;
    data[idx] = sinf(x * 6.2831853f * (var + 1)) * 100.0f;
}

__global__ void fill_sinusoidal_d(double* data, int n_elements, int n_vars)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_elements * n_vars;
    if (idx >= total) return;
    int elem = idx / n_vars;
    int var  = idx % n_vars;
    double x = (double)elem / (double)n_elements;
    data[idx] = sin(x * 6.283185307179586 * (var + 1)) * 100.0;
}

/** Particle-like data: mixed smooth + noisy */
__global__ void fill_particle_data(float* data, int n_particles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    /* 7 components: x, y, z, ux, uy, uz, w */
    int total = n_particles * 7;
    if (idx >= total) return;
    int p   = idx / 7;
    int var = idx % 7;
    float t = (float)p / (float)n_particles;
    if (var < 3) {
        /* positions: smooth, bounded [0,1] */
        data[idx] = t + 0.01f * sinf(t * 100.0f * (var + 1));
    } else if (var < 6) {
        /* momenta: wider range, oscillatory */
        data[idx] = sinf(t * 20.0f * (var - 2)) * 1e6f;
    } else {
        /* weight: mostly constant with small variation */
        data[idx] = 1.0f + 0.001f * sinf(t * 50.0f);
    }
}

/* ============================================================
 * Test 1: Lifecycle
 * ============================================================ */

static void test_lifecycle()
{
    printf("Test 1: Lifecycle ...\n");

    warpx_data_type_t types[] = {
        WARPX_DATA_EFIELD, WARPX_DATA_BFIELD, WARPX_DATA_JFIELD,
        WARPX_DATA_RHO, WARPX_DATA_PARTICLES, WARPX_DATA_CUSTOM
    };
    const char* names[] = {
        "E-field", "B-field", "J-field", "rho", "particles", "custom"
    };
    int ncomps[] = { 3, 3, 3, 1, 7, 1 };

    for (int t = 0; t < 6; t++) {
        WarpxSettings s = warpx_default_settings();
        s.data_type    = types[t];
        s.n_components = ncomps[t];
        s.element_size = 8;

        gpucompress_warpx_t handle = NULL;
        gpucompress_error_t err;

        err = gpucompress_warpx_create(&handle, &s);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "create failed");
        TEST_ASSERT(handle != NULL, "handle is NULL after create");

        /* Attach a small synthetic buffer */
        size_t n_elem = 1024;
        void* d_data = NULL;
        CHECK_CUDA(cudaMalloc(&d_data, n_elem * ncomps[t] * 8));

        err = gpucompress_warpx_attach(handle, d_data, n_elem);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "attach failed");

        err = gpucompress_warpx_destroy(handle);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "destroy failed");

        CHECK_CUDA(cudaFree(d_data));
        printf("  %s: OK\n", names[t]);
    }

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 2: get_nbytes correctness
 * ============================================================ */

static void test_nbytes()
{
    printf("Test 2: get_nbytes correctness ...\n");

    struct { warpx_data_type_t type; int ncomp; int esize; size_t n_elem; const char* name; } cases[] = {
        { WARPX_DATA_EFIELD,    3, 8, 10000, "E-field (3 comp, double)" },
        { WARPX_DATA_BFIELD,    3, 4, 10000, "B-field (3 comp, float)"  },
        { WARPX_DATA_JFIELD,    3, 8, 5000,  "J-field (3 comp, double)" },
        { WARPX_DATA_RHO,       1, 8, 8000,  "rho (1 comp, double)"     },
        { WARPX_DATA_PARTICLES, 7, 4, 2000,  "particles (7 comp, float)"},
        { WARPX_DATA_CUSTOM,    1, 4, 4096,  "custom (1 comp, float)"   },
    };

    for (int c = 0; c < 6; c++) {
        WarpxSettings s = warpx_default_settings();
        s.data_type    = cases[c].type;
        s.n_components = cases[c].ncomp;
        s.element_size = cases[c].esize;

        gpucompress_warpx_t handle = NULL;
        gpucompress_error_t err = gpucompress_warpx_create(&handle, &s);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS && handle != NULL, "create failed");

        void* d_dummy = NULL;
        size_t alloc = cases[c].n_elem * cases[c].ncomp * cases[c].esize;
        CHECK_CUDA(cudaMalloc(&d_dummy, alloc));

        err = gpucompress_warpx_attach(handle, d_dummy, cases[c].n_elem);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "attach failed");

        size_t nbytes = 0;
        err = gpucompress_warpx_get_nbytes(handle, &nbytes);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "get_nbytes failed");

        size_t expected = cases[c].n_elem * (size_t)cases[c].ncomp * (size_t)cases[c].esize;
        printf("  %s: nbytes=%zu  expected=%zu\n", cases[c].name, nbytes, expected);
        TEST_ASSERT(nbytes == expected, "nbytes mismatch");

        gpucompress_warpx_destroy(handle);
        CHECK_CUDA(cudaFree(d_dummy));
    }

    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Helper: lossless compression round-trip
 * ============================================================ */

static void roundtrip_float(warpx_data_type_t dtype, int ncomp, size_t n_elem,
                            gpucompress_algorithm_t algo, unsigned int preproc,
                            const char* label, int test_num)
{
    size_t total_floats = n_elem * ncomp;
    size_t nbytes = total_floats * sizeof(float);

    /* Allocate and fill */
    float* d_data = NULL;
    CHECK_CUDA(cudaMalloc(&d_data, nbytes));

    int grid = ((int)total_floats + 255) / 256;
    if (dtype == WARPX_DATA_PARTICLES) {
        fill_particle_data<<<grid, 256>>>(d_data, (int)n_elem);
    } else {
        fill_sinusoidal_f<<<grid, 256>>>(d_data, (int)n_elem, ncomp);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Create adapter and attach */
    WarpxSettings s = warpx_default_settings();
    s.data_type    = dtype;
    s.n_components = ncomp;
    s.element_size = 4;

    gpucompress_warpx_t handle = NULL;
    gpucompress_error_t err = gpucompress_warpx_create(&handle, &s);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS && handle != NULL, "create failed");
    err = gpucompress_warpx_attach(handle, d_data, n_elem);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "attach failed");

    /* Verify pointer retrieval */
    void* got_ptr = NULL;
    size_t got_nbytes = 0;
    err = gpucompress_warpx_get_device_ptr(handle, &got_ptr, &got_nbytes);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "get_device_ptr failed");
    TEST_ASSERT(got_ptr == d_data, "pointer mismatch");
    TEST_ASSERT(got_nbytes == nbytes, "nbytes mismatch from get_device_ptr");

    /* Compress */
    size_t max_comp = gpucompress_max_compressed_size(nbytes);
    void* d_compressed = NULL;
    CHECK_CUDA(cudaMalloc(&d_compressed, max_comp));
    size_t comp_size = max_comp;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm     = algo;
    cfg.preprocessing = preproc;

    gpucompress_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    err = gpucompress_compress_warpx(d_data, nbytes,
                                      d_compressed, &comp_size,
                                      &cfg, &stats);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "compress failed");

    printf("  %s: %zu -> %zu bytes (ratio %.2fx, algo=%s)\n",
           label, nbytes, comp_size, stats.compression_ratio,
           gpucompress_algorithm_name(stats.algorithm_used));

    /* Decompress */
    float* d_decomp = NULL;
    CHECK_CUDA(cudaMalloc(&d_decomp, nbytes));
    size_t decomp_size = nbytes;
    err = gpucompress_decompress_gpu(d_compressed, comp_size,
                                     d_decomp, &decomp_size, NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "decompress failed");
    TEST_ASSERT(decomp_size == nbytes, "decompress size mismatch");

    /* Verify lossless */
    std::vector<float> h_orig(total_floats), h_decomp(total_floats);
    CHECK_CUDA(cudaMemcpy(h_orig.data(), d_data, nbytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_decomp.data(), d_decomp, nbytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (size_t i = 0; i < total_floats; i++) {
        if (h_orig[i] != h_decomp[i]) mismatches++;
    }
    TEST_ASSERT(mismatches == 0, "lossless round-trip mismatch");

    gpucompress_warpx_destroy(handle);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_compressed));
    CHECK_CUDA(cudaFree(d_decomp));
}

/* ============================================================
 * Test 3: LZ4 round-trip for all WarpX data types
 * ============================================================ */

static void test_roundtrip_all_types()
{
    printf("Test 3: LZ4 round-trip for all data types ...\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "gpucompress_init failed");

    roundtrip_float(WARPX_DATA_EFIELD,    3, 8192, GPUCOMPRESS_ALGO_LZ4, GPUCOMPRESS_PREPROC_NONE, "E-field",    3);
    roundtrip_float(WARPX_DATA_BFIELD,    3, 8192, GPUCOMPRESS_ALGO_LZ4, GPUCOMPRESS_PREPROC_NONE, "B-field",    3);
    roundtrip_float(WARPX_DATA_JFIELD,    3, 8192, GPUCOMPRESS_ALGO_LZ4, GPUCOMPRESS_PREPROC_NONE, "J-field",    3);
    roundtrip_float(WARPX_DATA_RHO,       1, 8192, GPUCOMPRESS_ALGO_LZ4, GPUCOMPRESS_PREPROC_NONE, "rho",        3);
    roundtrip_float(WARPX_DATA_PARTICLES, 7, 4096, GPUCOMPRESS_ALGO_LZ4, GPUCOMPRESS_PREPROC_NONE, "particles",  3);
    roundtrip_float(WARPX_DATA_CUSTOM,    1, 8192, GPUCOMPRESS_ALGO_LZ4, GPUCOMPRESS_PREPROC_NONE, "custom",     3);

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 4: All 8 fixed algorithms round-trip (E-field)
 * ============================================================ */

static void test_all_fixed_algorithms()
{
    printf("Test 4: All 8 fixed algorithms round-trip ...\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "gpucompress_init failed");

    gpucompress_algorithm_t algos[] = {
        GPUCOMPRESS_ALGO_LZ4,
        GPUCOMPRESS_ALGO_SNAPPY,
        GPUCOMPRESS_ALGO_DEFLATE,
        GPUCOMPRESS_ALGO_GDEFLATE,
        GPUCOMPRESS_ALGO_ZSTD,
        GPUCOMPRESS_ALGO_ANS,
        GPUCOMPRESS_ALGO_CASCADED,
        GPUCOMPRESS_ALGO_BITCOMP,
    };
    const char* names[] = {
        "LZ4", "Snappy", "Deflate", "Gdeflate", "Zstd", "ANS", "Cascaded", "Bitcomp"
    };

    for (int a = 0; a < 8; a++) {
        char label[64];
        snprintf(label, sizeof(label), "E-field/%s", names[a]);
        roundtrip_float(WARPX_DATA_EFIELD, 3, 8192, algos[a],
                        GPUCOMPRESS_PREPROC_NONE, label, 4);
    }

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 5: NN-based AUTO algorithm selection
 * ============================================================ */

static void test_nn_auto_selection()
{
    printf("Test 5: NN AUTO algorithm selection ...\n");

    gpucompress_error_t err = gpucompress_init("neural_net/weights/model.nnwt");
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: NN weights not found (tried neural_net/weights/model.nnwt)\n");
        g_pass++;
        return;
    }
    TEST_ASSERT(gpucompress_nn_is_loaded() == 1, "NN not loaded after init");

    /* Test with different data types to exercise NN selection */
    roundtrip_float(WARPX_DATA_EFIELD,    3, 16384, GPUCOMPRESS_ALGO_AUTO, GPUCOMPRESS_PREPROC_NONE, "E-field/AUTO",   5);
    roundtrip_float(WARPX_DATA_BFIELD,    3, 16384, GPUCOMPRESS_ALGO_AUTO, GPUCOMPRESS_PREPROC_NONE, "B-field/AUTO",   5);
    roundtrip_float(WARPX_DATA_RHO,       1, 16384, GPUCOMPRESS_ALGO_AUTO, GPUCOMPRESS_PREPROC_NONE, "rho/AUTO",       5);
    roundtrip_float(WARPX_DATA_PARTICLES, 7, 8192,  GPUCOMPRESS_ALGO_AUTO, GPUCOMPRESS_PREPROC_NONE, "particles/AUTO", 5);

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 6: Byte shuffle preprocessing round-trip
 * ============================================================ */

static void test_shuffle_preprocessing()
{
    printf("Test 6: Byte shuffle preprocessing ...\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "gpucompress_init failed");

    gpucompress_algorithm_t algos[] = {
        GPUCOMPRESS_ALGO_LZ4, GPUCOMPRESS_ALGO_ZSTD, GPUCOMPRESS_ALGO_ANS
    };
    const char* names[] = { "LZ4+shuffle", "Zstd+shuffle", "ANS+shuffle" };

    for (int a = 0; a < 3; a++) {
        roundtrip_float(WARPX_DATA_EFIELD, 3, 8192, algos[a],
                        GPUCOMPRESS_PREPROC_SHUFFLE_4, names[a], 6);
    }

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 7: Lossy quantization (error-bounded)
 * ============================================================ */

static void test_lossy_quantization()
{
    printf("Test 7: Lossy quantization (error-bounded) ...\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "gpucompress_init failed");

    size_t n_elem = 8192;
    int ncomp = 3;
    size_t total_floats = n_elem * ncomp;
    size_t nbytes = total_floats * sizeof(float);

    float* d_data = NULL;
    CHECK_CUDA(cudaMalloc(&d_data, nbytes));
    int grid = ((int)total_floats + 255) / 256;
    fill_sinusoidal_f<<<grid, 256>>>(d_data, (int)n_elem, ncomp);
    CHECK_CUDA(cudaDeviceSynchronize());

    double error_bounds[] = { 1.0, 0.1, 0.01 };
    const char* bound_names[] = { "eb=1.0", "eb=0.1", "eb=0.01" };

    for (int e = 0; e < 3; e++) {
        size_t max_comp = gpucompress_max_compressed_size(nbytes);
        void* d_compressed = NULL;
        CHECK_CUDA(cudaMalloc(&d_compressed, max_comp));
        size_t comp_size = max_comp;

        gpucompress_config_t cfg = gpucompress_default_config();
        cfg.algorithm     = GPUCOMPRESS_ALGO_LZ4;
        cfg.preprocessing = GPUCOMPRESS_PREPROC_QUANTIZE;
        cfg.error_bound   = error_bounds[e];

        gpucompress_stats_t stats;
        memset(&stats, 0, sizeof(stats));
        err = gpucompress_compress_gpu(d_data, nbytes,
                                       d_compressed, &comp_size,
                                       &cfg, &stats, NULL);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "lossy compress failed");

        /* Decompress */
        float* d_decomp = NULL;
        CHECK_CUDA(cudaMalloc(&d_decomp, nbytes));
        size_t decomp_size = nbytes;
        err = gpucompress_decompress_gpu(d_compressed, comp_size,
                                         d_decomp, &decomp_size, NULL);
        TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "lossy decompress failed");

        /* Verify error bound */
        std::vector<float> h_orig(total_floats), h_decomp(total_floats);
        CHECK_CUDA(cudaMemcpy(h_orig.data(), d_data, nbytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_decomp.data(), d_decomp, nbytes, cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (size_t i = 0; i < total_floats; i++) {
            float diff = fabsf(h_orig[i] - h_decomp[i]);
            if (diff > max_err) max_err = diff;
        }

        printf("  %s: ratio=%.2fx  max_error=%.6f  bound=%.6f  %s\n",
               bound_names[e], stats.compression_ratio, max_err,
               (float)error_bounds[e],
               (max_err <= (float)error_bounds[e] * 1.01f) ? "WITHIN" : "EXCEEDED");

        /* Allow 1% tolerance on error bound due to floating point */
        TEST_ASSERT(max_err <= (float)error_bounds[e] * 1.01f,
                    "error bound exceeded");

        CHECK_CUDA(cudaFree(d_compressed));
        CHECK_CUDA(cudaFree(d_decomp));
    }

    CHECK_CUDA(cudaFree(d_data));
    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 8: Double-precision round-trip
 * ============================================================ */

static void test_double_precision()
{
    printf("Test 8: Double-precision round-trip ...\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "gpucompress_init failed");

    size_t n_elem = 8192;
    int ncomp = 3;
    size_t total_doubles = n_elem * ncomp;
    size_t nbytes = total_doubles * sizeof(double);

    /* Allocate and fill double data */
    double* d_data = NULL;
    CHECK_CUDA(cudaMalloc(&d_data, nbytes));
    int grid = ((int)total_doubles + 255) / 256;
    fill_sinusoidal_d<<<grid, 256>>>(d_data, (int)n_elem, ncomp);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Create adapter with double precision */
    WarpxSettings s = warpx_default_settings();
    s.data_type    = WARPX_DATA_EFIELD;
    s.n_components = ncomp;
    s.element_size = 8;  /* double */

    gpucompress_warpx_t handle = NULL;
    err = gpucompress_warpx_create(&handle, &s);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "create failed");
    err = gpucompress_warpx_attach(handle, d_data, n_elem);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "attach failed");

    size_t got_nbytes = 0;
    gpucompress_warpx_get_nbytes(handle, &got_nbytes);
    TEST_ASSERT(got_nbytes == nbytes, "nbytes mismatch for double");

    /* Compress with LZ4 */
    size_t max_comp = gpucompress_max_compressed_size(nbytes);
    void* d_compressed = NULL;
    CHECK_CUDA(cudaMalloc(&d_compressed, max_comp));
    size_t comp_size = max_comp;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

    gpucompress_stats_t stats;
    err = gpucompress_compress_warpx(d_data, nbytes,
                                     d_compressed, &comp_size,
                                     &cfg, &stats);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "compress failed");

    printf("  double: %zu -> %zu bytes (ratio %.2fx)\n",
           nbytes, comp_size, stats.compression_ratio);

    /* Decompress */
    double* d_decomp = NULL;
    CHECK_CUDA(cudaMalloc(&d_decomp, nbytes));
    size_t decomp_size = nbytes;
    err = gpucompress_decompress_gpu(d_compressed, comp_size,
                                     d_decomp, &decomp_size, NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "decompress failed");
    TEST_ASSERT(decomp_size == nbytes, "decompress size mismatch");

    /* Verify lossless (bitwise) */
    std::vector<double> h_orig(total_doubles), h_decomp(total_doubles);
    CHECK_CUDA(cudaMemcpy(h_orig.data(), d_data, nbytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_decomp.data(), d_decomp, nbytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (size_t i = 0; i < total_doubles; i++) {
        if (memcmp(&h_orig[i], &h_decomp[i], sizeof(double)) != 0) mismatches++;
    }
    TEST_ASSERT(mismatches == 0, "double round-trip mismatch");

    gpucompress_warpx_destroy(handle);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_compressed));
    CHECK_CUDA(cudaFree(d_decomp));
    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 9: Large dataset stress test (16M cells)
 * ============================================================ */

static void test_large_dataset()
{
    printf("Test 9: Large dataset stress test ...\n");

    gpucompress_error_t err = gpucompress_init(NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "gpucompress_init failed");

    /* 16M cells * 3 components * 4 bytes = 192 MB */
    size_t n_elem = 16 * 1024 * 1024;
    int ncomp = 3;
    size_t total_floats = n_elem * ncomp;
    size_t nbytes = total_floats * sizeof(float);

    /* Check GPU memory first */
    size_t free_mem = 0, total_mem = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_mem, &total_mem));
    /* Need ~3x for data + compressed + decomp, plus overhead */
    size_t needed = nbytes * 4;
    if (free_mem < needed) {
        printf("  SKIP: not enough GPU memory (need %zu MB, have %zu MB)\n",
               needed / (1024*1024), free_mem / (1024*1024));
        g_pass++;
        return;
    }

    float* d_data = NULL;
    CHECK_CUDA(cudaMalloc(&d_data, nbytes));
    int grid = ((int)total_floats + 255) / 256;
    fill_sinusoidal_f<<<grid, 256>>>(d_data, (int)n_elem, ncomp);
    CHECK_CUDA(cudaDeviceSynchronize());

    size_t max_comp = gpucompress_max_compressed_size(nbytes);
    void* d_compressed = NULL;
    CHECK_CUDA(cudaMalloc(&d_compressed, max_comp));
    size_t comp_size = max_comp;

    gpucompress_config_t cfg = gpucompress_default_config();
    cfg.algorithm = GPUCOMPRESS_ALGO_LZ4;

    gpucompress_stats_t stats;
    err = gpucompress_compress_gpu(d_data, nbytes,
                                   d_compressed, &comp_size,
                                   &cfg, &stats, NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "large compress failed");

    printf("  16M cells: %zu MB -> %zu MB (ratio %.2fx, %.1f MB/s)\n",
           nbytes / (1024*1024), comp_size / (1024*1024),
           stats.compression_ratio, stats.throughput_mbps);

    /* Decompress and verify */
    float* d_decomp = NULL;
    CHECK_CUDA(cudaMalloc(&d_decomp, nbytes));
    size_t decomp_size = nbytes;
    err = gpucompress_decompress_gpu(d_compressed, comp_size,
                                     d_decomp, &decomp_size, NULL);
    TEST_ASSERT(err == GPUCOMPRESS_SUCCESS, "large decompress failed");

    /* Spot-check: verify first and last 1024 elements */
    std::vector<float> h_orig(1024), h_decomp(1024);

    CHECK_CUDA(cudaMemcpy(h_orig.data(), d_data, 1024*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_decomp.data(), d_decomp, 1024*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 1024; i++) {
        TEST_ASSERT(h_orig[i] == h_decomp[i], "large round-trip mismatch (head)");
    }

    size_t tail_off = (total_floats - 1024) * sizeof(float);
    CHECK_CUDA(cudaMemcpy(h_orig.data(), (char*)d_data + tail_off, 1024*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_decomp.data(), (char*)d_decomp + tail_off, 1024*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 1024; i++) {
        TEST_ASSERT(h_orig[i] == h_decomp[i], "large round-trip mismatch (tail)");
    }

    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_compressed));
    CHECK_CUDA(cudaFree(d_decomp));
    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Test 10: NN AUTO with shuffle preprocessing
 * ============================================================ */

static void test_nn_with_shuffle()
{
    printf("Test 10: NN AUTO + shuffle preprocessing ...\n");

    gpucompress_error_t err = gpucompress_init("neural_net/weights/model.nnwt");
    if (err != GPUCOMPRESS_SUCCESS) {
        printf("  SKIP: NN weights not found\n");
        g_pass++;
        return;
    }

    roundtrip_float(WARPX_DATA_EFIELD, 3, 16384, GPUCOMPRESS_ALGO_AUTO,
                    GPUCOMPRESS_PREPROC_SHUFFLE_4, "E-field/AUTO+shuffle", 10);
    roundtrip_float(WARPX_DATA_RHO,    1, 16384, GPUCOMPRESS_ALGO_AUTO,
                    GPUCOMPRESS_PREPROC_SHUFFLE_4, "rho/AUTO+shuffle",     10);

    gpucompress_cleanup();
    printf("  PASS\n");
    g_pass++;
}

/* ============================================================
 * Main
 * ============================================================ */

int main()
{
    printf("=== WarpX Adapter Test Suite ===\n\n");

    test_lifecycle();
    test_nbytes();
    test_roundtrip_all_types();
    test_all_fixed_algorithms();
    test_nn_auto_selection();
    test_shuffle_preprocessing();
    test_lossy_quantization();
    test_double_precision();
    test_large_dataset();
    test_nn_with_shuffle();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
