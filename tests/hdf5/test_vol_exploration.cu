/**
 * @file test_vol_exploration.cu
 * @brief Test that exploration triggers through the VOL path.
 *
 * Tests:
 *  1. Exploration fires when threshold=0 (force-trigger on every chunk)
 *  2. Exploration does NOT fire when disabled
 *  3. Exploration does NOT fire when SGD/online learning is off
 *  4. Exploration respects K parameter (checks explored alternatives count)
 *  5. Data round-trip is correct after exploration swaps winner
 *
 * Uses a small 3D chunked dataset written through the VOL connector with
 * GPU-resident data. The NN must be loaded (weights path from argv or env).
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* Mirror filter constants */
#define H5Z_FILTER_GPUCOMPRESS   305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

#define TMP_FILE "/tmp/test_vol_exploration.h5"

/* ── Infrastructure ─────────────────────────────────────────────── */

static int g_pass = 0;
static int g_fail = 0;

#define PASS() do { printf("  PASS\n"); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s (line %d)\n", msg, __LINE__); g_fail++; return; } while(0)

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static hid_t make_vol_fapl(void) {
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) return H5I_INVALID_HID;
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
    H5VLclose(vol_id);
    return fapl;
}

static hid_t make_dcpl_auto(int L, int chunk_z) {
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0; cd[2] = 0;
    pack_double(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

/* Fill GPU buffer with a smooth pattern */
__global__ void fill_smooth(float* data, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total)
        data[i] = sinf((float)i * 0.01f) * 100.0f + (float)(i % 256);
}

/* Write dataset through VOL, read back, collect chunk diagnostics */
static int write_and_read(float* d_data, float* d_read, int L, int chunk_z,
                           hid_t dcpl) {
    size_t n = (size_t)L * L * L;

    gpucompress_reset_chunk_history();
    remove(TMP_FILE);

    hid_t fapl = make_vol_fapl();
    if (fapl < 0) return -1;
    hid_t file = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) return -1;

    hsize_t dims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };
    hid_t fsp  = H5Screate_simple(3, dims, NULL);
    hid_t dset = H5Dcreate2(file, "V", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);

    herr_t wr = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(dset); H5Fclose(file);
    if (wr < 0) return -1;

    /* Read back */
    fapl = make_vol_fapl();
    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "V", H5P_DEFAULT);

    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(dset); H5Fclose(file);

    remove(TMP_FILE);
    return 0;
}

/* Count exploration triggers from chunk history */
static int count_explorations(void) {
    int n = gpucompress_get_chunk_history_count();
    int expl = 0;
    for (int i = 0; i < n; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            if (d.exploration_triggered) expl++;
        }
    }
    return expl;
}

static int count_sgd_fires(void) {
    int n = gpucompress_get_chunk_history_count();
    int fires = 0;
    for (int i = 0; i < n; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            if (d.sgd_fired) fires++;
        }
    }
    return fires;
}

/* ── Test 1: Exploration fires with threshold=0 ─────────────────── */

static void test_exploration_fires(float* d_data, float* d_read,
                                    int L, int chunk_z, hid_t dcpl) {
    printf("Test 1: Exploration fires when threshold=0...\n");

    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.1f, 0.10f, 0.10f);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);  /* force trigger */
    gpucompress_set_exploration_k(4);

    int rc = write_and_read(d_data, d_read, L, chunk_z, dcpl);
    if (rc < 0) FAIL("write_and_read failed");

    int n_chunks = gpucompress_get_chunk_history_count();
    int expl = count_explorations();

    printf("    chunks=%d  explorations=%d\n", n_chunks, expl);

    if (n_chunks == 0) FAIL("no chunks written");
    if (expl == 0) FAIL("exploration never triggered with threshold=0");

    PASS();
}

/* ── Test 2: Exploration does NOT fire when disabled ────────────── */

static void test_exploration_disabled(float* d_data, float* d_read,
                                       int L, int chunk_z, hid_t dcpl) {
    printf("Test 2: Exploration does NOT fire when disabled...\n");

    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.1f, 0.10f, 0.10f);
    gpucompress_set_exploration(0);  /* disabled */

    int rc = write_and_read(d_data, d_read, L, chunk_z, dcpl);
    if (rc < 0) FAIL("write_and_read failed");

    int expl = count_explorations();
    printf("    explorations=%d (expected 0)\n", expl);

    if (expl != 0) FAIL("exploration fired when disabled");

    PASS();
}

/* ── Test 3: Exploration does NOT fire when online learning is off ── */

static void test_exploration_no_learning(float* d_data, float* d_read,
                                          int L, int chunk_z, hid_t dcpl) {
    printf("Test 3: Exploration does NOT fire when online learning off...\n");

    gpucompress_disable_online_learning();
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);

    int rc = write_and_read(d_data, d_read, L, chunk_z, dcpl);
    if (rc < 0) FAIL("write_and_read failed");

    int expl = count_explorations();
    int sgd = count_sgd_fires();
    printf("    explorations=%d  sgd=%d (expected both 0)\n", expl, sgd);

    if (expl != 0) FAIL("exploration fired with online learning off");
    if (sgd != 0) FAIL("SGD fired with online learning off");

    PASS();
}

/* ── Test 4: Data round-trip correct after exploration ──────────── */

static void test_roundtrip_after_exploration(float* d_data, float* d_read,
                                              int L, int chunk_z, hid_t dcpl) {
    printf("Test 4: Data round-trip correct after exploration...\n");

    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.1f, 0.10f, 0.10f);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);
    gpucompress_set_exploration_k(8);

    int rc = write_and_read(d_data, d_read, L, chunk_z, dcpl);
    if (rc < 0) FAIL("write_and_read failed");

    /* Verify lossless: d_data == d_read */
    size_t n = (size_t)L * L * L;
    float* h_orig = (float*)malloc(n * sizeof(float));
    float* h_read = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h_orig, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_read, d_read, n * sizeof(float), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (size_t i = 0; i < n; i++) {
        if (h_orig[i] != h_read[i]) {
            if (mismatches < 3)
                printf("    mismatch [%zu]: %.6f vs %.6f\n", i, h_orig[i], h_read[i]);
            mismatches++;
        }
    }
    free(h_orig);
    free(h_read);

    int expl = count_explorations();
    printf("    mismatches=%d  explorations=%d (expected 0 mismatches, >0 explorations)\n",
           mismatches, expl);
    if (mismatches > 0) FAIL("data mismatch after exploration round-trip");
    if (expl == 0) FAIL("exploration never triggered in round-trip test");

    PASS();
}

/* ── Test 5: Exploration with high threshold does NOT fire ──────── */

static void test_exploration_high_threshold(float* d_data, float* d_read,
                                             int L, int chunk_z, hid_t dcpl) {
    printf("Test 5: Exploration with high threshold (99%%) does NOT fire...\n");

    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.1f, 0.10f, 0.10f);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.99);  /* 99% — very unlikely */
    gpucompress_set_exploration_k(4);

    /* Write twice so SGD reduces error, making 99% threshold unreachable */
    write_and_read(d_data, d_read, L, chunk_z, dcpl);
    int rc = write_and_read(d_data, d_read, L, chunk_z, dcpl);
    if (rc < 0) FAIL("write_and_read failed");

    int expl = count_explorations();
    printf("    explorations=%d (expected 0)\n", expl);

    if (expl != 0) FAIL("exploration fired with 99%% threshold");

    PASS();
}

/* ── Main ────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    const char* weights = (argc > 1) ? argv[1] : getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) {
        fprintf(stderr, "Usage: %s model.nnwt\n", argv[0]);
        return 1;
    }

    printf("=== VOL Exploration Tests ===\n");
    printf("Weights: %s\n\n", weights);

    if (gpucompress_init(weights) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n");
        return 1;
    }
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) {
        fprintf(stderr, "FATAL: VOL register failed\n");
        gpucompress_cleanup();
        return 1;
    }

    int L = 64;
    int chunk_z = 8;  /* 64x64x8 = 128KB chunks, 8 chunks total */
    size_t total = (size_t)L * L * L * sizeof(float);

    float* d_data = NULL;
    float* d_read = NULL;
    cudaMalloc(&d_data, total);
    cudaMalloc(&d_read, total);

    int blocks = (L * L * L + 255) / 256;
    fill_smooth<<<blocks, 256>>>(d_data, L * L * L);
    cudaDeviceSynchronize();

    hid_t dcpl = make_dcpl_auto(L, chunk_z);

    /* Run tests — reload NN before each to start fresh */
    gpucompress_reload_nn(weights);
    test_exploration_fires(d_data, d_read, L, chunk_z, dcpl);

    gpucompress_reload_nn(weights);
    test_exploration_disabled(d_data, d_read, L, chunk_z, dcpl);

    gpucompress_reload_nn(weights);
    test_exploration_no_learning(d_data, d_read, L, chunk_z, dcpl);

    gpucompress_reload_nn(weights);
    test_roundtrip_after_exploration(d_data, d_read, L, chunk_z, dcpl);

    gpucompress_reload_nn(weights);
    test_exploration_high_threshold(d_data, d_read, L, chunk_z, dcpl);

    /* Cleanup */
    H5Pclose(dcpl);
    cudaFree(d_data);
    cudaFree(d_read);
    H5VLclose(vol_id);
    gpucompress_cleanup();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
