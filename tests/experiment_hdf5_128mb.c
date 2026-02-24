/**
 * @file experiment_hdf5_128mb.c
 * @brief HDF5 experiment: single 128MB float32 dataset with mixed data patterns
 *
 * Generates 8 x 16MB chunks with wildly different data characteristics so the
 * NN picks different algorithms per chunk. Single dataset, single buffer.
 *
 * Chunk patterns:
 *   0: constant (42.0)              — near-zero entropy
 *   1: smooth sine wave             — low entropy, high smoothness
 *   2: pure random (LCG)            — max entropy, no structure
 *   3: sawtooth ramp                — monotonic, medium entropy
 *   4: sparse (99% zero, 1% spikes) — very low entropy
 *   5: gaussian noise (Box-Muller)   — high entropy, smooth distribution
 *   6: step function (8 levels)     — low entropy, discontinuous
 *   7: high-freq sine + noise       — medium entropy, chaotic derivative
 *
 * Usage:
 *   LD_LIBRARY_PATH=/tmp/lib:build ./build/experiment_hdf5_128mb <nn_weights.nnwt>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

/* 128 MB of float32 = 32M floats */
#define TOTAL_FLOATS  (32 * 1024 * 1024)
#define TOTAL_BYTES   (TOTAL_FLOATS * sizeof(float))

/* 16 MB chunks = 4M floats per chunk => 8 chunks */
#define CHUNK_FLOATS  (4 * 1024 * 1024)
#define N_CHUNKS      (TOTAL_FLOATS / CHUNK_FLOATS)

#define HDF5_FILE     "/tmp/experiment_hdf5_128mb.h5"
#define DSET_NAME     "data"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Simple LCG PRNG */
static uint32_t lcg_state;
static void     lcg_seed(uint32_t s) { lcg_state = s; }
static uint32_t lcg_next(void) { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state; }
static float    lcg_float(void) { return (float)(lcg_next() >> 8) / 16777216.0f; } /* [0, 1) */

static const char* chunk_labels[N_CHUNKS] = {
    "constant (42.0)",
    "smooth sine",
    "pure random",
    "sawtooth ramp",
    "sparse (99% zero)",
    "gaussian noise",
    "step function (8 levels)",
    "high-freq sine + noise"
};

static void fill_chunk(float* buf, int chunk_id) {
    size_t N = CHUNK_FLOATS;

    switch (chunk_id) {
    case 0: /* constant */
        for (size_t i = 0; i < N; i++)
            buf[i] = 42.0f;
        break;

    case 1: /* smooth sine — very low frequency */
        for (size_t i = 0; i < N; i++)
            buf[i] = 1000.0f * sinf(2.0f * (float)M_PI * i / (float)N);
        break;

    case 2: /* pure random [−1000, 1000] */
        lcg_seed(0xDEADBEEF);
        for (size_t i = 0; i < N; i++)
            buf[i] = lcg_float() * 2000.0f - 1000.0f;
        break;

    case 3: /* sawtooth ramp 0..1 repeating every 1024 samples */
        for (size_t i = 0; i < N; i++)
            buf[i] = (float)(i % 1024) / 1024.0f;
        break;

    case 4: /* sparse: 99% zero, 1% random spikes */
        lcg_seed(0xCAFEBABE);
        for (size_t i = 0; i < N; i++) {
            if ((lcg_next() % 100) == 0)
                buf[i] = lcg_float() * 10000.0f - 5000.0f;
            else
                buf[i] = 0.0f;
        }
        break;

    case 5: /* gaussian noise via Box-Muller, wide spread */
        lcg_seed(0x12345678);
        for (size_t i = 0; i < N; i += 2) {
            float u1 = lcg_float() * 0.9999f + 0.0001f;
            float u2 = lcg_float();
            float mag = sqrtf(-2.0f * logf(u1));
            buf[i]     = mag * cosf(2.0f * (float)M_PI * u2) * 500.0f;
            if (i + 1 < N)
                buf[i+1] = mag * sinf(2.0f * (float)M_PI * u2) * 500.0f;
        }
        break;

    case 6: /* step function: 8 discrete levels */
        for (size_t i = 0; i < N; i++) {
            int level = (int)(i / (N / 8));
            buf[i] = (float)level * 100.0f;
        }
        break;

    case 7: /* high-frequency sine + random noise */
        lcg_seed(0xABCD1234);
        for (size_t i = 0; i < N; i++)
            buf[i] = 500.0f * sinf(2.0f * (float)M_PI * i * 0.3f)
                   + (lcg_float() - 0.5f) * 1000.0f;
        break;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <nn_weights.nnwt>\n", argv[0]);
        return 1;
    }

    const char* weights_path = argv[1];

    /* ---- Generate mixed data ---- */
    float* data = (float*)malloc(TOTAL_BYTES);
    if (!data) { perror("malloc"); return 1; }

    printf("=== HDF5 128MB Mixed-Data Experiment (NN Auto-Select) ===\n\n");
    printf("Generating %d chunks x %d MB:\n", N_CHUNKS,
           (int)(CHUNK_FLOATS * sizeof(float) / (1024 * 1024)));
    for (int c = 0; c < N_CHUNKS; c++) {
        fill_chunk(data + (size_t)c * CHUNK_FLOATS, c);
        printf("  Chunk %d: %s\n", c, chunk_labels[c]);
    }
    printf("\n");

    /* ---- Init library with NN ---- */
    gpucompress_error_t rc = gpucompress_init(weights_path);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init: %s\n", gpucompress_error_string(rc));
        free(data);
        return 1;
    }

    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "FATAL: NN weights did not load from %s\n", weights_path);
        free(data);
        gpucompress_cleanup();
        return 1;
    }

    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "FATAL: H5Z_gpucompress_register failed\n");
        free(data);
        gpucompress_cleanup();
        return 1;
    }

    printf("NN weights: %s\n", weights_path);
    printf("HDF5 file:  %s\n", HDF5_FILE);
    printf("Dataset:    \"%s\"\n\n", DSET_NAME);

    /* ---- Create HDF5 file and dataset ---- */
    remove(HDF5_FILE);

    hid_t file_id = H5Fcreate(HDF5_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) { fprintf(stderr, "H5Fcreate failed\n"); free(data); gpucompress_cleanup(); return 1; }

    hsize_t dims[1] = {TOTAL_FLOATS};
    hid_t dspace = H5Screate_simple(1, dims, NULL);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = {CHUNK_FLOATS};
    H5Pset_chunk(dcpl, 1, chunk_dims);

    /* ALGO_AUTO: let NN pick per chunk */
    herr_t hs = H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 0, 0.0);
    if (hs < 0) { fprintf(stderr, "H5Pset_gpucompress failed\n"); free(data); gpucompress_cleanup(); return 1; }

    hid_t dset = H5Dcreate2(file_id, DSET_NAME, H5T_NATIVE_FLOAT,
                             dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) { fprintf(stderr, "H5Dcreate2 failed\n"); free(data); gpucompress_cleanup(); return 1; }

    /* ---- Write ---- */
    H5Z_gpucompress_reset_chunk_tracking();

    double t0 = time_ms();
    hs = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    double t1 = time_ms();

    if (hs < 0) { fprintf(stderr, "H5Dwrite failed\n"); free(data); gpucompress_cleanup(); return 1; }

    H5Z_gpucompress_write_chunk_attr(dset);

    /* ---- Summary table ---- */
    int n_chunks = H5Z_gpucompress_get_chunk_count();
    printf("\n  Chunk | Pattern                  | NN chose\n");
    printf("  ------+--------------------------+---------\n");
    for (int i = 0; i < n_chunks; i++) {
        int a = H5Z_gpucompress_get_chunk_algorithm(i);
        printf("  %5d | %-24s | %s\n", i, chunk_labels[i],
               gpucompress_algorithm_name((gpucompress_algorithm_t)a));
    }
    printf("\n");

    H5Dclose(dset);
    H5Pclose(dcpl);
    H5Sclose(dspace);
    H5Fclose(file_id);

    /* ---- Read back and verify ---- */
    file_id = H5Fopen(HDF5_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset = H5Dopen2(file_id, DSET_NAME, H5P_DEFAULT);
    hsize_t storage = H5Dget_storage_size(dset);

    float* read_buf = (float*)malloc(TOTAL_BYTES);
    if (!read_buf) { fprintf(stderr, "malloc failed\n"); free(data); gpucompress_cleanup(); return 1; }
    memset(read_buf, 0, TOTAL_BYTES);

    double t2 = time_ms();
    hs = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, read_buf);
    double t3 = time_ms();

    H5Dclose(dset);
    H5Fclose(file_id);

    if (hs < 0) {
        fprintf(stderr, "H5Dread failed\n");
        free(data); free(read_buf); gpucompress_cleanup();
        return 1;
    }

    /* Byte-for-byte verification */
    int mismatch = 0;
    if (memcmp(data, read_buf, TOTAL_BYTES) != 0) {
        mismatch = 1;
        for (size_t i = 0; i < TOTAL_FLOATS; i++) {
            if (data[i] != read_buf[i]) {
                int chunk = (int)(i / CHUNK_FLOATS);
                fprintf(stderr, "FAIL: mismatch at [%zu] (chunk %d, %s): wrote %.8g, read %.8g\n",
                        i, chunk, chunk_labels[chunk], data[i], read_buf[i]);
                break;
            }
        }
    }

    /* ---- Results ---- */
    double ratio = (storage > 0) ? (double)TOTAL_BYTES / (double)storage : 0.0;
    double write_ms = t1 - t0;
    double read_ms  = t3 - t2;

    printf("========================================\n");
    printf("  Results\n");
    printf("========================================\n");
    printf("Storage:    %llu KB (%.2fx ratio)\n", (unsigned long long)storage / 1024, ratio);
    printf("Write:      %.1f ms (%.0f MB/s)\n", write_ms,
           TOTAL_BYTES / (1024.0 * 1024.0) / (write_ms / 1000.0));
    printf("Read:       %.1f ms (%.0f MB/s)\n", read_ms,
           TOTAL_BYTES / (1024.0 * 1024.0) / (read_ms / 1000.0));
    printf("Verify:     %s\n", mismatch ? "FAIL — data mismatch!" : "PASS — lossless round-trip");

    struct stat st;
    if (stat(HDF5_FILE, &st) == 0) {
        printf("File size:  %.1f MB (%s)\n", st.st_size / (1024.0 * 1024.0), HDF5_FILE);
    }
    printf("========================================\n");

    free(data);
    free(read_buf);
    gpucompress_cleanup();

    return mismatch ? 1 : 0;
}
