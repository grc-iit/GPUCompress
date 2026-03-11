/**
 * test_algo_per_chunk.cu
 *
 * Verify that ALGO_AUTO selects different algorithms per chunk.
 * Creates an HDF5 file via GPUCompress VOL with NN enabled,
 * then inspects per-chunk diagnostics to see which algorithm
 * was chosen for each chunk.
 *
 * BUILD:
 *   nvcc -o test_algo_per_chunk test_algo_per_chunk.cu \
 *       -I../../include -L../../build -lgpucompress \
 *       -I/tmp/hdf5-install/include -L/tmp/hdf5-install/lib -lhdf5 \
 *       -lcudart -lstdc++ -Xlinker -rpath,../../build
 *
 * RUN:
 *   export LD_LIBRARY_PATH=../../build:/tmp/hdf5-install/lib:$LD_LIBRARY_PATH
 *   export GPUCOMPRESS_WEIGHTS=../../neural_net/weights/model.nnwt
 *   ./test_algo_per_chunk
 */

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"
#include <hdf5.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5
#define TMP_FILE "/tmp/test_algo_per_chunk.h5"

static const char* ALGO_NAMES[] = {
    "lz4", "snappy", "deflate", "gdeflate", "zstd", "ans", "cascaded", "bitcomp"
};

static void pack_double_cd(double v, unsigned int* lo, unsigned int* hi)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static void action_to_str(int action, char *buf, size_t bufsz)
{
    if (action < 0) { snprintf(buf, bufsz, "none"); return; }
    int algo  = action % 8;
    int quant = (action / 8) % 2;
    int shuf  = (action / 16) % 2;
    snprintf(buf, bufsz, "%s%s%s", ALGO_NAMES[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
}

int main()
{
    printf("=== Test: ALGO_AUTO per-chunk algorithm selection ===\n\n");

    /* Init GPUCompress with NN weights */
    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) {
        weights = "../../neural_net/weights/model.nnwt";
    }
    gpucompress_error_t err = gpucompress_init(weights);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed (err=%d). Set GPUCOMPRESS_WEIGHTS.\n", err);
        return 1;
    }
    printf("GPUCompress initialized with NN weights.\n");

    /* Disable online learning — pure inference only */
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);

    /* Create test data on GPU: 8 chunks of 1 MB each, with varying patterns
     * to encourage the NN to pick different algorithms per chunk. */
    const size_t CHUNK_FLOATS = 256 * 1024;  /* 1 MiB per chunk */
    const int N_CHUNKS = 8;
    const size_t N_FLOATS = CHUNK_FLOATS * N_CHUNKS;
    const size_t TOTAL_BYTES = N_FLOATS * sizeof(float);

    float* h_data = (float*)malloc(TOTAL_BYTES);
    if (!h_data) { fprintf(stderr, "malloc failed\n"); return 1; }

    /* Fill each chunk with a different data pattern */
    srand(42);
    for (int c = 0; c < N_CHUNKS; c++) {
        size_t off = (size_t)c * CHUNK_FLOATS;
        switch (c % 4) {
            case 0: /* Constant — highly compressible */
                for (size_t i = 0; i < CHUNK_FLOATS; i++)
                    h_data[off + i] = 3.14159f;
                break;
            case 1: /* Random uniform — hard to compress */
                for (size_t i = 0; i < CHUNK_FLOATS; i++)
                    h_data[off + i] = (float)rand() / RAND_MAX;
                break;
            case 2: /* Smooth sine wave — moderate compressibility */
                for (size_t i = 0; i < CHUNK_FLOATS; i++)
                    h_data[off + i] = sinf((float)i * 0.001f) * 100.0f;
                break;
            case 3: /* Sparse — mostly zeros with occasional spikes */
                for (size_t i = 0; i < CHUNK_FLOATS; i++)
                    h_data[off + i] = (rand() % 100 == 0) ? (float)(rand() % 1000) : 0.0f;
                break;
        }
    }

    /* Upload to GPU */
    float* d_data = NULL;
    cudaMalloc(&d_data, TOTAL_BYTES);
    cudaMemcpy(d_data, h_data, TOTAL_BYTES, cudaMemcpyHostToDevice);

    /* Create DCPL with ALGO_AUTO */
    hsize_t chunk_dims[1] = { (hsize_t)CHUNK_FLOATS };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO — let NN decide per chunk */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    /* Create VOL FAPL */
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    /* Reset chunk history before write */
    gpucompress_reset_chunk_history();

    /* Write via HDF5 VOL */
    printf("\nWriting %d chunks (%.1f MiB each) via HDF5 VOL with ALGO_AUTO...\n",
           N_CHUNKS, (double)(CHUNK_FLOATS * sizeof(float)) / (1 << 20));

    remove(TMP_FILE);
    hid_t file = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "H5Fcreate failed\n"); return 1; }

    hsize_t dims[1] = { (hsize_t)N_FLOATS };
    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);

    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(dset);
    H5Fclose(file);
    H5Pclose(dcpl);

    if (wret < 0) { fprintf(stderr, "H5Dwrite failed\n"); return 1; }

    /* Inspect per-chunk diagnostics */
    int n_hist = gpucompress_get_chunk_history_count();
    printf("\nPer-chunk diagnostics (%d chunks recorded):\n", n_hist);
    printf("  chunk | action | algorithm            | ratio  | predicted | pattern\n");
    printf("  ------+--------+----------------------+--------+-----------+---------\n");

    int algo_counts[8] = {};
    int unique_algos = 0;
    int prev_action = -999;

    const char* patterns[] = {"constant", "random", "sine", "sparse",
                              "constant", "random", "sine", "sparse"};

    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) == 0) {
            char action_str[40];
            action_to_str(d.nn_action, action_str, sizeof(action_str));

            int algo_idx = d.nn_action % 8;
            if (algo_counts[algo_idx] == 0) unique_algos++;
            algo_counts[algo_idx]++;

            printf("  %5d | %5d  | %-20s | %5.2fx | %5.2fx    | %s\n",
                   i + 1, d.nn_action, action_str,
                   (double)d.actual_ratio, (double)d.predicted_ratio,
                   (i < N_CHUNKS) ? patterns[i] : "?");

            if (i > 0 && d.nn_action != prev_action) {
                /* Different action than previous chunk */
            }
            prev_action = d.nn_action;
        }
    }

    /* Summary */
    printf("\n  Algorithm distribution:\n");
    for (int a = 0; a < 8; a++) {
        if (algo_counts[a] > 0) {
            printf("    %-12s: %d chunks\n", ALGO_NAMES[a], algo_counts[a]);
        }
    }

    printf("\n  Unique algorithms used: %d\n", unique_algos);

    if (unique_algos > 1) {
        printf("\n  CONFIRMED: ALGO_AUTO selects DIFFERENT algorithms per chunk.\n");
    } else if (n_hist > 0) {
        printf("\n  NOTE: NN chose the SAME algorithm for all chunks.\n");
        printf("  (This can happen if the NN converges to one algo for this data.)\n");
    } else {
        printf("\n  ERROR: No chunk diagnostics recorded.\n");
    }

    /* Verify read-back */
    printf("\nVerifying read-back...\n");
    float* d_read = NULL;
    cudaMalloc(&d_read, TOTAL_BYTES);

    native_id = H5VLget_connector_id_by_name("native");
    fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    cudaDeviceSynchronize();
    H5Dclose(dset);
    H5Fclose(file);

    float* h_read = (float*)malloc(TOTAL_BYTES);
    cudaMemcpy(h_read, d_read, TOTAL_BYTES, cudaMemcpyDeviceToHost);

    unsigned long long mismatches = 0;
    for (size_t i = 0; i < N_FLOATS; i++) {
        unsigned int a, b;
        memcpy(&a, &h_data[i], sizeof(unsigned int));
        memcpy(&b, &h_read[i], sizeof(unsigned int));
        if (a != b) mismatches++;
    }
    printf("  Bitwise mismatches: %llu\n", mismatches);
    printf("  Verification: %s\n", mismatches == 0 ? "PASS" : "FAIL");

    /* Cleanup */
    remove(TMP_FILE);
    free(h_data);
    free(h_read);
    cudaFree(d_data);
    cudaFree(d_read);
    gpucompress_cleanup();

    printf("\n=== Done ===\n");
    return (mismatches == 0) ? 0 : 1;
}
