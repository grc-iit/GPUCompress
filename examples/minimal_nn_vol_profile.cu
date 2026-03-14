/**
 * minimal_nn_vol_profile.cu
 *
 * Minimal end-to-end example: GPU data → HDF5 VOL write (NN auto-select) → read back → verify.
 * 2 MB dataset, 1 MB chunk size. Designed for profiling with nsys/valgrind.
 *
 * Build: part of CMake (add_vol_demo). Run:
 *   ./minimal_nn_vol_profile                      # plain run
 *   nsys profile ./minimal_nn_vol_profile          # nsys profiling
 *   valgrind --tool=memcheck ./minimal_nn_vol_profile  # memcheck
 *   compute-sanitizer ./minimal_nn_vol_profile     # CUDA memcheck
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

#define H5Z_FILTER_GPUCOMPRESS 305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

/* ---- GPU data generation kernel ---- */
__global__ void generate_sincos_data(float* out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = (float)idx * 0.001f;
        out[idx] = sinf(x) * cosf(x * 0.3f) + 0.5f * sinf(x * 2.7f);
    }
}

/* ---- Helpers ---- */
static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static herr_t set_gpucompress_filter(hid_t dcpl, unsigned algo,
                                      unsigned preproc, unsigned shuf_size,
                                      double error_bound) {
    unsigned cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = algo;
    cd[1] = preproc;
    cd[2] = shuf_size;
    pack_double(error_bound, &cd[3], &cd[4]);
    return H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                         H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
}

static hid_t open_vol_file(const char* path, unsigned flags) {
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) return H5I_INVALID_HID;

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL) < 0) {
        H5Pclose(fapl); H5VLclose(vol_id);
        return H5I_INVALID_HID;
    }

    hid_t fid;
    if (flags & H5F_ACC_TRUNC)
        fid = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    else
        fid = H5Fopen(path, H5F_ACC_RDONLY, fapl);

    H5Pclose(fapl);
    H5VLclose(vol_id);
    return fid;
}

int main(void) {
    printf("=== Minimal NN + VOL Profiling Example ===\n");
    printf("    Dataset: 2 MB (512K floats), Chunk: 1 MB (256K floats)\n\n");

    const char* fname = "/tmp/minimal_nn_vol_profile.h5";
    const char* weights = "neural_net/weights/model.nnwt";
    const hsize_t N = 512 * 1024;          /* 512K floats = 2 MB */
    const hsize_t CHUNK = 256 * 1024;      /* 256K floats = 1 MB */
    const size_t bytes = N * sizeof(float);

    /* ---- Init gpucompress with NN weights ---- */
    gpucompress_error_t err = gpucompress_init(weights);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %s\n", gpucompress_error_string(err));
        /* Try without NN */
        err = gpucompress_init(NULL);
        if (err != GPUCOMPRESS_SUCCESS) {
            fprintf(stderr, "FATAL: gpucompress_init(NULL) failed\n");
            return 1;
        }
        printf("  WARNING: NN weights not loaded, using LZ4 fallback\n");
    }

    int nn_loaded = gpucompress_nn_is_loaded();
    printf("  NN loaded: %s\n", nn_loaded ? "yes" : "no");

    /* Suppress HDF5 error stack */
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    /* ---- Generate data on GPU ---- */
    printf("\n[1] Generating 2 MB sincos data on GPU...\n");
    float* d_data = NULL;
    if (cudaMalloc(&d_data, bytes) != cudaSuccess) {
        fprintf(stderr, "FATAL: cudaMalloc failed\n");
        gpucompress_cleanup();
        return 1;
    }

    int threads = 256;
    int blocks = ((int)N + threads - 1) / threads;
    generate_sincos_data<<<blocks, threads>>>(d_data, N);
    cudaDeviceSynchronize();
    printf("  Done.\n");

    /* ---- Reset chunk diagnostics ---- */
    gpucompress_reset_chunk_history();

    /* ---- Write via VOL (ALGO_AUTO = NN selection) ---- */
    printf("\n[2] Writing via HDF5 VOL (ALGO_AUTO, NN selection)...\n");
    {
        hid_t fid = open_vol_file(fname, H5F_ACC_TRUNC);
        if (fid < 0) {
            fprintf(stderr, "FATAL: open_vol_file write failed\n");
            cudaFree(d_data); gpucompress_cleanup();
            return 1;
        }

        hsize_t dims[1] = { N }, chunk[1] = { CHUNK };
        hid_t space = H5Screate_simple(1, dims, NULL);
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 1, chunk);

        /* ALGO_AUTO (0) + shuffle4 (0x02), lossless */
        set_gpucompress_filter(dcpl, 0 /*AUTO*/, 0x02, 4, 0.0);

        hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
        if (dset < 0) {
            fprintf(stderr, "FATAL: H5Dcreate2 failed\n");
            H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
            cudaFree(d_data); gpucompress_cleanup();
            return 1;
        }

        herr_t rc = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                              H5P_DEFAULT, d_data);
        if (rc < 0) {
            fprintf(stderr, "FATAL: H5Dwrite failed\n");
            H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
            cudaFree(d_data); gpucompress_cleanup();
            return 1;
        }

        H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);
        printf("  Write complete.\n");
    }

    /* ---- Print per-chunk diagnostics ---- */
    int n_chunks = gpucompress_get_chunk_history_count();
    printf("\n[3] Per-chunk diagnostics (%d chunks):\n", n_chunks);
    for (int i = 0; i < n_chunks; i++) {
        gpucompress_chunk_diag_t diag;
        if (gpucompress_get_chunk_diag(i, &diag) == 0) {
            int algo = (diag.nn_action >= 0) ? (diag.nn_action % 8) : -1;
            printf("  Chunk %d: action=%d algo=%d ratio=%.1f "
                   "stats=%.2fms nn=%.2fms comp=%.2fms "
                   "explore=%s sgd=%s\n",
                   i, diag.nn_action, algo,
                   diag.actual_ratio,
                   diag.stats_ms, diag.nn_inference_ms,
                   diag.compression_ms,
                   diag.exploration_triggered ? "yes" : "no",
                   diag.sgd_fired ? "yes" : "no");
        }
    }

    /* ---- Read back via VOL ---- */
    printf("\n[4] Reading back via HDF5 VOL...\n");
    float* d_read = NULL;
    cudaMalloc(&d_read, bytes);

    {
        hid_t fid = open_vol_file(fname, H5F_ACC_RDONLY);
        if (fid < 0) {
            fprintf(stderr, "FATAL: open_vol_file read failed\n");
            cudaFree(d_data); cudaFree(d_read); gpucompress_cleanup();
            return 1;
        }

        hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
        herr_t rc = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                             H5P_DEFAULT, d_read);
        if (rc < 0) {
            fprintf(stderr, "FATAL: H5Dread failed\n");
            H5Dclose(dset); H5Fclose(fid);
            cudaFree(d_data); cudaFree(d_read); gpucompress_cleanup();
            return 1;
        }

        H5Dclose(dset); H5Fclose(fid);
        printf("  Read complete.\n");
    }

    /* ---- Verify on host ---- */
    printf("\n[5] Verifying data correctness...\n");
    float* h_orig = (float*)malloc(bytes);
    float* h_read = (float*)malloc(bytes);
    cudaMemcpy(h_orig, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_read, d_read, bytes, cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (size_t i = 0; i < N && mismatches < 5; i++) {
        if (h_orig[i] != h_read[i]) {
            fprintf(stderr, "  MISMATCH [%zu]: orig=%.6f read=%.6f\n",
                    i, h_orig[i], h_read[i]);
            mismatches++;
        }
    }

    if (mismatches == 0)
        printf("  PASS: lossless round-trip verified (%zu floats)\n", (size_t)N);
    else
        printf("  FAIL: %d mismatches found\n", mismatches);

    /* ---- Cleanup ---- */
    free(h_orig); free(h_read);
    cudaFree(d_data); cudaFree(d_read);
    gpucompress_cleanup();

    printf("\n=== Done ===\n");
    return (mismatches == 0) ? 0 : 1;
}
