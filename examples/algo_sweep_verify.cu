/**
 * algo_sweep_verify.cu
 *
 * Generates Gray-Scott simulation data on GPU, then runs ALL 8 compression
 * algorithms (with and without shuffle) via the VOL connector.
 * Compares every algorithm's ratio against the NN's AUTO pick.
 *
 * Usage:
 *   ./algo_sweep_verify [model.nnwt] [--L 160] [--chunk_mb 4] [--steps 1000]
 *                       [--F 0.04] [--k 0.06075]
 *
 * Pattern reference (same as grayscott_vol_demo):
 *   --F 0.04   --k 0.06075   Spots (default)
 *   --F 0.035  --k 0.065     Stripes/labyrinth
 *   --F 0.014  --k 0.045     Chaos (hardest to compress)
 *   --F 0.04   --k 0.065     Sparse spots (easiest to compress)
 *   --F 0.03   --k 0.055     Coral/worms
 *   --F 0.025  --k 0.06      Mitosis
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "gpucompress_grayscott.h"
#include "gpucompress_hdf5_vol.h"
#include "hdf5/H5Zgpucompress.h"
#include "compression/compression_header.h"

/* ---- Timing -------------------------------------------------------- */
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- Bitwise verify kernel ---------------------------------------- */
__global__ void verify_kernel(const float *ref, const float *got,
                               unsigned int n, int *err) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && ref[i] != got[i]) atomicAdd(err, 1);
}

/* ---- Algorithm table ---------------------------------------------- */
static const char *algo_names[] = {
    "AUTO", "LZ4", "SNAPPY", "DEFLATE", "GDEFLATE",
    "ZSTD", "ANS", "CASCADED", "BITCOMP"
};

/* ---- CLI parsing -------------------------------------------------- */
static void parse_args(int argc, char **argv,
                       const char **weights, int *L, int *steps,
                       int *chunk_mb, float *F, float *k)
{
    *weights  = NULL;
    *L        = 160;
    *steps    = 1000;
    *chunk_mb = 4;
    *F        = 0.04f;
    *k        = 0.06075f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) {
            *L = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            *steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chunk_mb") == 0 && i + 1 < argc) {
            *chunk_mb = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--F") == 0 && i + 1 < argc) {
            *F = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            *k = (float)atof(argv[++i]);
        } else if (argv[i][0] != '-' && *weights == NULL) {
            *weights = argv[i];
        }
    }
}

/* ---- Run one algo config and return compressed size --------------- */
static size_t run_algo(float *d_src, size_t n_elem, int L, int chunk_z,
                       int n_chunks, hid_t fapl, int algo_id, int shuffle_sz,
                       double *out_write_s, double *out_read_s, int *out_pass)
{
    size_t nbytes = n_elem * sizeof(float);
    char fname[128];
    snprintf(fname, sizeof(fname), "/tmp/sweep_%s_%s.h5",
             algo_names[algo_id], shuffle_sz ? "shuf" : "noshuf");

    hsize_t dims[3]  = { (hsize_t)L, (hsize_t)L, (hsize_t)L };
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };

    /* Write */
    hid_t fid   = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    hid_t space = H5Screate_simple(3, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);
    H5Pset_gpucompress(dcpl, (gpucompress_algorithm_t)algo_id, 0, shuffle_sz, 0.0);

    hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT, space,
                             H5P_DEFAULT, dcpl, H5P_DEFAULT);

    double t0 = now_sec();
    H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_src);
    *out_write_s = now_sec() - t0;

    /* Measure compressed size */
    size_t total_comp = 0;
    for (int c = 0; c < n_chunks; c++) {
        hsize_t off[3] = { 0, 0, (hsize_t)c * chunk_z };
        hsize_t csz = 0;
        H5Dget_chunk_storage_size(dset, off, &csz);
        total_comp += csz;
    }

    H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl); H5Fclose(fid);

    /* Read back + verify */
    float *d_readback = NULL;
    cudaMalloc(&d_readback, nbytes);
    cudaMemset(d_readback, 0xCD, nbytes);

    fid  = H5Fopen(fname, H5F_ACC_RDONLY, fapl);
    dset = H5Dopen2(fid, "data", H5P_DEFAULT);

    t0 = now_sec();
    H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
    *out_read_s = now_sec() - t0;

    H5Dclose(dset); H5Fclose(fid);

    int *d_err;
    cudaMalloc(&d_err, sizeof(int));
    cudaMemset(d_err, 0, sizeof(int));
    unsigned int grid = ((unsigned int)n_elem + 255) / 256;
    verify_kernel<<<grid, 256>>>(d_src, d_readback, (unsigned int)n_elem, d_err);
    cudaDeviceSynchronize();
    int mismatches = 0;
    cudaMemcpy(&mismatches, d_err, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_err);
    cudaFree(d_readback);

    *out_pass = (mismatches == 0) ? 1 : 0;
    return total_comp;
}

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *weights;
    int L, steps, chunk_mb_val;
    float F, k;
    parse_args(argc, argv, &weights, &L, &steps, &chunk_mb_val, &F, &k);

    size_t n_elem = (size_t)L * L * L;
    size_t nbytes = n_elem * sizeof(float);
    int chunk_z   = (int)((size_t)chunk_mb_val * 1024 * 1024 / ((size_t)L * L * sizeof(float)));
    if (chunk_z < 1) chunk_z = 1;
    if (chunk_z > L) chunk_z = L;
    int n_chunks  = (L + chunk_z - 1) / chunk_z;
    double raw_mb = nbytes / (1024.0 * 1024.0);

    printf("=== Algorithm Sweep: Gray-Scott Data ===\n");
    printf("    Grid      : %d^3 = %.1f MB\n", L, raw_mb);
    printf("    Chunks    : %d x %d x %d  (%.1f MB x %d chunks)\n",
           L, L, chunk_z,
           (double)L * L * chunk_z * sizeof(float) / (1024.0 * 1024.0),
           n_chunks);
    printf("    Pattern   : F=%.4f  k=%.5f\n", F, k);
    printf("    Sim steps : %d\n", steps);
    printf("    Model     : %s\n\n", weights ? weights : "(none)");

    /* Init */
    gpucompress_init(weights);
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);

    H5Z_gpucompress_register();
    hid_t vol_id    = H5VL_gpucompress_register();
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl      = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    H5VL_gpucompress_set_trace(0);

    /* Generate Gray-Scott data on GPU */
    printf("Running Gray-Scott simulation (%d steps)...\n", steps);
    GrayScottSettings gs = gray_scott_default_settings();
    gs.L = L;
    gs.F = F;
    gs.k = k;

    gpucompress_grayscott_t sim = NULL;
    gpucompress_grayscott_create(&sim, &gs);
    gpucompress_grayscott_init(sim);

    double sim_t0 = now_sec();
    gpucompress_grayscott_run(sim, steps);
    cudaDeviceSynchronize();
    printf("Simulation done (%.3f s)\n\n", now_sec() - sim_t0);

    float *d_u = NULL, *d_v = NULL;
    gpucompress_grayscott_get_device_ptrs(sim, &d_u, &d_v);

    /* Table header */
    printf("+--------------+---------+--------------+---------+----------+----------+--------+\n");
    printf("| Algorithm    | Shuffle | Compressed   | Ratio   | Write    | Read     | Verify |\n");
    printf("+--------------+---------+--------------+---------+----------+----------+--------+\n");

    /* Store results for finding best */
    double best_ratio = 0.0;
    int    best_algo  = -1;
    int    best_shuf  = -1;

    double nn_ratio = 0.0;
    int    nn_algo  = -1;

    /* Run NN AUTO first */
    {
        double ws, rs; int pass;
        size_t comp = run_algo(d_v, n_elem, L, chunk_z, n_chunks, fapl,
                               GPUCOMPRESS_ALGO_AUTO, 4, &ws, &rs, &pass);
        double ratio = raw_mb / (comp / (1024.0 * 1024.0));
        nn_ratio = ratio;
        printf("| %-12s | %-7s | %8.3f MB  | %6.2fx | %6.3f s | %6.3f s | %-6s |  <- NN pick\n",
               "AUTO(NN)", "4-byte", comp / (1024.0 * 1024.0), ratio, ws, rs,
               pass ? "PASS" : "FAIL");

        /* Read back the header to see which algo NN actually chose */
        {
            hid_t fid  = H5Fopen("/tmp/sweep_AUTO_shuf.h5", H5F_ACC_RDONLY, fapl);
            hid_t dset = H5Dopen2(fid, "data", H5P_DEFAULT);
            hsize_t off[3] = {0, 0, 0};
            hsize_t csz = 0;
            H5Dget_chunk_storage_size(dset, off, &csz);
            if (csz > 0) {
                void *raw = malloc(csz);
                uint32_t filt = 0;
                size_t bufsz = csz;
                H5Dread_chunk(dset, H5P_DEFAULT, off, &filt, raw, &bufsz);
                CompressionHeader *hdr = (CompressionHeader *)raw;
                if (hdr->isValid()) {
                    nn_algo = hdr->getAlgorithmId();
                    printf("|              |         |   NN chose: %-8s                              |\n",
                           algo_names[nn_algo]);
                }
                free(raw);
            }
            H5Dclose(dset); H5Fclose(fid);
        }
    }

    printf("+--------------+---------+--------------+---------+----------+----------+--------+\n");

    /* Run all 8 algos x {no shuffle, 4-byte shuffle} */
    for (int algo = 1; algo <= 8; algo++) {
        for (int shuf = 0; shuf <= 4; shuf += 4) {
            double ws, rs; int pass;
            size_t comp = run_algo(d_v, n_elem, L, chunk_z, n_chunks, fapl,
                                   algo, shuf, &ws, &rs, &pass);
            double ratio = raw_mb / (comp / (1024.0 * 1024.0));

            if (ratio > best_ratio) {
                best_ratio = ratio;
                best_algo  = algo;
                best_shuf  = shuf;
            }

            const char *marker = "";
            if (algo == nn_algo && shuf == 4) marker = "  <- NN match";

            printf("| %-12s | %-7s | %8.3f MB  | %6.2fx | %6.3f s | %6.3f s | %-6s |%s\n",
                   algo_names[algo],
                   shuf ? "4-byte" : "none",
                   comp / (1024.0 * 1024.0), ratio, ws, rs,
                   pass ? "PASS" : "FAIL",
                   marker);
        }
    }

    printf("+--------------+---------+--------------+---------+----------+----------+--------+\n");

    /* Summary */
    printf("\n=== Results ===\n");
    printf("  Best overall : %s %s -> %.2fx\n",
           algo_names[best_algo],
           best_shuf ? "+shuffle" : "",
           best_ratio);
    printf("  NN picked    : %s +shuffle -> %.2fx\n",
           nn_algo >= 0 ? algo_names[nn_algo] : "?",
           nn_ratio);

    if (nn_algo == best_algo) {
        printf("  Verdict      : NN picked the BEST algorithm!\n");
    } else {
        double gap = best_ratio - nn_ratio;
        double pct = (gap / best_ratio) * 100.0;
        printf("  Verdict      : NN missed best by %.2fx (%.1f%% gap)\n", gap, pct);
        printf("                 Best was %s, NN chose %s\n",
               algo_names[best_algo],
               nn_algo >= 0 ? algo_names[nn_algo] : "?");
    }

    /* Cleanup */
    gpucompress_grayscott_destroy(sim);
    cudaFree(d_u); /* d_u freed by destroy, but safe */
    H5Pclose(fapl);
    H5VLclose(vol_id);
    gpucompress_cleanup();

    return 0;
}
