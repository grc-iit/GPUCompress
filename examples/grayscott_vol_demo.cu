/**
 * grayscott_vol_demo.cu
 *
 * Demonstrates the full GPUCompress pipeline with Gray-Scott reaction-diffusion:
 *   GPU simulation → NN compression → HDF5 via VOL → read-back → write
 *   decompressed copy → bitwise verify
 *
 * Produces two HDF5 files:
 *   1. /tmp/grayscott_compressed.h5   — written via VOL (NN-compressed chunks)
 *   2. /tmp/grayscott_decompressed.h5 — plain HDF5 (uncompressed, after round-trip)
 *
 * Verification: GPU kernel compares original vs decompressed bitwise.
 * You can also compare the two files externally with h5diff.
 *
 * Usage:
 *   ./grayscott_vol_demo [model.nnwt] [--L 128] [--steps 1000] [--chunk_mb 64]
 *
 * Options:
 *   model.nnwt   Path to NN weights. Omit to fall back to LZ4.
 *   --L N        Grid side length. Dataset size = N³ × 4 bytes.
 *   --steps N    Total simulation time-steps (default: 1000).
 *   --plotgap N  Snapshot every N steps (default: 0 = single snapshot at end).
 *   --chunk_mb M Chunk size in MB (auto-calculates chunk_z from L).
 *   --chunk_z Z  Set Z-dimension of chunk directly (alternative to --chunk_mb).
 *   --F val      Feed rate (default: 0.04). Controls pattern type.
 *   --k val      Kill rate (default: 0.06075). Controls pattern type.
 *
 * Dataset size reference:
 *   --L 128  →    8 MB        --L 640  →   1 GB
 *   --L 256  →   64 MB        --L 1000 →   4 GB
 *   --L 512  →  512 MB        --L 1260 →   8 GB
 *
 * Examples:
 *   # 8 MB dataset, default chunks (4 × 2 MB)
 *   ./grayscott_vol_demo model.nnwt --L 128 --steps 1000
 *
 *   # 64 MB dataset, 4 MB chunks (16 chunks)
 *   ./grayscott_vol_demo model.nnwt --L 256 --chunk_mb 4 --steps 1000
 *
 *   # 1 GB dataset, 64 MB chunks (16 chunks)
 *   ./grayscott_vol_demo model.nnwt --L 640 --chunk_mb 64 --steps 1000
 *
 *   # 4 GB dataset, 64 MB chunks (63 chunks)
 *   ./grayscott_vol_demo model.nnwt --L 1000 --chunk_mb 64 --steps 1000
 *
 *   # 8 GB dataset, 64 MB chunks (126 chunks) — needs ~30 GB GPU mem
 *   ./grayscott_vol_demo model.nnwt --L 1260 --chunk_mb 64 --steps 1000
 *
 *   # 4 GB dataset, 4 MB chunks (many small chunks, more NN decisions)
 *   ./grayscott_vol_demo model.nnwt --L 1000 --chunk_mb 4 --steps 1000
 *
 *   # Multiple snapshots: 5 snapshots of 64 MB each (plotgap = steps/n_snapshots)
 *   ./grayscott_vol_demo model.nnwt --L 256 --steps 5000 --plotgap 1000
 *
 *   # 10 snapshots of 1 GB, 64 MB chunks
 *   ./grayscott_vol_demo model.nnwt --L 640 --chunk_mb 64 --steps 10000 --plotgap 1000
 *
 *   # 20 snapshots of 4 GB, 64 MB chunks (long run)
 *   ./grayscott_vol_demo model.nnwt --L 1000 --chunk_mb 64 --steps 20000 --plotgap 1000
 *
 * Gray-Scott pattern reference (F, k):
 *   --F 0.04   --k 0.06075   Spots (default) — isolated blobs, large flat regions
 *   --F 0.035  --k 0.065     Stripes/labyrinth — winding connected structures
 *   --F 0.012  --k 0.05      Spots + waves — propagating fronts
 *   --F 0.025  --k 0.05      Solitons — pulsating spots
 *   --F 0.03   --k 0.055     Coral/worms — branching structures, higher entropy
 *   --F 0.014  --k 0.045     Chaos — turbulent, high entropy, harder to compress
 *   --F 0.04   --k 0.065     Sparse spots — very compressible, mostly empty
 *   --F 0.025  --k 0.06      Mitosis — splitting spots
 *
 * Examples with different patterns:
 *   # Stripes (higher entropy than spots)
 *   ./grayscott_vol_demo model.nnwt --L 160 --chunk_mb 4 --F 0.035 --k 0.065
 *
 *   # Chaos (hardest to compress)
 *   ./grayscott_vol_demo model.nnwt --L 160 --chunk_mb 4 --F 0.014 --k 0.045
 *
 *   # Sparse spots (easiest to compress)
 *   ./grayscott_vol_demo model.nnwt --L 160 --chunk_mb 4 --F 0.04 --k 0.065
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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

/* ---- Algorithm name table ----------------------------------------- */
static const char *algo_name(uint8_t id) {
    static const char *names[] = {
        "AUTO", "LZ4", "SNAPPY", "DEFLATE", "GDEFLATE",
        "ZSTD", "ANS", "CASCADED", "BITCOMP"
    };
    return (id < 9) ? names[id] : "?";
}

/* ---- CLI parsing -------------------------------------------------- */
static void parse_args(int argc, char **argv,
                       const char **weights, int *L, int *steps, int *plotgap,
                       int *chunk_z, int *chunk_mb, float *F, float *k)
{
    *weights  = NULL;
    *L        = 128;
    *steps    = 1000;
    *plotgap  = 0;   /* 0 = single snapshot at end */
    *chunk_z  = 0;   /* 0 = auto */
    *chunk_mb = 0;   /* 0 = not set */
    *F        = 0.04f;
    *k        = 0.06075f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) {
            *L = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            *steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--plotgap") == 0 && i + 1 < argc) {
            *plotgap = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chunk_z") == 0 && i + 1 < argc) {
            *chunk_z = atoi(argv[++i]);
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

/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    const char *weights;
    int L, steps, plotgap, chunk_z, chunk_mb;
    float F, k;
    parse_args(argc, argv, &weights, &L, &steps, &plotgap, &chunk_z, &chunk_mb, &F, &k);

    int n_snapshots = 1;
    if (plotgap > 0) {
        n_snapshots = steps / plotgap;
        if (n_snapshots < 1) n_snapshots = 1;
    } else {
        plotgap = steps;  /* single snapshot: run all steps then snapshot */
    }

    size_t n_elem  = (size_t)L * L * L;
    size_t nbytes  = n_elem * sizeof(float);
    if (chunk_mb > 0) {
        chunk_z = (int)((size_t)chunk_mb * 1024 * 1024 / ((size_t)L * L * sizeof(float)));
        if (chunk_z < 1) chunk_z = 1;
        if (chunk_z > L) chunk_z = L;
    } else if (chunk_z <= 0) {
        chunk_z = L / 4;
    }
    if (chunk_z < 1) chunk_z = 1;
    int n_chunks   = (L + chunk_z - 1) / chunk_z;

    printf("=== grayscott_vol_demo: Gray-Scott → NN Compression → HDF5 ===\n");
    printf("    Grid    : %d³ = %zu floats (%.1f MB)\n",
           L, n_elem, nbytes / (1024.0 * 1024.0));
    double chunk_size_mb = (double)L * L * chunk_z * sizeof(float) / (1024.0 * 1024.0);
    printf("    Chunks  : %d × %d × %d  (%d chunks, %.1f MB each)\n",
           L, L, chunk_z, n_chunks, chunk_size_mb);
    printf("    Steps   : %d  plotgap=%d  snapshots=%d\n", steps, plotgap, n_snapshots);
    printf("    Pattern : F=%.4f  k=%.5f\n", F, k);
    printf("    Model   : %s\n\n",
           weights ? weights : "(none — NN will fall back to LZ4)");

    /* ---- 1. Init GPUCompress + VOL ---- */
    gpucompress_error_t gerr = gpucompress_init(weights);
    if (gerr != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed: %d\n", gerr);
        return 1;
    }
    gpucompress_disable_online_learning();
    gpucompress_set_exploration(0);
    printf("NN model loaded: %s\n",
           gpucompress_nn_is_loaded() ? "yes" : "no (fallback to default)");

    H5Z_gpucompress_register();
    hid_t vol_id    = H5VL_gpucompress_register();
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl      = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    H5VL_gpucompress_set_trace(0);

    /* ---- 2. Create simulation ---- */
    GrayScottSettings s = gray_scott_default_settings();
    s.L = L;
    s.F = F;
    s.k = k;
    gpucompress_grayscott_t sim = NULL;
    gpucompress_grayscott_create(&sim, &s);
    gpucompress_grayscott_init(sim);

    /* ---- 3. Create HDF5 file with per-snapshot datasets ---- */
    const char *comp_fname   = "/tmp/grayscott_compressed.h5";
    const char *decomp_fname = "/tmp/grayscott_decompressed.h5";

    hsize_t dims[3]  = { (hsize_t)L, (hsize_t)L, (hsize_t)L };
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };

    hid_t comp_fid   = H5Fcreate(comp_fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    hid_t plain_fapl = H5Pcreate(H5P_FILE_ACCESS);
    hid_t decomp_fid = H5Fcreate(decomp_fname, H5F_ACC_TRUNC, H5P_DEFAULT, plain_fapl);

    float *d_readback = NULL;
    cudaMalloc(&d_readback, nbytes);

    int *d_err = NULL;
    cudaMalloc(&d_err, sizeof(int));

    int total_failures = 0;

    /* ---- 4. Snapshot loop ---- */
    for (int snap = 0; snap < n_snapshots; snap++) {
        /* Simulate plotgap steps */
        double sim_t0 = now_sec();
        gpucompress_grayscott_run(sim, plotgap);
        cudaDeviceSynchronize();
        double sim_dt = now_sec() - sim_t0;

        int cur_step = (snap + 1) * plotgap;
        float *d_u = NULL, *d_v = NULL;
        gpucompress_grayscott_get_device_ptrs(sim, &d_u, &d_v);

        char dset_name[32];
        snprintf(dset_name, sizeof(dset_name), "V_%04d", snap);

        printf("\n╔══ Snapshot %d/%d  (step %d, sim %.3f s) ═══════════════════╗\n",
               snap + 1, n_snapshots, cur_step, sim_dt);

        /* --- Write compressed --- */
        {
            hid_t space = H5Screate_simple(3, dims, NULL);
            hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
            H5Pset_chunk(dcpl, 3, cdims);
            H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 4, 0.0);

            hid_t dset = H5Dcreate2(comp_fid, dset_name, H5T_NATIVE_FLOAT, space,
                                     H5P_DEFAULT, dcpl, H5P_DEFAULT);

            H5VL_gpucompress_reset_stats();
            double t0 = now_sec();
            herr_t rc = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                                 H5S_ALL, H5S_ALL, H5P_DEFAULT, d_v);
            double dt = now_sec() - t0;

            int n_comp = 0;
            H5VL_gpucompress_get_stats(NULL, NULL, &n_comp, NULL);
            if (rc < 0 || n_comp != n_chunks) total_failures++;
            printf("  Write : %s  %.3f s  %.1f MB/s  (%d chunks)\n",
                   (rc >= 0 && n_comp == n_chunks) ? "OK  " : "FAIL",
                   dt, (nbytes / (1024.0 * 1024.0)) / dt, n_comp);

            /* Per-chunk algo report */
            int tally[9][2] = {};
            size_t total_comp = 0;
            int all_ok = 1;

            for (int c = 0; c < n_chunks; c++) {
                hsize_t off[3] = { 0, 0, (hsize_t)c * chunk_z };
                hsize_t csz = 0;
                H5Dget_chunk_storage_size(dset, off, &csz);
                total_comp += csz;

                void    *raw   = malloc(csz);
                uint32_t filt  = 0;
                size_t   bufsz = csz;
                H5Dread_chunk(dset, H5P_DEFAULT, off, &filt, raw, &bufsz);

                CompressionHeader *hdr = (CompressionHeader *)raw;
                if (!hdr->isValid()) { all_ok = 0; total_failures++; free(raw); continue; }
                if (hdr->hasQuantizationApplied()) {
                    printf("  *** FAIL chunk[%d]: quant set on lossless run!\n", c);
                    total_failures++;
                }
                uint8_t aid = hdr->getAlgorithmId();
                int shuf = hdr->hasShuffleApplied() ? 1 : 0;
                if (aid < 9) tally[aid][shuf]++;
                free(raw);
            }

            double raw_mb  = nbytes / (1024.0 * 1024.0);
            double comp_mb = total_comp / (1024.0 * 1024.0);
            printf("  Ratio : %.2fx  (%.1f MB → %.3f MB)  hdr=%s\n",
                   raw_mb / comp_mb, raw_mb, comp_mb, all_ok ? "OK" : "FAIL");
            printf("  Algo  :");
            for (int a = 1; a < 9; a++) {
                int n0 = tally[a][0], n1 = tally[a][1];
                if (n0 + n1 == 0) continue;
                if (n1) printf("  %s+shuf×%d", algo_name(a), n1);
                if (n0) printf("  %s×%d",       algo_name(a), n0);
            }
            printf("\n");

            H5Dclose(dset); H5Sclose(space); H5Pclose(dcpl);
        }

        /* --- Read back (decompress) --- */
        cudaMemset(d_readback, 0xCD, nbytes);
        {
            hid_t dset = H5Dopen2(comp_fid, dset_name, H5P_DEFAULT);

            H5VL_gpucompress_reset_stats();
            double t0 = now_sec();
            herr_t rc = H5Dread(dset, H5T_NATIVE_FLOAT,
                                H5S_ALL, H5S_ALL, H5P_DEFAULT, d_readback);
            double dt = now_sec() - t0;

            int n_decomp = 0;
            H5VL_gpucompress_get_stats(NULL, NULL, NULL, &n_decomp);
            if (rc < 0 || n_decomp != n_chunks) total_failures++;
            printf("  Read  : %s  %.3f s  %.1f MB/s\n",
                   (rc >= 0 && n_decomp == n_chunks) ? "OK  " : "FAIL",
                   dt, (nbytes / (1024.0 * 1024.0)) / dt);

            H5Dclose(dset);
        }

        /* --- Write decompressed to plain HDF5 --- */
        {
            float *h_buf = (float *)malloc(nbytes);
            cudaMemcpy(h_buf, d_readback, nbytes, cudaMemcpyDeviceToHost);

            hid_t space = H5Screate_simple(3, dims, NULL);
            hid_t dset  = H5Dcreate2(decomp_fid, dset_name, H5T_NATIVE_FLOAT, space,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, h_buf);
            H5Dclose(dset); H5Sclose(space);
            free(h_buf);
        }

        /* --- Bitwise verify --- */
        cudaMemset(d_err, 0, sizeof(int));
        unsigned int grid = ((unsigned int)n_elem + 255) / 256;
        verify_kernel<<<grid, 256>>>(d_v, d_readback, (unsigned int)n_elem, d_err);
        cudaDeviceSynchronize();
        int mismatches = 0;
        cudaMemcpy(&mismatches, d_err, sizeof(int), cudaMemcpyDeviceToHost);

        printf("  Verify: %s  (%zu / %zu)\n",
               mismatches == 0 ? "PASS" : "FAIL",
               n_elem - mismatches, n_elem);
        if (mismatches) total_failures++;

        printf("╚═══════════════════════════════════════════════════════════════╝\n");
    }

    /* ---- Summary ---- */
    printf("\n=== Summary: %d snapshot(s), %d failure(s) ===\n",
           n_snapshots, total_failures);
    printf("  Compressed file  : %s  (%d datasets: V_0000..V_%04d)\n",
           comp_fname, n_snapshots, n_snapshots - 1);
    printf("  Decompressed file: %s\n", decomp_fname);
    printf("  Lossless match   : %s\n", total_failures == 0 ? "YES" : "NO");

    /* ---- Cleanup ---- */
    cudaFree(d_err);
    cudaFree(d_readback);
    H5Fclose(comp_fid);
    H5Pclose(plain_fapl);
    H5Fclose(decomp_fid);
    H5Pclose(fapl);
    H5VLclose(vol_id);
    gpucompress_grayscott_destroy(sim);
    gpucompress_cleanup();

    return total_failures ? 1 : 0;
}
