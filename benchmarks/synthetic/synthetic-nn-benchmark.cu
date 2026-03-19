/**
 * @file synthetic-nn-benchmark.cu
 * @brief Synthetic heterogeneous data benchmark to trigger NN exploration.
 *
 * Generates chunks with wildly different data patterns (noise, smooth, spikes,
 * constants, periodic) so the NN picks different algorithms per chunk and
 * exploration fires frequently.  Runs only nn-rl+exp50 for a few timesteps.
 *
 * Usage:
 *   ./build/synthetic_nn_benchmark model.nnwt [--timesteps 20] [--L 128]
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

#define TMP_FILE "/tmp/synthetic_nn_test.h5"

/* ── Helpers ─────────────────────────────────────────────────────── */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static size_t file_size(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    off_t sz = lseek(fd, 0, SEEK_END);
    close(fd);
    return (sz < 0) ? 0 : (size_t)sz;
}

static void pack_double_cd(double v, unsigned int *lo, unsigned int *hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static hid_t make_vol_fapl(void) {
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);
    return fapl;
}

static hid_t make_dcpl_auto(int L, int chunk_z) {
    hsize_t cdims[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)chunk_z };
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 3, cdims);
    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0; /* ALGO_AUTO */
    cd[1] = 0;
    cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]); /* lossless */
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);
    return dcpl;
}

static void action_to_str(int action, char *buf, size_t bufsz) {
    static const char *names[] = {"lz4","snappy","deflate","gdeflate",
                                   "zstd","ans","cascaded","bitcomp"};
    if (action < 0) { snprintf(buf, bufsz, "none"); return; }
    int algo  = action % 8;
    int quant = (action / 8) % 2;
    int shuf  = (action / 16) % 2;
    snprintf(buf, bufsz, "%s%s%s", names[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
}

/* ── GPU kernel: fill heterogeneous patterns ────────────────────── */

__global__ void fill_heterogeneous(float *data, int L, int chunk_z,
                                    int n_chunks, unsigned int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = L * L * L;
    if (idx >= total) return;

    /* Which chunk does this element belong to? */
    int z = idx / (L * L);
    int chunk_id = z / chunk_z;

    /* Mix seed with chunk_id for per-chunk variation */
    unsigned int h = seed ^ (chunk_id * 2654435761u);

    /* Pattern selector: 5 patterns based on chunk_id */
    int pattern = chunk_id % 5;

    float val = 0.0f;
    int local = idx % (L * L * chunk_z);

    switch (pattern) {
    case 0:  /* Random noise — hard to compress */
    {
        curandState rng;
        curand_init(h + idx, 0, 0, &rng);
        val = curand_uniform(&rng) * 1000.0f;
        break;
    }
    case 1:  /* Constant — trivially compressible, huge ratio with any algo */
        val = 3.14159f;
        break;
    case 2:  /* Smooth low-entropy — small integers, great for delta/cascaded */
        val = (float)(local % 256);
        break;
    case 3:  /* Periodic repeating — good for LZ-family (snappy/lz4/zstd) */
        val = (float)((local % 32) * 100);
        break;
    case 4:  /* Sparse spikes — mostly zero with rare large values */
    {
        curandState rng;
        curand_init(h + idx, 0, 0, &rng);
        val = (curand_uniform(&rng) > 0.99f) ? 99999.0f : 0.0f;
        break;
    }
    }

    data[idx] = val;
}

/* ── Re-generate with shifted seed each timestep (preserves pattern diversity) ── */

/* ── Main ────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    const char *weights_path = NULL;
    int L = 128;
    int chunk_z = 4;
    int timesteps = 20;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (!weights_path && argv[i][0] != '-') {
            weights_path = argv[i];
        } else if (strcmp(argv[i], "--L") == 0 && i + 1 < argc) {
            L = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chunk-z") == 0 && i + 1 < argc) {
            chunk_z = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--timesteps") == 0 && i + 1 < argc) {
            timesteps = atoi(argv[++i]);
        }
    }

    if (!weights_path) weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s model.nnwt [--L 128] [--timesteps 20]\n", argv[0]);
        return 1;
    }

    size_t n_floats = (size_t)L * L * L;
    size_t total_bytes = n_floats * sizeof(float);
    double dataset_mb = (double)total_bytes / (1 << 20);
    int n_chunks = L / chunk_z;

    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║  Synthetic Heterogeneous NN Exploration Benchmark     ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n\n");
    printf("  Grid     : %d^3 = %zu floats (%.1f MB)\n", L, n_floats, dataset_mb);
    printf("  Chunks   : %d x %d x %d  (%d chunks, %.1f MB each)\n",
           L, L, chunk_z, n_chunks, dataset_mb / n_chunks);
    printf("  Timesteps: %d\n", timesteps);
    printf("  Patterns : noise / constant / gradient / sine / sparse (per chunk)\n");
    printf("  Weights  : %s\n\n", weights_path);

    /* Init GPUCompress + VOL */
    if (gpucompress_init(weights_path) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init failed\n"); return 1;
    }
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id == H5I_INVALID_HID) {
        fprintf(stderr, "FATAL: VOL register failed\n");
        gpucompress_cleanup(); return 1;
    }

    /* Configure nn-rl+exp50 */
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, 0.1f, 0.10f, 0.10f);
    gpucompress_set_exploration(1);
    gpucompress_set_exploration_threshold(0.0);
    gpucompress_set_exploration_k(8);

    /* Allocate GPU buffers */
    float *d_data = NULL, *d_read = NULL;
    cudaMalloc(&d_data, total_bytes);
    cudaMalloc(&d_read, total_bytes);

    hid_t dcpl = make_dcpl_auto(L, chunk_z);
    hsize_t dims3[3] = { (hsize_t)L, (hsize_t)L, (hsize_t)L };

    /* Initial fill */
    int threads = 256;
    int blocks = (n_floats + threads - 1) / threads;
    fill_heterogeneous<<<blocks, threads>>>(d_data, L, chunk_z, n_chunks, 12345);
    cudaDeviceSynchronize();

    printf("  %-4s  %-7s  %-7s  %-7s  %-8s  %-8s  %-8s  %-4s  %-5s\n",
           "T", "WrMs", "RdMs", "Ratio",
           "MAPE_R", "MAPE_C", "MAPE_D", "SGD", "Expl");
    printf("  ----  -------  -------  -------  "
           "--------  --------  --------  ----  -----\n");

    for (int t = 0; t < timesteps; t++) {
        /* Regenerate with different seed each timestep — preserves pattern diversity */
        fill_heterogeneous<<<blocks, threads>>>(d_data, L, chunk_z, n_chunks,
                                                 12345 + t * 7919);
        cudaDeviceSynchronize();

        gpucompress_reset_chunk_history();
        remove(TMP_FILE);

        /* Write */
        hid_t fapl = make_vol_fapl();
        hid_t file = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);

        hid_t fsp  = H5Screate_simple(3, dims3, NULL);
        hid_t dset = H5Dcreate2(file, "V", H5T_NATIVE_FLOAT,
                                 fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        H5Sclose(fsp);

        double tw0 = now_ms();
        H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
        H5Dclose(dset); H5Fclose(file);
        double write_ms = now_ms() - tw0;

        /* Read */
        fapl = make_vol_fapl();
        file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
        H5Pclose(fapl);
        dset = H5Dopen2(file, "V", H5P_DEFAULT);

        double tr0 = now_ms();
        H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
        cudaDeviceSynchronize();
        H5Dclose(dset); H5Fclose(file);
        double read_ms = now_ms() - tr0;

        /* Stats */
        size_t fsz = file_size(TMP_FILE);
        double ratio = (fsz > 0) ? (double)total_bytes / (double)fsz : 1.0;

        int n_hist = gpucompress_get_chunk_history_count();
        double mape_r = 0, mape_c = 0, mape_d = 0;
        int mr_cnt = 0, mc_cnt = 0, md_cnt = 0;
        int sgd_fires = 0, explorations = 0;

        for (int ci = 0; ci < n_hist; ci++) {
            gpucompress_chunk_diag_t d;
            if (gpucompress_get_chunk_diag(ci, &d) != 0) continue;
            if (d.sgd_fired) sgd_fires++;
            if (d.exploration_triggered) explorations++;
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
        mape_r = mr_cnt ? (mape_r / mr_cnt) * 100.0 : 0;
        mape_c = mc_cnt ? (mape_c / mc_cnt) * 100.0 : 0;
        mape_d = md_cnt ? (mape_d / md_cnt) * 100.0 : 0;

        printf("  %-4d  %6.0f  %6.0f   %5.2fx  %7.1f%%  %7.1f%%  %7.1f%%  %3d   %3d\n",
               t, write_ms, read_ms, ratio,
               mape_r, mape_c, mape_d, sgd_fires, explorations);

        /* Print per-chunk algo selection at milestones */
        if (t == 0 || t == timesteps - 1) {
            printf("    ── Algo selection at T=%d ──\n", t);
            for (int ci = 0; ci < n_hist; ci++) {
                gpucompress_chunk_diag_t d;
                if (gpucompress_get_chunk_diag(ci, &d) != 0) continue;
                if (ci < 5 || ci >= n_hist - 3 || ci % (n_hist / 8 + 1) == 0) {
                    char astr[40];
                    action_to_str(d.nn_action, astr, sizeof(astr));
                    printf("      C%3d [%-18s] ratio=%5.1fx  pred_r=%5.1fx%s%s\n",
                           ci + 1, astr, (double)d.actual_ratio,
                           (double)d.predicted_ratio,
                           d.sgd_fired ? " [SGD]" : "",
                           d.exploration_triggered ? " [EXP]" : "");
                }
            }
            printf("    ── end T=%d ──\n", t);
        }

        remove(TMP_FILE);
    }

    printf("\n=== Synthetic benchmark complete ===\n");

    /* Cleanup */
    cudaFree(d_data);
    cudaFree(d_read);
    H5Pclose(dcpl);
    H5VLclose(vol_id);
    gpucompress_cleanup();
    return 0;
}
