/**
 * @file test_vol_nn_predictions.cu
 * @brief HDF5 VOL end-to-end: GPU data generation -> write -> read with NN predictions.
 *
 * Generates 6 diverse data patterns on GPU, writes through the GPUCompress HDF5 VOL
 * with ALGO_AUTO (NN selects algorithm per chunk), then reads back to verify.
 * Prints per-chunk NN predictions vs actuals for all metrics.
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
#define TMP_FILE "/tmp/test_vol_nn_predictions.h5"
#define CSV_FILE "/tmp/test_vol_nn_predictions.csv"

static const char* ALGO_NAMES[] = {
    "LZ4", "Snappy", "Deflate", "GDeflate", "Zstd", "ANS", "Cascaded", "Bitcomp"
};

/* ============================================================
 * GPU Data Generation Kernels
 * ============================================================ */

__global__ void gen_constant(float* out, size_t n, float val) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = val;
}

__global__ void gen_sine_wave(float* out, size_t n, float freq) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = sinf((float)i * freq) * 100.0f;
}

__global__ void gen_random_uniform(float* out, size_t n, unsigned long long seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long x = (i + 1ULL) * 6364136223846793005ULL + seed;
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        out[i] = (float)(x & 0xFFFFFF) / (float)0xFFFFFF * 2.0f - 1.0f;
    }
}

__global__ void gen_sparse(float* out, size_t n, float density, unsigned long long seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long x = (i + 1ULL) * 6364136223846793005ULL + seed;
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        float r = (float)(x & 0xFFFFFF) / (float)0xFFFFFF;
        out[i] = (r < density) ? (float)((x >> 24) & 0xFFFF) : 0.0f;
    }
}

__global__ void gen_step_blocks(float* out, size_t n, int block_size) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        int block_id = (int)(i / block_size);
        out[i] = (float)(block_id * 7 % 256);
    }
}

__global__ void gen_linear_ramp(float* out, size_t n, float scale) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)i * scale;
}

/* ============================================================
 * Helpers
 * ============================================================ */

static void pack_double_cd(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits;
    memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

static void action_to_str(int action, char* buf, size_t bufsz) {
    if (action < 0) { snprintf(buf, bufsz, "none"); return; }
    int algo  = action % 8;
    int quant = (action / 8) % 2;
    int shuf  = (action / 16) % 2;
    snprintf(buf, bufsz, "%s%s%s", ALGO_NAMES[algo],
             shuf ? "+shuf" : "", quant ? "+quant" : "");
}

struct PatternInfo {
    const char* name;
    void (*launch)(float* d_ptr, size_t n);
};

static void launch_constant(float* d, size_t n) {
    int blocks = (int)((n + 255) / 256);
    gen_constant<<<blocks, 256>>>(d, n, 3.14159f);
}
static void launch_sine(float* d, size_t n) {
    int blocks = (int)((n + 255) / 256);
    gen_sine_wave<<<blocks, 256>>>(d, n, 0.001f);
}
static void launch_random(float* d, size_t n) {
    int blocks = (int)((n + 255) / 256);
    gen_random_uniform<<<blocks, 256>>>(d, n, 12345ULL);
}
static void launch_sparse(float* d, size_t n) {
    int blocks = (int)((n + 255) / 256);
    gen_sparse<<<blocks, 256>>>(d, n, 0.02f, 67890ULL);
}
static void launch_step_blocks(float* d, size_t n) {
    int blocks = (int)((n + 255) / 256);
    gen_step_blocks<<<blocks, 256>>>(d, n, 1024);
}
static void launch_ramp(float* d, size_t n) {
    int blocks = (int)((n + 255) / 256);
    gen_linear_ramp<<<blocks, 256>>>(d, n, 1e-5f);
}

static PatternInfo PATTERNS[] = {
    { "constant",    launch_constant },
    { "sine_wave",   launch_sine },
    { "random",      launch_random },
    { "sparse_2pct", launch_sparse },
    { "step_blocks", launch_step_blocks },
    { "linear_ramp", launch_ramp },
};
static const int N_PATTERNS = sizeof(PATTERNS) / sizeof(PATTERNS[0]);

/* Per-pattern accumulator for summary statistics */
struct PatternStats {
    float ratio_ape_sum, ct_ape_sum, dt_ape_sum;
    float ratio_act_sum, ratio_pred_sum;
    float ct_act_sum, ct_pred_sum;
    float dt_act_sum, dt_pred_sum;
    int   count, sgd_count;
    int   algo_hist[8];
};

/* ============================================================
 * Main
 * ============================================================ */
static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [--chunk-mb <MiB>] [--total-mb <MiB>] [--rl-lr <float>] [--mape-threshold <float>]\n"
        "  --chunk-mb        Chunk size in MiB (default: 1)\n"
        "  --total-mb        Total dataset size in MiB (default: 6, i.e. 1 chunk per pattern)\n"
        "  --rl-lr           SGD learning rate (default: 0.25)\n"
        "  --mape-threshold  MAPE threshold fraction (default: 0.20)\n", prog);
}

int main(int argc, char** argv)
{
    /* ---- Parse CLI args ---- */
    float  chunk_mb       = 1.0f;
    float  total_mb       = 0.0f;   /* 0 = auto (N_PATTERNS * chunk_mb) */
    float  rl_lr          = 0.25f;
    float  mape_threshold = 0.20f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--chunk-mb") && i + 1 < argc)
            chunk_mb = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--total-mb") && i + 1 < argc)
            total_mb = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--rl-lr") && i + 1 < argc)
            rl_lr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--mape-threshold") && i + 1 < argc)
            mape_threshold = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]); return 0;
        }
    }

    /* ---- Header ---- */
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║         HDF5 VOL  NN Prediction Demo                   ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\n\n");

    /* ---- Init GPUCompress ---- */
    const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights) weights = "neural_net/weights/model.nnwt";
    gpucompress_error_t err = gpucompress_init(weights);
    if (err != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init failed (err=%d). Set GPUCOMPRESS_WEIGHTS.\n", err);
        return 1;
    }

    /* ---- Configure RL ---- */
    gpucompress_enable_online_learning();
    gpucompress_set_reinforcement(1, rl_lr, mape_threshold, mape_threshold);
    gpucompress_set_exploration(0);

    /* ---- Compute sizes ---- */
    const size_t CHUNK_FLOATS = (size_t)(chunk_mb * 1024 * 1024 / sizeof(float));
    const size_t CHUNK_BYTES  = CHUNK_FLOATS * sizeof(float);
    const size_t TOTAL_BYTES  = (total_mb > 0)
        ? (size_t)(total_mb * 1024 * 1024)
        : CHUNK_BYTES * N_PATTERNS;
    const int    N_CHUNKS     = (int)(TOTAL_BYTES / CHUNK_BYTES);
    const size_t N_FLOATS     = (size_t)N_CHUNKS * CHUNK_FLOATS;

    /* Print config */
    printf("  Config\n");
    printf("  ├─ Chunk size    : %.0f MiB  (%zu floats)\n", chunk_mb, CHUNK_FLOATS);
    printf("  ├─ Total size    : %.0f MiB  (%d chunks)\n", (double)TOTAL_BYTES / (1 << 20), N_CHUNKS);
    printf("  ├─ Patterns      : %d (cycling)\n", N_PATTERNS);
    printf("  ├─ SGD lr        : %.3f\n", rl_lr);
    printf("  ├─ MAPE threshold: %.0f%%\n", mape_threshold * 100.0f);
    printf("  └─ Mode          : lossless, ALGO_AUTO\n");

    /* ---- Generate data on GPU ---- */
    float* d_data = NULL;
    cudaMalloc(&d_data, TOTAL_BYTES);

    printf("\n  Generating data on GPU ...");
    for (int c = 0; c < N_CHUNKS; c++) {
        PATTERNS[c % N_PATTERNS].launch(d_data + (size_t)c * CHUNK_FLOATS, CHUNK_FLOATS);
    }
    cudaDeviceSynchronize();
    printf(" done\n");

    /* ---- Create HDF5 file with VOL ---- */
    hid_t native_id = H5VLget_connector_id_by_name("native");
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_dims[1] = { (hsize_t)CHUNK_FLOATS };
    H5Pset_chunk(dcpl, 1, chunk_dims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS];
    cd[0] = 0;  cd[1] = 0;  cd[2] = 0;
    pack_double_cd(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS,
                  H5Z_FLAG_OPTIONAL, H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    /* ---- Write ---- */
    printf("  Writing via HDF5 VOL ...");
    fflush(stdout);
    gpucompress_reset_chunk_history();

    remove(TMP_FILE);
    hid_t file = H5Fcreate(TMP_FILE, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    if (file < 0) { fprintf(stderr, "\nH5Fcreate failed\n"); return 1; }

    hsize_t dims[1] = { (hsize_t)N_FLOATS };
    hid_t fsp  = H5Screate_simple(1, dims, NULL);
    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             fsp, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(fsp);

    herr_t wret = H5Dwrite(dset, H5T_NATIVE_FLOAT,
                           H5S_ALL, H5S_ALL, H5P_DEFAULT, d_data);
    H5Dclose(dset);
    H5Fclose(file);
    if (wret < 0) { fprintf(stderr, "\nH5Dwrite failed\n"); return 1; }
    printf(" done\n");

    /* ---- Read back ---- */
    printf("  Reading via HDF5 VOL ...");
    fflush(stdout);

    float* d_read = NULL;
    cudaMalloc(&d_read, TOTAL_BYTES);
    cudaMemset(d_read, 0, TOTAL_BYTES);

    fapl = H5Pcreate(H5P_FILE_ACCESS);
    native_id = H5VLget_connector_id_by_name("native");
    H5Pset_fapl_gpucompress(fapl, native_id, NULL);
    H5VLclose(native_id);

    file = H5Fopen(TMP_FILE, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    if (file < 0) { fprintf(stderr, "\nH5Fopen failed\n"); return 1; }

    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    herr_t rret = H5Dread(dset, H5T_NATIVE_FLOAT,
                          H5S_ALL, H5S_ALL, H5P_DEFAULT, d_read);
    H5Dclose(dset);
    H5Fclose(file);
    if (rret < 0) { fprintf(stderr, "\nH5Dread failed\n"); return 1; }
    printf(" done\n");

    /* ============================================================
     * Collect per-chunk diagnostics into per-pattern stats
     * ============================================================ */
    int n_hist = gpucompress_get_chunk_history_count();

    PatternStats pstats[N_PATTERNS];
    memset(pstats, 0, sizeof(pstats));

    int total_sgd = 0;

    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) != 0) continue;

        int pat = i % N_PATTERNS;
        PatternStats& ps = pstats[pat];
        ps.count++;
        if (d.sgd_fired) { ps.sgd_count++; total_sgd++; }

        int algo_idx = d.nn_action % 8;
        ps.algo_hist[algo_idx]++;

        /* Ratio */
        ps.ratio_act_sum  += d.actual_ratio;
        ps.ratio_pred_sum += d.predicted_ratio;
        if (d.actual_ratio > 0 && d.predicted_ratio > 0)
            ps.ratio_ape_sum += fabsf(d.actual_ratio - d.predicted_ratio) / d.actual_ratio * 100.0f;

        /* Comp time */
        ps.ct_act_sum  += d.compression_ms;
        ps.ct_pred_sum += d.predicted_comp_time;
        if (d.compression_ms > 0 && d.predicted_comp_time > 0)
            ps.ct_ape_sum += fabsf(d.compression_ms - d.predicted_comp_time) / d.compression_ms * 100.0f;

        /* Decomp time */
        ps.dt_act_sum  += d.decompression_ms;
        ps.dt_pred_sum += d.predicted_decomp_time;
        if (d.decompression_ms > 0 && d.predicted_decomp_time > 0)
            ps.dt_ape_sum += fabsf(d.decompression_ms - d.predicted_decomp_time) / d.decompression_ms * 100.0f;
    }

    /* ============================================================
     * Write ALL chunk results to CSV + print terminal every 3 chunks
     * ============================================================ */
    FILE* csv = fopen(CSV_FILE, "w");
    if (csv) {
        fprintf(csv, "chunk,pattern,nn_pick,action,"
                     "ratio_actual,ratio_predicted,ratio_mape,"
                     "comp_ms_actual,comp_ms_predicted,comp_mape,"
                     "decomp_ms_actual,decomp_ms_predicted,decomp_mape,"
                     "sgd_fired\n");
    }

    printf("\n");
    printf("  ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    char title_buf[128];
    snprintf(title_buf, sizeof(title_buf),
             "  PER-CHUNK NN PREDICTIONS vs ACTUALS  (every 3rd; all %d in CSV)", n_hist);
    printf("  │%-97s│\n", title_buf);
    printf("  ├──────┬──────────────┬────────────────┬───────────────────┬───────────────────┬───────────────────┤\n");
    printf("  │  #   │ pattern      │ NN pick        │   RATIO           │   COMP TIME (ms)  │  DECOMP TIME (ms) │\n");
    printf("  │      │              │                │  actual predicted │  actual predicted │  actual predicted │\n");
    printf("  │      │              │                │          MAPE     │          MAPE     │          MAPE     │\n");
    printf("  ├──────┼──────────────┼────────────────┼───────────────────┼───────────────────┼───────────────────┤\n");

    for (int i = 0; i < n_hist; i++) {
        gpucompress_chunk_diag_t d;
        if (gpucompress_get_chunk_diag(i, &d) != 0) continue;

        char action_str[20];
        action_to_str(d.nn_action, action_str, sizeof(action_str));

        float r_mape  = (d.actual_ratio > 0 && d.predicted_ratio > 0)
            ? fabsf(d.actual_ratio - d.predicted_ratio) / d.actual_ratio * 100.0f : 0.0f;
        float ct_mape = (d.compression_ms > 0 && d.predicted_comp_time > 0)
            ? fabsf(d.compression_ms - d.predicted_comp_time) / d.compression_ms * 100.0f : 0.0f;
        float dt_mape = (d.decompression_ms > 0 && d.predicted_decomp_time > 0)
            ? fabsf(d.decompression_ms - d.predicted_decomp_time) / d.decompression_ms * 100.0f : 0.0f;

        /* CSV: every chunk */
        if (csv) {
            fprintf(csv, "%d,%s,%s,%d,%.4f,%.4f,%.2f,%.4f,%.4f,%.2f,%.4f,%.4f,%.2f,%d\n",
                    i, PATTERNS[i % N_PATTERNS].name, action_str, d.nn_action,
                    d.actual_ratio, d.predicted_ratio, r_mape,
                    d.compression_ms, d.predicted_comp_time, ct_mape,
                    d.decompression_ms, d.predicted_decomp_time, dt_mape,
                    d.sgd_fired ? 1 : 0);
        }

        /* Terminal: every 3rd chunk + always the last chunk */
        if (i % 3 == 0 || i == n_hist - 1) {
            printf("  │ %4d │ %-12s │ %-14s │ %5.1fx  %5.1fx    │ %6.2f  %6.2f    │ %6.2f  %6.2f    │\n",
                   i, PATTERNS[i % N_PATTERNS].name, action_str,
                   d.actual_ratio, d.predicted_ratio,
                   d.compression_ms, d.predicted_comp_time,
                   d.decompression_ms, d.predicted_decomp_time);
            printf("  │      │              │  %s         │       MAPE %5.1f%% │       MAPE %5.1f%% │       MAPE %5.1f%% │\n",
                   d.sgd_fired ? "SGD Y" : "     ",
                   r_mape, ct_mape, dt_mape);
        }
    }

    printf("  └──────┴──────────────┴────────────────┴───────────────────┴───────────────────┴───────────────────┘\n");

    if (csv) {
        fclose(csv);
        printf("  CSV written to %s\n", CSV_FILE);
    }

    /* ============================================================
     * Per-pattern summary table
     * ============================================================ */
    printf("\n");
    printf("  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  SUMMARY BY PATTERN  (averaged over %d chunks per pattern)                                 │\n",
           n_hist / N_PATTERNS);
    printf("  ├──────────────┬──────────────────┬─────────────┬─────────────┬─────────────┬────────────────┤\n");
    printf("  │ pattern      │ top algorithm    │ ratio MAPE  │ comp MAPE   │ decomp MAPE │  SGD updates   │\n");
    printf("  ├──────────────┼──────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤\n");

    float total_ratio_ape = 0, total_ct_ape = 0, total_dt_ape = 0;
    int   total_count = 0;

    for (int p = 0; p < N_PATTERNS; p++) {
        PatternStats& ps = pstats[p];
        if (ps.count == 0) continue;

        /* Find top algo for this pattern */
        int top_algo = 0;
        for (int a = 1; a < 8; a++)
            if (ps.algo_hist[a] > ps.algo_hist[top_algo]) top_algo = a;

        float r_mape  = ps.ratio_ape_sum / ps.count;
        float ct_mape = ps.ct_ape_sum / ps.count;
        float dt_mape = ps.dt_ape_sum / ps.count;

        total_ratio_ape += ps.ratio_ape_sum;
        total_ct_ape    += ps.ct_ape_sum;
        total_dt_ape    += ps.dt_ape_sum;
        total_count     += ps.count;

        printf("  │ %-12s │ %-16s │   %6.1f%%   │   %6.1f%%   │   %6.1f%%   │ %4d / %-4d     │\n",
               PATTERNS[p].name, ALGO_NAMES[top_algo],
               r_mape, ct_mape, dt_mape,
               ps.sgd_count, ps.count);
    }

    printf("  ├──────────────┼──────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤\n");

    float avg_r  = total_count ? total_ratio_ape / total_count : 0;
    float avg_ct = total_count ? total_ct_ape / total_count : 0;
    float avg_dt = total_count ? total_dt_ape / total_count : 0;

    printf("  │ ALL          │                  │   %6.1f%%   │   %6.1f%%   │   %6.1f%%   │ %4d / %-4d     │\n",
           avg_r, avg_ct, avg_dt, total_sgd, n_hist);
    printf("  └──────────────┴──────────────────┴─────────────┴─────────────┴─────────────┴────────────────┘\n");

    /* ============================================================
     * Verify GPU data integrity (lossless)
     * ============================================================ */
    printf("\n  Verification: ");
    fflush(stdout);

    float* h_orig = (float*)malloc(TOTAL_BYTES);
    float* h_read = (float*)malloc(TOTAL_BYTES);
    cudaMemcpy(h_orig, d_data, TOTAL_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_read, d_read, TOTAL_BYTES, cudaMemcpyDeviceToHost);

    int pass = 1;
    int n_failed = 0;
    for (int c = 0; c < N_CHUNKS; c++) {
        size_t off = (size_t)c * CHUNK_FLOATS;
        bool chunk_ok = true;
        for (size_t j = 0; j < CHUNK_FLOATS; j++) {
            if (h_orig[off + j] != h_read[off + j]) {
                chunk_ok = false;
                break;
            }
        }
        if (!chunk_ok) {
            if (n_failed == 0)
                printf("\n");
            printf("    FAIL: chunk %d (%s)\n", c, PATTERNS[c % N_PATTERNS].name);
            n_failed++; pass = 0;
        }
    }

    if (pass)
        printf("%d/%d chunks bit-exact\n", N_CHUNKS, N_CHUNKS);
    else
        printf("    %d/%d chunks failed!\n", n_failed, N_CHUNKS);

    printf("\n  %s\n\n", pass ? "=== ALL PASS ===" : "=== SOME FAILED ===");

    /* ---- Cleanup ---- */
    free(h_orig);
    free(h_read);
    cudaFree(d_data);
    cudaFree(d_read);
    H5Pclose(dcpl);
    remove(TMP_FILE);
    gpucompress_cleanup();

    return pass ? 0 : 1;
}
