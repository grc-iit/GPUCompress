/**
 * @file bench_lr_sweep.c
 * @brief LR x MAPE Sweep Benchmark for NN Online Adaptation
 *
 * Phase 1: Run all 16 static configs (8 algos x 2 shuffle) per pattern to find
 *          the true best possible compression ratio.
 * Phase 2: Sweep SGD learning rate {0.1..0.9} x MAPE threshold {0.05,0.10,0.20,0.50}
 *          with exploration OFF (reinforcement only).  Track per-chunk compressed
 *          sizes to measure how fast the NN adapts to match the best static ratio.
 *
 * Each (lr, mape) combo gets a fresh NN via cleanup/init cycle.
 *
 * Outputs:
 *   - Aggregate CSV:  pattern,mode,algorithm,shuffle,lr,mape_thr,ratio,...
 *   - Per-chunk CSV:  pattern,lr,mape_thr,chunk_id,ratio,sgd_fired,action_label
 *
 * Usage:
 *   ./build/bench_lr_sweep neural_net/weights/model.nnwt [--output agg.csv] [--chunk-csv chunks.csv]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <hdf5.h>

#include "gpucompress.h"
#include "hdf5/H5Zgpucompress.h"

/* ============================================================
 * Constants
 * ============================================================ */

#define CHUNK_FLOATS   (1024 * 1024)           /* 1M floats = 4 MB per chunk */
#define CHUNK_BYTES    (CHUNK_FLOATS * sizeof(float))
#define DATASET_FLOATS (256 * 1024 * 1024)     /* 256M floats = 1 GB */
#define DATASET_BYTES  ((size_t)DATASET_FLOATS * sizeof(float))
#define N_CHUNKS       (DATASET_FLOATS / CHUNK_FLOATS)  /* 256 */

#define N_ALGOS        8
#define N_STATIC       (N_ALGOS * 2)           /* 8 algos x 2 shuffle */

#define TMP_HDF5       "/tmp/bm_lr_sweep_tmp.h5"
#define DEFAULT_CSV    "tests/bench_lr_sweep_results/bench_lr_sweep.csv"
#define DEFAULT_CHUNK_CSV "tests/bench_lr_sweep_results/bench_lr_sweep_chunks.csv"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================
 * Sweep Grid
 * ============================================================ */

static const float LR_VALUES[]   = { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f };
static const float MAPE_VALUES[] = { 0.05f, 0.10f, 0.20f, 0.50f };

#define N_LR   (int)(sizeof(LR_VALUES)   / sizeof(LR_VALUES[0]))
#define N_MAPE (int)(sizeof(MAPE_VALUES) / sizeof(MAPE_VALUES[0]))

/* ============================================================
 * Pattern Configuration — all 8 patterns
 * ============================================================ */

typedef struct {
    const char *name;
    const char *desc;
    int         fill_id;
} pattern_cfg_t;

static const pattern_cfg_t patterns[] = {
    { "ramp",          "linear ramp",              3 },
    { "sparse",        "sparse (99% zero)",        4 },
    { "gaussian",      "gaussian noise",           5 },
    { "hf_sine_noise", "high-freq sine + noise",   7 },
};
#define N_PATTERNS (int)(sizeof(patterns) / sizeof(patterns[0]))

/* ============================================================
 * Static config table
 * ============================================================ */

typedef struct {
    gpucompress_algorithm_t algo;
    unsigned int            shuffle_sz;
} static_cfg_t;

/* ============================================================
 * PRNG (simple LCG)
 * ============================================================ */

static uint32_t lcg_state;
static void     lcg_seed(uint32_t s) { lcg_state = s; }
static uint32_t lcg_next(void)       { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state; }
static float    lcg_float(void)      { return (float)(lcg_next() >> 8) / 16777216.0f; }

/* ============================================================
 * Timing
 * ============================================================ */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ============================================================
 * Data Pattern Generator
 * ============================================================ */

static void fill_dataset(float *buf, int id) {
    const size_t N = DATASET_FLOATS;
    switch (id) {
    case 0:
        for (size_t i = 0; i < N; i++) buf[i] = 42.0f;
        break;
    case 1:
        for (size_t i = 0; i < N; i++)
            buf[i] = 1000.0f * sinf(2.0f * (float)M_PI * i / (float)N);
        break;
    case 2:
        lcg_seed(0xDEADBEEF);
        for (size_t i = 0; i < N; i++) buf[i] = lcg_float() * 2000.0f - 1000.0f;
        break;
    case 3:
        for (size_t i = 0; i < N; i++) buf[i] = (float)i / (float)N;
        break;
    case 4:
        lcg_seed(0xCAFEBABE);
        for (size_t i = 0; i < N; i++) {
            if ((lcg_next() % 100) == 0) buf[i] = lcg_float() * 10000.0f - 5000.0f;
            else buf[i] = 0.0f;
        }
        break;
    case 5:
        lcg_seed(0x12345678);
        for (size_t i = 0; i < N; i += 2) {
            float u1 = lcg_float() * 0.9999f + 0.0001f;
            float u2 = lcg_float();
            float mag = sqrtf(-2.0f * logf(u1));
            buf[i] = mag * cosf(2.0f * (float)M_PI * u2) * 500.0f;
            if (i + 1 < N) buf[i + 1] = mag * sinf(2.0f * (float)M_PI * u2) * 500.0f;
        }
        break;
    case 6:
        for (size_t i = 0; i < N; i++) buf[i] = (float)(i / (N / 8)) * 100.0f;
        break;
    case 7:
        lcg_seed(0xABCD1234);
        for (size_t i = 0; i < N; i++)
            buf[i] = 500.0f * sinf(2.0f * (float)M_PI * i * 0.3f) + (lcg_float() - 0.5f) * 1000.0f;
        break;
    }
}

/* ============================================================
 * Action label helper
 * ============================================================ */

static void action_label(int action, char *buf, size_t bufsz) {
    if (action < 0) { snprintf(buf, bufsz, "?"); return; }
    int algo_idx = action % 8;
    int use_shuf = (action / 16) % 2;
    snprintf(buf, bufsz, "%s%s",
             gpucompress_algorithm_name((gpucompress_algorithm_t)(algo_idx + 1)),
             use_shuf ? "+shuf" : "");
}

/* ============================================================
 * Result Storage
 * ============================================================ */

#define MAX_RESULTS 2048

typedef struct {
    char         pattern[32];
    char         mode[16];       /* "static" or "nn" */
    char         algorithm[32];
    unsigned int shuffle;
    float        lr;
    float        mape_thr;
    double       ratio;
    double       write_ms;
    double       read_ms;
    double       write_mbps;
    double       read_mbps;
    int          sgd_fired_count;
    int          converged_chunk; /* first chunk where rolling-16 ratio >= 99% best static, -1 if never */
} sweep_result_t;

static sweep_result_t g_results[MAX_RESULTS];
static int            g_nresults = 0;

/* Per-chunk CSV file handle */
static FILE *g_chunk_csv = NULL;

/* ============================================================
 * run_static — single static algo+shuffle HDF5 write+read
 * ============================================================ */

static int run_static(const float *data, size_t n_floats,
                      gpucompress_algorithm_t algo, unsigned int shuffle_sz,
                      sweep_result_t *r)
{
    size_t orig_bytes = n_floats * sizeof(float);
    remove(TMP_HDF5);

    hid_t file = H5Fcreate(TMP_HDF5, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) return -1;

    hsize_t dims[1]       = { n_floats };
    hsize_t chunk_dims[1] = { CHUNK_FLOATS };
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);

    herr_t hs = H5Pset_gpucompress(dcpl, algo, 0, shuffle_sz, 0.0);
    if (hs < 0) {
        H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
        remove(TMP_HDF5); return -1;
    }

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) {
        H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
        remove(TMP_HDF5); return -1;
    }

    double t0 = time_ms();
    hs = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    if (hs < 0) {
        H5Dclose(dset); H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
        remove(TMP_HDF5); return -1;
    }
    H5Dclose(dset); H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
    double t1 = time_ms();

    /* Read back */
    file = H5Fopen(TMP_HDF5, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { remove(TMP_HDF5); return -1; }
    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    if (dset < 0) { H5Fclose(file); remove(TMP_HDF5); return -1; }

    hsize_t storage = H5Dget_storage_size(dset);

    float *rbuf = (float *)malloc(orig_bytes);
    if (!rbuf) { H5Dclose(dset); H5Fclose(file); remove(TMP_HDF5); return -1; }

    double t2 = time_ms();
    hs = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
    H5Dclose(dset); H5Fclose(file);
    double t3 = time_ms();

    if (hs < 0) { free(rbuf); remove(TMP_HDF5); return -1; }

    for (size_t i = 0; i < n_floats; i++) {
        if (data[i] != rbuf[i]) {
            fprintf(stderr, "VERIFY FAIL: static mismatch at [%zu]\n", i);
            free(rbuf); remove(TMP_HDF5); return -1;
        }
    }
    free(rbuf);

    snprintf(r->algorithm, sizeof(r->algorithm), "%s",
             gpucompress_algorithm_name(algo));
    r->shuffle    = shuffle_sz;
    r->ratio      = (storage > 0) ? (double)orig_bytes / (double)storage : 0.0;
    r->write_ms   = t1 - t0;
    r->read_ms    = t3 - t2;
    double mb     = orig_bytes / (1024.0 * 1024.0);
    r->write_mbps = (r->write_ms > 0) ? mb / (r->write_ms / 1000.0) : 0.0;
    r->read_mbps  = (r->read_ms  > 0) ? mb / (r->read_ms  / 1000.0) : 0.0;
    r->lr         = 0.0f;
    r->mape_thr   = 0.0f;
    r->sgd_fired_count = 0;
    r->converged_chunk = -1;

    remove(TMP_HDF5);
    return 0;
}

/* ============================================================
 * run_nn — NN HDF5 write+read with per-chunk ratio extraction
 *
 * Writes per-chunk data to g_chunk_csv if open.
 * Computes converged_chunk: first chunk where a rolling-16 average
 * of per-chunk ratios reaches >= target_ratio * 0.99.
 * ============================================================ */

static int run_nn(const float *data, size_t n_floats,
                  double target_ratio, sweep_result_t *r)
{
    size_t orig_bytes = n_floats * sizeof(float);
    remove(TMP_HDF5);

    hid_t file = H5Fcreate(TMP_HDF5, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) return -1;

    hsize_t dims[1]       = { n_floats };
    hsize_t chunk_dims[1] = { CHUNK_FLOATS };
    hid_t dspace = H5Screate_simple(1, dims, NULL);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, chunk_dims);

    if (H5Pset_gpucompress(dcpl, GPUCOMPRESS_ALGO_AUTO, 0, 0, 0.0) < 0) {
        H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
        remove(TMP_HDF5); return -1;
    }

    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_FLOAT,
                             dspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    if (dset < 0) {
        H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
        remove(TMP_HDF5); return -1;
    }

    gpucompress_reset_chunk_history();
    double t0 = time_ms();
    herr_t hs = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                          H5P_DEFAULT, data);
    if (hs < 0) {
        H5Dclose(dset); H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
        remove(TMP_HDF5); return -1;
    }
    H5Dclose(dset); H5Pclose(dcpl); H5Sclose(dspace); H5Fclose(file);
    double t1 = time_ms();

    /* Collect per-chunk diagnostics */
    int n_hist = gpucompress_get_chunk_history_count();
    int sgd_count = 0;
    int last_action = -1;

    int    chunk_sgd[N_CHUNKS];
    int    chunk_action[N_CHUNKS];
    memset(chunk_sgd, 0, sizeof(chunk_sgd));
    memset(chunk_action, -1, sizeof(chunk_action));

    for (int i = 0; i < N_CHUNKS && i < n_hist; i++) {
        gpucompress_chunk_diag_t hd;
        if (gpucompress_get_chunk_diag(i, &hd) == 0) {
            chunk_sgd[i]    = hd.sgd_fired;
            chunk_action[i] = hd.nn_action;
            sgd_count      += hd.sgd_fired;
            last_action     = hd.nn_action;
        }
    }

    /* Reopen to get per-chunk compressed sizes + storage + read-back */
    file = H5Fopen(TMP_HDF5, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) { remove(TMP_HDF5); return -1; }
    dset = H5Dopen2(file, "data", H5P_DEFAULT);
    if (dset < 0) { H5Fclose(file); remove(TMP_HDF5); return -1; }

    hsize_t storage = H5Dget_storage_size(dset);

    /* Per-chunk compressed sizes */
    double chunk_ratios[N_CHUNKS];
    memset(chunk_ratios, 0, sizeof(chunk_ratios));
    {
        hid_t cspace = H5Dget_space(dset);
        for (int i = 0; i < N_CHUNKS; i++) {
            hsize_t coff; unsigned fmask; haddr_t caddr; hsize_t csz = 0;
            if (H5Dget_chunk_info(dset, cspace, (hsize_t)i,
                                  &coff, &fmask, &caddr, &csz) >= 0 && csz > 0) {
                chunk_ratios[i] = (double)CHUNK_BYTES / (double)csz;
            }
        }
        H5Sclose(cspace);
    }

    /* Timed read-back */
    float *rbuf = (float *)malloc(orig_bytes);
    if (!rbuf) { H5Dclose(dset); H5Fclose(file); remove(TMP_HDF5); return -1; }

    double t2 = time_ms();
    hs = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rbuf);
    H5Dclose(dset); H5Fclose(file);
    double t3 = time_ms();

    if (hs < 0) { free(rbuf); remove(TMP_HDF5); return -1; }

    for (size_t i = 0; i < n_floats; i++) {
        if (data[i] != rbuf[i]) {
            fprintf(stderr, "VERIFY FAIL: NN mismatch at [%zu]\n", i);
            free(rbuf); remove(TMP_HDF5); return -1;
        }
    }
    free(rbuf);

    /* Write per-chunk CSV and compute convergence */
    double threshold = target_ratio * 0.99;
    int converged = -1;
    double rolling_sum = 0.0;
    int    rolling_w   = 16;

    for (int i = 0; i < N_CHUNKS; i++) {
        rolling_sum += chunk_ratios[i];
        if (i >= rolling_w) rolling_sum -= chunk_ratios[i - rolling_w];

        int w = (i < rolling_w) ? (i + 1) : rolling_w;
        double rolling_avg = rolling_sum / w;

        if (converged < 0 && i >= rolling_w - 1 && rolling_avg >= threshold)
            converged = i;

        /* Write per-chunk row */
        if (g_chunk_csv) {
            char albl[32];
            action_label(chunk_action[i], albl, sizeof(albl));
            fprintf(g_chunk_csv, "%s,%.2f,%.2f,%d,%.4f,%.4f,%d,%s\n",
                    r->pattern, r->lr, r->mape_thr, i,
                    chunk_ratios[i], rolling_avg,
                    chunk_sgd[i], albl);
        }
    }

    action_label(last_action, r->algorithm, sizeof(r->algorithm));
    r->shuffle         = 0;
    r->ratio           = (storage > 0) ? (double)orig_bytes / (double)storage : 0.0;
    r->write_ms        = t1 - t0;
    r->read_ms         = t3 - t2;
    double mb          = orig_bytes / (1024.0 * 1024.0);
    r->write_mbps      = (r->write_ms > 0) ? mb / (r->write_ms / 1000.0) : 0.0;
    r->read_mbps       = (r->read_ms  > 0) ? mb / (r->read_ms  / 1000.0) : 0.0;
    r->sgd_fired_count = sgd_count;
    r->converged_chunk = converged;

    remove(TMP_HDF5);
    return 0;
}

/* ============================================================
 * CSV Output
 * ============================================================ */

static void write_csv(const char *path) {
    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "ERROR: cannot write %s\n", path); return; }

    fprintf(fp, "pattern,mode,algorithm,shuffle,lr,mape_thr,ratio,"
                "write_ms,read_ms,write_mbps,read_mbps,"
                "sgd_fired_count,converged_chunk\n");

    for (int i = 0; i < g_nresults; i++) {
        sweep_result_t *r = &g_results[i];
        fprintf(fp, "%s,%s,%s,%u,%.2f,%.2f,%.4f,"
                    "%.2f,%.2f,%.1f,%.1f,%d,%d\n",
                r->pattern, r->mode, r->algorithm, r->shuffle,
                r->lr, r->mape_thr,
                r->ratio, r->write_ms, r->read_ms,
                r->write_mbps, r->read_mbps,
                r->sgd_fired_count, r->converged_chunk);
    }

    fclose(fp);
    printf("\nCSV written: %s (%d rows)\n", path, g_nresults);
}

/* ============================================================
 * Console Grid: rows=LR, cols=MAPE
 * ============================================================ */

static void print_grid(const char *pat_name, double best_ratio,
                       const char *best_cfg) {
    double grid[9][4];
    int    sgd_grid[9][4];
    int    conv_grid[9][4];
    memset(grid, 0, sizeof(grid));
    memset(sgd_grid, 0, sizeof(sgd_grid));
    memset(conv_grid, -1, sizeof(conv_grid));

    for (int i = 0; i < g_nresults; i++) {
        sweep_result_t *r = &g_results[i];
        if (strcmp(r->pattern, pat_name) != 0 || strcmp(r->mode, "nn") != 0)
            continue;
        int li = -1, mi = -1;
        for (int l = 0; l < N_LR; l++)
            if (fabsf(r->lr - LR_VALUES[l]) < 0.001f) { li = l; break; }
        for (int m = 0; m < N_MAPE; m++)
            if (fabsf(r->mape_thr - MAPE_VALUES[m]) < 0.001f) { mi = m; break; }
        if (li >= 0 && mi >= 0) {
            grid[li][mi]     = r->ratio;
            sgd_grid[li][mi] = r->sgd_fired_count;
            conv_grid[li][mi] = r->converged_chunk;
        }
    }

    printf("\n=== LR x MAPE Sweep — %s (best static: %.2fx %s) ===\n",
           pat_name, best_ratio, best_cfg);

    /* Ratio grid */
    printf("\n  Final Compression Ratio (* = within 1%% of best static):\n");
    printf("  %8s |", "LR\\MAPE");
    for (int m = 0; m < N_MAPE; m++) printf(" %8.2f", MAPE_VALUES[m]);
    printf("\n  ---------+");
    for (int m = 0; m < N_MAPE; m++) printf("---------");
    printf("\n");
    for (int l = 0; l < N_LR; l++) {
        printf("  %8.1f |", LR_VALUES[l]);
        for (int m = 0; m < N_MAPE; m++) {
            if (grid[l][m] > 0) {
                char mark = (grid[l][m] >= best_ratio * 0.99) ? '*' : ' ';
                printf(" %6.2fx%c", grid[l][m], mark);
            } else
                printf("    FAIL ");
        }
        printf("\n");
    }

    /* Convergence grid */
    printf("\n  Converged at chunk (-1 = never reached 99%% of best):\n");
    printf("  %8s |", "LR\\MAPE");
    for (int m = 0; m < N_MAPE; m++) printf(" %8.2f", MAPE_VALUES[m]);
    printf("\n  ---------+");
    for (int m = 0; m < N_MAPE; m++) printf("---------");
    printf("\n");
    for (int l = 0; l < N_LR; l++) {
        printf("  %8.1f |", LR_VALUES[l]);
        for (int m = 0; m < N_MAPE; m++) {
            if (grid[l][m] > 0)
                printf(" %8d", conv_grid[l][m]);
            else
                printf("     FAIL");
        }
        printf("\n");
    }

    /* SGD count grid */
    printf("\n  SGD Fire Count:\n");
    printf("  %8s |", "LR\\MAPE");
    for (int m = 0; m < N_MAPE; m++) printf(" %8.2f", MAPE_VALUES[m]);
    printf("\n  ---------+");
    for (int m = 0; m < N_MAPE; m++) printf("---------");
    printf("\n");
    for (int l = 0; l < N_LR; l++) {
        printf("  %8.1f |", LR_VALUES[l]);
        for (int m = 0; m < N_MAPE; m++) {
            if (grid[l][m] > 0)
                printf(" %8d", sgd_grid[l][m]);
            else
                printf("     FAIL");
        }
        printf("\n");
    }
    printf("\n");
}

/* ============================================================
 * Static summary table
 * ============================================================ */

static void print_static_summary(double best_ratios[], char best_cfgs[][64]) {
    printf("\n");
    printf("========================================================\n");
    printf("  PHASE 1 RESULTS: Best Static Config Per Pattern\n");
    printf("========================================================\n\n");
    printf("  %-14s | %10s | %-20s\n", "Pattern", "Ratio", "Best Config");
    printf("  %-14s-+-%10s-+-%-20s\n",
           "--------------", "----------", "--------------------");
    for (int p = 0; p < N_PATTERNS; p++) {
        printf("  %-14s | %8.2fx | %-20s\n",
               patterns[p].name, best_ratios[p], best_cfgs[p]);
    }
    printf("\n");
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char **argv) {
    const char *csv_path       = DEFAULT_CSV;
    const char *chunk_csv_path = DEFAULT_CHUNK_CSV;
    const char *weights_path   = NULL;
    const char *pattern_filter = NULL;  /* NULL = run all */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            csv_path = argv[++i];
        else if (strcmp(argv[i], "--chunk-csv") == 0 && i + 1 < argc)
            chunk_csv_path = argv[++i];
        else if (strcmp(argv[i], "--pattern") == 0 && i + 1 < argc)
            pattern_filter = argv[++i];
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [weights.nnwt] [--pattern NAME] [--output agg.csv] [--chunk-csv chunks.csv]\n"
                   "  Patterns: ramp, sparse, gaussian, hf_sine_noise\n"
                   "  Or set GPUCOMPRESS_WEIGHTS env var\n", argv[0]);
            return 0;
        } else if (!weights_path)
            weights_path = argv[i];
    }

    if (!weights_path)
        weights_path = getenv("GPUCOMPRESS_WEIGHTS");
    if (!weights_path) {
        fprintf(stderr, "Usage: %s [weights.nnwt] [--output agg.csv] [--chunk-csv chunks.csv]\n"
                        "  Or set GPUCOMPRESS_WEIGHTS env var\n", argv[0]);
        return 1;
    }

    /* Count patterns that will actually run */
    int n_active = 0;
    for (int p = 0; p < N_PATTERNS; p++) {
        if (!pattern_filter || strcmp(pattern_filter, patterns[p].name) == 0)
            n_active++;
    }

    printf("=== LR x MAPE Sweep Benchmark ===\n\n");
    printf("Weights:     %s\n", weights_path);
    printf("Agg CSV:     %s\n", csv_path);
    printf("Chunk CSV:   %s\n", chunk_csv_path);
    if (pattern_filter)
        printf("Filter:      %s\n", pattern_filter);
    printf("Chunk:       %zu MB float32 (%zu floats)\n",
           CHUNK_BYTES / (1024 * 1024), (size_t)CHUNK_FLOATS);
    printf("Dataset:     %zu MB float32 (%zu floats, %d chunks)\n",
           DATASET_BYTES / (1024 * 1024), (size_t)DATASET_FLOATS, N_CHUNKS);
    printf("Patterns:    %d\n", n_active);
    printf("Static:      %d configs (8 algos x 2 shuffle)\n", N_STATIC);
    printf("LR:          %d values (0.1 .. 0.9)\n", N_LR);
    printf("MAPE:        %d values (0.05, 0.10, 0.20, 0.50)\n", N_MAPE);
    printf("NN runs:     %d per pattern (%d total)\n",
           N_LR * N_MAPE, N_LR * N_MAPE * n_active);
    printf("Exploration: OFF (reinforcement only)\n\n");

    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    /* Build static config table */
    static_cfg_t cfgs[N_STATIC];
    int ncfg = 0;
    for (int a = 1; a <= N_ALGOS; a++) {
        for (int s = 0; s <= 1; s++) {
            cfgs[ncfg].algo       = (gpucompress_algorithm_t)a;
            cfgs[ncfg].shuffle_sz = s ? 4 : 0;
            ncfg++;
        }
    }

    /* Initial init */
    gpucompress_error_t rc = gpucompress_init(weights_path);
    if (rc != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "FATAL: gpucompress_init: %s\n", gpucompress_error_string(rc));
        return 1;
    }
    if (!gpucompress_nn_is_loaded()) {
        fprintf(stderr, "FATAL: NN weights did not load from %s\n", weights_path);
        gpucompress_cleanup();
        return 1;
    }
    if (H5Z_gpucompress_register() < 0) {
        fprintf(stderr, "FATAL: H5Z_gpucompress_register failed\n");
        gpucompress_cleanup();
        return 1;
    }

    /* Per-pattern best static tracking */
    double best_ratios[N_PATTERNS];
    char   best_cfgs[N_PATTERNS][64];
    memset(best_ratios, 0, sizeof(best_ratios));
    memset(best_cfgs, 0, sizeof(best_cfgs));

    /* ============================================================
     * PHASE 1: Run all 16 static configs per pattern
     * ============================================================ */
    printf("============================================================\n");
    printf("  PHASE 1: Static Baselines (%d configs x %d patterns)\n",
           N_STATIC, n_active);
    printf("============================================================\n\n");

    for (int p = 0; p < N_PATTERNS; p++) {
        if (pattern_filter && strcmp(pattern_filter, patterns[p].name) != 0)
            continue;

        printf("Pattern %d/%d: %s (%zu MB)\n",
               p + 1, N_PATTERNS, patterns[p].desc,
               DATASET_BYTES / (1024 * 1024));

        float *data = (float *)malloc(DATASET_BYTES);
        if (!data) { perror("malloc"); return 1; }
        fill_dataset(data, patterns[p].fill_id);

        int ok = 0, fail = 0;
        double best = 0.0;
        int    best_idx = -1;

        for (int c = 0; c < ncfg; c++) {
            if (g_nresults >= MAX_RESULTS) break;
            sweep_result_t *r = &g_results[g_nresults];
            if (run_static(data, DATASET_FLOATS,
                           cfgs[c].algo, cfgs[c].shuffle_sz, r) == 0) {
                snprintf(r->pattern, sizeof(r->pattern), "%s", patterns[p].name);
                snprintf(r->mode, sizeof(r->mode), "static");
                if (r->ratio > best) {
                    best = r->ratio;
                    best_idx = g_nresults;
                }
                g_nresults++;
                ok++;
            } else {
                fail++;
            }
        }

        if (best_idx >= 0) {
            sweep_result_t *br = &g_results[best_idx];
            best_ratios[p] = best;
            snprintf(best_cfgs[p], sizeof(best_cfgs[p]), "%s%s",
                     br->algorithm, br->shuffle ? "+shuf" : "");
            printf("  Best: %.2fx (%s), %d/%d ok\n",
                   best, best_cfgs[p], ok, ncfg);
        }
        if (fail > 0) printf("  %d configs failed\n", fail);

        free(data);
    }

    print_static_summary(best_ratios, best_cfgs);

    /* ============================================================
     * PHASE 2: NN sweep — LR x MAPE per pattern with per-chunk tracking
     * ============================================================ */
    printf("============================================================\n");
    printf("  PHASE 2: NN Sweep (LR x MAPE, exploration OFF)\n");
    printf("============================================================\n\n");

    /* Open per-chunk CSV */
    g_chunk_csv = fopen(chunk_csv_path, "w");
    if (g_chunk_csv) {
        fprintf(g_chunk_csv, "pattern,lr,mape_thr,chunk_id,"
                             "chunk_ratio,rolling_avg,sgd_fired,action\n");
    } else {
        fprintf(stderr, "WARNING: cannot write chunk CSV %s\n", chunk_csv_path);
    }

    for (int p = 0; p < N_PATTERNS; p++) {
        if (pattern_filter && strcmp(pattern_filter, patterns[p].name) != 0)
            continue;

        printf("Pattern %d/%d: %s (target: %.2fx %s)\n",
               p + 1, N_PATTERNS, patterns[p].desc,
               best_ratios[p], best_cfgs[p]);

        float *data = (float *)malloc(DATASET_BYTES);
        if (!data) { perror("malloc"); return 1; }
        fill_dataset(data, patterns[p].fill_id);

        int combo = 0, total = N_LR * N_MAPE;
        for (int li = 0; li < N_LR; li++) {
            for (int mi = 0; mi < N_MAPE; mi++) {
                combo++;
                if (g_nresults >= MAX_RESULTS) break;

                float lr   = LR_VALUES[li];
                float mape = MAPE_VALUES[mi];

                printf("  [%2d/%d] lr=%.1f mape=%.2f ... ",
                       combo, total, lr, mape);
                fflush(stdout);

                /* Fresh weights: cleanup + init + re-register */
                gpucompress_cleanup();
                rc = gpucompress_init(weights_path);
                if (rc != GPUCOMPRESS_SUCCESS) {
                    printf("FAILED (init: %s)\n", gpucompress_error_string(rc));
                    continue;
                }
                if (H5Z_gpucompress_register() < 0) {
                    printf("FAILED (register)\n");
                    continue;
                }

                /* Configure: reinforcement only, no exploration */
                gpucompress_enable_online_learning();
                gpucompress_set_reinforcement(1, lr, mape, 0.0f);
                gpucompress_set_exploration(0);

                sweep_result_t *r = &g_results[g_nresults];
                /* Pre-fill pattern/lr/mape so run_nn can write chunk CSV */
                snprintf(r->pattern, sizeof(r->pattern), "%s", patterns[p].name);
                r->lr       = lr;
                r->mape_thr = mape;

                if (run_nn(data, DATASET_FLOATS, best_ratios[p], r) == 0) {
                    snprintf(r->mode, sizeof(r->mode), "nn");

                    char mark = (r->ratio >= best_ratios[p] * 0.99) ? '*' : ' ';
                    printf("%.2fx%c conv@%d SGD=%d (%s)\n",
                           r->ratio, mark, r->converged_chunk,
                           r->sgd_fired_count, r->algorithm);
                    g_nresults++;
                } else {
                    printf("FAILED\n");
                }
            }
        }

        print_grid(patterns[p].name, best_ratios[p], best_cfgs[p]);
        free(data);
    }

    /* Close chunk CSV */
    if (g_chunk_csv) {
        fclose(g_chunk_csv);
        g_chunk_csv = NULL;
        printf("Chunk CSV written: %s\n", chunk_csv_path);
    }

    /* Aggregate CSV */
    write_csv(csv_path);

    gpucompress_cleanup();
    printf("Done. %d results recorded.\n", g_nresults);
    return 0;
}
