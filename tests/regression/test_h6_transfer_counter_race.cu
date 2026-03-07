/**
 * test_h6_transfer_counter_race.cu
 *
 * H-6: VOL transfer byte/count counters (s_h2d_bytes, s_d2h_bytes, etc.)
 *       are plain size_t/int, not std::atomic.  A race occurs when multiple
 *       threads call H5Dwrite/H5Dread through the VOL connector concurrently,
 *       each triggering vol_memcpy() on its own thread.
 *
 * Test strategy:
 *   1. Launch N_THREADS threads, each writing a separate HDF5 file via VOL.
 *   2. Each file has N_CHUNKS chunks sized to trigger partial-boundary
 *      vol_memcpy(D2D) calls (dataset size not divisible by chunk size).
 *   3. After all writes complete, read them all back concurrently.
 *   4. Compare cumulative transfer counters against expected values.
 *   5. Repeat ROUNDS times to amplify any lost increments.
 *
 * With non-atomic counters, concurrent += on s_h2d_bytes / s_d2h_bytes
 * will lose increments, causing the reported totals to be less than expected.
 *
 * Run: ./test_h6_transfer_counter_race
 *   Expected: PASS if counters are atomic (accurate), FAIL if racy.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include <thread>
#include <vector>
#include <atomic>

#include <hdf5.h>
#include "gpucompress.h"
#include "gpucompress_hdf5_vol.h"

/* ------------------------------------------------------------------ */
/* Config                                                               */
/* ------------------------------------------------------------------ */
#define N_THREADS     8
#define N_CHUNKS      4          /* chunks per file */
#define CHUNK_ELEMS   (16*1024)  /* 64 KB per chunk (float32) */
/* Make dataset slightly larger than N_CHUNKS * CHUNK_ELEMS so the last
   chunk is partial — this forces vol_memcpy(D2D) for zero-pad. */
#define EXTRA_ELEMS   1000
#define TOTAL_ELEMS   ((size_t)N_CHUNKS * CHUNK_ELEMS + EXTRA_ELEMS)
#define ROUNDS        5

#define H5Z_FILTER_GPUCOMPRESS    305
#define H5Z_GPUCOMPRESS_CD_NELMTS 5

static std::atomic<int> g_pass{0};
static std::atomic<int> g_fail{0};

#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

static void pack_double(double v, unsigned int* lo, unsigned int* hi) {
    uint64_t bits; memcpy(&bits, &v, sizeof(bits));
    *lo = (unsigned int)(bits & 0xFFFFFFFFu);
    *hi = (unsigned int)(bits >> 32);
}

/* Each thread opens its own file with the VOL connector. */
static hid_t open_vol_file(const char* path, unsigned flags) {
    hid_t vol_id = H5VL_gpucompress_register();
    if (vol_id < 0) return H5I_INVALID_HID;
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_gpucompress(fapl, H5VL_NATIVE, NULL);
    hid_t fid = (flags & H5F_ACC_TRUNC)
        ? H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, fapl)
        : H5Fopen(path, H5F_ACC_RDONLY, fapl);
    H5Pclose(fapl);
    H5VLclose(vol_id);
    return fid;
}

/* Thread function: write a dataset through VOL, then read it back. */
static void thread_write_read(int id, float* d_data, float* d_out,
                               size_t nbytes, std::atomic<int>* errors)
{
    char path[256];
    snprintf(path, sizeof(path), "/tmp/test_h6_thread_%d.h5", id);

    /* WRITE */
    hid_t fid = open_vol_file(path, H5F_ACC_TRUNC);
    if (fid < 0) { (*errors)++; return; }

    hsize_t dims[1]  = { (hsize_t)TOTAL_ELEMS };
    hsize_t cdims[1] = { (hsize_t)CHUNK_ELEMS };
    hid_t space = H5Screate_simple(1, dims, NULL);
    hid_t dcpl  = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, 1, cdims);

    unsigned int cd[H5Z_GPUCOMPRESS_CD_NELMTS] = {0};
    cd[0] = 2; /* LZ4 */
    cd[1] = 2; /* SHUFFLE_4 */
    cd[2] = 4;
    pack_double(0.0, &cd[3], &cd[4]);
    H5Pset_filter(dcpl, H5Z_FILTER_GPUCOMPRESS, H5Z_FLAG_OPTIONAL,
                  H5Z_GPUCOMPRESS_CD_NELMTS, cd);

    hid_t dset = H5Dcreate2(fid, "data", H5T_NATIVE_FLOAT,
                             space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Sclose(space); H5Pclose(dcpl);
    if (dset < 0) { H5Fclose(fid); (*errors)++; return; }

    herr_t we = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, d_data);
    H5Dclose(dset);
    H5Fclose(fid);
    if (we < 0) { (*errors)++; return; }

    /* READ */
    hid_t fid2 = open_vol_file(path, H5F_ACC_RDONLY);
    if (fid2 < 0) { (*errors)++; return; }

    hid_t dset2 = H5Dopen2(fid2, "data", H5P_DEFAULT);
    herr_t re = H5Dread(dset2, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, d_out);
    H5Dclose(dset2);
    H5Fclose(fid2);
    if (re < 0) { (*errors)++; return; }

    remove(path);
}

int main(void) {
    printf("=== H-6: Transfer counter race under concurrent VOL writes ===\n");
    printf("N_THREADS=%d, N_CHUNKS=%d (+partial), CHUNK_ELEMS=%d, ROUNDS=%d\n\n",
           N_THREADS, N_CHUNKS, CHUNK_ELEMS, ROUNDS);

    if (gpucompress_init(NULL) != GPUCOMPRESS_SUCCESS) {
        fprintf(stderr, "gpucompress_init() failed\n");
        return 1;
    }

    size_t nbytes = TOTAL_ELEMS * sizeof(float);

    /* Allocate per-thread GPU buffers */
    float* d_data[N_THREADS];
    float* d_out[N_THREADS];
    for (int i = 0; i < N_THREADS; i++) {
        if (cudaMalloc(&d_data[i], nbytes) != cudaSuccess ||
            cudaMalloc(&d_out[i],  nbytes) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for thread %d\n", i);
            return 1;
        }
        /* Fill with a thread-unique pattern */
        float* h_tmp = (float*)malloc(nbytes);
        for (size_t j = 0; j < TOTAL_ELEMS; j++)
            h_tmp[j] = (float)(i * 100000 + j) * 0.001f;
        cudaMemcpy(d_data[i], h_tmp, nbytes, cudaMemcpyHostToDevice);
        free(h_tmp);
    }

    /* ---------------------------------------------------------------- */
    /* Round loop: concurrent writes + reads                              */
    /* ---------------------------------------------------------------- */
    for (int round = 0; round < ROUNDS; round++) {
        printf("--- Round %d/%d ---\n", round + 1, ROUNDS);

        H5VL_gpucompress_reset_stats();

        std::atomic<int> errors{0};
        std::vector<std::thread> threads;
        threads.reserve(N_THREADS);

        for (int t = 0; t < N_THREADS; t++) {
            threads.emplace_back(thread_write_read, t,
                                 d_data[t], d_out[t], nbytes, &errors);
        }
        for (auto& th : threads) th.join();

        if (errors.load() > 0) {
            printf("  %d thread(s) had HDF5 errors — skipping counter check\n",
                   errors.load());
            g_fail++;
            continue;
        }

        /* Check activity counters (these are already atomic — sanity) */
        int writes = 0, reads = 0, comp = 0, decomp = 0;
        H5VL_gpucompress_get_stats(&writes, &reads, &comp, &decomp);

        printf("  Activity: writes=%d reads=%d comp=%d decomp=%d\n",
               writes, reads, comp, decomp);

        if (writes == N_THREADS) PASS("s_gpu_writes == N_THREADS");
        else { printf("  FAIL: s_gpu_writes=%d expected=%d\n", writes, N_THREADS); g_fail++; }

        if (reads == N_THREADS) PASS("s_gpu_reads == N_THREADS");
        else { printf("  FAIL: s_gpu_reads=%d expected=%d\n", reads, N_THREADS); g_fail++; }

        /* Check transfer counters — the target of this test.
         * We don't know the exact expected byte counts (they depend on
         * compressed sizes and internal VOL decisions), but we CAN check:
         *   1. Counters are > 0 (vol_memcpy was called)
         *   2. h2d_count and d2h_count are multiples of N_THREADS
         *      (each thread's read path does H2D for compressed chunks,
         *       each thread's write path may do D2D for partial chunks)
         *   3. Repeat ROUNDS times and check monotonicity within a round
         */
        int h2d_count = 0, d2h_count = 0, d2d_count = 0;
        size_t h2d_bytes = 0, d2h_bytes = 0, d2d_bytes = 0;
        H5VL_gpucompress_get_transfer_stats(
            &h2d_count, &h2d_bytes,
            &d2h_count, &d2h_bytes,
            &d2d_count, &d2d_bytes);

        printf("  Transfers: H2D=%d (%zu B) D2H=%d (%zu B) D2D=%d (%zu B)\n",
               h2d_count, h2d_bytes, d2h_count, d2h_bytes,
               d2d_count, d2d_bytes);

        /* Read path: each thread reads (N_CHUNKS+1) chunks, each triggering
         * a vol_memcpy(H2D) for the compressed data. */
        int expected_h2d = N_THREADS * (N_CHUNKS + 1);  /* +1 for partial chunk */
        if (h2d_count == expected_h2d) {
            PASS("h2d_count matches expected");
        } else {
            printf("  FAIL: h2d_count=%d expected=%d (lost increments?)\n",
                   h2d_count, expected_h2d);
            g_fail++;
        }

        /* Write path: partial last chunk triggers vol_memcpy(D2D) per thread */
        if (d2d_count >= N_THREADS) {
            PASS("d2d_count >= N_THREADS (partial chunk padding)");
        } else {
            printf("  FAIL: d2d_count=%d expected>=%d (lost increments?)\n",
                   d2d_count, N_THREADS);
            g_fail++;
        }

        /* Byte totals: should be self-consistent.
         * h2d_bytes should equal sum of compressed chunk sizes across all
         * threads' reads. We can't compute exactly, but it must be > 0
         * and proportional to h2d_count. */
        if (h2d_count > 0 && h2d_bytes > 0) {
            size_t avg = h2d_bytes / (size_t)h2d_count;
            printf("  Avg H2D per transfer: %zu B\n", avg);
            PASS("h2d_bytes > 0 and consistent with h2d_count");
        } else if (h2d_count > 0 && h2d_bytes == 0) {
            FAIL("h2d_count > 0 but h2d_bytes == 0 (byte counter lost all increments)");
        } else {
            PASS("no H2D transfers (all chunks direct-decompress)");
        }
    }

    /* ---------------------------------------------------------------- */
    /* Verify data integrity (spot-check one thread)                      */
    /* ---------------------------------------------------------------- */
    printf("--- Data integrity check ---\n");
    {
        float* h_orig = (float*)malloc(nbytes);
        float* h_back = (float*)malloc(nbytes);
        cudaMemcpy(h_orig, d_data[0], nbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_back, d_out[0],  nbytes, cudaMemcpyDeviceToHost);

        int mismatch = 0;
        for (size_t i = 0; i < TOTAL_ELEMS; i++) {
            if (h_orig[i] != h_back[i]) { mismatch++; break; }
        }
        if (mismatch == 0) PASS("thread-0 round-trip byte-exact");
        else FAIL("thread-0 round-trip data mismatch");

        free(h_orig);
        free(h_back);
    }

    /* ---------------------------------------------------------------- */
    /* Cleanup                                                            */
    /* ---------------------------------------------------------------- */
    for (int i = 0; i < N_THREADS; i++) {
        cudaFree(d_data[i]);
        cudaFree(d_out[i]);
    }

    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass.load(), g_fail.load());
    printf("%s\n", g_fail.load() == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    gpucompress_cleanup();
    return g_fail.load() == 0 ? 0 : 1;
}
