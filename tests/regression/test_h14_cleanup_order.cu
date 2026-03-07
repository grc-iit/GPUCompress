/**
 * test_h14_cleanup_order.cu
 *
 * H-14: gpucompress_cleanup() called before cudaFree() on user buffers.
 *       Verify that cudaFree works correctly after library cleanup, since
 *       cleanup does NOT call cudaDeviceReset().
 *
 * Test strategy:
 *   1. gpucompress_init()
 *   2. cudaMalloc user buffers
 *   3. Do a compression operation (optional, ensures library is "warm")
 *   4. gpucompress_cleanup()
 *   5. cudaFree user buffers — should succeed (cudaSuccess)
 *   6. Verify CUDA context is still functional after cleanup
 *   7. Also test the correct order (cudaFree before cleanup) for comparison
 *
 * Run: ./test_h14_cleanup_order
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpucompress.h"

static int g_pass = 0;
static int g_fail = 0;
#define PASS(msg) do { printf("  PASS: %s\n", msg); g_pass++; } while(0)
#define FAIL(msg) do { printf("  FAIL: %s\n", msg); g_fail++; } while(0)

#define BUF_SIZE (1024 * 1024)  /* 1 MB */

int main(void) {
    printf("=== H-14: cudaFree after gpucompress_cleanup() ===\n\n");

    /* ---- Test 1: cudaFree AFTER cleanup (the H-14 pattern) ---- */
    printf("--- Test 1: cudaFree after gpucompress_cleanup ---\n");
    {
        gpucompress_error_t err = gpucompress_init(NULL);
        if (err != GPUCOMPRESS_SUCCESS) {
            printf("  SKIP: gpucompress_init failed (%d)\n", err);
            return 1;
        }

        /* Allocate user buffers */
        void *d_buf1 = NULL, *d_buf2 = NULL;
        cudaError_t ce1 = cudaMalloc(&d_buf1, BUF_SIZE);
        cudaError_t ce2 = cudaMalloc(&d_buf2, BUF_SIZE);

        if (ce1 != cudaSuccess || ce2 != cudaSuccess) {
            printf("  SKIP: cudaMalloc failed\n");
            if (d_buf1) cudaFree(d_buf1);
            if (d_buf2) cudaFree(d_buf2);
            gpucompress_cleanup();
            return 1;
        }
        PASS("cudaMalloc succeeded for user buffers");

        /* Cleanup library first (H-14 pattern) */
        gpucompress_cleanup();
        PASS("gpucompress_cleanup() returned");

        /* Now try cudaFree */
        cudaError_t f1 = cudaFree(d_buf1);
        cudaError_t f2 = cudaFree(d_buf2);

        if (f1 == cudaSuccess && f2 == cudaSuccess) {
            PASS("cudaFree succeeded after cleanup (CUDA context alive)");
        } else {
            printf("  FAIL: cudaFree returned %d, %d\n", (int)f1, (int)f2);
            g_fail++;
        }
    }

    /* ---- Test 2: CUDA context still functional after cleanup ---- */
    printf("\n--- Test 2: CUDA context functional after cleanup ---\n");
    {
        /* Try a fresh cudaMalloc + cudaFree after library cleanup */
        void *d_test = NULL;
        cudaError_t ce = cudaMalloc(&d_test, 1024);
        if (ce == cudaSuccess) {
            PASS("cudaMalloc works after gpucompress_cleanup");
            cudaFree(d_test);
        } else {
            FAIL("cudaMalloc failed after cleanup (CUDA context may be dead)");
        }

        /* Try cudaMemset */
        ce = cudaMalloc(&d_test, 1024);
        if (ce == cudaSuccess) {
            ce = cudaMemset(d_test, 0, 1024);
            if (ce == cudaSuccess) {
                PASS("cudaMemset works after cleanup");
            } else {
                FAIL("cudaMemset failed after cleanup");
            }
            cudaFree(d_test);
        }
    }

    /* ---- Test 3: Correct order (cudaFree before cleanup) ---- */
    printf("\n--- Test 3: Correct order (cudaFree before cleanup) ---\n");
    {
        gpucompress_error_t err = gpucompress_init(NULL);
        if (err != GPUCOMPRESS_SUCCESS) {
            printf("  SKIP: gpucompress_init failed\n");
        } else {
            void *d_buf = NULL;
            cudaMalloc(&d_buf, BUF_SIZE);

            cudaError_t f = cudaFree(d_buf);
            if (f == cudaSuccess) {
                PASS("cudaFree before cleanup succeeded");
            } else {
                FAIL("cudaFree before cleanup failed");
            }

            gpucompress_cleanup();
            PASS("gpucompress_cleanup after cudaFree succeeded");
        }
    }

    /* ---- Test 4: Re-init after cleanup ---- */
    printf("\n--- Test 4: Re-init after cleanup ---\n");
    {
        gpucompress_error_t err = gpucompress_init(NULL);
        if (err == GPUCOMPRESS_SUCCESS) {
            PASS("gpucompress_init succeeds after prior cleanup");

            /* Verify library is functional */
            if (gpucompress_is_initialized()) {
                PASS("library reports initialized after re-init");
            } else {
                FAIL("library reports not initialized after re-init");
            }

            gpucompress_cleanup();
        } else {
            FAIL("gpucompress_init failed after prior cleanup");
        }
    }

    /* ---- Summary ---- */
    printf("\n=== Summary: %d pass, %d fail ===\n", g_pass, g_fail);
    printf("%s\n", g_fail == 0 ? "OVERALL: PASS" : "OVERALL: FAIL");
    return g_fail == 0 ? 0 : 1;
}
