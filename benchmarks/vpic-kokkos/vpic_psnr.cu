/**
 * @file vpic_psnr.cu
 * @brief GPU PSNR computation for the VPIC benchmark.
 *
 * Single-pass reduction kernel computes MSE and data range between
 * original and decompressed GPU buffers. Called after read-back,
 * outside the timed section.
 *
 * Pattern follows statsPass1Kernel in src/stats/stats_kernel.cu.
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>

/* ============================================================
 * Device helpers: CAS-based atomicMin/Max for float
 * (same implementation as stats_kernel.cu)
 * ============================================================ */

__device__ static void atomicMinFloat(float* address, float val) {
    float old = *address;
    while (val < old) {
        unsigned int assumed = __float_as_uint(old);
        unsigned int result = atomicCAS((unsigned int*)address, assumed, __float_as_uint(val));
        if (result == assumed) break;
        old = __uint_as_float(result);
    }
}

__device__ static void atomicMaxFloat(float* address, float val) {
    float old = *address;
    while (val > old) {
        unsigned int assumed = __float_as_uint(old);
        unsigned int result = atomicCAS((unsigned int*)address, assumed, __float_as_uint(val));
        if (result == assumed) break;
        old = __uint_as_float(result);
    }
}

/* ============================================================
 * Reduction result (device-side, 24 bytes)
 * ============================================================ */

struct PsnrReduction {
    double sum_sq_diff;   // sum of (original[i] - decompressed[i])^2
    float  vmin;          // min(original)
    float  vmax;          // max(original)
    float  max_abs_err;   // max(|original[i] - decompressed[i]|)
};

/* ============================================================
 * Single-pass PSNR reduction kernel
 * ============================================================ */

static constexpr int PSNR_BLOCK = 256;
static constexpr int PSNR_MAX_BLOCKS = 1024;

__global__ void psnrReductionKernel(
    const float* __restrict__ original,
    const float* __restrict__ decompressed,
    size_t n,
    PsnrReduction* __restrict__ result)
{
    // Per-thread accumulators
    double t_sse = 0.0;
    float  t_min = FLT_MAX;
    float  t_max = -FLT_MAX;
    float  t_max_err = 0.0f;

    // Grid-stride loop
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        float a = original[i];
        float b = decompressed[i];
        double diff = (double)a - (double)b;
        t_sse += diff * diff;
        float abs_err = fabsf(a - b);
        if (abs_err > t_max_err) t_max_err = abs_err;
        if (a < t_min) t_min = a;
        if (a > t_max) t_max = a;
    }

    // Warp-level reduction (warp size = 32)
    for (int offset = 16; offset > 0; offset >>= 1) {
        t_sse += __shfl_down_sync(0xFFFFFFFF, t_sse, offset);
        float other_min = __shfl_down_sync(0xFFFFFFFF, t_min, offset);
        float other_max = __shfl_down_sync(0xFFFFFFFF, t_max, offset);
        float other_err = __shfl_down_sync(0xFFFFFFFF, t_max_err, offset);
        if (other_min < t_min) t_min = other_min;
        if (other_max > t_max) t_max = other_max;
        if (other_err > t_max_err) t_max_err = other_err;
    }

    // Shared memory for inter-warp reduction
    __shared__ double s_sse[PSNR_BLOCK / 32];
    __shared__ float  s_min[PSNR_BLOCK / 32];
    __shared__ float  s_max[PSNR_BLOCK / 32];
    __shared__ float  s_err[PSNR_BLOCK / 32];

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    if (lane == 0) {
        s_sse[warp] = t_sse;
        s_min[warp] = t_min;
        s_max[warp] = t_max;
        s_err[warp] = t_max_err;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp == 0) {
        int n_warps = PSNR_BLOCK / 32;
        t_sse = (lane < n_warps) ? s_sse[lane] : 0.0;
        t_min = (lane < n_warps) ? s_min[lane] : FLT_MAX;
        t_max = (lane < n_warps) ? s_max[lane] : -FLT_MAX;
        t_max_err = (lane < n_warps) ? s_err[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            t_sse += __shfl_down_sync(0xFFFFFFFF, t_sse, offset);
            float other_min = __shfl_down_sync(0xFFFFFFFF, t_min, offset);
            float other_max = __shfl_down_sync(0xFFFFFFFF, t_max, offset);
            float other_err = __shfl_down_sync(0xFFFFFFFF, t_max_err, offset);
            if (other_min < t_min) t_min = other_min;
            if (other_max > t_max) t_max = other_max;
            if (other_err > t_max_err) t_max_err = other_err;
        }

        // Lane 0 atomically contributes to global result
        if (lane == 0) {
            atomicAdd(&result->sum_sq_diff, t_sse);
            atomicMinFloat(&result->vmin, t_min);
            atomicMaxFloat(&result->vmax, t_max);
            atomicMaxFloat(&result->max_abs_err, t_max_err);
        }
    }
}

/* ============================================================
 * Host wrapper: compute PSNR, RMSE, and max point-wise error
 * ============================================================ */

/* Forward declaration */
extern "C" double vpic_compute_quality_gpu(
    const float* d_original, const float* d_decompressed, size_t n_floats,
    double* out_rmse, double* out_max_err);

extern "C" double vpic_compute_psnr_gpu(
    const float* d_original,
    const float* d_decompressed,
    size_t n_floats)
{
    /* Backward-compatible: returns PSNR only. Use vpic_compute_quality_gpu for all metrics. */
    double rmse_out, max_err_out;
    return vpic_compute_quality_gpu(d_original, d_decompressed, n_floats, &rmse_out, &max_err_out);
}

extern "C" double vpic_compute_quality_gpu(
    const float* d_original,
    const float* d_decompressed,
    size_t n_floats,
    double* out_rmse,
    double* out_max_err)
{
    if (out_rmse) *out_rmse = 0.0;
    if (out_max_err) *out_max_err = 0.0;

    if (!d_original || !d_decompressed || n_floats == 0)
        return 120.0;

    // Allocate result on device (once, reused across calls — single-threaded benchmark)
    static PsnrReduction* d_result = nullptr;
    if (!d_result) {
        cudaError_t err = cudaMalloc(&d_result, sizeof(PsnrReduction));
        if (err != cudaSuccess) {
            fprintf(stderr, "vpic_psnr: cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return -1.0;
        }
    }

    // Initialize
    PsnrReduction init = {0.0, FLT_MAX, -FLT_MAX, 0.0f};
    cudaMemcpy(d_result, &init, sizeof(PsnrReduction), cudaMemcpyHostToDevice);

    // Launch kernel
    int n_blocks = (int)((n_floats + PSNR_BLOCK - 1) / PSNR_BLOCK);
    if (n_blocks > PSNR_MAX_BLOCKS) n_blocks = PSNR_MAX_BLOCKS;
    psnrReductionKernel<<<n_blocks, PSNR_BLOCK>>>(
        d_original, d_decompressed, n_floats, d_result);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "vpic_psnr: kernel failed: %s\n", cudaGetErrorString(err));
        return -1.0;
    }

    // Read result back
    PsnrReduction h_result;
    cudaMemcpy(&h_result, d_result, sizeof(PsnrReduction), cudaMemcpyDeviceToHost);

    // Compute metrics
    double mse = h_result.sum_sq_diff / (double)n_floats;
    double data_range = (double)h_result.vmax - (double)h_result.vmin;
    double rmse = sqrt(mse);

    if (out_rmse) *out_rmse = rmse;
    if (out_max_err) *out_max_err = (double)h_result.max_abs_err;

    if (mse == 0.0 || data_range == 0.0)
        return 120.0;  // Perfect reconstruction or constant data

    /* PSNR using R = data_range = max(original) - min(original).
     * This is the HPC/scientific data convention (used by SZ, ZFP, etc.)
     * For image data, R would be 255 (8-bit) or 65535 (16-bit). */
    return 10.0 * log10((data_range * data_range) / mse);
}
