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
    double sum_abs_err;   // sum of |original[i] - decompressed[i]|
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
    double t_sae = 0.0;
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
        t_sae += (double)abs_err;
        if (abs_err > t_max_err) t_max_err = abs_err;
        if (a < t_min) t_min = a;
        if (a > t_max) t_max = a;
    }

    // Warp-level reduction (warp size = 32)
    for (int offset = 16; offset > 0; offset >>= 1) {
        t_sse += __shfl_down_sync(0xFFFFFFFF, t_sse, offset);
        t_sae += __shfl_down_sync(0xFFFFFFFF, t_sae, offset);
        float other_min = __shfl_down_sync(0xFFFFFFFF, t_min, offset);
        float other_max = __shfl_down_sync(0xFFFFFFFF, t_max, offset);
        float other_err = __shfl_down_sync(0xFFFFFFFF, t_max_err, offset);
        if (other_min < t_min) t_min = other_min;
        if (other_max > t_max) t_max = other_max;
        if (other_err > t_max_err) t_max_err = other_err;
    }

    // Shared memory for inter-warp reduction
    __shared__ double s_sse[PSNR_BLOCK / 32];
    __shared__ double s_sae[PSNR_BLOCK / 32];
    __shared__ float  s_min[PSNR_BLOCK / 32];
    __shared__ float  s_max[PSNR_BLOCK / 32];
    __shared__ float  s_err[PSNR_BLOCK / 32];

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    if (lane == 0) {
        s_sse[warp] = t_sse;
        s_sae[warp] = t_sae;
        s_min[warp] = t_min;
        s_max[warp] = t_max;
        s_err[warp] = t_max_err;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp == 0) {
        int n_warps = PSNR_BLOCK / 32;
        t_sse = (lane < n_warps) ? s_sse[lane] : 0.0;
        t_sae = (lane < n_warps) ? s_sae[lane] : 0.0;
        t_min = (lane < n_warps) ? s_min[lane] : FLT_MAX;
        t_max = (lane < n_warps) ? s_max[lane] : -FLT_MAX;
        t_max_err = (lane < n_warps) ? s_err[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            t_sse += __shfl_down_sync(0xFFFFFFFF, t_sse, offset);
            t_sae += __shfl_down_sync(0xFFFFFFFF, t_sae, offset);
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
            atomicAdd(&result->sum_abs_err, t_sae);
            atomicMinFloat(&result->vmin, t_min);
            atomicMaxFloat(&result->vmax, t_max);
            atomicMaxFloat(&result->max_abs_err, t_max_err);
        }
    }
}

/* ============================================================
 * Global SSIM reduction kernel
 *
 * Computes global (single-window) SSIM over the entire buffer.
 * Five accumulators: sum_x, sum_y, sum_xx, sum_yy, sum_xy.
 * SSIM formula evaluated on host from these 5 scalars.
 * ============================================================ */

struct SsimReduction {
    double sum_x;    // sum of original values
    double sum_y;    // sum of decompressed values
    double sum_xx;   // sum of original^2
    double sum_yy;   // sum of decompressed^2
    double sum_xy;   // sum of original * decompressed
};

__global__ void ssimReductionKernel(
    const float* __restrict__ original,
    const float* __restrict__ decompressed,
    size_t n,
    SsimReduction* __restrict__ result)
{
    double t_sx = 0.0, t_sy = 0.0;
    double t_sxx = 0.0, t_syy = 0.0, t_sxy = 0.0;

    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < n; i += stride) {
        double x = (double)original[i];
        double y = (double)decompressed[i];
        t_sx  += x;
        t_sy  += y;
        t_sxx += x * x;
        t_syy += y * y;
        t_sxy += x * y;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        t_sx  += __shfl_down_sync(0xFFFFFFFF, t_sx,  offset);
        t_sy  += __shfl_down_sync(0xFFFFFFFF, t_sy,  offset);
        t_sxx += __shfl_down_sync(0xFFFFFFFF, t_sxx, offset);
        t_syy += __shfl_down_sync(0xFFFFFFFF, t_syy, offset);
        t_sxy += __shfl_down_sync(0xFFFFFFFF, t_sxy, offset);
    }

    __shared__ double s_sx[PSNR_BLOCK / 32];
    __shared__ double s_sy[PSNR_BLOCK / 32];
    __shared__ double s_sxx[PSNR_BLOCK / 32];
    __shared__ double s_syy[PSNR_BLOCK / 32];
    __shared__ double s_sxy[PSNR_BLOCK / 32];

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    if (lane == 0) {
        s_sx[warp]  = t_sx;
        s_sy[warp]  = t_sy;
        s_sxx[warp] = t_sxx;
        s_syy[warp] = t_syy;
        s_sxy[warp] = t_sxy;
    }
    __syncthreads();

    if (warp == 0) {
        int n_warps = PSNR_BLOCK / 32;
        t_sx  = (lane < n_warps) ? s_sx[lane]  : 0.0;
        t_sy  = (lane < n_warps) ? s_sy[lane]  : 0.0;
        t_sxx = (lane < n_warps) ? s_sxx[lane] : 0.0;
        t_syy = (lane < n_warps) ? s_syy[lane] : 0.0;
        t_sxy = (lane < n_warps) ? s_sxy[lane] : 0.0;

        for (int offset = 16; offset > 0; offset >>= 1) {
            t_sx  += __shfl_down_sync(0xFFFFFFFF, t_sx,  offset);
            t_sy  += __shfl_down_sync(0xFFFFFFFF, t_sy,  offset);
            t_sxx += __shfl_down_sync(0xFFFFFFFF, t_sxx, offset);
            t_syy += __shfl_down_sync(0xFFFFFFFF, t_syy, offset);
            t_sxy += __shfl_down_sync(0xFFFFFFFF, t_sxy, offset);
        }

        if (lane == 0) {
            atomicAdd(&result->sum_x,  t_sx);
            atomicAdd(&result->sum_y,  t_sy);
            atomicAdd(&result->sum_xx, t_sxx);
            atomicAdd(&result->sum_yy, t_syy);
            atomicAdd(&result->sum_xy, t_sxy);
        }
    }
}

/* ============================================================
 * Host wrapper: compute PSNR, RMSE, max/mean point-wise error, SSIM
 * ============================================================ */

/* Forward declaration */
extern "C" double vpic_compute_quality_gpu(
    const float* d_original, const float* d_decompressed, size_t n_floats,
    double* out_rmse, double* out_max_err, double* out_mean_err, double* out_ssim);

extern "C" double vpic_compute_psnr_gpu(
    const float* d_original,
    const float* d_decompressed,
    size_t n_floats)
{
    /* Backward-compatible: returns PSNR only. */
    double rmse_out, max_err_out;
    return vpic_compute_quality_gpu(d_original, d_decompressed, n_floats,
                                    &rmse_out, &max_err_out, nullptr, nullptr);
}

extern "C" double vpic_compute_quality_gpu(
    const float* d_original,
    const float* d_decompressed,
    size_t n_floats,
    double* out_rmse,
    double* out_max_err,
    double* out_mean_err,
    double* out_ssim)
{
    if (out_rmse) *out_rmse = 0.0;
    if (out_max_err) *out_max_err = 0.0;
    if (out_mean_err) *out_mean_err = 0.0;
    if (out_ssim) *out_ssim = 1.0;

    if (!d_original || !d_decompressed || n_floats == 0)
        return INFINITY;

    // Allocate results on device (once, reused — single-threaded benchmark)
    static PsnrReduction* d_result = nullptr;
    static SsimReduction* d_ssim_result = nullptr;
    if (!d_result) {
        cudaError_t err = cudaMalloc(&d_result, sizeof(PsnrReduction));
        if (err != cudaSuccess) {
            fprintf(stderr, "vpic_psnr: cudaMalloc failed: %s\n", cudaGetErrorString(err));
            return -1.0;
        }
    }
    if (!d_ssim_result && out_ssim) {
        cudaError_t serr = cudaMalloc(&d_ssim_result, sizeof(SsimReduction));
        if (serr != cudaSuccess) {
            fprintf(stderr, "vpic_psnr: cudaMalloc(SSIM) failed: %s\n",
                    cudaGetErrorString(serr));
        }
    }

    // Initialize and launch PSNR reduction
    PsnrReduction init = {0.0, 0.0, FLT_MAX, -FLT_MAX, 0.0f};
    cudaMemcpy(d_result, &init, sizeof(PsnrReduction), cudaMemcpyHostToDevice);

    int n_blocks = (int)((n_floats + PSNR_BLOCK - 1) / PSNR_BLOCK);
    if (n_blocks > PSNR_MAX_BLOCKS) n_blocks = PSNR_MAX_BLOCKS;
    psnrReductionKernel<<<n_blocks, PSNR_BLOCK>>>(
        d_original, d_decompressed, n_floats, d_result);

    // Launch SSIM reduction (serialized after PSNR on default stream)
    if (out_ssim && d_ssim_result) {
        SsimReduction ssim_init = {0.0, 0.0, 0.0, 0.0, 0.0};
        cudaMemcpy(d_ssim_result, &ssim_init, sizeof(SsimReduction), cudaMemcpyHostToDevice);
        ssimReductionKernel<<<n_blocks, PSNR_BLOCK>>>(
            d_original, d_decompressed, n_floats, d_ssim_result);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "vpic_psnr: kernel failed: %s\n", cudaGetErrorString(err));
        return -1.0;
    }

    // Read PSNR result
    PsnrReduction h_result;
    err = cudaMemcpy(&h_result, d_result, sizeof(PsnrReduction), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "vpic_psnr: PSNR memcpy failed: %s\n", cudaGetErrorString(err));
        return -1.0;
    }

    // Compute metrics
    double mse = h_result.sum_sq_diff / (double)n_floats;
    double data_range = (double)h_result.vmax - (double)h_result.vmin;
    double rmse = sqrt(mse);

    if (out_rmse) *out_rmse = rmse;
    if (out_max_err) *out_max_err = (double)h_result.max_abs_err;
    if (out_mean_err) *out_mean_err = h_result.sum_abs_err / (double)n_floats;

    // Compute SSIM from 5 accumulators
    if (out_ssim && d_ssim_result) {
        SsimReduction h_ssim;
        err = cudaMemcpy(&h_ssim, d_ssim_result, sizeof(SsimReduction), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            fprintf(stderr, "vpic_psnr: SSIM memcpy failed: %s\n", cudaGetErrorString(err));
            *out_ssim = -1.0;
        } else {
            double N = (double)n_floats;
            double mu_x  = h_ssim.sum_x / N;
            double mu_y  = h_ssim.sum_y / N;
            double var_x = h_ssim.sum_xx / N - mu_x * mu_x;
            double var_y = h_ssim.sum_yy / N - mu_y * mu_y;
            double cov   = h_ssim.sum_xy / N - mu_x * mu_y;
            /* Stabilization constants: K1=0.01, K2=0.03, L=data_range */
            double L  = data_range > 0.0 ? data_range : 1.0;
            double C1 = (0.01 * L) * (0.01 * L);
            double C2 = (0.03 * L) * (0.03 * L);
            *out_ssim = ((2.0 * mu_x * mu_y + C1) * (2.0 * cov + C2)) /
                        ((mu_x * mu_x + mu_y * mu_y + C1) * (var_x + var_y + C2));
        }
    }

    if (mse == 0.0 || data_range == 0.0)
        return INFINITY;  // Perfect reconstruction or constant data

    /* PSNR using R = data_range = max(original) - min(original).
     * HPC/scientific data convention (SZ, ZFP, etc.) */
    return 10.0 * log10((data_range * data_range) / mse);
}
