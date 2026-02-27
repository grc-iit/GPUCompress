/**
 * @file auto_stats_gpu.h
 * @brief Shared definition of AutoStatsGPU — device-side statistics buffer.
 *
 * Extracted from stats_kernel.cu so that nn_gpu.cu can reference the same
 * struct for the fused stats→NN inference pipeline.
 */

#ifndef AUTO_STATS_GPU_H
#define AUTO_STATS_GPU_H

#include <cstddef>

struct __align__(8) AutoStatsGPU {
    // Pass 1 outputs
    double sum;              // sum of all elements (for mean)
    double abs_diff_sum;     // sum of |x[i+1] - 2*x[i] + x[i-1]|
    float  vmin;             // data min
    float  vmax;             // data max
    size_t num_elements;

    // Pass 2 output
    double mad_sum;          // sum of |x[i] - mean|

    // Entropy pipeline output
    double entropy;

    // Finalize outputs (contiguous for single D->H copy)
    double mad_normalized;
    double deriv_normalized;
    int    state;
    int    action;

    // Error level (input, set from host)
    int    error_level;
};

#endif // AUTO_STATS_GPU_H
