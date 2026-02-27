/**
 * @file nn_reinforce.cpp
 * @brief Online NN reinforcement — STUBBED OUT (F9: GPU SGD replaces CPU path).
 *
 * All functions are no-ops. Kept for backward compatibility.
 * GPU-native SGD is now in nn_gpu.cu (nnSGDKernel + runNNSGD).
 */

#include "nn/nn_reinforce.h"
#include <cstddef>

extern "C" int nn_reinforce_init(const void* /*d_weights*/) {
    return 0;  // no-op
}

extern "C" void nn_reinforce_add_sample(const float /*input_raw*/[15],
                                         double /*actual_ratio*/,
                                         double /*actual_comp_time*/,
                                         double /*actual_decomp_time*/,
                                         double /*actual_psnr*/) {
    // no-op
}

extern "C" int nn_reinforce_apply(void* /*d_weights*/, float /*learning_rate*/) {
    return 0;  // no-op
}

extern "C" void nn_reinforce_get_last_stats(float* grad_norm, int* num_samples,
                                              int* was_clipped) {
    if (grad_norm)   *grad_norm   = 0.0f;
    if (num_samples) *num_samples = 0;
    if (was_clipped) *was_clipped = 0;
}

extern "C" void nn_reinforce_cleanup(void) {
    // no-op
}
