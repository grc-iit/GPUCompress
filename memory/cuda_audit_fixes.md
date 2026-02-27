# CUDA Audit Fix Plan — Key Decisions & Issues Found

## Bug Found in Original Fix Plan
- **F2 (stats workspace pre-alloc)**: Original plan omitted the `h_init` H→D copy after `cudaMemsetAsync`. Without it, `vmin` would be 0.0 instead of `FLT_MAX`, corrupting atomicMin results. **Fixed in updated plan.**

## Omissions Fixed
- **F12**: Decompression timing events `dt0/dt1` (lines 750-801) were not addressed. Now explicitly covered — reuse `g_t_start/g_t_stop` sequentially.
- **F2**: `AutoStatsGPU` was defined locally in `stats_kernel.cu` but needed by `nn_gpu.cu`. Fixed by extracting to `src/stats/auto_stats_gpu.h`.

## F9 Decision: Keep NN Inference on GPU (not CPU)
User explicitly requested GPU-only path. Three optimizations applied:
1. **Parallelize kernel**: `<<<1,32>>>` → `<<<32,128>>>` (one block per config, 128 threads = hidden dim)
2. **Consolidate D→H**: 4 separate copies → 1 packed `NNConfigOutput[32]` (384B)
3. **Fuse stats→NN**: NN kernel reads `d_stats` directly (no host round-trip). Each block normalizes MAD/deriv internally (6 ops, replaces `finalizeStatsOnlyKernel`).

New combined function: `runStatsAndNNInference()` — enqueues stats kernels + NN kernel + 2 D→H copies + 1 sync.

Host stats copy kept for OOD detection, experience logging, and stats output struct.

## Execution Order Change
F2 moved before F10/F9 because pre-allocated workspace and shared `AutoStatsGPU` header are prerequisites for both.

## Files
- Audit: `CUDA_MEMCPY_AUDIT.md`
- Fix Plan: `CUDA_MEMCPY_FIX_PLAN.md`
