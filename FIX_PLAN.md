# Fix Plan: GPUCompress 20 Confirmed Bugs

## Context

Two rounds of deep codebase investigation + audit identified 22 bugs. After critical review by verification agents, 2 fixes were dropped (L3: filter_mask is input-only in HDF5; M18: benchmark intentionally accumulates learning). The remaining 20 bugs are confirmed with reviewed fixes.

## Execution Order

Fixes grouped by file, ordered by severity. Each fix is self-contained.

---

## Group 1: `src/api/gpucompress_api.cpp` (9 bugs)

### C1 — Exploration winner headers missing setAlgorithmId()
- **Lines ~996, ~2171**: Insert `alt_hdr.setAlgorithmId((uint8_t)alt_algo);` before the header is written (memcpy on host path, cudaMemcpyAsync on GPU path)
- `alt_algo` is in scope at both locations (verified)

### C2 — Global g_t_start/g_t_stop timing events are thread-unsafe
- **Reviewer correction**: Per-call event creation is too expensive (cudaEventCreate is a sync point)
- **Fix**: Protect usage with a new `static std::mutex g_timing_mutex;` declared near line 122. Wrap lines 623-639 and 833-903 (all timing event usage in host compress) with `std::lock_guard<std::mutex> timing_lk(g_timing_mutex);`
- Keep the pre-allocated globals (they're fine when serialized)

### H3 — Host-path ALGO_AUTO uses global singleton stats/inference buffers
- **Reviewer correction**: `g_stats_mutex` doesn't exist — must create it
- **Fix**: Add `static std::mutex g_auto_mutex;` near line 122. Wrap the entire ALGO_AUTO block (lines 457-476: stats computation + inference) with this mutex. This serializes host-path AUTO calls but is correct.
- The GPU path uses per-CompContext buffers and doesn't need this mutex.

### H4 — Host-path SGD missing mutex
- **Line ~1067**: Wrap the `runNNSGD(...)` call with `std::lock_guard<std::mutex> sgd_lk(g_sgd_mutex);`
- `g_sgd_mutex` already exists at line 122

### H7 — Host decompress skips version check
- **Line 1167**: Change `if (header.magic != COMPRESSION_MAGIC)` → `if (!header.isValid())`
- `isValid()` checks both magic AND version range [1, COMPRESSION_HEADER_VERSION]

### H8 — Zero compressed_size bypasses validation
- **After line ~1178**: Add:
  ```cpp
  if (compressed_size == 0) {
      if (header.original_size == 0) { *output_size = 0; return GPUCOMPRESS_SUCCESS; }
      return GPUCOMPRESS_ERROR_INVALID_HEADER;
  }
  ```
- **Reviewer correction**: Must allow legitimate empty-data case (original_size==0)

### M7 — acquireCompContext falls off end of non-void function
- **After line ~242** (for-loop closing brace): Add `return nullptr;`
- The only callsite (line 1659) already checks for nullptr

### M9 — Primary exploration sample records psnr=0.0
- **Reviewer correction**: Don't hardcode 120.0 — use quant_result.psnr when quantization was applied
- **Lines 733-735**: Replace last `0.0` with:
  ```cpp
  double primary_psnr = (d_quantized && quant_result.isValid()) ? quant_result.psnr : 120.0;
  ```
- **Lines 1927-1929**: Same fix for GPU path

### M10 — compress_gpu ignores caller's stream_arg
- **Lines 1661-1662**: Remove `(void)stream_arg;`. After acquiring ctx->stream, add bidirectional event sync using **pre-allocated** events from the CompContext (ctx->t_start can be reused as a sync event):
  ```cpp
  cudaStream_t stream = ctx->stream;
  cudaStream_t caller_stream = stream_arg ? static_cast<cudaStream_t>(stream_arg) : nullptr;
  if (caller_stream) {
      cudaEventRecord(ctx->t_start, caller_stream);
      cudaStreamWaitEvent(stream, ctx->t_start, 0);
  }
  ```
- Before return: record completion on ctx->stream and make caller's stream wait:
  ```cpp
  if (caller_stream) {
      cudaEventRecord(ctx->t_stop, stream);
      cudaStreamWaitEvent(caller_stream, ctx->t_stop, 0);
  }
  ```
- Uses pre-allocated per-context events (no per-call creation overhead)

---

## Group 2: `src/nn/nn_gpu.cu` (3 bugs)

### H1 — Non-Ctx inference paths don't wait for SGD completion
- **In `runNNInference()` (~line 1074)** and **`runNNFusedInference()` (~line 1137)**: Add before kernel launch:
  ```cpp
  if (g_sgd_ever_fired.load(std::memory_order_acquire))
      cudaStreamWaitEvent(stream, g_sgd_done, 0);
  ```
- **In `runNNSGD()` (after line 1222)**: Add after SGD kernel launch:
  ```cpp
  cudaEventRecord(g_sgd_done, stream);
  g_sgd_ever_fired.store(true, std::memory_order_release);
  ```
- **Reviewer note**: Non-Ctx SGD runs on caller's stream (not g_sgd_stream). Recording g_sgd_done on that stream is correct — the event signals "SGD kernel on THIS stream completed." Future inference on any stream will wait for it. Since non-Ctx SGD also calls `cudaStreamSynchronize(stream)` before returning (line 1235), the event is guaranteed recorded before the function exits.

### M6 — Non-atomic global flags
- **Line 62**: `static bool g_nn_loaded` → `static std::atomic<bool> g_nn_loaded{false};`
- **Line 63**: `static NNRankCriterion g_rank_criterion` → `static std::atomic<int> g_rank_criterion{NN_RANK_BY_RATIO};`
- Update all read sites to use `.load()`, all write sites to use `.store()`
- **Reviewer caveat**: Atomics don't protect compound operations (check g_nn_loaded + read d_nn_weights). This is acceptable — the sequence `if (!g_nn_loaded) return -1;` then `use d_nn_weights` is safe because weights are set BEFORE g_nn_loaded is set to true (release ordering).

### M12 — SGD skips PSNR gradient for lossless results
- **Lines 637-645**: Replace the if/else with:
  ```cpp
  {
      float psnr_val = (sample.actual_psnr <= 0.0f) ? 120.0f : sample.actual_psnr;
      float clamped = fminf(psnr_val, 120.0f);
      float y_std3 = weights->y_stds[3];
      if (y_std3 < 1e-8f) y_std3 = 1e-8f;
      s_d3[3] = s_y[3] - (clamped - weights->y_means[3]) / y_std3;
  }
  ```

---

## Group 3: `src/preprocessing/quantization_kernels.cu` (3 bugs)

### C4 — Missing int32 clamping
- **After line ~76** (after int16 clamp): Add:
  ```cpp
  else if (sizeof(OutputT) == 4) {
      quantized = fmax(-2147483648.0, fmin(2147483647.0, quantized));
  }
  ```
- Double literals are used, so float64 precision represents these values exactly

### H6 — int cast overflow in CUB DeviceReduce
- **Lines ~197, 205, 209**: Change `(int)num_elements` → `(int)num_elements` with a preceding guard:
  ```cpp
  if (num_elements > (size_t)INT_MAX) {
      return cudaErrorInvalidValue;  // or handle gracefully
  }
  ```
- **Reviewer note**: CUB int64_t support is version-dependent. A size guard is safer and more portable than casting to int64_t.

### H9 — Forced INT8 silently violates error bound
- **After line ~397** (after precision switch): Add warning:
  ```cpp
  if (precision == 8 && data_range > 0) {
      double max_quant = data_range / (2.0 * config.error_bound);
      if (max_quant > 127.0)
          fprintf(stderr, "gpucompress WARNING: forced INT8 cannot represent data range "
                  "(need %.0f levels, int8 max=127). Error bound may be violated.\n", max_quant);
  }
  ```

---

## Group 4: `src/preprocessing/byte_shuffle_kernels.cu` (2 bugs)

### M15 — element_size parameter ignored, always uses <4>
- **Lines 238-240**: Replace `(void)element_size; byte_shuffle_kernel_specialized<4><<<...>>>` with a switch:
  ```cpp
  switch (element_size) {
      case 1: byte_shuffle_kernel_specialized<1><<<...>>>(...); break;
      case 2: byte_shuffle_kernel_specialized<2><<<...>>>(...); break;
      case 4: byte_shuffle_kernel_specialized<4><<<...>>>(...); break;
      case 8: byte_shuffle_kernel_specialized<8><<<...>>>(...); break;
      default: byte_shuffle_kernel_specialized<4><<<...>>>(...); break;
  }
  ```
- Apply same pattern to `launch_byte_unshuffle` (lines 260-262)
- Add explicit template instantiations at bottom of file for `<1>`, `<2>`, `<8>`

### M16 — Use-after-free in DeviceChunkArrays
- **In `byte_shuffle_simple()` before line 310** (`return device_output`): Add `cudaStreamSynchronize(stream);`
- **In `byte_unshuffle_simple()` before line ~352**: Same fix
- **Reviewer note**: This makes the simple API synchronous, but it already calls `cudaStreamSynchronize` inside `createDeviceChunkArrays`. The API was never truly async.

---

## Group 5: `src/hdf5/H5Zgpucompress.c` (2 bugs)

### L1 — Data race in ensure_initialized()
- Replace lines 141-151 with `pthread_once` pattern:
  ```c
  static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
  static int g_init_result = -1;
  static void do_initialize(void) {
      const char* weights = getenv("GPUCOMPRESS_WEIGHTS");
      if (gpucompress_init(weights) == GPUCOMPRESS_SUCCESS) {
          g_gpucompress_initialized = 1;
          g_init_result = 0;
      }
  }
  static int ensure_initialized(void) {
      pthread_once(&g_init_once, do_initialize);
      return g_init_result;
  }
  ```
- `#include <pthread.h>` already present (line 25 verified)

### L2 — Shuffle size parsing only handles 4
- **Lines 248-250**: Expand to warn on unsupported sizes:
  ```c
  if (cd_nelmts >= 3) {
      if (cd_values[2] == 4) {
          config.preprocessing |= GPUCOMPRESS_PREPROC_SHUFFLE_4;
      } else if (cd_values[2] != 0) {
          fprintf(stderr, "gpucompress HDF5 filter: shuffle element_size=%u not supported "
                  "(only 0 and 4 are supported), ignoring shuffle\n", cd_values[2]);
      }
  }
  ```

---

## Group 6: `benchmarks/grayscott/grayscott-benchmark.cu` (1 bug)

### M17 — gather_chunk_kernel OOB read on last chunk
- **Modify `gather_chunk` signature** to accept `actual_cz`:
  ```cpp
  static void gather_chunk(float *dst, const float *src, int L, int chunk_z, int actual_cz, int chunk_idx) {
      size_t chunk_floats = (size_t)L * L * actual_cz;
      int blocks = (int)((chunk_floats + 255) / 256);
      if (blocks > 2048) blocks = 2048;
      int k0 = chunk_idx * chunk_z;  // offset uses original chunk_z
      gather_chunk_kernel<<<blocks, 256>>>(dst, src, L, actual_cz, k0);
  }
  ```
- **Also fix `scatter_chunk`** with the same pattern (reviewer identified same OOB bug)
- Update all call sites (~lines 383, 482, 535) to pass `actual_cz`

---

## Group 7: Python files (4 bugs)

### H5 — `neural_net/inference/evaluate.py` uses wrong normalization stats
- Pass `checkpoint` to `evaluate_ranking()` and use checkpoint stats:
  ```python
  x_means = checkpoint.get('x_means', data['x_means'])
  x_stds = checkpoint.get('x_stds', data['x_stds'])
  y_means = checkpoint.get('y_means', data['y_means'])
  y_stds = checkpoint.get('y_stds', data['y_stds'])
  ```

### M13 — `neural_net/export/export_weights.py` verify_export compares incompatible values
- Normalize `test_input` before passing to model:
  ```python
  test_input_norm = (test_input - torch.from_numpy(xm)) / torch.from_numpy(np.clip(xs, 1e-8, None))
  expected_output = model(test_input_norm).numpy()[0]
  ```
- Then denormalize: `expected_output = expected_output * ys + ym` to match manual path

### M19 — `neural_net/core/data.py` missing NaN handling
- **Defensive fix**: Before return in `compute_stats_cpu()`:
  ```python
  entropy = 0.0 if np.isnan(entropy) else float(entropy)
  mad = 0.0 if np.isnan(mad) else float(mad)
  second_derivative = 0.0 if np.isnan(second_derivative) else float(second_derivative)
  ```
- Reviewer noted current code may already prevent NaN via `data_range < 1e-30` guard, but defensive guards are cheap

### M20 — `neural_net/training/cross_validate.py` missing PSNR fillna
- **Line 47**: Insert `.fillna(120.0)` after `.replace(...)`:
  ```python
  sub_df['psnr_clamped'] = sub_df['psnr_db'].replace([np.inf, -np.inf], 120.0).fillna(120.0).clip(upper=120.0).astype(np.float32)
  ```

---

## Dropped Fixes (from reviewer feedback)

| Bug | Reason Dropped |
|-----|---------------|
| **L3** (filter_mask not copied back) | HDF5's `chunk_read.filters` is input-only. Copying back is incorrect API usage. |
| **M18** (Phase 5 inherits weights) | Intentional experimental design — Phase 5 tests accumulated learning from Phase 4. |

---

## Verification Plan

1. **Build**: `make` or `cmake --build build` — all 7 modified .cu/.c/.cpp files must compile cleanly
2. **Unit tests**: Run existing test suite (if any) to check for regressions
3. **C1 verification**: Compress with ALGO_AUTO + exploration enabled, decompress the result — must succeed (previously would use wrong algorithm)
4. **M7 verification**: Compiler should no longer warn about "control reaches end of non-void function"
5. **M17 verification**: Run grayscott-benchmark with L not divisible by chunk_z — no CUDA memory errors
6. **Python verification**: Run `python -m neural_net.export.export_weights` — verify_export should still pass with aligned normalization
