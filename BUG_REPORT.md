# GPUCompress — Consolidated Bug Report

Two rounds of deep investigation + audit verification across the entire codebase.

---

## Critical (data corruption / undefined behavior)

### C1 — Exploration winner headers missing algorithm ID

- **File:** `src/api/gpucompress_api.cpp` lines 975–996, 2150–2171
- **Description:** When level-2 exploration finds a better alternative configuration, the winner's `CompressionHeader` is written without calling `setAlgorithmId()`. The primary path correctly sets it (lines 682, 1881), but the exploration winner path does not.
- **Impact:** Decompression reads the wrong algorithm ID from the header and dispatches to the wrong nvCOMP decompressor. This causes silent data corruption or decompression failure.
- **Fix:** Add `alt_hdr.setAlgorithmId((uint8_t)alt_algo);` after line 996 and after line 2171.

### C2 — Global CUDA timing events with no mutex

- **File:** `src/api/gpucompress_api.cpp` lines 125–126, 623–639
- **Description:** `g_t_start` and `g_t_stop` are global CUDA event handles used for timing compression. Multiple threads calling `gpucompress_compress()` concurrently share these events with no synchronization.
- **Impact:** Race condition corrupts elapsed-time measurements. Since these timings feed into the SGD reinforcement signal, the neural network trains on garbage reward values.
- **Fix:** Move timing events into per-call locals, or protect with `g_sgd_mutex`.

### C4 — Missing int32 clamping in quantization kernel

- **File:** `src/preprocessing/quantization_kernels.cu` lines 71–78
- **Description:** `quantize_linear_kernel` clamps to int8 and int16 ranges but has no clamping path for int32 quantization. Float values outside `[-2^31, 2^31-1]` undergo undefined signed integer overflow.
- **Impact:** Silent data corruption — quantized values wrap around to opposite sign. Particularly dangerous for scientific data with large dynamic range.
- **Fix:** Add `case 32:` clamping branch to `[-2147483648, 2147483647]`.

---

## High (incorrect results / security boundary)

### H1 — Non-Ctx inference reads weights mid-SGD

- **File:** `src/nn/nn_gpu.cu` lines 1116, 1213
- **Description:** The non-CompContext inference path does not call `cudaStreamWaitEvent(g_sgd_done)` before reading NN weights. The non-CompContext SGD path never records the `g_sgd_done` event after finishing.
- **Impact:** Inference reads partially-updated weights during concurrent SGD, producing unpredictable algorithm rankings.
- **Fix:** Add `cudaStreamWaitEvent(stream, g_sgd_done, 0)` before inference launch; record `g_sgd_done` after SGD kernel completes.

### H3 — Host-path ALGO_AUTO uses global singleton buffers

- **File:** `src/api/gpucompress_api.cpp` lines 460, 468
- **Description:** When `gpucompress_compress()` (host path) runs with `ALGO_AUTO`, it uses global singleton GPU buffers for feature extraction and NN inference. The GPU-path (`compress_gpu()`) correctly uses per-CompContext buffers.
- **Impact:** Concurrent host-path ALGO_AUTO calls overwrite each other's feature vectors and inference results, causing wrong algorithm selection.
- **Fix:** Allocate per-call inference buffers, or serialize access with a mutex.

### H4 — Host-path SGD missing mutex

- **File:** `src/api/gpucompress_api.cpp` line 1067 vs 2228
- **Description:** `compress_gpu()` correctly acquires `g_sgd_mutex` before launching SGD. `gpucompress_compress()` (host path) at line 1067 does not.
- **Impact:** Concurrent host-path compressions race on weight updates, corrupting the NN model.
- **Fix:** Wrap the host-path SGD call in `std::lock_guard<std::mutex>(g_sgd_mutex)`.

### H5 — Evaluation script uses wrong normalization stats

- **File:** `neural_net/inference/evaluate.py` lines 258–264
- **Description:** Loads a trained model from a checkpoint file but computes fresh normalization statistics from `encode_and_split()` on the current dataset, rather than using the statistics saved with the checkpoint.
- **Impact:** Feature scaling mismatch between training and evaluation — model sees shifted/scaled inputs, accuracy degrades silently.
- **Fix:** Save and load normalization stats (means, stds) alongside the model checkpoint.

### H6 — int cast overflow in CUB DeviceReduce

- **File:** `src/preprocessing/quantization_kernels.cu` lines 197–210
- **Description:** `num_elements` (size_t) is cast to `int` when passed to CUB `DeviceReduce::Min`/`Max`. For arrays larger than 2^31 elements (~8 GB of float32), this overflows.
- **Impact:** CUB operates on a truncated count, computing wrong min/max. Quantization scale factor is wrong, corrupting all quantized values.
- **Fix:** Use `size_t` or `int64_t` overload; CUB supports `num_items` as 64-bit since CUDA 11.

### H7 — Host decompress skips version validation

- **File:** `src/api/gpucompress_api.cpp` line 1167
- **Description:** `gpucompress_decompress()` checks only the 4-byte magic (`"GPUC"`) but does not call `header.isValid()` which also checks the version field. The GPU-path `gpucompress_decompress_gpu()` at line 2321 correctly calls `isValid()`.
- **Impact:** Accepts headers from future incompatible versions. Decompression proceeds with misinterpreted fields, causing silent corruption or crashes.
- **Fix:** Replace magic-only check with `header.isValid()`.

### H8 — Zero compressed_size bypasses validation

- **File:** `src/api/gpucompress_api.cpp` line 1186
- **Description:** When `compressed_size == 0` is stored in the header, the host decompress path bypasses all payload validation and passes a zero-length buffer to the nvCOMP decompressor.
- **Impact:** nvCOMP receives invalid input — undefined behavior ranging from crashes to memory corruption.
- **Fix:** Reject `compressed_size == 0` as invalid after header parsing.

### H9 — Forced INT8 silently violates error bound

- **File:** `src/preprocessing/quantization_kernels.cu` lines 391–397
- **Description:** When the user forces INT8 precision, values outside `[-128, 127]` after scaling are silently clamped. The clamping error can exceed the user-specified error bound, but no warning is emitted.
- **Impact:** Violates the stated error bound guarantee for large-range data. Users relying on error-bounded lossy compression get silently worse accuracy.
- **Fix:** Compute and report actual max clamping error; warn or error if it exceeds the user's bound.

---

## Medium (logic errors / robustness)

### M6 — Non-atomic global flags

- **File:** `src/nn/nn_gpu.cu`
- **Description:** `g_nn_loaded` (bool) and `g_rank_criterion` (int) are plain global variables read by multiple threads. No `std::atomic` or mutex protection.
- **Impact:** Torn reads possible on architectures with non-atomic bool/int stores. Could cause one thread to see a partially-updated flag and skip NN initialization or use the wrong ranking criterion.
- **Fix:** Declare as `std::atomic<bool>` and `std::atomic<int>`.

### M7 — acquireCompContext falls off end of function

- **File:** `src/api/gpucompress_api.cpp` lines 233–243
- **Description:** After the condition variable wakes and the for-loop scans all `N_COMP_CTX` slots, if no slot is found free (shouldn't happen given the CV predicate, but possible under spurious wakeup), the function falls off the end of a non-void function with no `return`.
- **Impact:** Undefined behavior — the caller receives a garbage pointer, leading to memory corruption.
- **Fix:** Add `return nullptr;` (or `__builtin_unreachable()`) after the for-loop, and check for nullptr at the call site.

### M9 — Primary exploration sample records psnr=0.0

- **File:** `src/api/gpucompress_api.cpp` lines 733–735, 1927–1929
- **Description:** When the primary compression is lossless (no quantization), the exploration sample records `psnr = 0.0` instead of the library's convention of `120.0` for lossless.
- **Impact:** SGD receives a misleadingly low PSNR for what was actually a perfect result, biasing the NN toward lossy configurations.
- **Fix:** Set `psnr = 120.0f` when quantization is disabled (lossless).

### M10 — compress_gpu ignores caller's stream

- **File:** `src/api/gpucompress_api.cpp` line 1662
- **Description:** `compress_gpu()` accepts a `stream_arg` parameter but internally uses `ctx->stream` (the CompContext's dedicated stream) for all operations.
- **Impact:** Breaks the caller's stream-ordering guarantees. Work that the caller expected to be ordered with respect to `stream_arg` may execute out of order.
- **Fix:** Use `stream_arg` when provided (non-null), falling back to `ctx->stream`.

### M12 — SGD skips gradient for lossless results

- **File:** `src/nn/nn_gpu.cu` lines 637–645
- **Description:** The SGD kernel skips the PSNR loss gradient when `actual_psnr <= 0.0`. Lossless compressions report `psnr = 0.0` (due to M9), so the NN never learns from lossless results.
- **Impact:** Lossless configurations are invisible to training, degrading the NN's ability to recommend them.
- **Fix:** Map `actual_psnr <= 0.0` to `120.0` before gradient computation, consistent with the library's convention.

### M13 — Export verification compares incompatible values

- **File:** `neural_net/export/export_weights.py` lines 178–197
- **Description:** `verify_export()` compares `model(raw_input)` (which internally normalizes) against `denorm(manual_forward(normalize(raw_input)))` (which double-normalizes then denormalizes). These are logically different computations.
- **Impact:** Verification always "passes" vacuously — it cannot catch export errors because it's comparing apples to oranges.
- **Fix:** Compare `model(raw_input)` against `manual_forward(normalize(raw_input))` (without the denorm wrapper), or normalize inputs before both paths identically.

### M15 — Byte shuffle ignores element_size parameter

- **File:** `src/preprocessing/byte_shuffle_kernels.cu` lines 238, 260
- **Description:** Both `launch_byte_shuffle` and `launch_byte_unshuffle` explicitly discard the `element_size` parameter with `(void)element_size` and always instantiate the `<4>` (float32) specialization.
- **Impact:** Byte shuffle is silently wrong for non-float32 types (e.g., float64, int16). Data is shuffled assuming 4-byte elements regardless of actual type.
- **Fix:** Dispatch to appropriate template specialization based on `element_size`, or add runtime assert if only 4-byte is supported.

### M16 — Byte shuffle use-after-free race

- **File:** `src/preprocessing/byte_shuffle_kernels.cu` line 310, `src/compression/util.h` lines 41–43
- **Description:** `byte_shuffle_simple()` returns the output pointer but does not synchronize the stream before the `DeviceChunkArrays` destructor runs. The destructor calls `cudaFree` on `d_input_ptrs`, `d_output_ptrs`, and `d_sizes` while the async shuffle kernel may still be reading them.
- **Impact:** Use-after-free — the kernel reads freed GPU memory. May cause silent corruption, wrong results, or CUDA errors depending on timing.
- **Fix:** Add `cudaStreamSynchronize(stream)` before `DeviceChunkArrays` goes out of scope, or use `cudaFreeAsync` on the same stream.

### M17 — Gray-Scott benchmark gather kernel OOB read

- **File:** `benchmarks/grayscott/grayscott-benchmark.cu` lines 147–169
- **Description:** `gather_chunk_kernel` computes the actual Z-dimension of the last chunk (`actual_cz`) but passes the full `chunk_z` to the inner loop bound. For the final chunk where `actual_cz < chunk_z`, threads read beyond allocated memory.
- **Impact:** Out-of-bounds GPU memory read on the last chunk. May read garbage, cause CUDA memory errors, or silently produce wrong compressed data.
- **Fix:** Use `actual_cz` instead of `chunk_z` in the loop bound.

### M18 — Benchmark Phase 5 inherits mutated weights

- **File:** `benchmarks/grayscott/grayscott-benchmark.cu` lines 1035–1069
- **Description:** Phase 5 (exhaustive search) runs after Phase 4 (online learning with SGD). The NN weights have been modified by Phase 4's SGD updates. Phase 5 uses these mutated weights for its NN-predicted comparisons.
- **Impact:** Exhaustive search vs NN comparison uses a different NN state than the original pre-trained model, making the comparison unfair and the benchmark results misleading.
- **Fix:** Call `gpucompress_reload_nn()` before Phase 5 to restore original weights.

### M19 — No NaN handling in training feature extraction

- **File:** `neural_net/core/data.py`
- **Description:** Entropy, MAD (mean absolute deviation), and second_derivative features can produce NaN for degenerate inputs (e.g., constant arrays). No NaN check or fill is performed before these values enter the training pipeline.
- **Impact:** A single NaN row propagates through normalization, loss computation, and gradient updates, silently degrading or destroying model training.
- **Fix:** Add `.fillna(0.0)` or appropriate sentinel values after feature computation.

### M20 — Missing PSNR NaN fill in cross-validation

- **File:** `neural_net/training/cross_validate.py` line 47
- **Description:** PSNR values are used directly without `.fillna(120.0)`. Lossless configurations may have NaN or inf PSNR in the training data.
- **Impact:** NaN propagates through the loss function during cross-validation, corrupting fold metrics and potentially the model.
- **Fix:** Add `.fillna(120.0)` for PSNR column before training.

---

## Low (cosmetic / defense-in-depth)

### L1 — Data race in filter plugin initialization

- **File:** `src/hdf5/H5Zgpucompress.c` lines 141–151
- **Description:** `ensure_initialized()` uses a plain `int` flag without atomics or mutex. Two threads could both see `initialized == 0` and both call `gpucompress_init()`.
- **Impact:** Benign in practice — `gpucompress_init()` is internally idempotent. Double initialization wastes a few microseconds but causes no corruption.
- **Fix:** Use `pthread_once()` or `std::call_once` for correctness.

### L2 — Shuffle size parsing incomplete

- **File:** `src/hdf5/H5Zgpucompress.c` lines 248–250
- **Description:** The filter plugin's `cd_values[2]` parsing only handles value `4` for byte shuffle element size. Values `2` and `8` are documented in the public header (`include/gpucompress_hdf5.h`) but silently ignored.
- **Impact:** Users who set shuffle size 2 or 8 via the HDF5 filter API get no shuffle applied, with no error or warning.
- **Fix:** Add cases for 2 and 8, or return an error for unsupported values.

### L3 — VOL read path doesn't copy filter_mask

- **File:** `src/hdf5/H5VLgpucompress.cu` line 929
- **Description:** `read_chunk_from_native` allocates a local `filter_mask` but never copies the value back to the caller.
- **Impact:** Cosmetic within the VOL's own read/write cycle (the VOL always writes with compression, so filter_mask is unused). Would matter only if cross-path interop with the standard HDF5 filter pipeline is attempted.
- **Fix:** Copy `filter_mask` back to the caller's output parameter if provided.

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High | 6 |
| Medium | 10 |
| Low | 3 |
| **Total** | **22** |

### Architectural Notes

- **Host path vs GPU path asymmetry:** `compress_gpu()` is well-hardened (CompContext pool, per-slot buffers, `g_sgd_mutex`). Nearly all concurrency bugs live in `gpucompress_compress()` which uses unprotected global singletons.
- **NN training pipeline:** A separate cluster of data-quality bugs (NaN handling, normalization mismatch, PSNR convention) silently degrade model accuracy over time.
- **Preprocessing:** Quantization and byte shuffle both have edge-case bugs that only manifest with specific data types or sizes, making them hard to catch in standard testing.
