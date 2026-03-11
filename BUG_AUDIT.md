# GPUCompress Bug Audit Report

**Date:** 2026-03-10
**Scope:** Full codebase audit across 6 areas — compression/decompression, NN/SGD, preprocessing, HDF5 VOL, benchmark methodology, and Python training pipeline.
**Method:** Initial automated sweep followed by deep-dive investigation with 6 parallel agents confirming each finding against source code.

---

## Confirmed Bugs (Fixed)

### 1. Memory leak on NN failure — `src/api/gpucompress_api.cpp:540`
- **Severity:** High
- **Issue:** `d_input` GPU buffer allocated at line 421 but not freed when NN inference fails and returns `GPUCOMPRESS_ERROR_NN_NOT_LOADED`.
- **Fix:** Added `cudaFree(d_input)` before the early return.

### 2. TOCTOU race in `gpucompress_get_chunk_diag` — `src/api/gpucompress_api.cpp:1567`
- **Severity:** High
- **Issue:** `g_chunk_history_count` checked *outside* the mutex, then struct copied *inside* the mutex. A concurrent writer could be mid-write to that index between the check and the copy.
- **Fix:** Moved mutex acquisition before the count check so the entire read is atomic.

### 3. `allocSGDBuffers` returns void — `src/nn/nn_gpu.cu:797`
- **Severity:** High
- **Issue:** Function returns `void`, so callers can't detect `cudaMalloc` failure. The caller at line 1226 already checks `d_sgd_grad_buffer == nullptr` after the call, but the init-time caller at line 966 does not.
- **Fix:** Changed return type to `bool` (returns `false` on any allocation failure).

### 4. `effective_eb` vs `error_bound` mismatch — `src/preprocessing/quantization_kernels.cu:477`
- **Severity:** Medium
- **Issue:** `QuantizationResult.error_bound` stored the original `config.error_bound`, but quantization actually used `effective_eb` (adjusted for float precision, int32 overflow limits). Dequantization works correctly (uses `scale_factor`), but any code using `error_bound` for validation/PSNR gets the wrong value.
- **Fix:** Changed `result.error_bound = config.error_bound` to `result.error_bound = effective_eb`.

### 5. `io_done_flag` data race — `src/hdf5/H5VLgpucompress.cu:1047`
- **Severity:** Low-Medium
- **Issue:** Declared as plain `bool` but accessed across threads. Currently protected by `io_mtx` in all access sites, but fragile — any future access outside the mutex would be undefined behavior.
- **Fix:** Changed to `std::atomic<bool>`.

---

## False Positives (Investigated and Dismissed)

### Compression/Decompression
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| Division by zero when `compressed_size == 0` | False positive | nvcomp never returns 0 for valid compression; minor defensive concern at most |
| Missing `cudaStreamSynchronize` before reading compressed size | False positive | `cudaEventSynchronize(tl_t_stop)` provides implicit sync; explicit sync at line 729 before D->H copy |
| Integer overflow in `n_chunks * chunk_size` | False positive | All operations use `size_t` on 64-bit; no actual overflow path exists |

### NN Inference & SGD
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| `log1pf()` domain violation producing NaN | False positive | All inputs (ratio, times) are physically positive; `log1pf(x)` valid for x > -1 |
| Race condition in SGD weight updates | False positive | Perfect thread partitioning — each of 128 threads owns distinct weight indices (verified by test) |
| Gradient clipping divide-by-zero | False positive | Guard `norm > 1.0f` prevents division; norm=0 uses identity scaling `clip_scale = 1.0f` |

### Preprocessing
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| Static buffer race in `quantize()` | False positive | Thread-safe overload exists with per-slot buffers (`ctx->d_range_min/max`); API correctly uses it |

### HDF5 VOL Plugin
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| Worker pool deadlock on error | False positive | Errors caught via `worker_err`; sentinels guarantee orderly shutdown |
| Prefetch thread deadlock on early exit | False positive | Explicit cleanup path at `done_read` resets semaphore and unblocks thread |
| Sentinel shutdown deadlock | False positive | Proper condition variable signaling after each sentinel push |

### Benchmark Methodology
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| Missing `cudaDeviceSynchronize` before write timer stop | False positive | VOL's `H5Dwrite` joins all worker threads before returning (synchronous) |
| GPU cache warmup bias across phases | Minor concern | Negligible for I/O-bound storage benchmarking; phases process different compression configs |
| Throughput calculation ambiguity | False positive | Intentional design: wall-clock for throughput, GPU-time for breakdown; correctly labeled |

### Python Training Pipeline
| Alleged Issue | Verdict | Reason |
|---------------|---------|--------|
| `std()` uses ddof=0 instead of ddof=1 | False positive | Large N (100K+ rows) makes difference negligible (<0.05%); consistent across train/inference |

---

## Additional Findings (No Issues)

The Python training pipeline deep-dive also verified:
- No data leakage between train/val splits (correctly splits by file)
- Loss function (MSELoss) correctly applied to normalized targets
- Weight export format matches C++ loader exactly (verified field-by-field)
- Feature normalization consistent between Python training and CUDA inference
- SGD target normalization in active learning matches training normalization

---

---

## Benchmark Measurement Issues (Found 2026-03-11)

**Scope:** Audit of metric correctness in both `benchmarks/grayscott/grayscott-benchmark.cu` and `benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx`.

### Issue 6: VPIC oracle uses raw POSIX I/O instead of HDF5 VOL — `vpic_benchmark_deck.cxx:338-396`
- **Severity:** High
- **Affects:** VPIC benchmark only (grayscott is already fixed)
- **Issue:** The exhaustive (oracle) phase writes/reads via raw POSIX `open()/write()/read()` to a `.bin` file, while all NN phases go through HDF5 VOL (`H5Dwrite`/`H5Dread`). This gives the oracle an unfair throughput advantage by bypassing all HDF5 overhead (B-tree allocation, chunk indexing, metadata writes, superblock). The comparison between oracle and NN phases is apples-to-oranges.
- **Evidence:** Oracle E2E write (line 338): `int fd = open(TMP_ORACLE, O_WRONLY | ...)` followed by `write(fd, h_comp_buf, comp_size)`. NN phases (line 449): `H5Dwrite(dset, H5T_NATIVE_FLOAT, ...)`.
- **Grayscott comparison:** Grayscott fixed this — its oracle Stage 2 routes through HDF5 VOL with `make_dcpl_fixed()` and the majority-vote algorithm.
- **Fix:** Rewrite VPIC oracle Stage 2 to use HDF5 VOL with a per-chunk fixed algorithm, matching the grayscott approach.
- **Status:** Open

### Issue 7: VPIC oracle ratio excludes file format overhead — `vpic_benchmark_deck.cxx:332,408`
- **Severity:** High
- **Affects:** VPIC benchmark only
- **Issue:** Oracle compression ratio is computed as `total_bytes / total_best_compressed` (line 332) using raw compressed chunk sizes from `gpucompress_compress_gpu()`, excluding any file format overhead. All NN phases compute ratio as `total_bytes / file_size` (line 562) using the actual HDF5 file size on disk, which includes metadata. This makes the oracle ratio appear artificially higher than the NN phases even when compression is identical.
- **Evidence:** Line 408: `r->ratio = oracle_ratio;` (raw compressed), vs line 562: `r->ratio = (double)total_bytes / (double)(fbytes ? fbytes : 1);` (file-based).
- **Grayscott comparison:** Grayscott oracle uses `file_size(tmp_file)` for ratio since it writes through HDF5.
- **Fix:** Addressed automatically when Issue 6 is fixed (oracle goes through HDF5, ratio computed from file size).
- **Status:** Open

### Issue 8: Missing `cudaDeviceSynchronize` before write timer stop — `vpic_benchmark_deck.cxx:458-463`
- **Severity:** Medium
- **Affects:** VPIC benchmark (all phases); grayscott NN phases have the same pattern
- **Issue:** The write timer stops after `H5Dclose` + `H5Fclose` but without an explicit `cudaDeviceSynchronize()` after `H5Dwrite`. The VOL's internal workers do synchronous `cudaMemcpy D→H` before writing to disk, so GPU work is implicitly synced. However, this is fragile — if the VOL implementation changes to use async copies, the timer would undercount.
- **Evidence:** VPIC `run_phase()` (line 458-462): no CUDA sync between `H5Dwrite` and timer stop. Contrast with VPIC no-comp (same function) and grayscott `run_phase_nocomp()` (line 623) which do include `cudaDeviceSynchronize()`.
- **Impact:** Currently benign (VOL workers sync internally), but inconsistent across phases.
- **Fix:** Add `cudaDeviceSynchronize()` after `H5Dwrite` in `run_phase()` for consistency.
- **Status:** Open

### Issue 9: Per-chunk timers mix CUDA events with CPU wall-clock — `src/api/gpucompress_api.cpp`
- **Severity:** Low
- **Affects:** Both benchmarks (they read from the same `gpucompress_chunk_diag_t`)
- **Issue:** `compression_ms` is measured via CUDA events (`cudaEventElapsedTime` — pure GPU kernel time), while `nn_inference_ms`, `preprocessing_ms`, `exploration_ms`, and `sgd_update_ms` are all measured via `std::chrono::steady_clock` (CPU wall-clock). When summed into `total_tracked` GPU-time, the components use different timing domains. With concurrent GPU work on multiple streams, wall-clock timers include sync contention while CUDA events do not.
- **Evidence:**
  - `compression_ms`: CUDA events at `gpucompress_api.cpp` lines ~685-695
  - `nn_inference_ms`: `steady_clock` at lines ~470, 518
  - `preprocessing_ms`: `steady_clock` at lines ~585, 607
- **Impact:** The percentage breakdown is approximately correct (all components experience similar sync overhead), but the absolute values are not strictly comparable. NN inference time is inflated by `cudaStreamSynchronize` contention when multiple workers share the GPU.
- **Fix:** Document the caveat, or switch all timers to CUDA events for consistency.
- **Status:** Open

### Issue 10: Write/read timers include HDF5 close overhead — Both benchmarks
- **Severity:** Low
- **Affects:** Both benchmarks, all phases
- **Issue:** Both `write_ms` and `read_ms` include `H5Dclose` + `H5Fclose` in the timed region. These calls flush HDF5 metadata (B-tree finalization, superblock update, `fdatasync`). This inflates both write and read throughput denominators beyond the pure data transfer path.
- **Evidence:** Grayscott `run_phase_vol()` lines 707-712; VPIC `run_phase()` lines 458-463.
- **Impact:** Consistent across all HDF5 phases, so relative comparisons are valid. Absolute throughput numbers are slightly conservative (lower than true data-path throughput).
- **Fix:** No fix needed — this is an acceptable end-to-end measurement convention. Could optionally move `H5Dclose`/`H5Fclose` outside the timer for pure data throughput.
- **Status:** Accepted (by design)

### Issue 11: Cumulative GPU-time sum exceeds wall-clock with concurrent workers — Both benchmarks
- **Severity:** Low
- **Affects:** Both benchmarks (overhead breakdown table)
- **Issue:** The "Total GPU-time" column sums per-chunk `nn_inference_ms + preprocessing_ms + compression_ms + exploration_ms + sgd_update_ms` across all chunks. With 8 concurrent VOL workers processing chunks in parallel, this cumulative sum can be up to 8x the wall-clock time. For example, 17 chunks × 50ms NN each = 850ms cumulative, but wall-clock is only ~200ms because 8 run concurrently.
- **Evidence:** Both benchmarks show `total GPU-time` > `write_ms` in the overhead breakdown table.
- **Impact:** Both benchmarks document this correctly in the table header ("8 concurrent workers") and show wall-clock separately. The per-component **percentages** are valid since all components are summed the same way.
- **Fix:** No fix needed — already documented. The percentages show correct relative cost.
- **Status:** Accepted (documented)

---

## Summary

**5 confirmed bugs fixed, 6 benchmark measurement issues documented, 14 false positives dismissed.**

| # | Bug | Severity | Status |
|---|-----|----------|--------|
| 1 | Memory leak on NN failure path | High | Fixed |
| 2 | TOCTOU race in chunk_diag reader | High | Fixed |
| 3 | allocSGDBuffers silent failure | High | Fixed |
| 4 | effective_eb vs error_bound mismatch | Medium | Fixed |
| 5 | io_done_flag not atomic | Low-Med | Fixed |
| 6 | VPIC oracle uses raw POSIX I/O instead of HDF5 | High | Open |
| 7 | VPIC oracle ratio excludes file overhead | High | Open |
| 8 | Missing cudaDeviceSynchronize before write timer | Medium | Open |
| 9 | Per-chunk timers mix CUDA events with wall-clock | Low | Open |
| 10 | Write/read timers include HDF5 close overhead | Low | Accepted |
| 11 | Cumulative GPU-time exceeds wall-clock | Low | Accepted |
