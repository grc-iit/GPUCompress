# GPUCompress Audit Findings

**Generated:** 2026-03-14  
**Specification:** auditAGENT.md  
**Scope:** Static analysis and execution modeling per Stages 1–4 (structural mapping, static graphs, execution modeling, formal verification).

---

## Architecture Summary (Stage 1)

- **Entry points:** `gpu_compress` / `gpu_decompress` (CLI), `gpucompress_compress` / `gpucompress_decompress` / `gpucompress_decompress_gpu` (API), HDF5 filter `H5Z_filter_gpucompress`, HDF5 VOL `H5VLgpucompress` (dataset read/write).
- **Build:** CMake; C/C++/CUDA; nvCOMP (ManagerFactory), HDF5 C, optional GDS/cuFile.
- **CUDA usage:** Default stream, per-context stream and events, dedicated SGD stream/event, VOL gather/scatter streams; device allocations in API, VOL, and NN paths.
- **nvCOMP:** Compression/decompression via `createCompressionManager` / `createDecompressionManager`; managers created per call, not long-lived.
- **HDF5:** Filter path (host buffers); VOL path (GPU buffers, chunk-by-chunk compress/decompress, native chunk I/O).

---

## Finding 1: Unchecked `cudaStreamCreate` / `cudaEventCreate` in `gpucompress_init`

| Field | Value |
|-------|--------|
| **Title** | SGD stream and event creation success ignored in `gpucompress_init` |
| **Category** | Runtime |
| **Severity** | High |
| **Affected files** | `src/api/gpucompress_api.cpp` |

**Execution path:**  
`gpucompress_init()` → `cudaStreamCreate(&g_default_stream)` (checked) → `cudaStreamCreate(&g_sgd_stream)` → `cudaEventCreate(&g_sgd_done)` → … (no check on the last two).

**Reasoning:**  
- API contract: every CUDA creation can fail (e.g. device OOM, driver error).  
- If `cudaStreamCreate(&g_sgd_stream)` or `cudaEventCreate(&g_sgd_done)` fails, the code still sets `g_initialized = true` and continues.  
- Later, `nn_gpu.cu` and API code use `g_sgd_stream` and `g_sgd_done` (e.g. `cudaStreamWaitEvent(stream, g_sgd_done, 0)`).  
- Using an invalid stream or event is undefined behavior and can cause hangs, wrong ordering, or crashes.

**Why this is a bug:**  
The library reports success and then uses resources that may never have been created. That violates the assumption “after init success, all global CUDA objects are valid” and leads to undefined behavior on first use of SGD or related paths.

---

## Finding 2: `initCompContextPool()` return value ignored in `gpucompress_init`

| Field | Value |
|-------|--------|
| **Title** | CompContext pool init failure leaves library in inconsistent state |
| **Category** | Runtime / Logical |
| **Severity** | High |
| **Affected files** | `src/api/gpucompress_api.cpp` |

**Execution path:**  
`gpucompress_init()` → `gpucompress::initCompContextPool()` (return value not used) → bandwidth probe, optional NN load → `g_initialized.store(true)`.

**Reasoning:**  
- `initCompContextPool()` can return `-1` on any `cudaStreamCreate`, `cudaEventCreate`, or `cudaMalloc` failure for a pool slot.  
- The caller does not check this return value.  
- After a failed pool init, `g_initialized` is still set to true.  
- Context-based compress path calls `acquireCompContext()` which waits on `g_pool_free_count > 0`. If no slot was ever marked free (all inited slots failed, or init failed before marking any), `g_pool_free_count` can remain 0 and callers block forever.  
- Even if some slots were inited, a partial failure leaves the pool in an undefined state (some slots initialized, some not) with no cleanup of the successful allocations before returning -1.

**Why this is a bug:**  
The library can report successful initialization while the context pool is failed or partially initialized, leading to deadlock or use of uninitialized/partially initialized contexts.

---

## Finding 3: Resource leak and partial state on `initCompContextPool()` failure

| Field | Value |
|-------|--------|
| **Title** | CompContext pool leaks streams, events, and device memory on first allocation failure |
| **Category** | Runtime / Memory |
| **Severity** | Medium |
| **Affected files** | `src/api/gpucompress_api.cpp` |

**Execution path:**  
`initCompContextPool()` loops over `N_COMP_CTX`; for each slot it creates stream, events, and device buffers. On first failure (e.g. `cudaMalloc` for slot `i`), it returns `-1` without destroying or freeing resources already created for slots `0..i-1`.

**Reasoning:**  
- Each iteration allocates: one stream, six events, and multiple device buffers.  
- The code uses early `return -1` on any failure and has no cleanup block for previously created resources.  
- After return, those resources are never freed (destroyCompContextPool is only called from gpucompress_cleanup, and if init is considered “failed” by a future fix, cleanup might not run).  
- So: partial allocation + leak on any pool init failure.

**Why this is a bug:**  
Violates cudaMalloc/cudaStreamCreate/cudaEventCreate vs cudaFree/cudaStreamDestroy/cudaEventDestroy symmetry and can exhaust device (and host) resources over repeated init failures.

---

## Finding 4: VOL `new_obj()` uses `calloc` result without NULL check

| Field | Value |
|-------|--------|
| **Title** | H5VLgpucompress `new_obj()` dereferences result of `calloc()` without checking for NULL |
| **Category** | Runtime / Corruption risk |
| **Severity** | High |
| **Affected files** | `src/hdf5/H5VLgpucompress.cu` |

**Execution path:**  
Any VOL callback that wraps an underlying object (e.g. dataset create/open, file create/open, group create, attribute open) calls `new_obj(under, vol_id)`. `new_obj()` does `calloc(1, sizeof(H5VL_gpucompress_t))` and then assigns to `o->under_object`, `o->under_vol_id`, etc. There is no check for `o == NULL`.

**Reasoning:**  
- C standard: `calloc` can return NULL on allocation failure.  
- If `calloc` returns NULL, the next line dereferences `o` → undefined behavior (typically SIGSEGV).  
- Regression test `test_h7_null_calloc` documents this (fault injection via LD_PRELOAD to force calloc to return NULL).  
- Many call sites use the result of `new_obj()` without checking (e.g. `H5VL_gpucompress_dataset_create` assigns to `dset` and later uses `dset->dcpl_id`).

**Why this is a bug:**  
Under OOM or allocator failure, the VOL connector crashes instead of propagating an HDF5 error, and can corrupt process state.

---

## Finding 5: VOL gather and scatter stream creation success not checked

| Field | Value |
|-------|--------|
| **Title** | `cudaStreamCreate` for gather_stream and scatter_stream in H5VLgpucompress not checked |
| **Category** | Runtime / Stream correctness |
| **Severity** | High |
| **Affected files** | `src/hdf5/H5VLgpucompress.cu` |

**Execution path:**  
- **Write path:** Chunked GPU write creates `gather_stream` with `cudaStreamCreate(&gather_stream)` (no check). Later: `gather_chunk_kernel<<<..., gather_stream>>>`, `cudaStreamSynchronize(gather_stream)`, `cudaStreamDestroy(gather_stream)`.  
- **Read path:** Chunked GPU read creates `scatter_stream` with `cudaStreamCreate(&scatter_stream)` (no check). Later: `scatter_chunk_kernel<<<..., scatter_stream>>>`, `cudaStreamSynchronize(scatter_stream)`, `if (scatter_stream) cudaStreamDestroy(scatter_stream)`.

**Reasoning:**  
- If `cudaStreamCreate` fails, the stream handle is undefined (e.g. left null or garbage).  
- Passing that handle to kernel launch or sync is undefined behavior.  
- Calling `cudaStreamDestroy` on an invalid handle is also undefined.  
- So: unchecked creation → possible use of invalid stream → UB and possible hang or crash.

**Why this is a bug:**  
Stream correctness requires that every stream used in launch/sync/destroy was successfully created. Ignoring creation failure violates that and can cause non-deterministic failures under load or low resources.

---

## Finding 6: `gpucompress_decompress_gpu` does not handle zero compressed size

| Field | Value |
|-------|--------|
| **Title** | Zero-byte compressed payload not handled in GPU decompression path |
| **Category** | Logical / nvCOMP correctness |
| **Severity** | Medium |
| **Affected files** | `src/api/gpucompress_api.cpp` |

**Execution path:**  
`gpucompress_decompress_gpu(d_input, input_size, d_output, output_size, stream)` → read header from device → `compressed_size = header.compressed_size`. If header has `original_size == 0` and `compressed_size == 0` (or valid zero-byte payload), the code still does `createDecompressionManager(d_compressed_data, stream)` and `configure_decompression` / `decompress` on a zero-byte input.

**Reasoning:**  
- Host path `gpucompress_decompress()` explicitly handles `compressed_size == 0`: if `header.original_size == 0` it sets `*output_size = 0` and returns success; otherwise returns invalid header error.  
- GPU path has no equivalent. Passing zero-byte compressed data to nvcomp may be invalid or poorly defined.  
- Regression test `test_h8_zero_compressed_size` indicates zero compressed size is a defined scenario that must be handled.  
- Even if nvcomp tolerates 0 bytes, not setting `*output_size = 0` and short-circuiting is inconsistent with the host API and can confuse callers.

**Why this is a bug:**  
Logical inconsistency with host decompression and missing handling of a valid edge case (zero-length chunk), which can lead to wrong behavior or reliance on nvcomp behavior that may not be guaranteed.

---

## Finding 7: H5Z filter decompress path returns 0 on error (HDF5 convention)

| Field | Value |
|-------|--------|
| **Title** | HDF5 filter returns 0 on decompression failure without pushing HDF5 error stack |
| **Category** | HDF5 safety / Logical |
| **Severity** | Low |
| **Affected files** | `src/hdf5/H5Zgpucompress.c` |

**Execution path:**  
Filter callback `H5Z_filter_gpucompress` in reverse (decompress) mode: `gpucompress_decompress(...)`; if `err != GPUCOMPRESS_SUCCESS`, it does `H5free_memory(new_buf); return 0;`. HDF5 filter documentation typically uses return 0 to indicate failure, but the application has no way to distinguish “decompress failed” from other failures and the HDF5 error stack may not be set by the filter.

**Reasoning:**  
- Returning 0 signals failure to HDF5, which is correct for “must fail” semantics.  
- The filter does not push a descriptive error onto the HDF5 error stack (e.g. via H5Epush), so debugging and error reporting are harder.  
- Same pattern on compression side: `return 0` on compress failure without enriching HDF5 error state.

**Why this is a bug/risk:**  
Incorrect or incomplete error reporting: callers see a generic failure and cannot tell that the GPUCompress decompression (or compression) step failed, and have no structured HDF5 error to inspect.

---

## Finding 8: VOL fallback write/read use `assert` after failed `malloc`

| Field | Value |
|-------|--------|
| **Title** | Assert used for allocation failure in VOL fallback paths |
| **Category** | Runtime |
| **Severity** | Low |
| **Affected files** | `src/hdf5/H5VLgpucompress.cu` |

**Execution path:**  
`gpu_fallback_dh_write` and the GPU fallback read path allocate a host staging buffer with `malloc(total_bytes)`. Code uses `assert(h_tmp && "...")` and then `if (!h_tmp) return -1;`. In Release builds with NDEBUG, the assert is removed; the `if (!h_tmp) return -1` remains, so behavior is correct. If the assert were ever the only check, removal in Release would be wrong; as written, the redundant assert is defensive but the pattern is brittle (future edits could drop the `if`).

**Reasoning:**  
- Allocation failure should always be handled by returning an error, not by assert (which is disabled in NDEBUG).  
- Current code does both; the real error handling is the `if (!h_tmp) return -1`.  
- Risk is low but the pattern is worth noting for future changes.

**Why this is a risk:**  
Reliance on assert for “must not happen” in code paths that can legitimately fail (OOM) is fragile; the code is correct today but the style could lead to bugs if someone removes the explicit check.

---

## Summary Table

| # | Title | Category | Severity |
|---|--------|----------|----------|
| 1 | Unchecked g_sgd_stream / g_sgd_done creation in gpucompress_init | Runtime | High |
| 2 | initCompContextPool return value ignored | Runtime / Logical | High |
| 3 | initCompContextPool leaks on partial failure | Runtime / Memory | Medium |
| 4 | new_obj() calloc result not checked (VOL) | Runtime / Corruption risk | High |
| 5 | VOL gather/scatter cudaStreamCreate not checked | Runtime / Stream | High |
| 6 | gpucompress_decompress_gpu zero compressed_size not handled | Logical / nvCOMP | Medium |
| 7 | H5Z filter error return without HDF5 error stack | HDF5 safety | Low |
| 8 | VOL fallback assert + malloc failure pattern | Runtime | Low |

---

## Clarifications (per audit protocol)

The following were not assumed; they would refine the audit if provided:

1. **Entry point file:** Analyzed all: CLI (`compress.cpp`, `decompress.cpp`), API (`gpucompress_api.cpp`), HDF5 filter (`H5Zgpucompress.c`), HDF5 VOL (`H5VLgpucompress.cu`).
2. **Typical execution scenario:** Single-GPU, host and device buffers; VOL path used for chunked GPU read/write.
3. **Unified memory:** Not assumed; explicit device and host allocations observed.
4. **GPUDirect:** Optional (cuFile in CLI); not assumed for core API/VOL paths.
5. **Multi-GPU:** Not assumed; single device and default stream used in API.

---

*End of audit findings. No fixes were proposed unless explicitly requested.*
