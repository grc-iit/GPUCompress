# AUDIT_REPORT.md — Verification Results

**Verified:** 2026-03-14  
**Source:** `AUDIT_REPORT.md` (23 findings across CRITICAL/HIGH/MEDIUM/LOW)

---

## Summary

| Category | Total | Confirmed | Wrong | Overstated |
|----------|-------|-----------|-------|------------|
| CRITICAL | 8 | 5 | 1 | 2 |
| HIGH | 7 | 5 | 2 | 0 |
| MEDIUM | 6 | 5 | 1 | 0 |
| LOW | 2 | 2 | 0 | 0 |
| **Total** | **23** | **17** | **4** | **2** |

---

## Wrong Findings (4)

### C4: Chunk History `fetch_add` Outside Mutex Scope — WRONG

**Report claim:** Two threads can concurrently call `g_chunk_history_count.fetch_add(1)` and race on the `realloc`, because the `fetch_add` is outside the mutex scope.

**Actual code** (`src/api/gpucompress_api.cpp:1282-1293`):

```cpp
    /* Append to per-chunk history — grow array as needed */
    {
        std::lock_guard<std::mutex> lk(g_chunk_history_mutex);   // line 1284
        int idx = g_chunk_history_count.fetch_add(1);            // line 1285
        if (idx >= g_chunk_history_cap) {
            int new_cap = (g_chunk_history_cap == 0) ? 4096 : g_chunk_history_cap * 2;
            auto* p = static_cast<gpucompress_chunk_diag_t*>(
                realloc(g_chunk_history, (size_t)new_cap * sizeof(gpucompress_chunk_diag_t)));
            if (p) { g_chunk_history = p; g_chunk_history_cap = new_cap; }
        }
        if (idx < g_chunk_history_cap) {
            gpucompress_chunk_diag_t *h = &g_chunk_history[idx];
```

**Why this is wrong:** The `fetch_add` at line 1285 is inside the `std::lock_guard` scope — the lock is acquired at line 1284 and released when the block closes. Two threads cannot execute `fetch_add` concurrently because the mutex serializes entry. The report's scenario ("Thread A gets idx=4095, Thread B gets idx=4096, both concurrently") is impossible.

---

### H3: Per-Chunk Stream Create/Destroy in VOL — WRONG

**Report claim:** `gather_stream` and `scatter_stream` are created and destroyed for every chunk, causing per-chunk overhead.

**Actual code — write path** (`src/hdf5/H5VLgpucompress.cu`):

- Stream created **once** at line 1143, before the chunk loop:
  ```cpp
  cudaStream_t gather_stream = nullptr;
  cudaStreamCreate(&gather_stream);        // line 1143
  // ...
  for (size_t ci = 0; ci < total_chunks && ret == 0; ci++) {
      // ... chunk loop uses gather_stream ...
  } /* chunk loop */
  ```
- Stream destroyed **once** at line 1250, after the loop:
  ```cpp
  cudaStreamDestroy(gather_stream);        // line 1250
  ```

**Actual code — read path:** Same pattern. `scatter_stream` created at line 1504 (before loop), destroyed at line 1697 (after loop).

**Why this is wrong:** The streams are per-operation, not per-chunk. One create and one destroy per dataset write/read. The report misread the scope of the chunk loop relative to the stream lifecycle.

---

### H6: Exploration Loop Leaks `d_alt_out` if `compress()` Throws — WRONG

**Report claim:** If `alt_comp->compress()` throws at line 999, `d_alt_out` is never freed.

**Actual code** (`src/api/gpucompress_api.cpp:990-1184`):

```cpp
                    uint8_t* d_alt_out = nullptr;             // line 990 — outer scope
                    try {
                        // ...
                        if (cudaMalloc(&d_alt_out, alt_max) == cudaSuccess) {
                            // ...
                            alt_comp->compress(d_alt_input, d_alt_out, alt_cc);  // line 999
                            // ...
                        }
                    } catch (...) {                           // line 1181
                        // Compression failed for this config, skip it
                    }
                    if (d_alt_out) cudaFree(d_alt_out);      // line 1184 — always runs
```

**Why this is wrong:** `d_alt_out` is declared at line 990 in the outer scope, before the `try`. If `compress()` throws, the `catch(...)` at line 1181 catches the exception, and execution falls through to line 1184 where `cudaFree(d_alt_out)` runs unconditionally. There is no leak — the cleanup always executes regardless of whether the try block succeeds or throws.

---

### M1: HDF5 Streams Created But Never Destroyed — WRONG

**Report claim:** `gather_stream` and `scatter_stream` are created but `cudaStreamDestroy()` is never called, causing a CUDA resource leak.

**Actual code:**

- `gather_stream` destroyed at line 1250:
  ```cpp
  cudaStreamDestroy(gather_stream);                    // line 1250
  ```
- `scatter_stream` destroyed at line 1697:
  ```cpp
  if (scatter_stream) cudaStreamDestroy(scatter_stream); // line 1697
  ```

**Why this is wrong:** Both streams are destroyed at the end of their enclosing write/read operation scope. No resource leak exists. This finding contradicts H3 (which claims per-chunk create/destroy) — both are wrong in opposite directions, indicating the agent did not trace the stream lifecycle through the code.

---

## Overstated Findings (2)

### C7: Use-After-Free Check on `d_quantized` — Severity Overstated

**Report claim (CRITICAL):** After `cudaFree(d_quantized)` at line 846, the pointer value is read at line 890 (`if (d_quantized && quant_result.isValid())`), which is undefined behavior because the pointer is indeterminate after free.

**Assessment:** Technically correct per strict C++14 standard interpretation — reading the value of a pointer after `free`/`cudaFree` (without dereferencing) is formally UB. However:

- `cudaFree` does not modify the caller's pointer variable. The variable still holds the old non-null address.
- No real implementation will crash or produce wrong behavior from this read.
- The pointer is never dereferenced — only its non-null-ness is checked.
- This is a style/pedantic issue, not a crash or corruption risk.

**Correct severity:** LOW, not CRITICAL.

### C5: `cudaDeviceSynchronize()` in VOL Read Loop — Severity Overstated

**Report claim (CRITICAL):** Two `cudaDeviceSynchronize()` calls per chunk in the read loop drain all GPU work and serialize the device.

**Assessment:** The finding is factually correct — `cudaDeviceSynchronize()` at lines 1635 and 1640 does stall the device. However, this is a **performance** issue, not a correctness bug. No data corruption, crash, or silent error results. The calls exist to bracket wall-clock timing (`clock_gettime`). Labeling a performance bottleneck as CRITICAL alongside use-after-free and data races is a severity miscategorization.

**Correct severity:** HIGH (performance), not CRITICAL.

---

## Confirmed Findings (17)

All remaining findings were verified against the source code and are accurate:

| Finding | Verdict | Key Evidence |
|---------|---------|-------------|
| C1 | Confirmed | `d_nn_weights` is plain pointer (nn_gpu.cu:59), TOCTOU between `g_nn_loaded` check and kernel launch (nn_gpu.cu:1352-1360) while `gpucompress_reload_nn` holds `g_init_mutex` but inference does not |
| C2 | Confirmed | `g_online_learning_enabled` is plain `bool` (gpucompress_api.cpp:102), written at line 1781 and read at line 860 without lock |
| C3 | Confirmed | `realloc` at line 1710 with no lock, concurrent reader at line 1931-1933 with no lock |
| C6 | Confirmed | `H5Pcopy` return unchecked at lines 1986, 2006, 2043, 2052 |
| C8 | Confirmed | `quantize_simple` 5-arg overload uses global `d_range_min/max` (quantization_kernels.cu:327-333); source code comment at line 325-326 explicitly documents the race |
| H1 | Confirmed | Multiple redundant sync points verified |
| H2 | Confirmed | Per-call `cudaMalloc`/`cudaFree` in preprocessing kernels |
| H4 | Confirmed | nvcomp managers created per compress call |
| H5 | Confirmed | `configure_compression` result not validated (line 719) |
| H7 | Confirmed | `header_size + max_compressed_size` unchecked for overflow (line 724) |
| M2 | Confirmed | `g_sgd_ever_fired` set before stream sync (nn_gpu.cu:1306-1307); benign on x86 |
| M3 | Confirmed | Plain `float` ranking weights written without synchronization (line 1838-1840) |
| M4 | Confirmed | Native byte order in header; fail-safe via magic check |
| M5 | Confirmed | No size validation after `dequantize_simple` (line 1457-1464) |
| M6 | Confirmed | No bounds check on NN action before `static_cast` to enum (line 613) |
| L1 | Confirmed | Duplicate global and per-context range buffers |
| L2 | Confirmed | Three separate 8-byte D→H copies instead of one 24-byte copy |
