# Plan: Add ALGO_MEMCPY Passthrough Algorithm for Pipeline Baseline

## Context

Our "no-comp" benchmark baseline takes a serial D→H fallback path (568 MB/s) while all compressed algorithms use the 3-stage pipelined VOL (1100+ MB/s). This **confounds pipeline parallelism with compression benefit**, making LZ4 appear 2x faster than no-comp despite only 1.02x compression ratio. An SC reviewer flagged this as a **Major issue** that risks rejection.

**Goal:** Add ALGO_MEMCPY (algorithm ID 9) — a passthrough that exercises the full 3-stage VOL pipeline but replaces nvCOMP with `cudaMemcpyAsync D→D`. This decomposes the speedup into:
1. "Native HDF5" (serial fallback) — what happens without GPUCompress
2. "Pipeline Only" (MEMCPY) — pipeline architecture contribution alone
3. Compressed algorithms — compression contribution on top of pipeline

## Design Decisions

- **MEMCPY is NOT part of NN auto-selection.** The NN uses 32 actions (8 algos × 2 quant × 2 shuffle) with warp-level CUDA operations. Adding a 9th algorithm would require retraining. MEMCPY is benchmark-only, specified explicitly via `config.algorithm = 9`.
- **Intercept BEFORE action encoding.** The non-AUTO path does `(cfg.algorithm - 1) % 8` which aliases MEMCPY(9) to LZ4(0). Must short-circuit before this line.
- **Skip ALL preprocessing.** No quantization, no shuffle — pure passthrough.
- **MUST acquire CompContext.** For measurement fairness — pool acquisition overhead is real pipeline cost all compressors pay. Use `ctx->stream` for CUDA ops (VOL workers pass `stream_arg=NULL`; using null stream would serialize all worker streams).
- **Synchronous header copy.** `cudaMemcpyAsync H→D` from a stack-allocated struct is UB on non-null streams. Use synchronous `cudaMemcpy` for the 64-byte header.

## Implementation Steps

### Step 1: Add enum value
**File:** `include/gpucompress.h:56`
```c
GPUCOMPRESS_ALGO_BITCOMP  = 8,  /**< Lossless for scientific data */
GPUCOMPRESS_ALGO_MEMCPY   = 9   /**< Passthrough (device memcpy); benchmarking only */
```

### Step 2: Update algorithm name helpers
**File:** `src/api/gpucompress_api.cpp`
- Line 121-124: Add `"memcpy"` to `ALGORITHM_NAMES[]`
- Line 393: Change `idx <= 8` → `idx <= 9`
- Line 402: Change `i <= 8` → `i <= 9`

### Step 3: Intercept MEMCPY in compression (critical)
**File:** `src/api/gpucompress_compress.cpp:60`

Insert inside `if (cfg.algorithm != GPUCOMPRESS_ALGO_AUTO)`, BEFORE the `int action = (cfg.algorithm - 1) % 8` line:

```cpp
if (cfg.algorithm == GPUCOMPRESS_ALGO_MEMCPY) {
    // Acquire CompContext for fairness (pool overhead is real pipeline cost)
    ContextGuard guard{gpucompress::acquireCompContext()};
    if (!guard.ctx) return GPUCOMPRESS_ERROR_CUDA_FAILED;
    cudaStream_t stream = guard.ctx->stream;

    size_t total = GPUCOMPRESS_HEADER_SIZE + input_size;
    if (total > *output_size) { *output_size = total; return GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL; }
    
    CompressionHeader hdr;
    hdr.original_size = input_size;
    hdr.compressed_size = input_size;
    hdr.shuffle_element_size = 0;
    hdr.quant_flags = 0;
    hdr.setAlgorithmId(GPUCOMPRESS_ALGO_MEMCPY);
    
    uint8_t* out = (uint8_t*)d_output;
    // Sync header copy (stack struct, can't use async on non-null stream)
    cudaMemcpy(out, &hdr, sizeof(hdr), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(out + GPUCOMPRESS_HEADER_SIZE, d_input, input_size,
                     cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    *output_size = total;
    if (stats) { stats->predicted_ratio = 1.0; }
    return GPUCOMPRESS_SUCCESS;
}
```

**Key fixes from reviewer feedback:**
- Acquires CompContext (fairness: pays pool acquisition overhead like real compressors)
- Uses `ctx->stream` (VOL workers pass `stream_arg=NULL`; null stream would serialize all workers)
- Synchronous `cudaMemcpy` for 64-byte header (async from stack is UB on non-null streams)

### Step 4: Intercept MEMCPY in decompression
**File:** `src/api/gpucompress_compress.cpp:~1033`

**CRITICAL:** Insert BEFORE `createDecompressionManager()` call (line ~1036), not after. Raw MEMCPY payload is not nvCOMP-formatted; `createDecompressionManager` will crash on it.

After header is read and `d_compressed_data` pointer is computed (line ~1032), before line 1036:
```cpp
if (header.getAlgorithmId() == GPUCOMPRESS_ALGO_MEMCPY) {
    if (header.original_size > *output_size) {
        *output_size = header.original_size;
        return GPUCOMPRESS_ERROR_BUFFER_TOO_SMALL;
    }
    const uint8_t* payload = (const uint8_t*)d_input + GPUCOMPRESS_HEADER_SIZE;
    cudaMemcpyAsync(d_output, payload, header.original_size, cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);
    *output_size = header.original_size;
    return GPUCOMPRESS_SUCCESS;
}
```

### Step 5: Update Python benchmark
**File:** `scripts/gpucompress_hdf5.py:1014`
```python
("no-comp-vol", 9),  # ALGO_MEMCPY through VOL pipeline (true passthrough baseline)
```

### Step 6: Build and test
```bash
cmake --build build
# Smoke test
LD_LIBRARY_PATH=/tmp/hdf5-install/lib:build python3 scripts/train_and_export_checkpoints.py \
    --model resnet18 --epochs 1 --checkpoint-epochs 1 --hdf5-direct --benchmark
```

## What Does NOT Change

| Component | Why |
|-----------|-----|
| NN kernels / action space (32 actions) | MEMCPY is never auto-selected |
| `internal.hpp` decodeAction / toInternalAlgorithm | Intercepted before these are called |
| `CompressionAlgorithm` enum (nvCOMP internal) | Not an nvCOMP algorithm |
| VOL worker code | Workers call `gpucompress_compress_gpu()` transparently |
| Chunk header format | 4-bit algo ID field supports 0-15; value 9 fits |
| SGD / exploration / online learning | MEMCPY bypasses all of these |

## Verification

1. **Smoke test:** ResNet-18, 1 epoch, --benchmark → verify no-comp-vol appears with ratio ≈ 1.0x and 0 mismatches
2. **Pipeline check:** Verify no-comp-vol has non-zero stage1/drain/io_drain timing (confirms pipeline was used)
3. **Throughput comparison:** no-comp-vol should be close to bitcomp throughput (both go through pipeline with minimal work)
4. **ViT-Base run:** 1 epoch lossless+lossy → verify the 3-baseline decomposition in plots

## Reviewer Findings (4 agents)

### Addressed in plan:
- **SC-reviewer:** Plan is strong. Acquire CompContext for fairness. Add to generic-benchmark.cu too.
- **Perf-optimizer:** Null stream bug (fixed: use ctx->stream). Stack header async (fixed: sync copy). CompContext required (fixed).
- **gpucompress-reviewer:** Decompression crash if check is after createDecompressionManager (fixed: insert before). Stack async UB (fixed).
- **Timing-auditor:** Per-chunk VOL timing overlay will be zeroed (acceptable). 5ms clamp distorts MEMCPY compression_ms (use raw only). write_ms wall-clock is correct.

### Accepted trade-offs:
- D→H copy transfers full uncompressed data (larger than compressed). This is intentional — measures pipeline with worst-case I/O volume.
- Per-chunk diagnostics (pool_wait, d2h, io_wait) are zeroed because diag_slot doesn't propagate through public API. Acceptable since wall-clock write_ms is the primary metric.
- `compression_ms` will show 5ms floor (clamped). Only `compression_ms_raw` is meaningful for MEMCPY.

### Future work (not blocking):
- Add MEMCPY as a phase in generic-benchmark.cu for C-level benchmarks
- Add round-trip test in tests/ directory

## Critical Files

| File | Action |
|------|--------|
| `include/gpucompress.h` | Add ALGO_MEMCPY=9 to enum |
| `src/api/gpucompress_api.cpp` | Add "memcpy" name, update bounds |
| `src/api/gpucompress_compress.cpp` | Intercept in compress + decompress (with CompContext) |
| `scripts/gpucompress_hdf5.py` | Change no-comp-vol from 8→9 |
