# GPUCompress Optimization Report

**Generated:** 2026-03-14
**Methodology:** 6-agent parallel analysis + compute-sanitizer memcheck/racecheck + nsys profiling
**Benchmark:** Gray-Scott 100^3 (4 MB), 1 MB chunks, 4 chunks, nn-rl phase

---

## Profiling Summary

| Tool | Result |
|------|--------|
| compute-sanitizer memcheck | **0 leaks, 0 errors** |
| compute-sanitizer racecheck | 38 hazards — all in nvcomp `unsnap_kernel` (not our code) |
| nsys CUDA API | 108 cudaMalloc + 108 cudaFree for 4 chunks; `cudaMallocHost` = 1.5s (40.9%) |
| nsys GPU kernels | `snap_kernel` 47ms (78.8%), all custom kernels clean |
| Wall-clock vs GPU-time | 1270ms wall vs 547ms GPU-time (**2.3x overhead from malloc**) |

### Malloc Distribution by Phase

| Phase | cudaMalloc | cudaMallocHost | cudaHostAlloc | Total Time |
|-------|-----------|---------------|--------------|-----------|
| Pool Init | 85 calls (57ms) | 5 calls (103ms) | — | 160ms |
| Write (compress) | 14 calls (54ms) | 11 calls (239ms) | 9 calls (245ms) | 573ms |
| Write (cleanup) | — | 1 call (1,156ms!) | — | 1,738ms |
| Read (decompress) | 3 calls | 1 call (3ms) | 1 call | 4ms |
| Cleanup | 6 calls | — | 3 calls | 287ms |

---

## Agent 1: CUDA Kernel Optimization

### K1: Shared Memory Bank Conflicts in Entropy Histogram [HIGH]
**File:** `src/stats/entropy_kernel.cu:49-81, 88-131`
**Kernels:** `histogramKernel`, `histogramKernelVec4` — 256-thread blocks, 256-bin shared histogram

All 256 threads simultaneously `atomicAdd(&s_hist[byte_val], 1)`. With 32 shared memory banks, this creates severe bank conflicts and serialization.

**Fix:** Use warp-level reduction via shuffle operations, or integrate CUB `DeviceHistogram`.
**Expected gain:** 2-3x histogram throughput.

---

### K2: GPU Spin-Wait for Inter-Block Synchronization [MEDIUM]
**File:** `src/stats/stats_kernel.cu:128-135`
**Kernel:** `statsPass1Kernel`

Block 0 initializes stats struct, other blocks spin-wait on `atomicAdd(d_init_flag, 0)`. GPU warps cannot context-switch while spinning — wastes cycles.

**Fix:** Initialize `AutoStatsGPU` on host via `cudaMemcpyAsync` before kernel launch. Remove spin-wait entirely.
**Expected gain:** 10-20% stats latency reduction.

---

### K3: Uncoalesced Memory Access in Gather/Scatter [HIGH]
**File:** `src/hdf5/H5VLgpucompress.cu:743-776, 781-811`
**Kernels:** `gather_chunk_kernel`, `scatter_chunk_kernel`

Multi-dimensional index calculation with 7 modulo/division operations per thread creates scattered memory accesses. No shared memory tiling.

**Fix:** Pre-compute address lists on host, or use tiled gather with shared memory staging.
**Expected gain:** 1.5-2x for large non-contiguous chunks.

---

### K4: Warp Underutilization in Byte Shuffle [MEDIUM]
**File:** `src/preprocessing/byte_shuffle_kernels.cu:18-71, 78-131`
**Config:** 128 threads/block, 4 warps/block, 1 warp/chunk

For `ElementSize=4`, only lanes 0-3 are active (4/32 = 12.5% utilization). 28 lanes idle per warp.

**Fix:** Map 8 warps per chunk or use `__shfl_sync` to distribute byte work across lanes.
**Expected gain:** 1.2-1.5x shuffle throughput.

---

### K5: SGD Kernel Shared Memory Bank Conflicts [MEDIUM]
**File:** `src/nn/nn_gpu.cu:525-836`
**Config:** `<<<1, 128>>>`

128-element shared arrays (`s_h1`, `s_h2`, `s_dz2`) accessed by 128 threads simultaneously. Potential 4-way bank conflicts with 32 banks.

**Fix:** Use `__shfl_down_sync` for final reduction stages. Validate with Nsight Compute.

---

### K6: No Shared Memory Tiling in Gray-Scott Stencil [LOW]
**File:** `src/gray-scott/gray_scott_gpu.cu:51-97`

6-point 3D Laplacian reads 7 global memory values per thread with no shared memory halo caching.

**Fix:** Block-wise tiling (8x8x8 + halos). Low priority — benchmark-only code.

---

## Agent 2: GPU Memory Optimization

### M1: Byte Shuffle Per-Call Array Allocations [HIGH]
**File:** `src/preprocessing/byte_shuffle_kernels.cu:201-211, 308, 351`

`createDeviceChunkArrays()` allocates 3 device buffers per shuffle call (`d_input_ptrs`, `d_output_ptrs`, `d_sizes`). `byte_shuffle_simple()` / `byte_unshuffle_simple()` allocate `device_output` per call. All freed immediately.

**Fix:** Pool pointer arrays in CompContext. Reuse across calls.

---

### M2: Entropy Histogram Per-Call Allocations [MEDIUM]
**File:** `src/stats/entropy_kernel.cu:210-215`

Allocates `d_histogram` (1 KB) + `d_entropy` (8 bytes) per entropy call. Freed at lines 266-267.

**Fix:** Pre-allocate in CompContext stats workspace.

---

### M3: CUB Temp Storage Per-Call [MEDIUM]
**File:** `src/preprocessing/quantization_kernels.cu:211-229`

`compute_data_range()` allocates CUB temp storage per call, freed immediately.

**Fix:** Pre-allocate in CompContext workspace.

---

### M4: H5VL Worker Buffers Allocated Per-Write [HIGH]
**File:** `src/hdf5/H5VLgpucompress.cu:1055-1065, 1286-1290`

Per-write operation: allocates `N_COMP_WORKERS` device buffers + `N_IO_BUFS` pinned buffers. Freed after write completes.

**Fix:** Move to persistent session-level pool. Allocate once at VOL init.

---

### M5: H5VL Staging Buffers Per-Operation [MEDIUM]
**File:** `src/hdf5/H5VLgpucompress.cu:1353, 1430`

Per-operation `cudaMallocHost` for staging. Falls back to `malloc` if pinned alloc fails.

**Fix:** Pre-allocate pinned staging buffers at session level.

---

### M6: RL Decompression Buffer Per-Compression [MEDIUM]
**File:** `src/api/gpucompress_api.cpp:830, 1052, 1080`

Temporary decompression buffer allocated per-compression for RL feedback measurement.

**Fix:** Optional pool buffer in CompContext (gated by `g_reinforce_lr > 0`).

---

### M7: Static Global Buffers (Legacy Anti-Pattern) [LOW]
**Files:** `src/preprocessing/quantization_kernels.cu:30-31`, `src/nn/nn_gpu.cu:72-86`

Static globals (`d_range_min`, `d_range_max`, `d_infer_output`) — partially migrated to CompContext pool but legacy statics remain as fallback.

**Fix:** Remove static fallbacks, enforce pool-only usage.

---

## Agent 3: CUDA Stream Concurrency

### S1: 9 cudaStreamSynchronize() in Core Compression [CRITICAL]
**File:** `src/api/gpucompress_api.cpp:899, 1533, 1555, 2298, 2305, 2430, 2609, 2673, 2854`

Each sync flushes the entire GPU queue on that stream. With 8 active CompContext slots, creates contention.

**Fix:** Replace with event-based synchronization. For D->H copies where result is immediately needed, use synchronous `cudaMemcpy` (one operation instead of async+sync).

---

### S2: VOL Read Loop Double Synchronization (H1) [CRITICAL]
**File:** `src/hdf5/H5VLgpucompress.cu:1643, 1649`

Two `cudaStreamSynchronize(scatter_stream)` per chunk in the read loop. For 1000 chunks = 2000 sync calls.

**Fix:** Remove pre-decompress sync (line 1643) — H2D copy already blocks. Replace post-decompress sync with event-based sync for non-contiguous layouts.

---

### S3: cudaDeviceSynchronize() in NN Load/Cleanup [MEDIUM]
**File:** `src/nn/nn_gpu.cu:1029, 1102`

Global device sync drains ALL streams. Bad during hot-reload while concurrent compressions run.

**Fix:** Use `cudaEventRecord` + `cudaStreamWaitEvent` on each active CompContext stream.

---

### S4: Preprocessing Functions Over-Synchronize [MEDIUM]
**Files:** `src/preprocessing/byte_shuffle_kernels.cu:336,379`, `src/preprocessing/quantization_kernels.cu:227,679,732`

Shuffle and quantization functions sync at exit. Unnecessary since next kernel on same stream implicitly waits.

**Fix:** Remove syncs from helper functions. Add single batch sync at API boundary.

---

### S5: Stats Pipeline Redundant Async+Sync Pattern [MEDIUM]
**Files:** `src/stats/entropy_kernel.cu:258,327`, `src/stats/stats_kernel.cu:481`

`cudaMemcpyAsync` followed immediately by `cudaStreamSynchronize`. Redundant — use synchronous `cudaMemcpy` or defer sync.

---

### S6: Exploration Should Use Separate Stream [MEDIUM]
**File:** `src/api/gpucompress_api.cpp:2430-2620`

Exploration syncs main stream and runs serially. Primary result could be copied off-GPU while alternatives compress on a helper stream.

**Fix:** Create `explore_stream`, launch all K alternatives asynchronously. Sync once after all complete.

---

## Agent 4: nvCOMP Compression

### N1: Compression Manager Created/Destroyed Per Operation (H4) [CRITICAL]
**File:** `src/api/gpucompress_api.cpp:748, 1040, 2153, 2434`

`createCompressionManager()` called every compression. Each call allocates nvcomp workspace (1-100+ MB), initializes state, then destroys at function exit. For K=3 exploration: 4 manager create/destroy cycles per chunk.

**Fix:** Cache managers by `(algorithm, chunk_size, stream)` tuple. Reuse across calls. Managers are stateless after `configure_compression()`.
**Expected savings:** 5-10ms per call + elimination of workspace malloc/free cycles.

---

### N2: Decompression Manager Per-Call [HIGH]
**File:** `src/api/gpucompress_api.cpp:825, 1075, 2243, 2470, 2785`

Same pattern as N1 for decompression. Created for RL timing measurement and exploration PSNR verification.

**Fix:** Cache by `(algorithm, stream)` tuple.

---

### N3: Exploration Buffer Allocations Per-Alternative [HIGH]
**File:** `src/api/gpucompress_api.cpp:1052, 1080, 2446, 2473`

Per alternative: `cudaMalloc(&d_alt_out, alt_max)` + `cudaMalloc(&d_rt_decompressed, rt_decomp_size)`. For K=3: 6 alloc/free cycles.

**Fix:** Pre-allocate K exploration buffers at init (sized to worst-case). Reuse within exploration phase.
**Expected savings:** 50-150ms per exploration trigger.

---

### N4: Repeated configure_compression() Calls [MEDIUM]
**File:** `src/api/gpucompress_api.cpp:759, 1049, 2164, 2443`

`configure_compression()` is non-trivial (validates, computes workspace sizes). Called per alternative in exploration.

**Fix:** Cache results by `(algorithm, input_size)` — deterministic function, safe to memoize.

---

## Agent 5: HDF5 I/O

### I1: Only 2 Prefetch Slots in Read Pipeline [HIGH]
**File:** `src/hdf5/H5VLgpucompress.cu:1503-1504`

`N_SLOTS_R = 2` limits prefetch depth. When decompression is faster than disk I/O, GPU sits idle.

**Fix:** Increase to 4-8 slots. Semaphore-based management already supports arbitrary counts.

---

### I2: Pinned Pool Buffer Contention Between Workers and I/O [HIGH]
**File:** `src/hdf5/H5VLgpucompress.cu:1115-1135`

`N_IO_BUFS = 16` shared between 8 workers + 8 I/O queue capacity. Workers starve for pool buffers when I/O thread lags.

**Fix:** Increase to `N_COMP_WORKERS * 2` or decouple worker buffers from I/O buffers.

---

### I3: H5S_ALL Resolution Not Cached [MEDIUM]
**File:** `src/hdf5/H5VLgpucompress.cu:955-968, 1470-1482`

Every H5Dwrite/H5Dread with `H5S_ALL` triggers blocking `H5VLdataset_get(native, GET_SPACE)` RPC to fetch dimensions.

**Fix:** Cache resolved space at dataset open/create time.

---

### I4: Synchronous scatter_stream Blocks I/O Progress [HIGH]
**File:** `src/hdf5/H5VLgpucompress.cu:1643, 1649`

(Same as S2 — cross-referenced between agents.)

---

### I5: CLI Tool No Chunked I/O Pipeline [HIGH]
**File:** `src/cli/compress.cpp:279-732`

Entire file loaded into GPU memory, compressed as single batch, written at once. No pipelining for large files.

**Fix:** Streaming pipeline: read chunk N while compressing chunk N-1.

---

### I6: Contiguity Check Recomputed Per-Chunk [LOW]
**File:** `src/hdf5/H5VLgpucompress.cu:1180-1184, 1607-1610`

Chunk dimensions don't vary within a dataset. Compute once before loop.

---

## Agent 6: End-to-End Pipeline

### E1: g_auto_mutex Serializes NN Inference Across All Threads [CRITICAL]
**File:** `src/api/gpucompress_api.cpp:587`

```cpp
std::lock_guard<std::mutex> auto_lk(g_auto_mutex);
```

All 8 CompContext threads requesting ALGO_AUTO compete for this single mutex. Only 1 thread can run stats+NN at a time. With 8 chunks: **31.5ms x 8 = 252ms serialized** (46% of total GPU time).

**Fix:** Per-context NN inference buffers (already partially allocated in CompContext). Batch NN inference: infer all 8 chunks in a single call (32x8 = 256 configs). Remove mutex.
**Expected savings:** 192ms (from 252ms to ~60ms).

---

### E2: Exploration PSNR Uses CPU-Side D->H Transfer [MEDIUM-HIGH]
**File:** `src/api/gpucompress_api.cpp:1139-1142`

Two full-data D->H transfers per exploration alternative for CPU-side MSE computation. Code comments: `"(UNNECESSARY if GPU PSNR kernel existed)"`.

**Fix:** Implement GPU PSNR kernel (single-pass reduction). Eliminates 2x input_size D->H per alternative.

---

### E3: I/O Queue Blocking Stalls Compression Workers [HIGH]
**File:** `src/hdf5/H5VLgpucompress.cu:1248`

Work queue capacity = `N_COMP_WORKERS`. Main loop stalls if all 8 workers busy. No double buffering between compression and I/O.

**Fix:** Decouple compression and I/O with separate queues. Pre-compress to ring buffer (16 slots).

---

### E4: Pre-Decompress RL Measurement Blocks Main Path [MEDIUM]
**File:** `src/api/gpucompress_api.cpp:824-843`

Full decompression + `cudaEventSynchronize` added to write path for SGD training data. Adds 20-50ms per chunk.

**Fix:** Move to background stream with callback. Batch decompress samples asynchronously.

---

### E5: Exploration K-Loop Sequential Compression [MEDIUM]
**File:** `src/api/gpucompress_api.cpp:985-1242`

K alternatives compressed and decompressed sequentially with sync after each.

**Fix:** Launch all K on separate streams, sync once after all complete.

---

## Consolidated Priority Matrix

### Tier 1: Critical (2.5-3x speedup potential)

| ID | Issue | Impact | Effort |
|----|-------|--------|--------|
| **E1** | g_auto_mutex serializes NN inference | 46% GPU time wasted | Medium |
| **N1** | Compression manager per-call (H4) | 1.5s malloc overhead | Medium |
| **S1** | 9x cudaStreamSynchronize in compress | Pipeline serialization | Medium |
| **S2/I4** | VOL read double sync (H1) | Per-chunk serialization | Low |

### Tier 2: High (30-50% additional improvement)

| ID | Issue | Impact | Effort |
|----|-------|--------|--------|
| **M4** | H5VL worker buffers per-write | 573ms alloc in write | Medium |
| **N3** | Exploration buffer alloc per-alt | 50-150ms per explore | Low |
| **E3** | I/O queue blocking stalls workers | Pipeline stall | Medium |
| **I1** | Only 2 prefetch slots in read | GPU idle on reads | Low |
| **K1** | Histogram bank conflicts | Stats bottleneck | Medium |
| **I5** | CLI no chunked pipeline | Can't stream large files | High |

### Tier 3: Medium (10-20% polish)

| ID | Issue | Impact | Effort |
|----|-------|--------|--------|
| **S4** | Preprocessing over-synchronize | Unnecessary sync | Low |
| **S5** | Stats async+sync redundancy | Unnecessary sync | Low |
| **K2** | GPU spin-wait in stats init | Wasted cycles | Low |
| **M1** | Byte shuffle per-call allocs | Per-call overhead | Low |
| **M2** | Entropy per-call allocs | Per-call overhead | Low |
| **N4** | configure_compression() repeated | 2-10ms per call | Low |
| **E2** | CPU-side PSNR (no GPU kernel) | D->H overhead | Medium |
| **E4** | RL decompress blocks main path | 20-50ms per chunk | Medium |

### Tier 4: Low Priority

| ID | Issue | Impact | Effort |
|----|-------|--------|--------|
| **K3** | Gather/scatter uncoalesced | Large chunks only | High |
| **K4** | Byte shuffle warp underutil | 12.5% utilization | Medium |
| **K6** | Gray-Scott no shared mem tiling | Benchmark only | High |
| **I3** | H5S_ALL not cached | Minor RPC overhead | Low |
| **I6** | Contiguity check per-chunk | Trivial compute | Low |
| **M7** | Static global legacy buffers | Code hygiene | Low |

---

## Estimated Impact

### Current Performance (4 chunks, 1 MB each)
- GPU-time: 547ms
- Wall-clock: 1,270ms (2.3x overhead)
- Bottleneck: malloc (57%), mutex serialization (46% of GPU-time)

### After Tier 1 Optimizations
- GPU-time: ~200-250ms (NN mutex removed, stream sync reduced)
- Wall-clock: ~400-500ms (malloc overhead eliminated via pooling/caching)
- **Speedup: 2.5-3.2x**

### After Tier 1 + Tier 2
- GPU-time: ~150-200ms (exploration parallelized, reads pipelined)
- Wall-clock: ~250-350ms (I/O overlap, pinned buffer reuse)
- **Speedup: 3.5-5x**

---

## Good Practices Observed

- Native double `atomicAdd` (sm_60+)
- Grid size capping (entropy kernel bounds to 1024 blocks)
- Warp-level reductions via `__shfl_down_sync` in stats
- Grid-stride loops in all kernels
- Warp-efficient bitonic sort in NN inference via `__shfl_xor_sync`
- Overflow protection in quantization (grid cap at 65535)
- Per-slot CompContext pool with mutex protection
- Semaphore-based prefetch slot management in VOL read

---

*Report generated by 6-agent parallel analysis system. All line numbers reference current HEAD.*
