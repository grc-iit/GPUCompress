# Benchmark Metrics Enhancement Plan

## Motivation

The current benchmarks (Gray-Scott and VPIC) measure coarse wall-clock times
(`write_ms`, `read_ms`) and a few flags (`sgd_fired`, `exploration_triggered`).
This makes it impossible to answer questions like:

- How much time does NN inference add per chunk?
- What fraction of write time is preprocessing vs. compression vs. exploration?
- How expensive is an SGD weight update?
- What is the per-chunk compression ratio distribution?

This plan adds fine-grained per-chunk timing and ratio metrics to the library's
diagnostic infrastructure and updates both benchmarks to report them.

---

## Current State

### `gpucompress_stats_t` (per-call, in `include/gpucompress.h:122-139`)

Already has:
- `actual_comp_time_ms` -- primary compression kernel time
- `predicted_ratio`, `predicted_comp_time_ms` -- NN predictions
- `compression_ratio` -- actual ratio
- `sgd_fired`, `exploration_triggered` -- flags
- `nn_original_action`, `nn_final_action`

### `gpucompress_chunk_diag_t` (per-chunk history, in `include/gpucompress.h:530-535`)

Currently only stores:
```c
typedef struct {
    int nn_action;
    int nn_original_action;
    int exploration_triggered;
    int sgd_fired;
} gpucompress_chunk_diag_t;
```

**Problem**: No timing data, no ratios, no NN prediction accuracy data.
The benchmark can only aggregate flags, not timing breakdowns.

### Benchmark CSV columns (current)

**Aggregate CSV** (`benchmark_*_vol.csv`):
```
phase, L, steps, F, k, chunk_z, n_chunks,
sim_ms, write_ms, read_ms, file_mib, orig_mib, ratio,
write_mibps, read_mibps, mismatches, sgd_fires, explorations
```

**Chunk CSV** (`benchmark_*_vol_chunks.csv`):
```
phase, chunk, nn_action, explored, sgd_fired
```

---

## Plan

### Step 1: Extend `gpucompress_chunk_diag_t`

**File**: `include/gpucompress.h`

Add timing and ratio fields to the per-chunk diagnostic struct:

```c
typedef struct {
    /* Existing fields */
    int    nn_action;
    int    nn_original_action;
    int    exploration_triggered;
    int    sgd_fired;

    /* NEW: per-chunk timing breakdown (ms, 0.0 if not applicable) */
    float  nn_inference_ms;       /* stats kernels + NN forward pass */
    float  preprocessing_ms;      /* quantization + byte shuffle */
    float  compression_ms;        /* primary nvCOMP kernel only */
    float  exploration_ms;        /* exploration loop (0 if not triggered) */
    float  sgd_update_ms;         /* SGD weight update (0 if not fired) */

    /* NEW: per-chunk ratio and prediction accuracy */
    float  actual_ratio;          /* input_size / compressed_size */
    float  predicted_ratio;       /* NN-predicted ratio (0 if not ALGO_AUTO) */
} gpucompress_chunk_diag_t;
```

### Step 2: Instrument timing in the compress paths

Both `gpucompress_compress` (host path, ~line 390) and `gpucompress_compress_gpu`
(GPU path, ~line 1580) follow the same structure. Add CUDA event timing around
each stage. Use the per-context event pairs (`ctx->t_start`/`ctx->t_stop` for
GPU path, `g_t_start`/`g_t_stop` for host path).

#### 2a: NN inference timing

**Where**: Around the `runStatsKernelsNoSync` + `runNNFusedInference` block.

Host path (`gpucompress_compress`):
- Start event before `runStatsKernelsNoSync` (~line 444)
- Stop event after `runNNFusedInference` returns (~line 455)
- Store elapsed as `nn_inference_ms`

GPU path (`gpucompress_compress_gpu`):
- Same pattern around ~lines 1652-1663

#### 2b: Preprocessing timing

**Where**: Around the quantization + shuffle block.

Host path:
- Start event before quantization check (~line 513)
- Stop event after shuffle completes (~line 554)
- Store elapsed as `preprocessing_ms`

GPU path:
- Same pattern around ~lines 1700-1750 (approximate)

Note: If neither quantization nor shuffle is applied, `preprocessing_ms = 0`.

#### 2c: Compression timing (already exists)

The primary compression kernel is already timed via `primary_comp_time_ms`.
Just propagate this value to the chunk diag struct.

#### 2d: Exploration timing

**Where**: Around the exploration loop.

Host path:
- Start event before the exploration `if` block (~line 686)
- Stop event after exploration completes (~line 1035)
- Store elapsed as `exploration_ms`

GPU path:
- Same pattern around ~lines 1850-2190 (approximate)

Note: If exploration is not triggered, `exploration_ms = 0`.

#### 2e: SGD update timing

**Where**: Around `runNNSGD`.

Host path:
- Start event before `runNNSGD` (~line 1057)
- Stop event after it returns (~line 1062)
- Store elapsed as `sgd_update_ms`

GPU path:
- Same pattern around ~lines 2180-2195 (approximate)

Note: If SGD does not fire, `sgd_update_ms = 0`.

### Step 3: Populate chunk diag with new fields

**File**: `src/api/gpucompress_api.cpp`

In both chunk history recording blocks (host ~lines 1096-1113, GPU ~lines 2227-2244),
add the new fields:

```c
h->nn_inference_ms       = nn_inference_ms;
h->preprocessing_ms      = preprocessing_ms;
h->compression_ms        = primary_comp_time_ms;
h->exploration_ms        = exploration_ms;
h->sgd_update_ms         = sgd_update_ms;
h->actual_ratio          = (float)input_size / (float)(compressed_size > 0 ? compressed_size : 1);
h->predicted_ratio       = predicted_ratio;
```

### Step 4: Update benchmark chunk CSV

**Files**:
- `tests/benchmarks/benchmark_grayscott_vol.cu`
- `tests/benchmarks/vpic_benchmark_deck.cxx`

Update `write_chunk_csv` to emit the new columns:

```csv
phase, chunk, nn_action, explored, sgd_fired,
nn_inference_ms, preprocessing_ms, compression_ms, exploration_ms, sgd_update_ms,
actual_ratio, predicted_ratio
```

### Step 5: Compute and emit aggregate timing metrics

**Files**: Same benchmark files.

After reading all chunk diags, compute:

| Aggregate metric          | Derivation                                               |
|---------------------------|----------------------------------------------------------|
| `total_nn_inference_ms`   | Sum of `nn_inference_ms` across all chunks               |
| `total_preprocessing_ms`  | Sum of `preprocessing_ms` across all chunks              |
| `total_compression_ms`    | Sum of `compression_ms` across all chunks                |
| `total_exploration_ms`    | Sum of `exploration_ms` across all chunks                |
| `total_sgd_ms`            | Sum of `sgd_update_ms` across all chunks                 |
| `overhead_ms`             | `write_ms - max(per-worker cumulative)` (VOL/HDF5/memcpy)|
| `ratio_min`               | Min `actual_ratio` across chunks                         |
| `ratio_max`               | Max `actual_ratio` across chunks                         |
| `ratio_stddev`            | Stddev of `actual_ratio` across chunks                   |
| `mean_prediction_error`   | Mean of `abs(predicted - actual) / actual` (MAPE)        |

Add these to the aggregate CSV:

```csv
phase, ..., (existing columns), ...,
total_nn_inference_ms, total_preprocessing_ms, total_compression_ms,
total_exploration_ms, total_sgd_ms, overhead_ms,
ratio_min, ratio_max, ratio_stddev, mean_prediction_error_pct
```

### Step 6: Update `PhaseResult` struct

**Files**: Both benchmark files.

Add fields to `PhaseResult`:

```c
struct PhaseResult {
    /* ... existing fields ... */

    /* NEW: timing breakdown (summed across chunks) */
    double total_nn_inference_ms;
    double total_preprocessing_ms;
    double total_compression_ms;
    double total_exploration_ms;
    double total_sgd_ms;

    /* NEW: ratio distribution */
    double ratio_min;
    double ratio_max;
    double ratio_stddev;

    /* NEW: NN accuracy */
    double mean_prediction_error_pct;  /* MAPE */
};
```

### Step 7: Update the summary table

Update `print_summary_table` to show the timing breakdown:

```
Phase       | Sim   | Write | Read  | Ratio | File  | Comp  | Preproc | NN Inf | Expl  | SGD   |
            | (ms)  |(MiB/s)|(MiB/s)|       |(MiB)  | (ms)  |  (ms)   | (ms)   | (ms)  | (ms)  |
no-comp     |  ...  |  ...  |  ...  |  ...  |  ...  |  ...  |   0.0   |   0.0  |  0.0  |  0.0  |
static      |  ...  |  ...  |  ...  |  ...  |  ...  |  ...  |  12.3   |   0.0  |  0.0  |  0.0  |
nn          |  ...  |  ...  |  ...  |  ...  |  ...  |  ...  |   5.1   |   8.2  |  0.0  |  0.0  |
nn-rl       |  ...  |  ...  |  ...  |  ...  |  ...  |  ...  |   5.1   |   8.2  |  0.0  |  3.4  |
nn-rl+exp50 |  ...  |  ...  |  ...  |  ...  |  ...  |  ...  |   5.1   |   8.2  | 45.0  |  3.4  |
```

---

## Implementation Notes

### Event reuse

The host path has one pair of global timing events (`g_t_start`/`g_t_stop`).
The GPU path has per-context events (`ctx->t_start`/`ctx->t_stop`). Since we
need to time multiple stages sequentially within a single compress call, we can
reuse the same event pair for each stage (record start, record stop, read
elapsed, move to next stage). No new event objects needed.

### Overhead calculation

`overhead_ms` represents time spent in HDF5/VOL plumbing, D2H/H2D memcpy, and
thread coordination -- everything not captured by the per-chunk timers. Since
VOL uses `N_COMP_WORKERS` parallel threads, the sum of per-chunk times will
exceed wall time. The correct overhead calculation is:

```
overhead_ms = write_ms - max_worker_cumulative_ms
```

where `max_worker_cumulative_ms` is the longest-running worker's total
(inference + preprocess + compress + explore + sgd). Since we don't currently
track which chunk went to which worker, a simpler approximation is:

```
overhead_ms_approx = write_ms - (total_compression_ms / N_COMP_WORKERS)
```

This is an approximation. A more precise approach would require per-worker
tracking (out of scope for this change).

### Static and no-comp phases

For `no-comp` and `static` phases, NN/SGD/exploration fields will all be zero.
`preprocessing_ms` will be non-zero for `static` (shuffle is applied).
`compression_ms` will be non-zero for `static` (lz4). This naturally produces
the right breakdown.

### Backward compatibility

The `gpucompress_chunk_diag_t` struct is extended with new fields at the end.
Existing code that reads only the first 4 fields will still work (C struct
layout). The struct size increases from 16 bytes to 44 bytes. No ABI break
for callers that use `gpucompress_get_chunk_diag()` since the function copies
into a caller-provided struct.

---

## File Change Summary

| File | Change |
|------|--------|
| `include/gpucompress.h` | Extend `gpucompress_chunk_diag_t` with 7 new fields |
| `src/api/gpucompress_api.cpp` | Add event timing around 4 stages in both compress paths; populate new chunk diag fields |
| `tests/benchmarks/benchmark_grayscott_vol.cu` | Update `PhaseResult`, chunk CSV, aggregate CSV, summary table |
| `tests/benchmarks/vpic_benchmark_deck.cxx` | Same updates as Gray-Scott benchmark |

Estimated: ~200 lines of new/modified code across 4 files.

---

## Metrics Summary Table

| Metric | Granularity | Source | Currently Available |
|--------|-------------|--------|---------------------|
| Simulation time | per-phase | benchmark timer | Yes |
| Total write time | per-phase | benchmark timer | Yes |
| Total read time | per-phase | benchmark timer | Yes |
| File size / ratio | per-phase | benchmark stat() | Yes |
| Write/Read throughput | per-phase | derived | Yes |
| Mismatches | per-phase | GPU kernel | Yes |
| SGD fire count | per-phase | chunk diag sum | Yes |
| Exploration count | per-phase | chunk diag sum | Yes |
| NN action | per-chunk | chunk diag | Yes |
| NN inference time | per-chunk | CUDA events | **No -- add** |
| Preprocessing time | per-chunk | CUDA events | **No -- add** |
| Compression kernel time | per-chunk | CUDA events | Exists in `stats_t`, **not in chunk diag** |
| Exploration time | per-chunk | CUDA events | **No -- add** |
| SGD update time | per-chunk | CUDA events | **No -- add** |
| Per-chunk actual ratio | per-chunk | computed | **No -- add** |
| Per-chunk predicted ratio | per-chunk | NN output | **No -- add** |
| Ratio min/max/stddev | per-phase | derived from chunks | **No -- add** |
| Mean prediction error | per-phase | derived from chunks | **No -- add** |
| Total compression time | per-phase | sum of chunk times | **No -- add** |
| HDF5/VOL overhead | per-phase | derived | **No -- add** |
