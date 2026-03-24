# Kendall Tau Ranking Quality Measurement

## Context

The NN's primary job is ranking ~32 compressor configs and picking the best one per chunk. MAPE/MAE/R² measure prediction accuracy but not **decision quality** — whether the NN correctly identifies the top algorithm. Kendall's tau directly measures ranking quality. The SC reviewer identified this as essential for the "online learning" narrative.

At milestone timesteps, we perform a separate profiling pass: compress+decompress each chunk with all active configs (16 for lossless), compare predicted vs actual cost ranking, compute tau/top-1/regret. Milestones: T0, T5, T10, then every 10% of total timesteps (T10%, T20%, ... T100%). For ≥10 timesteps this gives ~13 milestones; for <10 timesteps, every timestep is a milestone.

## Design

### Approach: Separate Profiling Pass (No SGD Contamination)

After the normal timestep completes (H5Dwrite → H5Dread → verify → collect MAPE/MAE/R²), at milestones:

1. **Snapshot** all chunk diagnostics into a local array (before profiling pollutes chunk history)
2. For each chunk (raw GPU data still valid via `d_v`/`d_fields`/`d_data`):
   - Get NN's predicted ranking from the **snapshotted** diagnostics
   - Compress with each of 16 lossless configs via `gpucompress_compress_gpu()` with explicit algo (NOT ALGO_AUTO — so SGD/exploration code is never entered)
   - Pass non-null `gpucompress_stats_t*` to get `stats.actual_comp_time_ms` (internal CUDA events)
   - Decompress each via `gpucompress_decompress_gpu()`, timed with `cudaDeviceSynchronize()` + wall-clock
   - Run **3 repeats** per config, take **median** timing (addresses sub-ms measurement noise)
   - Apply **policy clamps** before cost: `comp_ms = max(5.0, comp_ms)`, `decomp_ms = max(5.0, decomp_ms)`, `ratio = min(100.0, ratio)`
   - Compute actual cost: `w0*comp_ms + w1*decomp_ms + w2*chunk_bytes/(ratio*bw)`
   - Compare predicted vs actual ranking → τ-b, top-1, regret
3. Flush nvcomp manager cache after profiling pass
4. Write per-chunk results to CSV, print aggregated summary

**Key safety properties:**
- No SGD contamination: explicit algo configs skip the SGD/exploration block (guarded by `cfg.algorithm == GPUCOMPRESS_ALGO_AUTO` at line 455)
- Chunk history pollution handled: diagnostics snapshotted before profiling begins
- `top_actions == nullptr` for non-AUTO phases: set `predicted_ranking_count = 0`, profiler skips these

### Step 1: Store Predicted Ranking in Chunk Diagnostics

The NN inference kernel already sorts all 32 configs by predicted cost into `d_fused_top_actions`. We just need to save this into chunk history.

**`include/gpucompress.h`** — Add to `gpucompress_chunk_diag_t`:
```c
int predicted_ranking[32];    /* action IDs sorted by predicted cost (best first) */
int predicted_ranking_count;  /* number of valid entries (0 if non-AUTO path) */
```

**`src/api/internal.hpp`** — Add same fields to `ChunkDiagInput`

**`src/api/gpucompress_compress.cpp`** — In `gpucompress_compress_with_action_gpu()`:
- When `top_actions != nullptr` (AUTO path): copy into `ChunkDiagInput::predicted_ranking[]`, set count=32
- When `top_actions == nullptr` (non-AUTO): set `predicted_ranking_count = 0`

**`src/api/gpucompress_diagnostics.cpp`** — In `recordChunkDiagnostic()`, copy `predicted_ranking[]` and count into history entry

### Step 2: Shared Profiling Utility

**New file: `benchmarks/kendall_tau_profiler.cuh`** (CUDA header, included by `.cu` drivers)

For VPIC (`.cxx` file), create a thin wrapper: `benchmarks/vpic-kokkos/vpic_ranking_profiler.cu` that includes the `.cuh` and exposes a C-linkage function.

Core function:
```cpp
int run_ranking_profiler(
    const void* d_data,       // GPU original data
    size_t total_bytes,       // full dataset size
    size_t chunk_bytes,       // per-chunk size
    double error_bound,       // 0 for lossless
    float w0, float w1, float w2, float bw_bytes_per_ms,
    int n_repeats,            // timing repeats per config (default 3)
    FILE* csv,                // output CSV
    const char* phase_name,
    int timestep,
    RankingMilestoneResult* out
);
```

**Execution flow per milestone:**
1. `n_chunks = total_bytes / chunk_bytes`
2. Snapshot all chunk diagnostics: `std::vector<gpucompress_chunk_diag_t> diags(n_chunks)`
3. Determine active configs: lossless → filter using `decodeAction(id).use_quantization == false`
4. Allocate GPU buffers once: `d_comp_buf` (max compressed size), `d_decomp_buf` (chunk_bytes)
5. **Warmup pass**: compress one mid-dataset chunk with each of 16 configs (cold JIT/manager setup)
6. **Config-major iteration** (all chunks for config A, then config B — keeps nvcomp manager cache warm):
   - For each active config:
     - For each chunk:
       - `d_chunk = (uint8_t*)d_data + ci * chunk_bytes`
       - Build `gpucompress_config_t` with `algorithm = algo_idx + 1`, `preprocessing = shuffle ? SHUFFLE_4 : 0`
       - Run `n_repeats` compress+decompress cycles, take **median** comp_ms and decomp_ms
       - Compression: `gpucompress_compress_gpu(...)` with non-null `stats` → `stats.actual_comp_time_ms`
       - Decompression: `cudaDeviceSynchronize()` → wall-clock → `gpucompress_decompress_gpu(...)` → `cudaDeviceSynchronize()` → wall-clock
       - Apply policy clamps, compute actual_cost
       - Store in `results[ci][config_idx] = {action_id, ratio, comp_ms, decomp_ms, cost}`
7. For each chunk:
   - Sort configs by actual_cost → actual ranking
   - Get predicted ranking from `diags[ci].predicted_ranking[]`, filter to active configs using `decodeAction().use_quantization`
   - Compute **Kendall τ-b** (O(n²), n=16, tie-corrected)
   - Compute **top-1 accuracy**: `actual_ranking[0] == predicted_ranking[0]`
   - Compute **top-1 regret**: `actual_cost[predicted_best] / actual_cost[oracle_best]`
   - Write CSV row
8. Aggregate: mean τ ± std, top-1 fraction, mean regret
9. Flush nvcomp manager cache, free GPU buffers

### Step 3: Milestone Definition

```cpp
static bool is_ranking_milestone(int t, int total) {
    if (total < 10) return true;  // profile every timestep for short runs
    int last = total - 1;
    // T0, T5, T10 (early learning), then every 10% of total
    if (t == 0 || t == 5 || t == 10) return true;
    for (int pct = 10; pct <= 100; pct += 10) {
        if (t == last * pct / 100) return true;
    }
    return false;
}
```

For ≥10 timesteps: T0, T5, T10, T10%, T20%, T30%, T40%, T50%, T60%, T70%, T80%, T90%, T100% (~13 milestones, some may overlap with T5/T10). For <10 timesteps: every timestep is profiled.

### Step 4: Integrate into Benchmark Drivers

**Gray-Scott** (`benchmarks/grayscott/grayscott-benchmark.cu`):
- `#include "kendall_tau_profiler.cuh"`
- Open `ranking_csv` alongside `ts_csv` and `tc_csv`
- At milestone check (~line 1753), after existing chunk CSV writes:
  ```cpp
  if (is_ranking_milestone(t, timesteps) && ranking_csv) {
      run_ranking_profiler(d_v, total_bytes, chunk_bytes, error_bound,
                           w0, w1, w2, bw, 3, ranking_csv, phase_name, t, &tau_result);
  }
  ```

**VPIC** (`benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx`):
- Create `benchmarks/vpic-kokkos/vpic_ranking_profiler.cu` with C-linkage wrapper
- Link into VPIC build
- At milestone (~line 1373), call the wrapper function

**SDRBench** (`benchmarks/sdrbench/generic-benchmark.cu`):
- `#include "kendall_tau_profiler.cuh"`
- At milestone (~line 1535), call `run_ranking_profiler(d_data, ...)`

### Step 5: CSV Schema

File: `benchmark_*_ranking.csv`

Per-chunk rows:
```
phase,timestep,chunk,n_active_configs,kendall_tau_b,top1_correct,top1_regret,predicted_best,actual_best,predicted_best_cost,actual_best_cost,profiling_ms
```

### Step 6: Visualization

**`benchmarks/visualize.py`** — Add `make_ranking_quality_figure(csv_path, output_path)`:
- 3-panel plot: τ-b, top-1 accuracy, top-1 regret over milestone timesteps
- Aggregate per-chunk values to mean ± std per milestone per phase
- One line per phase (nn-rl, nn-rl+exp50)
- Reference lines: τ=0 (random), τ=1 (perfect); regret=1.0 (optimal)
- Following existing SC publication style

**`benchmarks/plots/generate_dataset_figures.py`** — Wire as `5e_ranking_quality.png`

## Files to Modify

| File | Change |
|------|--------|
| `include/gpucompress.h` | Add `predicted_ranking[32]`, `predicted_ranking_count` to `gpucompress_chunk_diag_t` |
| `src/api/internal.hpp` | Add same fields to `ChunkDiagInput` |
| `src/api/gpucompress_compress.cpp` | Copy `top_actions` into `ChunkDiagInput` (nullptr → count=0) |
| `src/api/gpucompress_diagnostics.cpp` | Copy ranking into history in `recordChunkDiagnostic()` |
| `benchmarks/kendall_tau_profiler.cuh` | **NEW** — shared profiling utility with Kendall τ-b computation |
| `benchmarks/vpic-kokkos/vpic_ranking_profiler.cu` | **NEW** — C-linkage wrapper for VPIC `.cxx` driver |
| `benchmarks/grayscott/grayscott-benchmark.cu` | Call profiler at milestones |
| `benchmarks/vpic-kokkos/vpic_benchmark_deck.cxx` | Call profiler at milestones via wrapper |
| `benchmarks/sdrbench/generic-benchmark.cu` | Call profiler at milestones |
| `benchmarks/visualize.py` | Add ranking quality plot |
| `benchmarks/plots/generate_dataset_figures.py` | Wire 5e plot |

## Review Findings Addressed

| # | Source | Issue | Resolution |
|---|--------|-------|------------|
| P0 | timing-auditor | `need_timing=false` when stats=NULL and learning disabled | Pass non-null `gpucompress_stats_t*`, read `stats.actual_comp_time_ms` |
| P0 | timing-auditor | External CUDA events can't bracket internal stream work | Use `stats.actual_comp_time_ms` for compression timing |
| P0 | timing-auditor | Cost formula missing policy clamps (5ms floor, 100x cap) | Apply clamps before cost computation |
| P1 | timing-auditor | No warmup for cold algorithms | Run throwaway compress per config before timed loop |
| P1 | timing-auditor | Decomp timing: no stats output | `cudaDeviceSynchronize()` + wall-clock bracketing |
| P1 | timing-auditor | Profiling appends to chunk history | Snapshot all diags into local vector first |
| P1 | timing-auditor | nvcomp manager cache thrash | Config-major iteration order + flush after profiling |
| C2 | gpucompress-reviewer | Profiling compressions pollute chunk history | Snapshot all diags into local vector first |
| C1 | gpucompress-reviewer | `top_actions == nullptr` for non-AUTO | Set `predicted_ranking_count = 0`, profiler checks before use |
| W1 | gpucompress-reviewer | VPIC `.cxx` can't include CUDA header | Separate `.cu` wrapper file for VPIC |
| P2 | timing-auditor | Filter predicted ranking robustly | Use `decodeAction().use_quantization` check |
| MF1 | sc-reviewer | Sub-ms measurement noise | 3 repeats per config, take median |
| MF2 | sc-reviewer | "top-3 regret" misnomer | Renamed to "top-1 regret" (cost of NN pick / oracle cost) |
| MF3 | sc-reviewer | Profiling overhead unquantified | Report `profiling_ms` in CSV |
| SC | sc-reviewer | Milestone deduplication | Dynamic milestones: T0,T5,T10 + every 10% for ≥10 timesteps |

## SC Paper Presentation (Recommended)

- **Figure A (headline):** 3-panel convergence (τ, top-1, regret) vs milestone. Lines for nn/nn-rl/nn-rl+exp50. ±1 std bands.
- **Figure B:** Regret CDF at T0 vs T100 — "X% of chunks within Y% of optimal"
- **Table:** Per-dataset summary: τ_T0, τ_T100, top1_T0, top1_T100, regret_T0, regret_T100, overhead_sec
- For n=16 configs, τ > 0.34 is significant at p < 0.01

## Verification

1. Build: `cmake --build build`
2. Run small GS: `L=64 CHUNK_MB=1 TIMESTEPS=10 RUNS=1 bash benchmarks/grayscott/run_gs_eval.sh`
3. Verify `benchmark_grayscott_ranking.csv` exists with 6 milestone × n_chunks rows
4. Verify τ ∈ [-1, 1], top1_correct ∈ {0, 1}, regret ≥ 1.0
5. Verify normal MAPE/MAE/R² unchanged (no SGD contamination — compare with/without profiling)
6. Run `generate_dataset_figures.py` → verify `5e_ranking_quality.png`
7. Repeat for small VPIC (`NX=32 TIMESTEPS=6`) and SDRBench
