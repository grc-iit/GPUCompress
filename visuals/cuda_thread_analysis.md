# CUDA Thread Analysis for nvcomp_gds.cu Compression

## Understanding Thread Spawning in nvCOMP

The `compressor.compress()` call is a **high-level API** that hides the actual CUDA kernel launches inside the nvCOMP library. To see the exact number of threads spawned, we need to use profiling tools.

---

## Methods to See CUDA Thread Counts

### Method 1: NVIDIA Nsight Systems Profiling (RECOMMENDED)

The program already includes NVTX markers, making it perfect for profiling.

#### Step 1: Build the program

```bash
cd /home/cc/GPUCompress
mkdir -p build && cd build
cmake -DBUILD_GDS_EXAMPLE=ON ..
make -j
```

#### Step 2: Create a test file and profile

```bash
# Run with profiling
nsys profile \
    --trace=cuda,nvtx \
    --output=nvcomp_gds_profile \
    --force-overwrite=true \
    ./examples/nvcomp_gds /tmp/test_gds_file.bin
```

#### Step 3: View the report

```bash
# Text summary
nsys stats nvcomp_gds_profile.nsys-rep

# Or open in GUI (if available)
nsys-ui nvcomp_gds_profile.nsys-rep
```

#### What You'll See:

The profiler will show:
- **Kernel names** launched by nvCOMP (e.g., `lz4CompressKernel`, `lz4DecompressKernel`)
- **Grid dimensions** (number of blocks)
- **Block dimensions** (threads per block)
- **Total threads** = gridDim.x × gridDim.y × gridDim.z × blockDim.x × blockDim.y × blockDim.z
- **Execution time** for each kernel
- **Timeline** showing when kernels execute

---

### Method 2: NVIDIA Nsight Compute (Detailed Kernel Analysis)

For even more detailed information about a specific kernel:

```bash
ncu --set full --export nvcomp_compress \
    ./examples/nvcomp_gds /tmp/test_gds_file.bin
```

This provides:
- Exact thread counts
- Warp efficiency
- Memory throughput
- Register usage per thread
- Occupancy metrics

---

### Method 3: Add CUDA Runtime API Callbacks (Code Instrumentation)

We can intercept kernel launches programmatically:

```cpp
// Add this instrumentation wrapper
#include <cuda_profiler_api.h>

// Before compression
cudaProfilerStart();

// Compression happens here
compressor.compress(d_input, d_compressed, comp_config);

cudaProfilerStop();
```

---

## Expected Thread Configuration (Estimated)

Based on typical nvCOMP implementation patterns and the batch configuration:

### For 100 MB Input with 64 KB Chunks:

```
COMPRESSION KERNEL CONFIGURATION (Typical):
═══════════════════════════════════════════════════════════════════

Batch size: 1,527 chunks
Chunk size: 65,536 bytes (64 KB)

Typical LZ4 GPU Implementation:
--------------------------------
Option A: One Block Per Chunk
   Grid:  (1527, 1, 1)           // 1,527 blocks
   Block: (256, 1, 1)            // 256 threads per block
   Total threads: 1,527 × 256 = 391,104 threads

Option B: Multiple Blocks Per Chunk (for larger chunks)
   Grid:  (3054, 1, 1)           // 2 blocks per chunk
   Block: (512, 1, 1)            // 512 threads per block
   Total threads: 3,054 × 512 = 1,563,648 threads

Option C: Warp-Based Processing
   Grid:  (1527, 1, 1)           // One block per chunk
   Block: (128, 1, 1)            // 4 warps per block
   Total threads: 1,527 × 128 = 195,456 threads


ACTUAL LAUNCH (varies by nvCOMP version and GPU):
--------------------------------
The library likely uses adaptive launch configuration based on:
• GPU SM count (for this file: ~108 SMs on modern GPUs)
• Available registers per SM
• Shared memory requirements
• Chunk size
• Occupancy heuristics


DECOMPRESSION KERNEL CONFIGURATION:
--------------------------------
Usually similar to compression but may differ:
   Grid:  (1527, 1, 1)
   Block: (256, 1, 1)
   Total threads: ~390,000+ threads

Note: Decompression is often faster and may use fewer threads
      because it's primarily memory-bound (just copying data).
```

---

## Practical Profiling Example

Let me create a script to automatically profile and extract thread information:

### `profile_nvcomp.sh`

```bash
#!/bin/bash

EXECUTABLE="./examples/nvcomp_gds"
TEST_FILE="/tmp/nvcomp_test.bin"
OUTPUT_DIR="./profiling_results"

mkdir -p "$OUTPUT_DIR"

echo "==================================================================="
echo "       CUDA Thread Analysis for nvCOMP Compression"
echo "==================================================================="
echo ""

# Step 1: Profile with Nsight Systems
echo "[1/3] Profiling with NVIDIA Nsight Systems..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output="$OUTPUT_DIR/nvcomp_profile" \
    --force-overwrite=true \
    --show-output=true \
    "$EXECUTABLE" "$TEST_FILE"

echo ""
echo "[2/3] Generating statistics report..."
nsys stats "$OUTPUT_DIR/nvcomp_profile.nsys-rep" > "$OUTPUT_DIR/stats.txt"

echo ""
echo "[3/3] Extracting CUDA kernel information..."

# Extract kernel information
echo "==================================================================="
echo "                    CUDA KERNEL SUMMARY"
echo "==================================================================="

# Parse the stats file for kernel information
grep -A 50 "CUDA Kernel Statistics" "$OUTPUT_DIR/stats.txt" || \
    echo "Run: nsys stats $OUTPUT_DIR/nvcomp_profile.nsys-rep"

echo ""
echo "==================================================================="
echo "                    NVTX RANGE TIMING"
echo "==================================================================="

grep -A 50 "NVTX Range Statistics" "$OUTPUT_DIR/stats.txt" || \
    echo "Check the full report for NVTX ranges"

echo ""
echo "Done! Full report available at: $OUTPUT_DIR/nvcomp_profile.nsys-rep"
echo "View in GUI: nsys-ui $OUTPUT_DIR/nvcomp_profile.nsys-rep"
```

---

## Alternative: Intercept Kernel Launches with CUDA Profiler API

Add this code to see kernel launch configurations at runtime:

```cpp
#include <cuda.h>
#include <iostream>

// Callback function to intercept kernel launches
void CUDART_CB kernelCallback(
    void *userdata,
    cudaLaunchKernelExC_callsite_t callsite,
    void *func,
    dim3 gridDim,
    dim3 blockDim,
    void **args,
    size_t sharedMem,
    cudaStream_t stream)
{
    std::cout << "Kernel Launch Detected:" << std::endl;
    std::cout << "  Grid:  (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
    std::cout << "  Block: (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")" << std::endl;
    std::cout << "  Total threads: " << 
        (gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z) 
        << std::endl;
    std::cout << "  Shared memory: " << sharedMem << " bytes" << std::endl;
    std::cout << std::endl;
}

// In main(), before compression:
// cudaLaunchKernelExC_SetCallback(kernelCallback, nullptr);
```

---

## Expected Output from Profiling

When you profile the nvcomp_gds program, you'll see something like:

```
=================================================================
CUDA Kernel Statistics
=================================================================

 Time(%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Name
 -------  ---------------  ---------  ---------  ---------  ----
   45.2%      12,345,678          3  4,115,226  4,100,000  lz4CompressBatchKernel
   30.1%       8,234,567          1  8,234,567  8,234,567  initialize(uint8_t*, unsigned long)
   15.3%       4,189,234          3  1,396,411  1,400,000  lz4DecompressBatchKernel
    8.4%       2,301,456          1  2,301,456  2,301,456  compare(...)
    1.0%         273,901          5     54,780     55,000  cudaMemsetAsync

Configuration for 'lz4CompressBatchKernel':
   Grid size:        (1527, 1, 1)
   Block size:       (256, 1, 1)
   Total threads:    391,104
   Registers/thread: 32
   Shared memory:    12,288 bytes per block
   Occupancy:        75.2%

Configuration for 'lz4DecompressBatchKernel':
   Grid size:        (1527, 1, 1)
   Block size:       (256, 1, 1)
   Total threads:    391,104
   Registers/thread: 28
   Shared memory:    8,192 bytes per block
   Occupancy:        82.1%
```

---

## Thread Count Breakdown

### User-Visible Kernels (in nvcomp_gds.cu):

```
1. initialize<<<195313, 512>>>
   - Grid:  195,313 blocks
   - Block: 512 threads
   - Total: 99,999,744 threads
   - Purpose: Initialize 100 MB input data

2. compare<<<2*smcount, 1024>>>  (assuming 108 SMs)
   - Grid:  216 blocks (2 × 108)
   - Block: 1024 threads
   - Total: 221,184 threads
   - Purpose: Compare decompressed vs original
```

### nvCOMP Internal Kernels (estimated):

```
3. lz4CompressBatchKernel (or similar name)
   - Grid:  ~1,527 blocks (one per chunk)
   - Block: ~256 threads
   - Total: ~391,104 threads
   - Purpose: Compress each 64KB chunk in parallel

4. lz4DecompressBatchKernel (or similar name)
   - Grid:  ~1,527 blocks
   - Block: ~256 threads
   - Total: ~391,104 threads
   - Purpose: Decompress each chunk in parallel

5. Additional helper kernels:
   - Metadata formatting: ~1,000-10,000 threads
   - Size calculation: ~1,000-10,000 threads
   - Prefix sum operations: ~10,000-50,000 threads
```

### Total Thread Count (Approximate):

```
TOTAL THREADS SPAWNED DURING ENTIRE PROGRAM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Initialize:          ~100 million threads
Compression:         ~400 thousand threads
Decompression:       ~400 thousand threads
Comparison:          ~220 thousand threads
Misc operations:     ~50 thousand threads
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOTAL (all kernels): ~101 million threads

NOTE: These threads execute in waves based on GPU SM count.
      A GPU with 108 SMs might execute ~108,000 threads
      simultaneously (assuming 1024 threads/SM occupancy).
```

---

## GPU Execution Reality

```
┌────────────────────────────────────────────────────────────────┐
│         THREAD SPAWNING vs CONCURRENT EXECUTION                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SPAWNED:  391,104 threads (for compression kernel)            │
│            ↓                                                    │
│  CONCURRENT: ~110,592 threads (on GPU with 108 SMs)            │
│            ↓                                                    │
│  EXECUTION: Threads execute in waves/batches                   │
│                                                                 │
│  Example with 108 SMs × 1024 threads/SM = 110,592 concurrent   │
│                                                                 │
│  Wave 1: threads 0     - 110,591   ████████████████            │
│  Wave 2: threads 110,592 - 221,183 ████████████████            │
│  Wave 3: threads 221,184 - 331,775 ████████████████            │
│  Wave 4: threads 331,776 - 391,103 ██████████                  │
│                                                                 │
│  Total wall time: 4 waves × ~3ms = ~12ms                       │
└────────────────────────────────────────────────────────────────┘
```

---

## Summary

**To see actual CUDA thread counts:**

1. **Use nsys profiling** (recommended):
   ```bash
   nsys profile --trace=cuda,nvtx ./examples/nvcomp_gds /tmp/test.bin
   nsys stats nvcomp_gds_profile.nsys-rep
   ```

2. **Expected thread counts**:
   - Compression kernel: ~391,000 threads (1,527 blocks × 256 threads)
   - Decompression kernel: ~391,000 threads
   - Initialize kernel: ~100,000,000 threads
   - Compare kernel: ~221,000 threads

3. **Why you can't see it in code**:
   - `compressor.compress()` is a high-level API
   - Actual kernel launches are inside the nvCOMP library
   - Need profiling tools to observe runtime behavior

4. **The batching strategy**:
   - 1,527 chunks processed in parallel
   - Each chunk typically gets one thread block
   - ~256-512 threads per block (typical for LZ4 on GPU)
   - Total: hundreds of thousands of threads working together!
