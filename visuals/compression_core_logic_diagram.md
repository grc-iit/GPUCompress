# Core Compression Logic - Deep Dive

## LZ4Manager Internal Architecture

### Reference: nvcomp_gds.cu Implementation

```
Line 114-127: COMPRESSOR CONFIGURATION
═══════════════════════════════════════════════════════════════════
constexpr size_t chunk_size = 1 << 16;  // 65,536 bytes (64 KB)

LZ4Manager compressor(
    chunk_size,                                      // Chunk size
    nvcompBatchedLZ4CompressDefaultOpts,            // Compression options
    nvcompBatchedLZ4DecompressDefaultOpts,          // Decompression options  
    stream                                           // CUDA stream
);

const CompressionConfig comp_config = compressor.configure_compression(n);
size_t lcompbuf = comp_config.max_compressed_buffer_size;
```

---

## Data Flow Through Compression Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    INPUT DATA TRANSFORMATION                             │
└─────────────────────────────────────────────────────────────────────────┘

STEP 1: INPUT DATA (Line 107-108)
───────────────────────────────────────────────────────────────────────────
   100 MB Contiguous Buffer (d_input)
   ┌────────────────────────────────────────────────────────────────────┐
   │ 0x00 0x01 0x02 ... 0xFF 0x00 0x01 ... (repeating pattern)        │
   └────────────────────────────────────────────────────────────────────┘
   
   Kernel: initialize<<<(n-1)/512 + 1, 512, 0, stream>>>(d_input, n)
           ▪ Each thread writes: data[i] = i & 0xff
           ▪ Creates sequential byte pattern
           ▪ Highly compressible (repeating 0-255 pattern)


STEP 2: CHUNKING (Implicit in LZ4Manager - Line 114)
───────────────────────────────────────────────────────────────────────────
   Logical Division into 64KB chunks:
   
   ┌─────────────────────────────────────────────────────────────────────┐
   │  Chunk 0         Chunk 1         Chunk 2               Chunk 1526   │
   │ ┌────────────┐ ┌────────────┐ ┌────────────┐        ┌──────────┐   │
   │ │ 64 KB      │ │ 64 KB      │ │ 64 KB      │  ....  │ 16.6 KB  │   │
   │ │ 0x00-0xFF  │ │ 0x00-0xFF  │ │ 0x00-0xFF  │        │ 0x00-0xC0│   │
   │ │ (repeats   │ │ (repeats   │ │ (repeats   │        │ (partial)│   │
   │ │  256 times)│ │  256 times)│ │  256 times)│        │          │   │
   │ └────────────┘ └────────────┘ └────────────┘        └──────────┘   │
   └─────────────────────────────────────────────────────────────────────┘
   
   Number of chunks = ceil(100,000,000 / 65,536) = 1,527 chunks
   
   Internal Metadata (managed by LZ4Manager):
   ┌──────────────────────────────────────────────────────┐
   │ void** chunk_ptrs[1527]     // Pointers to chunks    │
   │ size_t chunk_sizes[1527]    // Size of each chunk    │
   │ size_t compressed_sizes[1527] // Output sizes        │
   └──────────────────────────────────────────────────────┘


STEP 3: BATCH COMPRESSION (Line 167-168)
───────────────────────────────────────────────────────────────────────────
   compressor.compress(d_input, d_compressed, comp_config);
   
   ┌─────────────────────────────────────────────────────────────────────┐
   │              PARALLEL GPU COMPRESSION (1,527 chunks)                 │
   │                                                                      │
   │  GPU Thread Blocks: Each block handles one or more chunks           │
   │                                                                      │
   │  For Each Chunk i in [0, 1526]:                                     │
   │  ┌────────────────────────────────────────────────────────────────┐ │
   │  │  Thread Block i:                                               │ │
   │  │                                                                │ │
   │  │  1. Load chunk from d_input[i * chunk_size]                   │ │
   │  │                                                                │ │
   │  │  2. LZ4 Compression Algorithm:                                │ │
   │  │     a) Find repeating sequences (pattern matching)            │ │
   │  │     b) Replace with (offset, length) pairs                    │ │
   │  │     c) Encode literals and matches                            │ │
   │  │                                                                │ │
   │  │     Example for our sequential data:                          │ │
   │  │     Input:  0x00 0x01 0x02 ... 0xFF (repeated 256 times)     │ │
   │  │     Output: [Header] + [256 byte pattern] + [repeat count]   │ │
   │  │                                                                │ │
   │  │  3. Write compressed chunk to d_compressed                    │ │
   │  │                                                                │ │
   │  │  4. Record actual compressed size → compressed_sizes[i]       │ │
   │  └────────────────────────────────────────────────────────────────┘ │
   │                                                                      │
   │  All chunks processed in PARALLEL using GPU's thousands of cores    │
   └─────────────────────────────────────────────────────────────────────┘


STEP 4: COMPRESSED OUTPUT (Line 168)
───────────────────────────────────────────────────────────────────────────
   const size_t compressed_size = compressor.get_compressed_output_size();
   
   Compressed Buffer Layout:
   ┌─────────────────────────────────────────────────────────────────────┐
   │ [Metadata Header]                                                    │
   │  - Number of chunks: 1,527                                          │
   │  - Chunk offsets array                                              │
   │  - Chunk sizes array                                                │
   │                                                                      │
   │ [Compressed Chunk 0] (small, pattern detected)                      │
   │ [Compressed Chunk 1] (small, pattern detected)                      │
   │ [Compressed Chunk 2] (small, pattern detected)                      │
   │ ...                                                                  │
   │ [Compressed Chunk 1526] (small, pattern detected)                   │
   └─────────────────────────────────────────────────────────────────────┘
   
   Total size << 100 MB due to highly repetitive pattern


STEP 5: 4KB ALIGNMENT (Line 124, 171)
───────────────────────────────────────────────────────────────────────────
   lcompbuf = ((lcompbuf - 1) / 4096 + 1) * 4096;
   aligned_compressed_size = ((compressed_size - 1) / 4096 + 1) * 4096;
   
   ┌─────────────────────────────────────────────────────────────────────┐
   │ Compressed Data      │ Padding (zeros)              │               │
   │ (actual size)        │ to next 4KB boundary         │               │
   ├──────────────────────┴──────────────────────────────┤               │
   │              4KB aligned boundary                   │               │
   └─────────────────────────────────────────────────────────────────────┘
   
   Required for optimal GDS performance:
   • Unaligned I/O uses intermediate buffer (slower)
   • Aligned I/O enables direct DMA transfer (faster)
```

---

## LZ4 Compression Algorithm Details

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LZ4 COMPRESSION ALGORITHM                            │
│                   (Applied to each 64KB chunk)                          │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: One 64KB chunk with sequential bytes
───────────────────────────────────────────────────────────────────────────
Example first 1KB of chunk:
0x00 0x01 0x02 0x03 ... 0xFE 0xFF 0x00 0x01 0x02 ... 0xFE 0xFF 0x00 0x01 ...
└─────── pattern ──────┘      └─────── repeats ──────┘      └─ repeats...


ALGORITHM STEPS:
───────────────────────────────────────────────────────────────────────────
1. PATTERN DETECTION (Hash Table):
   • Build hash table of 4-byte sequences
   • Example: hash(0x00 0x01 0x02 0x03) → position
   • Scan input, lookup each 4-byte sequence
   • Find matches (same sequence seen before)

2. MATCH ENCODING:
   When pattern found:
   • Encode as: (offset, length)
   • offset: distance back to previous occurrence
   • length: how many bytes match
   
   Example:
   Position 256: 0x00 0x01 0x02 ... 0xFF (256 bytes)
   Matches Position 0!
   Encode as: offset=256, length=256
   
3. LITERAL ENCODING:
   When no match:
   • Encode byte(s) as-is (literal)
   • Use length prefix + bytes
   
4. OUTPUT FORMAT:
   [Token] [Literal Length] [Literals] [Match Offset] [Match Length]
   
   For our sequential pattern:
   • First 256 bytes: encoded as literals
   • Next 256 bytes: encoded as match (offset=256, len=256)
   • Remaining: more matches to first pattern
   
   HUGE COMPRESSION! 64KB → ~500 bytes


DECOMPRESSION (Reverse Process):
───────────────────────────────────────────────────────────────────────────
Read compressed stream:
1. Parse token
2. If literal: copy N bytes to output
3. If match: copy N bytes from output[current - offset]
4. Repeat until chunk complete

Fast because:
• No complex decoding (just copy operations)
• GPU parallel: each chunk decompressed by separate thread blocks
```

---

## Batch Processing Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│               WHY BATCH PROCESSING? (Key Design Decision)                │
└─────────────────────────────────────────────────────────────────────────┘

ALTERNATIVE 1: Single Large Compression
───────────────────────────────────────────────────────────────────────────
❌ Process entire 100 MB as one unit
   Problems:
   • Limited parallelism (1 compression job)
   • Large memory requirements for history buffer
   • Error in one byte corrupts entire file
   • Cannot seek/decompress specific portions
   • Worse GPU utilization


CHOSEN: Batch/Chunk Processing (64KB chunks)
───────────────────────────────────────────────────────────────────────────
✅ Process 1,527 chunks independently
   Benefits:
   • MASSIVE parallelism (1,527 parallel jobs)
   • Each chunk fits in GPU cache/shared memory
   • Error isolation (corrupt chunk doesn't affect others)
   • Random access (decompress specific chunks)
   • Optimal GPU utilization (thousands of cores busy)
   • Smaller memory footprint per compression unit


GPU THREAD MAPPING:
───────────────────────────────────────────────────────────────────────────
   Grid Layout (conceptual):
   
   ┌────────────────────────────────────────────────────────────────────┐
   │  Block 0      Block 1      Block 2            Block 1526           │
   │  ┌─────┐     ┌─────┐     ┌─────┐            ┌─────┐              │
   │  │ ... │     │ ... │     │ ... │     ....   │ ... │              │
   │  │threads│   │threads│   │threads│           │threads│            │
   │  │ ... │     │ ... │     │ ... │            │ ... │              │
   │  └─────┘     └─────┘     └─────┘            └─────┘              │
   │    │           │           │                   │                  │
   │    ▼           ▼           ▼                   ▼                  │
   │  Chunk 0    Chunk 1    Chunk 2              Chunk 1526           │
   └────────────────────────────────────────────────────────────────────┘
   
   Each block:
   • 256-1024 threads (typical)
   • Cooperate to compress one chunk
   • Use shared memory for hash table
   • Write results to global memory
   
   All blocks execute in parallel (GPU has many SMs)
```

---

## Memory Access Patterns

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY & ACCESS                             │
└─────────────────────────────────────────────────────────────────────────┘

COMPRESSION PHASE:
───────────────────────────────────────────────────────────────────────────
   Global Memory (GPU DRAM):
   ┌──────────────────────────┐
   │ d_input (100 MB)         │ ◄──┐
   │  - Read by all blocks    │    │ Coalesced reads
   │  - Each block reads its  │    │ (sequential access)
   │    assigned chunk        │    │
   └──────────────────────────┘    │
                                   │
   Shared Memory (On-chip):        │
   ┌──────────────────────────┐    │
   │ Per-block scratch        │ ◄──┘ Copy chunk here
   │  - Hash table            │    (fast access)
   │  - Working buffers       │
   │  - 48-96 KB per SM       │
   └──────────────────────────┘
          │
          │ Process chunk
          │ (LZ4 algorithm)
          ▼
   Global Memory (GPU DRAM):
   ┌──────────────────────────┐
   │ d_compressed             │ ◄── Write compressed output
   │  - Written by blocks     │    (scattered writes)
   │  - Variable size/chunk   │
   └──────────────────────────┘


GDS TRANSFER:
───────────────────────────────────────────────────────────────────────────
   GPU Memory              PCIe/NVLink              NVMe SSD
   ┌──────────────┐       ═══════════════▶        ┌──────────┐
   │ d_compressed │       DMA Transfer             │   File   │
   │ (GPU DRAM)   │       (no CPU copy)            │  System  │
   └──────────────┘                                └──────────┘
   
   Key: Buffer registered with GDS
        • Direct BAR (Base Address Register) mapping
        • GPU memory visible to storage controller
        • Zero-copy I/O
```

---

## Performance Characteristics

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PERFORMANCE METRICS                              │
└─────────────────────────────────────────────────────────────────────────┘

COMPRESSION SPEED:
───────────────────────────────────────────────────────────────────────────
   • 100 MB input
   • Sequential pattern (highly compressible)
   • GPU: ~10-50 GB/s compression throughput (depends on GPU)
   • Expected time: 2-10 ms
   • Compression ratio: ~300:1 for this pattern

FACTORS:
   ✓ GPU model (SM count, memory bandwidth)
   ✓ Data pattern (sequential = highly compressible)
   ✓ Chunk size (64KB = optimal for LZ4 on GPU)
   ✓ Batch size (1,527 chunks = good GPU saturation)


GDS I/O SPEED:
───────────────────────────────────────────────────────────────────────────
   • NVMe sequential read/write: ~7 GB/s (PCIe Gen4 x4)
   • GDS write time: compressed_size / 7 GB/s
   • GDS read time: compressed_size / 7 GB/s
   • No CPU overhead (DMA transfer)

TRADITIONAL I/O (without GDS):
   GPU → CPU → Page Cache → NVMe
   • 2x PCIe transfer (GPU↔CPU, CPU↔NVMe)
   • CPU memcpy overhead
   • ~2-3x slower than GDS


TOTAL PIPELINE:
───────────────────────────────────────────────────────────────────────────
   Initialize:        ~1 ms
   Compress:        ~5-10 ms
   GDS Write:       ~1-5 ms (depends on compressed size)
   GDS Read:        ~1-5 ms
   Decompress:      ~3-8 ms (faster than compression)
   Compare:         ~1-2 ms
   ─────────────────────────
   Total:          ~12-31 ms
```

---

## Code References

### Key Functions Called:

```cpp
// Line 117: Create LZ4 Manager
LZ4Manager compressor(chunk_size, compress_opts, decompress_opts, stream);

// Line 118: Configure compression (calculates buffer sizes)
const CompressionConfig comp_config = compressor.configure_compression(n);

// Line 167: Compress data (launches GPU kernels)
compressor.compress(d_input, d_compressed, comp_config);

// Line 168: Get actual compressed size
const size_t compressed_size = compressor.get_compressed_output_size(d_compressed);

// Line 183: Write to file with GDS
cuFileWrite(cf_handle, d_compressed, aligned_compressed_size, 0, 0);

// Line 200: Read from file with GDS
cuFileRead(cf_handle, d_compressed, aligned_compressed_size, 0, 0);

// Line 213-214: Configure decompression
const DecompressionConfig decomp_config = 
    compressor.configure_decompression(comp_config);

// Line 231: Decompress data (launches GPU kernels)
compressor.decompress(d_output, d_compressed, decomp_config);

// Line 233: Verify correctness
compare<<<2*smcount, 1024, 0, stream>>>(d_input, d_output, dh_invalid, n);
```

### Supporting Infrastructure:

```cpp
// Line 53-58: Initialize input data
__global__ void initialize(uint8_t* data, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    data[i] = i & 0xff;  // Sequential pattern
}

// Line 61-71: Compare decompressed with original
__global__ void compare(const uint8_t* ref, const uint8_t* val, 
                        int* invalid, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  while (i < n) {
    if (ref[i] != val[i])
      *invalid = 1;
    i += stride;
  }
}
```

---

## Summary

**DATA ORGANIZATION**: Batch of 1,527 chunks (not single large block)  
**CHUNK SIZE**: 64 KB (optimal for LZ4 GPU implementation)  
**PARALLELISM**: All chunks compressed/decompressed simultaneously on GPU  
**I/O METHOD**: GPU Direct Storage (zero-copy GPU↔NVMe transfers)  
**ALIGNMENT**: 4KB for optimal GDS performance  
**COMPRESSION**: LZ4 algorithm (fast, pattern-based, moderate ratio)  
**BENEFIT**: Massive speedup from GPU parallelism + direct storage access
