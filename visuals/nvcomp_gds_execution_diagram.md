# nvcomp_gds.cu - Complete Execution Flow Diagram

## Overview
This program demonstrates GPU Direct Storage (GDS) with nvCOMP LZ4 compression. The data is compressed as **BATCHES OF CHUNKS**, not as a single large memory space.

---

## Program Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MAIN EXECUTION FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

PHASE 1: INITIALIZATION
═══════════════════════════════════════════════════════════════════════
┌─────────────────┐
│  Open File      │  O_DIRECT flag (required for GDS)
│  with O_DIRECT  │  Creates/truncates file
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Allocate GPU    │  • d_input:      100 MB (uncompressed)
│ Memory Buffers  │  • d_output:     100 MB (decompressed result)
└────────┬────────┘  • d_compressed: Padded to 4KB alignment
         │
         ▼
┌─────────────────┐
│  Initialize     │  CUDA Kernel: initialize<<<>>>
│  Input Data     │  Sequential bytes: data[i] = i & 0xff
│  (Sequential)   │  Pattern: 0x00, 0x01, 0x02, ..., 0xFF, 0x00, ...
└────────┬────────┘
         │
         ▼


PHASE 2: COMPRESSION SETUP & BATCHING STRATEGY
═══════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────┐
│                   BATCHING CONFIGURATION                              │
│                                                                       │
│  chunk_size = 65,536 bytes (64 KB)                                   │
│  Total data = 100,000,000 bytes (100 MB)                             │
│                                                                       │
│  Number of chunks = 100,000,000 / 65,536 = 1,526 chunks              │
│                     (+ 1 partial chunk for remainder)                │
│                                                                       │
│  Total batch size = 1,527 chunks                                     │
└──────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────────────────────────────────────────────────────────┐
│               DATA ORGANIZATION IN MEMORY                            │
│                                                                      │
│  100 MB Input Buffer (d_input)                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Chunk 0  │ Chunk 1  │ Chunk 2  │ ... │ Chunk 1526         │    │
│  │ (64 KB)  │ (64 KB)  │ (64 KB)  │     │ (16,576 bytes)     │    │
│  └────────────────────────────────────────────────────────────┘    │
│     ▲          ▲          ▲                ▲                        │
│     │          │          │                │                        │
│  ptr[0]    ptr[1]    ptr[2]            ptr[1526]                    │
│                                                                      │
│  Batch Metadata (managed internally by LZ4Manager):                 │
│  • Array of pointers to each chunk                                  │
│  • Array of sizes for each chunk                                    │
│  • Compression state for each chunk                                 │
└─────────────────────────────────────────────────────────────────────┘

         │
         ▼

┌─────────────────┐
│  LZ4Manager     │  Constructor parameters:
│  Creation       │  • chunk_size: 65,536 bytes
│                 │  • Compression options: default
└────────┬────────┘  • Decompression options: default
         │            • Stream: async CUDA stream
         ▼
┌─────────────────┐
│  Configure      │  compressor.configure_compression(n)
│  Compression    │  Returns: CompressionConfig with:
│                 │  • max_compressed_buffer_size
└────────┬────────┘  • temp buffer requirements
         │
         ▼
┌─────────────────┐
│  Align Buffer   │  lcompbuf = ((lcompbuf - 1) / 4096 + 1) * 4096
│  to 4KB         │  Reason: Unaligned I/O uses extra memory copy
└────────┬────────┘  GDS requires 4KB alignment for best performance
         │
         ▼


PHASE 3: GDS SETUP
═══════════════════════════════════════════════════════════════════════
┌─────────────────┐
│  Initialize     │  cuFileDriverOpen()
│  cuFile Driver  │  Opens GDS subsystem
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Register File  │  cuFileHandleRegister()
│  Handle with    │  Associates file descriptor with GDS
│  GDS            │  Enables direct GPU-to-NVMe transfers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Register       │  cuFileBufRegister(d_compressed, lcompbuf, 0)
│  Buffer (opt)   │  OPTIONAL but recommended for best performance
│                 │  • Avoids extra memory copies
└────────┬────────┘  • Requires sufficient BAR memory on GPU
         │            • Fallback to internal buffer if fails
         ▼


PHASE 4: COMPRESSION (BATCH PROCESSING)
═══════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────┐
│                    BATCH COMPRESSION PROCESS                          │
│                                                                       │
│  LZ4Manager.compress() internally performs:                          │
│                                                                       │
│  FOR EACH CHUNK (1,527 chunks in parallel on GPU):                   │
│    1. Read chunk from d_input                                        │
│    2. Apply LZ4 compression algorithm                                │
│    3. Write compressed data to d_compressed                          │
│    4. Record compressed size                                         │
│                                                                       │
│  Execution Model: GPU parallelism across chunks                      │
│  • Multiple CUDA threads per chunk                                   │
│  • Chunks processed in parallel (batch processing)                   │
│  • Metadata stored for each compressed chunk                         │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Compress Data  │  compressor.compress(d_input, d_compressed, config)
│  (Asynchronous) │  Launches CUDA kernels for batch compression
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Get Compressed │  compressor.get_compressed_output_size()
│  Size           │  Returns actual compressed size
└────────┬────────┘  (typically smaller than 100 MB for sequential data)
         │
         ▼
┌─────────────────┐
│  Align Size     │  aligned_compressed_size = ((size - 1) / 4096 + 1) * 4096
│  to 4KB         │  Pad with zeros to next 4KB boundary
└────────┬────────┘
         │
         ▼


PHASE 5: GDS WRITE (GPU → NVMe Direct)
═══════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────┐
│                   GPU DIRECT STORAGE WRITE                            │
│                                                                       │
│  ┌──────────┐                                     ┌──────────────┐   │
│  │   GPU    │ ═══════════════════════════════════▶│  NVMe SSD    │   │
│  │  Memory  │  Direct DMA Transfer (no CPU copy)  │   Storage    │   │
│  │          │ ◀═══════════════════════════════════│              │   │
│  │d_compress│                                     │   File       │   │
│  └──────────┘                                     └──────────────┘   │
│                                                                       │
│  cuFileWrite(handle, d_compressed, aligned_size, offset, 0)          │
│                                                                       │
│  Benefits:                                                            │
│  • Zero-copy transfer (CPU not involved)                             │
│  • Higher bandwidth                                                   │
│  • Lower latency                                                      │
│  • Reduced CPU overhead                                              │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Write to File  │  cuFileWrite(cf_handle, d_compressed, aligned_size, 0, 0)
│  via GDS        │  Writes directly from GPU memory to NVMe
└────────┬────────┘  No CPU involvement in data transfer
         │
         ▼


PHASE 6: CLEANUP & READ PREPARATION
═══════════════════════════════════════════════════════════════════════
┌─────────────────┐
│  Clear          │  cudaMemsetAsync(d_compressed, 0xff, size, stream)
│  Compressed     │  Erases buffer to verify read operation
│  Buffer         │  (All bytes set to 0xFF)
└────────┬────────┘
         │
         ▼


PHASE 7: GDS READ (NVMe → GPU Direct)
═══════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────┐
│                   GPU DIRECT STORAGE READ                             │
│                                                                       │
│  ┌──────────┐                                     ┌──────────────┐   │
│  │   GPU    │ ◀═══════════════════════════════════│  NVMe SSD    │   │
│  │  Memory  │  Direct DMA Transfer (no CPU copy)  │   Storage    │   │
│  │          │                                     │              │   │
│  │d_compress│                                     │   File       │   │
│  └──────────┘                                     └──────────────┘   │
│                                                                       │
│  cuFileRead(handle, d_compressed, aligned_size, offset, 0)           │
│                                                                       │
│  Reads compressed data back from file directly into GPU memory       │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Read from File │  cuFileRead(cf_handle, d_compressed, aligned_size, 0, 0)
│  via GDS        │  Reads directly into GPU memory from NVMe
└────────┬────────┘
         │
         ▼


PHASE 8: DECOMPRESSION (BATCH PROCESSING)
═══════════════════════════════════════════════════════════════════════

┌─────────────────┐
│  Configure      │  compressor.configure_decompression(comp_config)
│  Decompression  │  Returns: DecompressionConfig
└────────┬────────┘  • decomp_data_size (should equal original 100 MB)
         │            • temp buffer requirements
         ▼

┌──────────────────────────────────────────────────────────────────────┐
│                   BATCH DECOMPRESSION PROCESS                         │
│                                                                       │
│  LZ4Manager.decompress() internally performs:                        │
│                                                                       │
│  FOR EACH CHUNK (1,527 chunks in parallel on GPU):                   │
│    1. Read compressed chunk from d_compressed                        │
│    2. Read chunk metadata (size, compression params)                 │
│    3. Apply LZ4 decompression algorithm                              │
│    4. Write decompressed data to d_output                            │
│                                                                       │
│  Execution Model: GPU parallelism across chunks                      │
│  • Multiple CUDA threads per chunk                                   │
│  • Chunks decompressed in parallel                                   │
│  • Results written to contiguous output buffer                       │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Decompress     │  compressor.decompress(d_output, d_compressed, decomp_config)
│  Data           │  Launches CUDA kernels for batch decompression
│  (Asynchronous) │  Restores original 100 MB data
└────────┬────────┘
         │
         ▼


PHASE 9: VERIFICATION
═══════════════════════════════════════════════════════════════════════

┌─────────────────┐
│  Compare        │  CUDA Kernel: compare<<<2*smcount, 1024, 0, stream>>>
│  Buffers        │  Compares d_input vs d_output byte-by-byte
│                 │  Sets *dh_invalid = 1 if any mismatch found
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Synchronize    │  cudaStreamSynchronize(stream)
│  Stream         │  Wait for comparison to complete
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Check Result   │  if (*dh_invalid) → FAILED
│                 │  else → PASSED
└────────┬────────┘
         │
         ▼


PHASE 10: CLEANUP
═══════════════════════════════════════════════════════════════════════
┌─────────────────┐
│  Deregister     │  cuFileBufDeregister(d_compressed)
│  Buffer         │  Releases BAR memory mapping
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Close File     │  close(fd)
│  & GDS Driver   │  cuFileDriverClose()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deallocate     │  compressor.deallocate_gpu_mem()
│  Compressor     │  Frees internal nvCOMP structures
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Free GPU       │  cudaFree(d_input, d_output, d_compressed)
│  Memory         │  cudaFreeHost(dh_invalid)
└────────┬────────┘  cudaStreamDestroy(stream)
         │
         ▼
┌─────────────────┐
│     DONE        │
└─────────────────┘


═══════════════════════════════════════════════════════════════════════
                    KEY INSIGHTS & DESIGN DECISIONS
═══════════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────────┐
│ 1. BATCHING STRATEGY                                                  │
│    ✓ Data is NOT compressed as single large block                    │
│    ✓ Data IS divided into 64KB chunks (batch processing)             │
│    ✓ Each chunk compressed independently                             │
│    ✓ Enables GPU parallelism (1,527 chunks in parallel)              │
│    ✓ Better error isolation (corrupt chunk doesn't affect others)    │
│    ✓ Random access possible (decompress specific chunks)             │
│                                                                       │
│ 2. MEMORY LAYOUT                                                      │
│    • Contiguous input buffer (100 MB)                                │
│    • Logically divided into chunks                                   │
│    • Chunk pointers managed by LZ4Manager                            │
│    • Compressed output also contiguous but variable per chunk        │
│                                                                       │
│ 3. GDS ALIGNMENT                                                      │
│    • 4KB alignment critical for performance                          │
│    • Unaligned I/O causes extra memory copy                          │
│    • Buffer registration optional but recommended                    │
│                                                                       │
│ 4. ASYNCHRONOUS EXECUTION                                             │
│    • All GPU operations use CUDA stream                              │
│    • Compression, decompression overlap with other work              │
│    • Synchronization only when checking results                      │
│                                                                       │
│ 5. COMPRESSION EFFICIENCY                                             │
│    • Sequential data (0x00-0xFF pattern) compresses very well        │
│    • LZ4 finds repeating patterns                                    │
│    • Actual compressed size << 100 MB                                │
└──────────────────────────────────────────────────────────────────────┘
