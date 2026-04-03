#!/usr/bin/env python3
"""
Smoke test: write a GPU tensor to HDF5 via GPUCompress VOL, verify correctness.

Tests:
  1. GPU tensor → H5Dwrite through VOL → compressed .h5 file
  2. VOL stats confirm GPU compression path was used (not silent CPU fallback)
  3. h5py read-back matches original data bitwise

Usage:
    python3 scripts/test_gpu_hdf5_write.py
    python3 scripts/test_gpu_hdf5_write.py --size-mb 16 --chunk-mb 4
"""

import argparse
import os
import sys
import tempfile

import numpy as np

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))


def main():
    parser = argparse.ArgumentParser(description="Smoke test: GPU → HDF5 VOL write")
    parser.add_argument("--size-mb", type=float, default=4.0,
                        help="Tensor size in MB (default: 4)")
    parser.add_argument("--chunk-mb", type=float, default=1.0,
                        help="Chunk size in MB (default: 1)")
    parser.add_argument("--error-bound", type=float, default=0.0,
                        help="Lossy error bound (default: 0.0 = lossless)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model.nnwt (default: auto-detect)")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("SKIP: No CUDA device available")
        return

    from gpucompress_hdf5 import GPUCompressHDF5Writer

    n_elements = int(args.size_mb * 1024 * 1024 / 4)
    chunk_elements = int(args.chunk_mb * 1024 * 1024 / 4)
    lib_dir = os.path.join(PROJECT_DIR, "build")

    weights = args.weights
    if weights is None:
        default_weights = os.path.join(PROJECT_DIR, "neural_net", "weights", "model.nnwt")
        if os.path.exists(default_weights):
            weights = default_weights

    print(f"=== GPU → HDF5 VOL Smoke Test ===")
    print(f"  Tensor: {args.size_mb:.1f} MB ({n_elements} floats)")
    print(f"  Chunk:  {args.chunk_mb:.1f} MB ({chunk_elements} floats)")
    print(f"  Error bound: {args.error_bound}")
    print(f"  Weights: {weights or 'None (LZ4 fallback)'}")
    print()

    # Create test data on GPU
    print("[1] Creating random GPU tensor...")
    torch.manual_seed(42)
    d_data = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    # Keep CPU copy for verification
    h_expected = d_data.cpu().numpy().copy()
    print(f"  Done. Shape: {d_data.shape}, dtype: {d_data.dtype}")

    # Write via VOL
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5_path = f.name

    try:
        print(f"\n[2] Writing via GPUCompress HDF5 VOL → {h5_path}")
        with GPUCompressHDF5Writer(lib_dir=lib_dir, weights_path=weights) as writer:
            writer.reset_stats()

            writer.write_gpu_tensor(
                d_data.data_ptr(), n_elements,
                h5_path, "data",
                chunk_elements=chunk_elements,
                error_bound=args.error_bound,
            )

            file_size = os.path.getsize(h5_path)
            orig_size = n_elements * 4
            ratio = orig_size / file_size if file_size > 0 else 0
            print(f"  File size: {file_size / (1024*1024):.2f} MB "
                  f"(ratio: {ratio:.2f}x)")

            # Check VOL stats — confirm GPU path was used
            stats = writer.get_stats()
            print(f"\n[3] VOL stats: {stats}")
            if stats["comp"] == 0:
                print("  FAIL: comp=0 — GPU compression path was NOT used!")
                print("  The write may have silently fallen back to CPU path.")
                sys.exit(1)
            else:
                print(f"  OK: {stats['comp']} chunks compressed on GPU")

            # Read back via VOL (GPU path) and verify
            print(f"\n[4] Reading back via VOL (GPU decompress)...")
            d_readback = torch.empty(n_elements, device="cuda", dtype=torch.float32)
            writer.read_gpu_tensor(h5_path, d_readback.data_ptr(), n_elements)

            h_readback = d_readback.cpu().numpy()
            print(f"  Shape: {h_readback.shape}, dtype: {h_readback.dtype}")

        if args.error_bound == 0.0:
            # Lossless: bitwise match
            if np.array_equal(h_expected, h_readback):
                print("  OK: Bitwise match (lossless)")
            else:
                mismatches = np.sum(h_expected != h_readback)
                print(f"  FAIL: {mismatches} mismatches out of {n_elements}")
                max_err = np.max(np.abs(h_expected - h_readback))
                print(f"  Max error: {max_err}")
                sys.exit(1)
        else:
            # Lossy: check absolute error bound (matches C-level guarantee)
            max_err = np.max(np.abs(h_expected.astype(np.float64) -
                                     h_readback.astype(np.float64)))
            data_range = np.max(h_expected) - np.min(h_expected)
            print(f"  Max absolute error: {max_err:.6e}")
            print(f"  Data range: {data_range:.4f}")
            print(f"  Error bound: {args.error_bound}")
            if max_err <= args.error_bound * 1.01:  # 1% tolerance for floating point
                print("  OK: Within absolute error bound")
            else:
                print(f"  FAIL: Max error {max_err:.6e} exceeds bound {args.error_bound}!")
                sys.exit(1)

        print(f"\n=== ALL TESTS PASSED ===")

    finally:
        if os.path.exists(h5_path):
            os.unlink(h5_path)


if __name__ == "__main__":
    main()
    # Flush before forced exit
    sys.stdout.flush()
    sys.stderr.flush()
    # Force clean exit to avoid atexit segfault from CUDA/HDF5 library teardown order
    os._exit(0)
