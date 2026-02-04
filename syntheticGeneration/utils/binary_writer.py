"""
Binary file I/O operations for GPUCompress compatibility.

Provides functions to write and read raw binary float32 data files
without headers, matching the GPUCompress input format.
"""

import numpy as np
import os
from typing import Tuple


def write_binary(filename: str, data: np.ndarray) -> str:
    """
    Write numpy array as raw binary float32 bytes.

    Args:
        filename: Output binary filename (typically .bin extension)
        data: Data array to write (will be converted to float32 and flattened)

    Returns:
        Path to created file

    Raises:
        ValueError: If data is not a numpy array
        IOError: If file write fails

    Example:
        >>> data = np.random.randn(1000, 1000).astype(np.float32)
        >>> write_binary('output.bin', data)
        'output.bin'
        >>> # File size will be 4,000,000 bytes (1000*1000*4)
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array")

    # Ensure data is float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    try:
        # Flatten and write raw bytes (no header)
        data.flatten().tofile(filename)
        return filename

    except Exception as e:
        raise IOError(f"Failed to write binary file {filename}: {e}")


def read_binary(filename: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Read raw binary file back as numpy array.

    Args:
        filename: Input binary filename
        dtype: Data type to interpret bytes as (default: float32)

    Returns:
        1D numpy array of the specified dtype

    Raises:
        IOError: If file read fails
        FileNotFoundError: If file doesn't exist

    Example:
        >>> data = read_binary('output.bin')
        >>> data.shape
        (1000000,)  # 1D array, reshape as needed
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Binary file not found: {filename}")

    try:
        data = np.fromfile(filename, dtype=dtype)
        return data

    except Exception as e:
        raise IOError(f"Failed to read binary file {filename}: {e}")


def get_binary_info(filename: str, dtype: np.dtype = np.float32) -> dict:
    """
    Get information about a binary file.

    Args:
        filename: Binary filename to inspect
        dtype: Expected data type (default: float32)

    Returns:
        Dictionary with file information:
        - file_size: Size in bytes
        - element_count: Number of elements
        - dtype: Data type string
        - dtype_size: Bytes per element

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Binary file not found: {filename}")

    file_size = os.path.getsize(filename)
    dtype_size = np.dtype(dtype).itemsize
    element_count = file_size // dtype_size

    return {
        'file_size': file_size,
        'element_count': element_count,
        'dtype': str(dtype),
        'dtype_size': dtype_size
    }


def print_binary_info(filename: str, dtype: np.dtype = np.float32):
    """
    Print information about a binary file.

    Args:
        filename: Binary filename to inspect
        dtype: Expected data type (default: float32)
    """
    try:
        info = get_binary_info(filename, dtype)

        print(f"\n{'=' * 70}")
        print(f"Binary File: {filename}")
        print(f"{'=' * 70}")

        # Format file size
        file_size = info['file_size']
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024**2:
            size_str = f"{file_size / 1024:.2f} KB"
        elif file_size < 1024**3:
            size_str = f"{file_size / (1024**2):.2f} MB"
        else:
            size_str = f"{file_size / (1024**3):.2f} GB"

        print(f"\nFile Statistics:")
        print(f"  File size:     {size_str} ({file_size:,} bytes)")
        print(f"  Element count: {info['element_count']:,}")
        print(f"  Data type:     {info['dtype']}")
        print(f"  Bytes/element: {info['dtype_size']}")

        # Verify file size is aligned to dtype
        remainder = file_size % info['dtype_size']
        if remainder != 0:
            print(f"\n  WARNING: File size not aligned to {info['dtype']} "
                  f"(remainder: {remainder} bytes)")

        print(f"{'=' * 70}\n")

    except Exception as e:
        print(f"Error reading binary file: {e}")
